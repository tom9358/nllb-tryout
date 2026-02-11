#!/usr/bin/env python3
"""
train_gos_transfer.py
---------------------
Fine-tune an existing GPT-2 model (e.g. Dutch) for a new language (e.g. Gronings)
by retraining only the token embeddings and then lightly fine-tuning the full model.
Uses dictionary-based token mapping for smart weight initialization.

OPTIMIZATIONS:
- Saves token mapping checkpoint to avoid re-running on crash
- Multiprocessing support for faster token mapping (Windows-compatible)
- Can resume from saved embeddings

Usage:

  # First run (full training):
  python train_gos_transfer.py --base_model GroNLP/gpt2-small-dutch --text kreuzeandtatoeba.txt --dict dict_clean.tsv --output_dir gos-transfer-gpt2 --epochs 3

  # Resume from checkpoint (if training crashed):
  python train_gos_transfer.py --base_model GroNLP/gpt2-small-dutch --text kreuzeandtatoeba.txt --dict dict_clean.tsv --output_dir gos-transfer-gpt2 --epochs 3 --resume_from_checkpoint

  # Reduce vocab size for faster token mapping:
  python train_gos_transfer.py --base_model GroNLP/gpt2-small-dutch --text kreuzeandtatoeba.txt --dict dict_clean.tsv --output_dir gos-transfer-gpt2 --epochs 3 --vocab_size 20000

  # Control number of CPU workers:
  python train_gos_transfer.py --base_model GroNLP/gpt2-small-dutch --text kreuzeandtatoeba.txt --dict dict_clean.tsv --output_dir gos-transfer-gpt2 --epochs 3 --num_workers 8

  # Test the trained model only:
  python test_gos_gpt2.py --model_dir gos-transfer-gpt2
"""
import argparse
from pathlib import Path
import torch
import pandas as pd
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from tqdm import tqdm
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count

# Helper function for parallel processing - MUST be at module level for Windows
def process_token_batch(args_tuple):
    tokens_batch, tokenizer_dutch_vocab, old_vocab_size, word_mapping = args_tuple
    batch_results = []
    
    for grn_token, grn_id in tokens_batch:
        clean_grn = grn_token.replace('Ġ', '').lower()
        
        # Check if this token exists in Dutch tokenizer
        if grn_token in tokenizer_dutch_vocab:
            dutch_id = tokenizer_dutch_vocab[grn_token]
            if dutch_id < old_vocab_size:
                batch_results.append((grn_id, dutch_id, 'direct'))
                continue
        
        # Try dictionary-based mapping
        if word_mapping:
            best_match = None
            best_score = 0
            
            for dutch_word, gronings_word in word_mapping.items():
                if clean_grn in gronings_word:
                    pos = gronings_word.find(clean_grn)
                    if pos != -1:
                        rel_pos = pos / len(gronings_word)
                        dutch_substr_start = int(rel_pos * len(dutch_word))
                        dutch_substr_end = dutch_substr_start + len(clean_grn)
                        
                        if dutch_substr_end <= len(dutch_word):
                            dutch_substr = dutch_word[dutch_substr_start:dutch_substr_end]
                            
                            for dutch_token, dutch_id in tokenizer_dutch_vocab.items():
                                clean_dutch = dutch_token.replace('Ġ', '').lower()
                                if dutch_substr in clean_dutch and dutch_id < old_vocab_size:
                                    score = len(dutch_substr) / max(len(clean_dutch), len(clean_grn))
                                    if score > best_score:
                                        best_match = dutch_id
                                        best_score = score
            
            if best_match is not None and best_score > 0.5:
                batch_results.append((grn_id, best_match, 'dict'))
    
    return batch_results

def main():
    # ============ Parse args ============
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Hugging Face model to reuse (e.g. GroNLP/gpt2-small-dutch)")
    parser.add_argument("--text", type=str, required=True, help="Path to corpus text file")
    parser.add_argument("--dict", type=str, help="Path to dictionary CSV (Dutch,Gronings columns)")
    parser.add_argument("--output_dir", type=str, default="grn-transfer-gpt2", help="Where to save model and tokenizer")
    parser.add_argument("--vocab_size", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--push_to_hub", action="store_true", help="Upload to Hugging Face after training")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from saved embedding checkpoint")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers for token mapping (default: CPU count)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    num_workers = args.num_workers or cpu_count()

    # ============ 1. Train a new tokenizer ============
    print("🔤 Training Gronings tokenizer...")
    tokenizer_grn = ByteLevelBPETokenizer()
    tokenizer_grn.train([args.text], vocab_size=args.vocab_size, min_frequency=2)
    tokenizer_grn.save_model(str(output_dir))

    # ============ 2. Load pretrained base model and tokenizers ============
    print(f"📦 Loading base model: {args.base_model}")
    config = AutoConfig.from_pretrained(args.base_model)
    model = GPT2LMHeadModel.from_pretrained(args.base_model)

    # Load both tokenizers
    tokenizer_dutch = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer_gronings = AutoTokenizer.from_pretrained(str(output_dir), use_fast=True)

    # Add padding token if missing
    if tokenizer_gronings.pad_token is None:
        tokenizer_gronings.pad_token = tokenizer_gronings.eos_token

    # ============ 3. Smart embedding initialization with dictionary ============
    print("🔁 Initializing embeddings with dictionary mapping...")

    # Get the old embedding layer
    old_emb = model.transformer.wte
    old_vocab_size, emb_dim = old_emb.weight.shape

    # Create new embedding layer
    new_vocab_size = len(tokenizer_gronings)
    new_emb = torch.nn.Embedding(new_vocab_size, emb_dim)

    # Check for saved checkpoint
    embedding_checkpoint = checkpoint_dir / "embedding_mapping.pt"
    if args.resume_from_checkpoint and embedding_checkpoint.exists():
        print(f"📥 Loading embedding checkpoint from {embedding_checkpoint}")
        checkpoint_data = torch.load(embedding_checkpoint)
        new_emb.weight.data = checkpoint_data['embedding_weights']
        mapped_tokens = checkpoint_data['mapped_tokens']
        direct_matches = checkpoint_data['direct_matches']
        subword_matches = checkpoint_data['subword_matches']
        print(f"✅ Loaded {len(mapped_tokens)} pre-mapped tokens from checkpoint")
        print(f"   - Direct matches: {direct_matches}")
        print(f"   - Dictionary-based matches: {subword_matches}")
        print(f"   - Random initialization: {new_vocab_size - len(mapped_tokens)}")
    else:
        # Initialize with random weights first
        torch.nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)

        # Dictionary-based mapping
        mapped_tokens = set()
        direct_matches = 0
        subword_matches = 0

        if args.dict:
            print("📖 Loading dictionary for token mapping...")
            if args.dict[-4:].lower() == ".tsv":
                print('tsv', end='')
                df = pd.read_csv(args.dict, header=None, names=['gronings', 'dutch'], sep='\t')
                print(' loaded')
            else:
                df = pd.read_csv(args.dict, header=None, names=['gronings', 'dutch'])
            
            # Create mapping from Dutch words to Gronings words
            word_mapping = {}
            for _, row in df.iterrows():
                if pd.notna(row['dutch']) and pd.notna(row['gronings']):
                    word_mapping[row['dutch'].strip().lower()] = row['gronings'].strip().lower()
            
            print(f"   Found {len(word_mapping)} word pairs in dictionary")
        else:
            word_mapping = {}

        # Prepare data for parallel processing
        vocab_items = list(tokenizer_gronings.get_vocab().items())
        batch_size = max(1, len(vocab_items) // num_workers)
        batches = [vocab_items[i:i+batch_size] for i in range(0, len(vocab_items), batch_size)]
        
        tokenizer_dutch_vocab = tokenizer_dutch.get_vocab()
        
        print(f"🚀 Mapping tokens using {num_workers} workers...")
        
        # Process in parallel
        pool_args = [(batch, tokenizer_dutch_vocab, old_vocab_size, word_mapping) for batch in batches]
        
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_token_batch, pool_args),
                total=len(batches),
                desc="Mapping tokens"
            ))
        
        # Aggregate results
        for batch_results in results:
            for grn_id, dutch_id, match_type in batch_results:
                new_emb.weight.data[grn_id] = old_emb.weight.data[dutch_id].clone()
                mapped_tokens.add(grn_id)
                if match_type == 'direct':
                    direct_matches += 1
                else:
                    subword_matches += 1

        print(f"✅ Initialized {len(mapped_tokens)} tokens from Dutch model")
        print(f"   - Direct matches: {direct_matches}")
        print(f"   - Dictionary-based matches: {subword_matches}")
        print(f"   - Random initialization: {new_vocab_size - len(mapped_tokens)}")
        
        # Save checkpoint
        print(f"💾 Saving embedding checkpoint to {embedding_checkpoint}")
        torch.save({
            'embedding_weights': new_emb.weight.data,
            'embedding_weights': new_emb.weight.data,
            'mapped_tokens': mapped_tokens,
            'direct_matches': direct_matches,
            'subword_matches': subword_matches,
        }, embedding_checkpoint)

    # Replace the embedding layer
    model.transformer.wte = new_emb
    model.resize_token_embeddings(new_vocab_size)

    # ============ 4. Freeze everything except embeddings ============
    for name, param in model.named_parameters():
        param.requires_grad = False

    for param in model.transformer.wte.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧊 Training only embeddings ({trainable_params:,}/{total_params:,} params ≈ {100*trainable_params/total_params:.2f}%)")

    # ============ 5. Prepare dataset ============
    print("📚 Loading dataset...")
    dataset = load_dataset("text", data_files={"train": args.text})

    def tokenize_function(examples):
        return tokenizer_gronings(
            examples["text"], 
            truncation=True,
            padding=False,
            max_length=args.block_size
        )

    tokenized = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

    def group_texts(examples):
        # Concatenate everything
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // args.block_size) * args.block_size
        # Split by block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_gronings,
        mlm=False,
    )

    # ============ 6a. Phase 1: Embedding-only training ============
    print("🧩 Phase 1: training embeddings only...")

    # Keep only embeddings trainable
    for name, param in model.named_parameters():
        param.requires_grad = False

    for param in model.transformer.wte.parameters():
        param.requires_grad = True

    training_args_phase1 = TrainingArguments(
        output_dir=str(output_dir / "phase1"),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=100,
        save_steps=500,
        eval_strategy="no",
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args_phase1,
        train_dataset=lm_dataset["train"],
        data_collator=data_collator,
    )

    # Train phase 1
    trainer.train()

    # ============ 6b. Phase 2: Light full fine-tuning ============
    print("⚖️ Phase 2: full-model equilibration...")

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    training_args_phase2 = TrainingArguments(
        output_dir=str(output_dir / "phase2"),
        num_train_epochs=args.epochs - 1,  # Remaining epochs
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,  # Much smaller learning rate
        warmup_steps=50,
        logging_steps=100,
        save_steps=500,
        eval_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args_phase2,
        train_dataset=lm_dataset["train"],
        data_collator=data_collator,
    )

    # Train phase 2
    trainer.train()

    # ============ 7. Save or push ============
    print("💾 Saving final model...")
    model.save_pretrained(output_dir)
    tokenizer_gronings.save_pretrained(output_dir)

    if args.push_to_hub:
        print("☁️ Uploading model to Hugging Face Hub...")
        model.push_to_hub(output_dir.name)
        tokenizer_gronings.push_to_hub(output_dir.name)

    print(f"\n🎉 Done. Model saved at: {output_dir}\n")

    # ============ 8. Example generation ============
    print("💬 Testing generation...")
    test_prompts = [
        "Waor komt gain",
        "Ik bin",
        "t Is n mooi"
    ]

    model.eval()
    device = next(model.parameters()).device  # Get model's device
    for prompt in test_prompts:
        inputs = tokenizer_gronings(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to model's device
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=50,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer_gronings.pad_token_id,
                eos_token_id=tokenizer_gronings.eos_token_id
            )
        print(f"Prompt: {prompt}")
        print(f"Generated: {tokenizer_gronings.decode(output[0], skip_special_tokens=True)}\n")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()