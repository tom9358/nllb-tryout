#!/usr/bin/env python3
"""
test_gos_gpt2.py
-----------------
Test a trained Gronings GPT-2 model with interactive generation.

Usage:
  # Test with default prompts:
  python test_gos_gpt2.py --model_dir gos-transfer-gpt2

  # Interactive mode:
  python test_gos_gpt2.py --model_dir gos-transfer-gpt2 --interactive

  # Custom prompts:
  python test_gos_gpt2.py --model_dir gos-transfer-gpt2 --prompts "Waor komt gain" "Ik bin" "Moarn is t"

  # Adjust generation parameters:
  python test_gos_gpt2.py --model_dir gos-transfer-gpt2 --max_length 100 --temperature 0.8 --top_p 0.9
"""

import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from pathlib import Path

def generate_text(model, tokenizer, prompt, device, max_length=50, temperature=0.9, top_p=0.95, num_return=1):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    results = []
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        results.append(text)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test a trained Gronings GPT-2 model")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--prompts", nargs="+", help="Custom prompts to test (space-separated)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode - enter prompts continuously")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--num_return", type=int, default=1, help="Number of generations per prompt")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"❌ Error: Model directory '{model_dir}' not found!")
        return
    
    print(f"📦 Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    print(f"🔤 Vocab size: {len(tokenizer)}")
    print()
    
    # Default test prompts
    default_prompts = [
        "Waor komt gain",
        "Ik bin",
        "t Is n mooi",
        "Doar hemmen wie",
        "Mor",
        "De lu",
    ]
    
    # Determine which prompts to use
    if args.interactive:
        print("🎮 Interactive mode - type prompts (Ctrl+C or 'quit' to exit)")
        print("=" * 60)
        try:
            while True:
                prompt = input("\n💬 Enter prompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue
                
                print(f"\n🔮 Generating from '{prompt}'...")
                results = generate_text(
                    model, tokenizer, prompt, device,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return=args.num_return
                )
                
                for i, result in enumerate(results, 1):
                    if args.num_return > 1:
                        print(f"\n  [{i}] {result}")
                    else:
                        print(f"\n  → {result}")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
    else:
        # Use custom prompts or defaults
        test_prompts = args.prompts if args.prompts else default_prompts
        
        print("💬 Testing generation...")
        print("=" * 60)
        
        for prompt in test_prompts:
            print(f"\n🔮 Prompt: '{prompt}'")
            results = generate_text(
                model, tokenizer, prompt, device,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return=args.num_return
            )
            
            for i, result in enumerate(results, 1):
                if args.num_return > 1:
                    print(f"  [{i}] {result}")
                else:
                    print(f"  → {result}")
        
        print("\n" + "=" * 60)
        print("✨ Testing complete!")

if __name__ == '__main__':
    main()