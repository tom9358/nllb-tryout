import re
import typing
import torch
from transformers import Adafactor, get_constant_schedule_with_warmup
import numpy as np
from .tokenizer_and_model_setup import setup_model_and_tokenizer, cleanup
from .config import config
import random
from sacremoses import MosesPunctNormalizer
import unicodedata
import sys
import pandas as pd
import matplotlib.pyplot as plt

def get_non_printing_char_replacer(replace_by: str = " ") -> typing.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }
    
    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)
    
    return replace_non_printing_char

def preproc(text: str):
    """Normalizes text a bit."""
    mpn = MosesPunctNormalizer(lang="en")
    mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]
    clean = mpn.normalize(text)
    clean = get_non_printing_char_replacer(" ")(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean

# List of synonym pairs
synonym_pairs_gos = [
    ('huus', 'hoes'), ('huzen', 'hoezen'), ('huuske', 'hoeske'), ('groag', 'geern'), ('raais', 'raaize'), ('kees', 'keze'), ('week', 'weke'),
    ('mìnsken', 'mìnsen'), ('uut', 'oet'), ('in', 'ien'), ('wer', 'wuir'), ('gebruuk', 'gebroek'), ('zuch', 'zok'), ('bruukst', 'broekst'), ('wind', 'wiend'),
    ('vanuut', 'vanoet'), ('wazzen', 'waren'), ('mekoar', 'nkander'), ('bruken', 'broeken'), ('zuch', 'zuk'), ('vis', 'visk'), ('olle', 'olde'),
    ('zuk', 'zok'), ('wotter', 'woater'), ('kraant', 'kraande'), ('haar', 'har'), ('bruuk', 'broek'), ('school', 'schoule'), ('iezer', 'iesder'),
    ('ais', 'ains'), ('hebben', 'hemmen'), ('zotterdag', 'zoaterdag'), ('bruukt', 'broekt'), ('bruukten', 'broekten'), ('iezern', 'iesdern'), ('kind', 'kiend'),
    ('mirreg', 'middag'), ('vast', 'vaast'), ('nacht', 'naacht'), ('kiender', 'kinder'), ('bruukte', 'broekte'), ('deus','deuze'), ('gelok', 'geluk')
]

def add_gronings_variations(sentences):
    # Gronings-specific removal of more or less optional diacritics
    if not random.getrandbits(2):
        sentences = [s.replace('ì', 'i').replace('è', 'e').replace('ò', 'o').replace('ó', 'o') for s in sentences]
    return sentences

def swap_synonyms(sentences, synonym_pairs, swap_prob=0.25):
    swapped_sentences = []

    for sent in sentences:
        words = sent.split()  # Split sentence into words
        new_words = []
        
        for word in words:
            replaced = False
            
            # Check each synonym pair
            for word1, word2 in synonym_pairs:
                if word == word1 and random.random() < swap_prob:
                    new_words.append(word2)  # Replace with the second variant
                    replaced = True
                    break
                elif word == word2 and random.random() < swap_prob:
                    new_words.append(word1)  # Replace with the first variant
                    replaced = True
                    break
            
            if not replaced:
                new_words.append(word)  # Keep the original word if not replaced
        
        # Reconstruct the sentence
        swapped_sentences.append(' '.join(new_words))
    
    return swapped_sentences

common_tatoeba_name = ["Tom", "Mary", "Sami", "John", "Maria"]

def add_data_variations(xx, yy, source_lang: str, target_lang: str, batch_size: int):
    # Randomly swap source and target languages
    if random.getrandbits(1):
        xx, yy, source_lang, target_lang = yy, xx, target_lang, source_lang

    if target_lang == 'gos_Latn':
        yy = add_gronings_variations(yy)
        yy = swap_synonyms(yy, synonym_pairs_gos)
    elif source_lang == 'gos_Latn':
        xx = add_gronings_variations(xx)
        xx = swap_synonyms(xx, synonym_pairs_gos)

    # Create more name variation (e.g., replacing "Tom")
    for i in range(len(xx)):
        for name in common_tatoeba_name:
            if name in xx[i] and name in yy[i]:
                namelist = ['Tom', 'Sam', 'Ben', 'Nick', 'Ed', 'Noah', 'Joey', 'Rick', 'Rob', 'Mick', 'Mike', 'Michael', 'Tim', 'Adam', 'Arnold', 'Lucas', 'Robin', 'James', 'Jim', 'Mary', 'Maria', 'Sami', 'John']
                othername = random.choice(namelist)
                xx[i] = xx[i].replace(name, othername)
                yy[i] = yy[i].replace(name, othername)

    # Small chance of uppercase transformation
    if not random.getrandbits(5):
        xx = [x.upper() for x in xx]
        yy = [y.upper() for y in yy]

    # chance of no capitalization at sentence start
    elif not random.getrandbits(3):
        xx = [x[:1].lower() + x[1:] for x in xx]
        yy = [y[:1].lower() + y[1:] for y in yy]

    # Small chance of random emoji at the end
    if not random.getrandbits(3):
        emojis = random.choices(["😊", "😂", "😍", "👍", "🔥", "🎉", "🌟", "😎", "🥳", '❤️', '💀', '😭', '🫶', '🤣', '😘', '🥺', '🤔', '🙏'], k=batch_size)
        xx = [xx[i] + emojis[i] for i in range(batch_size)]
        yy = [yy[i] + emojis[i] for i in range(batch_size)]

    # Small chance of sentence-final character deletion
    elif not random.getrandbits(3):
        xx = [x[:-1] if len(x) > 1 else x for x in xx]
        yy = [y[:-1] if len(y) > 1 else y for y in yy]

    return xx, yy

def get_batch_pairs(batch_size: int, corpus_objects, dataset: str = "train", max_chars=None, apply_variations=True):
    # Calculate weights based on dataset sizes
    weights = []
    for corp in corpus_objects:
        if dataset == "train":
            weights.append(len(corp.df_train))
        elif dataset == "validate":
            weights.append(len(corp.df_validate))
        else:
            raise ValueError(f"Invalid dataset specified: {dataset}. Choose from 'train' or 'validate'.")
    
    # Normalize weights for sampling
    weights = [w / sum(weights) for w in weights]
    corpus = np.random.choice(corpus_objects, p=weights, replace=False)

    # Sample the batch
    if dataset == "train":
        batch = corpus.df_train.sample(n=batch_size)
    elif dataset == "validate":
        batch = corpus.df_validate.sample(n=batch_size)

    # Preprocess sentences
    batch['source_sentence'] = batch['source_sentence'].apply(preproc)
    batch['target_sentence'] = batch['target_sentence'].apply(preproc)

    xx = batch['source_sentence'].tolist()
    yy = batch['target_sentence'].tolist()

    # Optional: Apply variations
    if apply_variations:
        xx, yy = add_data_variations(xx, yy, corpus.source_lang_nllb, corpus.target_lang_nllb, batch_size)

    # Trim sentences if max_chars is specified
    if max_chars:
        def truncate_at_space(sent, max_len):
            if len(sent) <= max_len:
                return sent
            # Find the last space before max_len
            truncate = sent[:max_len]
            return truncate[:truncate.rfind(" ")]

        xx = [truncate_at_space(x, max_chars) for x in xx]
        yy = [truncate_at_space(y, max_chars) for y in yy]

    return xx, yy, corpus.source_lang_nllb, corpus.target_lang_nllb

def train_model(model, tokenizer, corpus_objects):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #model should already be on the right device
    # model.to(device)
    device = next(model.parameters()).device

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"])
    
    cleanup()
    losses = []
    model.train()

    from tqdm.auto import trange
    tq = trange(1, config["training_steps"]+1, desc="Training Steps")

    for step in tq:
        try:
            xx, yy, lang1, lang2 = get_batch_pairs(config["batch_size"], corpus_objects, max_chars=config["max_chars"])
            
            tokenizer.src_lang = lang1
            x = tokenizer(xx, return_tensors='pt', padding='longest', truncation=True, max_length=config["max_length"]).to(device)
            tokenizer.src_lang = lang2
            y = tokenizer(yy, return_tensors='pt', padding='longest', truncation=True, max_length=config["max_length"]).to(device)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
            
            loss = model(**x, labels=y.input_ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            losses.append(loss.item())
            tq.set_postfix({'loss': np.mean(losses[-25:])})

            if step % 50 == 0 and step > 500:
                cleanup()
                # Save some intermediate checkpoints
                model.save_pretrained(config["MODEL_SAVE_PATH"] + f"/{step}")
                tokenizer.save_pretrained(config["MODEL_SAVE_PATH"] + f"/{step}")

        except RuntimeError as e:
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            print('Error:', e)
            continue

    # Final saving
    model.save_pretrained(config["MODEL_SAVE_PATH"] + f"/{step}")
    tokenizer.save_pretrained(config["MODEL_SAVE_PATH"] + f"/{step}")
    
    # Plotting and saving the losses
    plt.figure(figsize=(10, 5))
    pd.Series(losses).plot(label='Mean Loss')
    pd.Series(losses).ewm(span=30).mean().plot(label='Exponentially weighted moving average, 30 steps')
    pd.Series(losses).ewm(span=100).mean().plot(label='Exponentially weighted moving average, 100 steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    
    # Save the plot as an image
    loss_plot_path = config["MODEL_SAVE_PATH"] + "_final_loss_plot.png"
    plt.savefig(loss_plot_path)
    plt.close()

def main_train(corpus_objects):
    model, tokenizer = setup_model_and_tokenizer(config["modelname"], config["modelpath"], config["new_lang_nllb"], config["similar_lang_nllb"], device=config['device'])
    train_model(model, tokenizer, corpus_objects)
