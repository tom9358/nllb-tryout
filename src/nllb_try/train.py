import re
import torch
from transformers import Adafactor, get_constant_schedule_with_warmup
import numpy as np
from .tokenizer_and_model_setup import setup_model_and_tokenizer, cleanup
from .config import config
from tqdm.auto import trange
from sacremoses import MosesPunctNormalizer
import unicodedata
import sys
import pandas as pd
import matplotlib.pyplot as plt


mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [(re.compile(p), r) for p, r in mpn.substitutions]

non_printable_map = {
    ord(c): " "
    for c in (chr(i) for i in range(sys.maxunicode + 1))
    if unicodedata.category(c)[0] == "C"
}

def preproc(text: str) -> str:
    return unicodedata.normalize("NFKC", mpn.normalize(text).translate(non_printable_map))

# List of synonym pairs
synonym_pairs_gos = [
    ('huus', 'hoes'), ('huzen', 'hoezen'), ('huuske', 'hoeske'), ('groag', 'geern'), ('raais', 'raaize'), ('kees', 'keze'), ('week', 'weke'),
    ('mìnsken', 'mìnsen'), ('uut', 'oet'), ('in', 'ien'), ('wer', 'wuir'), ('gebruuk', 'gebroek'), ('zuch', 'zok'), ('bruukst', 'broekst'), ('wind', 'wiend'),
    ('vanuut', 'vanoet'), ('wazzen', 'waren'), ('mekoar', 'nkander'), ('bruken', 'broeken'), ('zuch', 'zuk'), ('vis', 'visk'), ('olle', 'olde'),
    ('zuk', 'zok'), ('wotter', 'woater'), ('kraant', 'kraande'), ('haar', 'har'), ('bruuk', 'broek'), ('school', 'schoule'), ('iezer', 'iesder'),
    ('ais', 'ains'), ('hebben', 'hemmen'), ('zotterdag', 'zoaterdag'), ('bruukt', 'broekt'), ('bruukten', 'broekten'), ('iezern', 'iesdern'), ('kind', 'kiend'),
    ('mirreg', 'middag'), ('vast', 'vaast'), ('nacht', 'naacht'), ('kiender', 'kinder'), ('bruukte', 'broekte'), ('deus','deuze'), ('gelok', 'geluk')
]

def add_gronings_variations(sentences: list[str]) -> list[str]:
    # Gronings-specific removal of more or less optional diacritics
    s = pd.Series(sentences)
    mask = np.random.rand(len(s)) < 0.25
    s.loc[mask] = s.loc[mask].str.replace('ì','i').str.replace('è','e').str.replace('ò','o').str.replace('ó','o')
    return s.tolist()

def swap_synonyms(sentences: list[str], synonym_pairs: list[tuple[str,str]], swap_prob: float = 0.25) -> list[str]:
    swapped_sentences = []

    for sent in sentences:
        words = sent.split()
        new_words = []

        for word in words:
            replaced = False

            # Check each synonym pair
            for word1, word2 in synonym_pairs:
                if word == word1 and np.random.rand() < swap_prob: #random.random() might be faster. or random.getrandbits(2) perhaps.
                    new_words.append(word2)
                    replaced = True
                    break
                elif word == word2 and np.random.rand() < swap_prob:
                    new_words.append(word1)
                    replaced = True
                    break

            if not replaced:
                new_words.append(word)  # Keep the original word if not replaced

        # Reconstruct the sentence
        swapped_sentences.append(' '.join(new_words))

    return swapped_sentences

common_tatoeba_name = ["Tom", "Mary", "Sami", "John", "Maria"]
namelist = ['Tom','Sam','Ben','Nick','Ed','Noah','Joey','Rick','Rob','Mick','Mike','Michael','Tim','Adam','Arnold','Lucas','Robin','James','Jim','Mary','Maria','Sami','John','Linda']
pattern_names = r'\b(' + '|'.join(map(re.escape, common_tatoeba_name)) + r')\b'

def apply_name_variation(xx, yy):
    # Create more name variation (e.g., replacing "Tom")
    xx = pd.Series(xx)
    yy = pd.Series(yy)
    mask = xx.str.contains(pattern_names) & yy.str.contains(pattern_names) & ((xx.str.extract(pattern_names, expand=False) == yy.str.extract(pattern_names, expand=False)))
    idxs = np.where(mask)[0]
    if len(idxs) > 0:
        rand_names = np.random.choice(namelist, size=len(idxs))
        for i, new_name in zip(idxs, rand_names):
            xx.iloc[i] = re.sub(pattern_names, new_name, xx.iloc[i])
            yy.iloc[i] = re.sub(pattern_names, new_name, yy.iloc[i])
    return xx, yy

def apply_variations(xx, yy):
    N = len(xx)
    idxs = np.random.permutation(N)
    n_upper = int(N / 32)
    n_nocap = int(N / 8)
    n_emoji = int(N / 8)
    n_delete = int(N / 8)

    # Uppercase transformation
    for arr in [xx, yy]:
        arr.iloc[idxs[:n_upper]] = arr.iloc[idxs[:n_upper]].str.upper()

    # No capitalization at sentence start
    sel = idxs[n_upper:n_upper + n_nocap]

    xx.iloc[sel] = xx.iloc[sel].apply(lambda s: s[0].lower() + s[1:] if s else s)
    yy.iloc[sel] = yy.iloc[sel].apply(lambda s: s[0].lower() + s[1:] if s else s)

    # Random emoji at the end
    emoji_idxs = idxs[n_upper + n_nocap: n_upper + n_nocap + n_emoji]
    emojis = np.random.choice(
        ["😊", "😂", "😍", "👍", "🔥", "🎉", "🌟", "😎", "🥳", '❤️', '💀', '😭', '🫶', '🤣', '😘', '🥺', '🤔', '🙏'],
        size=n_emoji)
    for k, i in enumerate(emoji_idxs):
        xx[i] += emojis[k]
        yy[i] += emojis[k]

    # Sentence-final character deletion
    for arr in [xx, yy]:
        for i in idxs[n_upper + n_nocap + n_emoji : n_upper + n_nocap + n_emoji + n_delete]:
            if len(arr[i]) > 1:
                arr[i] = arr[i][:-1]
    return xx, yy

def tokenize_mixed_langs(
    tokenizer, texts: list[str], langs: list[str], max_length: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns (input_ids, attention_mask) stacked tensors (len(texts), max_length)
    # Tokenizes a list of sentences and corresponding languages efficiently, handling mixed language batches by grouping sentences by language for faster batched tokenization.
    # Required for multilingual datasets when tokenizer must know the language per sentence (which NLLB does), allowing bulk tokenization while respecting per-sentence language settings.
    idxs_by_lang: dict[str, list[int]] = {}
    for i, lang in enumerate(langs):
        idxs_by_lang.setdefault(lang, []).append(i)
    input_ids_dict: dict[int, torch.Tensor] = {}
    attention_mask_dict: dict[int, torch.Tensor] = {}
    for lang, idxs in idxs_by_lang.items():
        batch_texts = [texts[i] for i in idxs]
        tokenizer.src_lang = lang
        feats = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        for j, i_global in enumerate(idxs):
            input_ids_dict[i_global] = feats['input_ids'][j]
            attention_mask_dict[i_global] = feats['attention_mask'][j]
    input_ids: list[torch.Tensor] = [input_ids_dict[i] for i in range(len(texts))]
    attention_mask: list[torch.Tensor] = [attention_mask_dict[i] for i in range(len(texts))]
    input_ids_tensor = torch.stack(input_ids).to(device)
    attention_mask_tensor = torch.stack(attention_mask).to(device)
    return input_ids_tensor, attention_mask_tensor

def train_model(model, tokenizer, corpus_objects: list) -> None:
    batch_size: int = config["batch_size"]
    max_length: int  = config["max_length"]
    num_epochs: int = config["num_epochs"]
    warmup_steps: int = config["warmup_steps"]
    model_save_path: str = config["MODEL_SAVE_PATH"]

    device = next(model.parameters()).device

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    cleanup()
    losses = []
    total_steps = 0
    # Combine
    dfs = []
    src_langs_all, tgt_langs_all = [], []
    for i, corpus in enumerate(corpus_objects):
        df = corpus.df_train.copy()
        df['corpus_idx'] = i
        dfs.append(df)
        src_langs_all.extend([corpus.source_lang_nllb] * len(df))
        tgt_langs_all.extend([corpus.target_lang_nllb] * len(df))
    df_all = pd.concat(dfs).reset_index(drop=True)
    N = len(df_all)

    # Preprocess alles in één keer
    orig_xx = df_all['source_sentence'].apply(preproc)
    orig_yy = df_all['target_sentence'].apply(preproc)
    srcs = np.array(src_langs_all, dtype=object)
    tgts = np.array(tgt_langs_all, dtype=object)

    for epoch in range(num_epochs):
        xx = orig_xx.copy()
        yy = orig_yy.copy()

        # Always add more name variety
        xx, yy = apply_name_variation(xx, yy)

        # Some additional data variation
        xx, yy = apply_variations(xx, yy)

        # Gronings-specific augmentation
        if np.any(tgts == 'gos_Latn'): # we should know where the gronings sentences are. TODO
            idxs = np.where(tgts == 'gos_Latn')[0]
            yy_idxs = list(yy[i] for i in idxs)
            yy_vals = add_gronings_variations(yy_idxs)
            yy_syns = swap_synonyms(yy_vals, synonym_pairs_gos)
            for k, i in enumerate(idxs):
                yy[i] = yy_syns[k]
        if np.any(srcs == 'gos_Latn'):
            idxs = np.where(srcs == 'gos_Latn')[0]
            xx_idxs = list(xx[i] for i in idxs)
            xx_vals = add_gronings_variations(xx_idxs)
            xx_syns = swap_synonyms(xx_vals, synonym_pairs_gos)
            for k, i in enumerate(idxs):
                xx[i] = xx_syns[k]

        # Randomly swap source and target languages
        swap_idxs = np.random.permutation(N)
        half = N // 2

        swap_mask = np.zeros(N, dtype=bool)
        swap_mask[:half] = True
        np.random.shuffle(swap_mask)

        xx_swapped = np.where(swap_mask, yy[swap_idxs], xx[swap_idxs])
        yy_swapped = np.where(swap_mask, xx[swap_idxs], yy[swap_idxs])
        src_swapped = np.where(swap_mask, tgts[swap_idxs], srcs[swap_idxs])
        tgt_swapped = np.where(swap_mask, srcs[swap_idxs], tgts[swap_idxs])

        # Shuffle
        final_idxs = np.random.permutation(N)
        df_all_aug = pd.DataFrame({
            "source_sentence": xx_swapped[final_idxs],
            "target_sentence": yy_swapped[final_idxs],
            "src_lang": src_swapped[final_idxs],
            "tgt_lang": tgt_swapped[final_idxs],
        })
        df_epoch = df_all_aug.sample(frac=1).reset_index(drop=True)
        # Bulk pre-tokenize all epoch data
        xx_texts = df_epoch['source_sentence'].tolist()
        yy_texts = df_epoch['target_sentence'].tolist()
        src_langs_epoch = df_epoch['src_lang'].tolist()
        tgt_langs_epoch = df_epoch['tgt_lang'].tolist()

        xx_input_ids, xx_attention = tokenize_mixed_langs(tokenizer, xx_texts, src_langs_epoch, max_length, device)
        yy_input_ids, yy_attention = tokenize_mixed_langs(tokenizer, yy_texts, tgt_langs_epoch, max_length, device)
        yy_input_ids[yy_input_ids == tokenizer.pad_token_id] = -100  # Masked loss targets

        n_samples_total = len(df_epoch)
        n_batches = int(np.ceil(n_samples_total / batch_size))
        tq = trange(n_batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step in tq:
            batch_start = step * batch_size
            batch_end = min((step+1)*batch_size, n_samples_total)
            # Just select slices! FAST!
            x = {
                "input_ids": xx_input_ids[batch_start:batch_end],
                "attention_mask": xx_attention[batch_start:batch_end]
            }
            y_input_ids_batch = yy_input_ids[batch_start:batch_end]
            loss = model(**x, labels=y_input_ids_batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            losses.append(loss.item())
            tq.set_postfix({'loss': np.mean(losses[-25:])})
            total_steps += 1

        print(f"Saving after epoch {epoch+1}")
        # Save checkpoints
        model.save_pretrained(model_save_path + f"/epoch{epoch+1}")
        tokenizer.save_pretrained(model_save_path + f"/epoch{epoch+1}")
        cleanup()

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
    plt.savefig(model_save_path + "_final_loss_plot.png")
    plt.close()

def main_train(corpus_objects: list):
    modelname: str = config["modelname"]
    modelpath: str = config["modelpath"]
    new_lang_nllb: str = config["new_lang_nllb"]
    similar_lang_nllb: str = config["similar_lang_nllb"]
    device: str = config["device"]

    model, tokenizer = setup_model_and_tokenizer(
        modelname,
        modelpath,
        new_lang_nllb,
        similar_lang_nllb,
        device=device
    )
    train_model(model, tokenizer, corpus_objects)
