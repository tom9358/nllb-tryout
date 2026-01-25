import re
import torch
from transformers import Adafactor, get_constant_schedule_with_warmup
import numpy as np
from .tokenizer_and_model_setup import setup_model_and_tokenizer, cleanup
from .config import config
from tqdm.auto import trange
from sacremoses import MosesPunctNormalizer
import random
import unicodedata
import sys
import pandas as pd
import matplotlib.pyplot as plt


mpn = MosesPunctNormalizer(lang="en")

non_printable_map = {
    ord(c): " "
    for c in (chr(i) for i in range(sys.maxunicode + 1))
    if unicodedata.category(c)[0] == "C"
}

def preproc(text: str) -> str:
    return unicodedata.normalize("NFKC", mpn.normalize(text).translate(non_printable_map))

# List of synonym pairs
synonym_pairs_gos = [
    ('huus', 'hoes'), ('huzen', 'hoezen'), ('huuske', 'hoeske'), ('groag', 'geern'), ('raais', 'raaize'), ('kees', 'keze'), ('week', 'weke'), ('mìns', 'mens'), ('mìnsk', 'mens'),
    ('mìnsen', 'mensen'), ('mìnsken', 'mìnsen'), ('uut', 'oet'), ('in', 'ien'), ('wer', 'wuir'), ('gebruuk', 'gebroek'), ('zuch', 'zok'), ('bruukst', 'broekst'), ('wind', 'wiend'),
    ('pampier', 'pepier'), ('vanuut', 'vanoet'), ('wazzen', 'waren'), ('mekoar', 'nkander'), ('bruken', 'broeken'), ('zuch', 'zuk'), ('vis', 'visk'), ('olle', 'olde'),
    ('zuk', 'zok'), ('wotter', 'woater'), ('kraant', 'kraande'), ('haar', 'har'), ('bruuk', 'broek'), ('school', 'schoule'), ('schoul', 'schoule'), ('iezer', 'iesder'),
    ('ais', 'ains'), ('hebben', 'hemmen'), ('zotterdag', 'zoaterdag'), ('bruukt', 'broekt'), ('bruukten', 'broekten'), ('iezern', 'iesdern'), ('kind', 'kiend'), ('altied', 'aaltied'),
    ('mirreg', 'middag'), ('vast', 'vaast'), ('nacht', 'naacht'), ('kiender', 'kinder'), ('bruukte', 'broekte'), ('deus','deuze'), ('gelok', 'geluk'), ('gang', 'gaang')
]

def add_gronings_variations(sentences: list[str]) -> list[str]:
    # Gronings-specific removal of more or less optional diacritics
    s = pd.Series(sentences)
    mask = np.random.rand(len(s)) < 0.25
    s.loc[mask] = s.loc[mask].str.replace('ì','i').str.replace('è','e').str.replace('ò','o').str.replace('ó','o')
    return s.tolist()

def swap_synonyms(
    sentences: list[str],
    synonym_pairs: list[tuple[str, str]],
    swap_prob_exponent: int = 2
) -> list[str]:
    lookup: dict[str, str] = {}

    for a, b in synonym_pairs:
        lookup[a] = b
        lookup[b] = a

    pats = '|'.join(map(re.escape, lookup.keys()))
    pattern = re.compile(rf'\b({pats})\b')

    def replacer(match):
        word = match.group(0)
        if not random.getrandbits(swap_prob_exponent):
            return lookup[word]
        return word

    s = pd.Series(sentences)
    swapped = s.str.replace(pattern, replacer, regex=True)
    return swapped.tolist()

common_tatoeba_name = ["Tom", "Mary", "Sami", "John", "Maria"]
namelist = np.array(['Tom','Sam','Ben','Nick','Ed','Noah','Joey','Rick','Rob','Mick','Mike','Michael','Tim','Adam','Arnold','Lucas','Robin','James','Jim','Mary','Maria','Sami','John','Linda'], dtype=object)
pattern_names = r'\b(' + '|'.join(map(re.escape, common_tatoeba_name)) + r')\b'
pattern_names_re = re.compile(pattern_names)

emoji_choices = np.array(["😊", "😂", "😍", "👍", "🔥", "🎉", "🌟", "😎", "🥳", '❤️', '💀', '😭', '🫶', '🤣', '😘', '🥺', '🤔', '🙏'], dtype=object)

def apply_variations(xx: pd.Series, yy: pd.Series) -> tuple[pd.Series, pd.Series]:
    N = len(xx)
    xx_vals = xx.to_numpy(dtype=object)
    yy_vals = yy.to_numpy(dtype=object)

    # Multiple-name safe replacement
    for i in range(N):
        s_x = xx_vals[i]
        s_y = yy_vals[i]
        matches_x = [m.group(0) for m in pattern_names_re.finditer(s_x)]
        matches_y = [m.group(0) for m in pattern_names_re.finditer(s_y)]
        if matches_x and matches_x == matches_y:
            rand_names = np.random.choice(namelist, size=len(matches_x))
            def replace_seq(original, rand_names_seq):
                out = []
                last_idx = 0
                for m, newname in zip(pattern_names_re.finditer(original), rand_names_seq):
                    out.append(original[last_idx:m.start()])
                    out.append(newname)
                    last_idx = m.end()
                out.append(original[last_idx:])
                return "".join(out)
            xx_vals[i] = replace_seq(s_x, rand_names)
            yy_vals[i] = replace_seq(s_y, rand_names)

    # General variations
    idxs = np.random.permutation(N)
    n_upper = N // 32
    n_nocap = N // 8
    n_emoji = N // 8
    n_delete = N // 8

    # Uppercase transformation
    upper_idxs = idxs[:n_upper]
    xx_vals[upper_idxs] = [s.upper() for s in xx_vals[upper_idxs]]
    yy_vals[upper_idxs] = [s.upper() for s in yy_vals[upper_idxs]]

    # No capitalization at sentence start
    sel = idxs[n_upper:n_upper + n_nocap]
    xx_vals[sel] = [s[0].lower() + s[1:] for s in xx_vals[sel]]
    yy_vals[sel] = [s[0].lower() + s[1:] for s in yy_vals[sel]]

    # Random emoji at the end
    emoji_idxs = idxs[n_upper + n_nocap: n_upper + n_nocap + n_emoji]
    emojis = np.random.choice(emoji_choices, size=n_emoji)
    xx_vals[emoji_idxs] = [s + emojis[k] for k, s in enumerate(xx_vals[emoji_idxs])]
    yy_vals[emoji_idxs] = [s + emojis[k] for k, s in enumerate(yy_vals[emoji_idxs])]

    # Sentence-final character deletion
    delete_idxs = idxs[n_upper + n_nocap + n_emoji : n_upper + n_nocap + n_emoji + n_delete]
    xx_vals[delete_idxs] = [s[:-1] if len(s) > 1 else s for s in xx_vals[delete_idxs]]
    yy_vals[delete_idxs] = [s[:-1] if len(s) > 1 else s for s in yy_vals[delete_idxs]]

    return pd.Series(xx_vals, index=xx.index), pd.Series(yy_vals, index=yy.index)


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
