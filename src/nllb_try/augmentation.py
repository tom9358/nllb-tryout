import re
import random
import sys
import unicodedata

import numpy as np
import pandas as pd
from sacremoses import MosesPunctNormalizer


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

mpn = MosesPunctNormalizer(lang="en")

non_printable_map = {
    ord(c): " "
    for c in (chr(i) for i in range(sys.maxunicode + 1))
    if unicodedata.category(c)[0] == "C"
}

def preproc(text: str) -> str:
    return unicodedata.normalize("NFKC", mpn.normalize(text).translate(non_printable_map))


# ---------------------------------------------------------------------------
# Gronings-specific augmentation
# ---------------------------------------------------------------------------

synonym_pairs_gos = [
    ('huus', 'hoes'), ('huzen', 'hoezen'), ('huuske', 'hoeske'), ('groag', 'geern'), ('raais', 'raaize'),
    ('kees', 'keze'), ('week', 'weke'), ('mìns', 'mens'), ('mìnsk', 'mens'), ('terug', 'terogge'),
    ('mìnsen', 'mensen'), ('mìnsken', 'mìnsen'), ('uut', 'oet'), ('in', 'ien'), ('wer', 'wuir'),
    ('gebruuk', 'gebroek'), ('zuch', 'zok'), ('bruukst', 'broekst'), ('wind', 'wiend'), ('rug', 'rogge'),
    ('pampier', 'pepier'), ('vanuut', 'vanoet'), ('wazzen', 'waren'), ('mekoar', 'nkander'),
    ('bruken', 'broeken'), ('zuch', 'zuk'), ('vis', 'visk'), ('ìnde', 'ende'), ('ìnd', 'end'),
    ('brug', 'brogge'), ('zuk', 'zok'), ('wotter', 'woater'), ('kraant', 'kraande'), ('haar', 'har'),
    ('bruuk', 'broek'), ('school', 'schoule'), ('schoul', 'schoule'), ('iezer', 'iesder'), ('kouk', 'kouke'),
    ('ais', 'ains'), ('hebben', 'hemmen'), ('zotterdag', 'zoaterdag'), ('bruukt', 'broekt'),
    ('bruukten', 'broekten'), ('iezern', 'iesdern'), ('kind', 'kiend'), ('altied', 'aaltied'),
    ('mirreg', 'middag'), ('vast', 'vaast'), ('nacht', 'naacht'), ('kiender', 'kinder'), ('kachel', 'kaggel'),
    ('deus','deuze'), ('gelok', 'geluk'), ('gang', 'gaang'), ('olle', 'olde'), ('bruukte', 'broekte')
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


# ---------------------------------------------------------------------------
# General parallel-corpus augmentation
# ---------------------------------------------------------------------------

common_tatoeba_name = ["Tom", "Mary", "Sami", "John", "Maria", "Mary"]
namelist = np.array(['Tom','Sam','Ben','Nick','Ed','Noah','Joey','Rick','Rob','Mick','Mike','Michael','Tim','Adam','Arnold','Lucas','Robin','James','Jim','Mary','Maria','Sami','John','Linda'], dtype=object)
pattern_names = r'\b(' + '|'.join(map(re.escape, common_tatoeba_name)) + r')\b'
pattern_names_re = re.compile(pattern_names)

emoji_choices = np.array([
    "😊", "😂", "😍", "👍", "🔥", "✨", "⭐", "😎", "😄", "❤️",
    "😱", "😭", "💕", "🤣", "😘", "😢", "🤔", "🙏", "🎁", "😉",
    "😅", "🙂", "👏", "😀", "😆", "😋", "😛", "😇", "🎵", "🌹",
], dtype=object)

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

    # Pilcrow insertion (teaches the model to preserve ¶ as a newline placeholder,
    # since SentencePiece normalizes actual \n to whitespace)
    n_newline = N // 8
    newline_idxs = idxs[n_upper + n_nocap + n_emoji + n_delete : n_upper + n_nocap + n_emoji + n_delete + n_newline]
    for arr in (xx_vals, yy_vals):
        s = pd.Series(arr[newline_idxs])
        # Append ¶ after sentence-ending punctuation, or at end of string
        arr[newline_idxs] = s.str.replace(r'([.!?])\s*$', r'\1¶', regex=True).to_numpy()
        # If no match (no final punctuation), just append ¶
        mask = ~pd.Series(arr[newline_idxs]).str.endswith('¶')
        arr[newline_idxs[mask]] = [v + '¶' for v in arr[newline_idxs[mask]]]

    return pd.Series(xx_vals, index=xx.index), pd.Series(yy_vals, index=yy.index)
