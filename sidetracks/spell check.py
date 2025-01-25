# I got sidetracked and wanted to (learn how to) do spell checking in python, and wanted to see if I could find spelling errors in the tatoeba sentences (not necessarily in Gronings).
# See my github for a spellchecker for Gronings. Should not be too hard to use it in python.

from spellchecker import SpellChecker
import pandas as pd
import nltk

lang_2 = 'nl'
lang_3 = 'nld'

# SpellChecker Nederlands
spell = SpellChecker(language=lang_2)

# Functie om spelfouten te vinden
def find_spelling_errors(sentence):
    words = nltk.word_tokenize(sentence)
    # Zoek naar woorden die niet correct gespeld zijn
    misspelled = spell.unknown(words)
    if misspelled:
        return list(misspelled)[0]
    else:
        return 0

df = pd.read_csv(
    f"../../tatoeba/{lang_3}_sentences.tsv",
    sep="\t", header=None, names=["id", "language", "source_sentence"])

# Pas de functie toe op de zinnen in de DataFrame
df['fouten'] = df['source_sentence'].apply(find_spelling_errors)

for _, row in df[df['fouten'] != 0].iterrows():
    if row['fouten'] not in []:#"'s","n't","'ll","'m","'ve","'re", "sami", "tv", "'d"]:
        print(row[['source_sentence', 'fouten']])
