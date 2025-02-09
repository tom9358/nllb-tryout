import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import TATOEBA_PATH
from downloadtatoeba import main_download

class ParallelCorpus:
    def __init__(self, source_lang_tatoeba, target_lang_tatoeba, source_lang_nllb, target_lang_nllb):
        main_download([source_lang_tatoeba, target_lang_tatoeba], redownload=False)

        self.source_lang = source_lang_tatoeba
        self.target_lang = target_lang_tatoeba
        self.source_lang_nllb = source_lang_nllb
        self.target_lang_nllb = target_lang_nllb
        self.df = self.load_and_format_parallel_sentences()
        self.df_train, self.df_temp = train_test_split(self.df, test_size=0.02, random_state=9358)
        self.df_validate, self.df_test = train_test_split(self.df_temp, test_size=0.5, random_state=9358)
        del self.df_temp

    def load_and_format_parallel_sentences(self):
        df_parallel = load_tatoeba(self.source_lang_tatoeba, self.target_lang_tatoeba)
        return df_parallel

def load_tatoeba(source_lang_tatoeba: str, target_lang_tatoeba: str):
    src_file = os.path.join(TATOEBA_PATH, f'{source_lang_tatoeba}_sentences.tsv')
    trg_file = os.path.join(TATOEBA_PATH, f'{target_lang_tatoeba}_sentences.tsv')
    link_file = os.path.join(TATOEBA_PATH, 'links.csv')

    src_sentences = pd.read_csv(src_file, sep="\t", header=None, names=["id", "language", "source_sentence"])
    trg_sentences = pd.read_csv(trg_file, sep="\t", header=None, names=["id", "language", "target_sentence"])
    link_sentences = pd.read_csv(link_file, sep="\t", header=None, names=["origin", "translation"])

    df_parallel = (link_sentences
        .merge(trg_sentences, left_on="origin", right_on="id")
        .merge(src_sentences, left_on="translation", right_on="id")\
            [["target_sentence", "source_sentence"]]
    )
    return df_parallel

def main_corpus(source_langs_tatoeba, source_langs_nllb):
    corpus_objects = []
    for i, source_lang_tatoeba, source_lang_nllb in enumerate(zip(source_langs_tatoeba, source_langs_nllb)):
        for target_lang_tatoeba, target_lang_nllb in zip(source_langs_tatoeba, source_langs_nllb)[i+1:]:
            print('Setting up parallel corpus for', source_lang_tatoeba, target_lang_tatoeba)
            corpus_objects.append(ParallelCorpus(source_lang_tatoeba, target_lang_tatoeba, source_lang_nllb, target_lang_nllb))
    return corpus_objects