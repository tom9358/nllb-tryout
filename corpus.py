import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import TATOEBA_PATH
from downloadtatoeba import main_download

class ParallelCorpus:
    def __init__(self, source_lang, target_lang):
        main_download([source_lang, target_lang], redownload=False)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.df = self.load_and_format_parallel_sentences()
        self.df_train, self.df_temp = train_test_split(self.df, test_size=0.02, random_state=9358)
        self.df_validate, self.df_test = train_test_split(self.df_temp, test_size=0.5, random_state=9358)
        del self.df_temp

    def load_and_format_parallel_sentences(self):
        df_parallel = load_tatoeba(self.source_lang, self.target_lang)
        df_parallel["source_lang_code"] = self.source_lang
        return df_parallel

def load_tatoeba(source_lang: str, trt_lang: str, max_pairs: int = None):
    src_file = os.path.join(TATOEBA_PATH, f'{source_lang}_sentences.tsv')
    trg_file = os.path.join(TATOEBA_PATH, f'{trt_lang}_sentences.tsv')
    link_file = os.path.join(TATOEBA_PATH, 'links.csv')

    src_sentences = pd.read_csv(src_file, sep="\t", names=["id", "language", source_lang])
    trg_sentences = pd.read_csv(trg_file, sep="\t", names=["id", "language", trt_lang])
    link_sentences = pd.read_csv(link_file, sep="\t", names=["origin", "translation"])

    df_parallel = (link_sentences
        .merge(trg_sentences, left_on="origin", right_on="id")
        .merge(src_sentences, left_on="translation", right_on="id")[["origin", "translation", trt_lang, source_lang]]
    )

    if max_pairs and max_pairs <= len(df_parallel):
        df_parallel = df_parallel.sample(max_pairs)
    return df_parallel[[trt_lang, source_lang]]

def main_corpus(source_langs):
    corpus_objects = []
    for i, src_lang in enumerate(source_langs):
        for tgt_lang in source_langs[i+1:]:
            print('Setting up parallel corpus for', src_lang, tgt_lang)
            corpus_objects.append(ParallelCorpus(src_lang, tgt_lang))
    return corpus_objects