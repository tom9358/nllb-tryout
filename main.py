from config import source_langs_tatoeba, source_langs_nllb, MODEL_SAVE_PATH, new_lang_nllb
from downloadtatoeba import main_download
from corpus import main_corpus
from train import main_train
from evaluate import main_evaluate
from tryout import main_tryout

def main():
    # Step 1: Download data
    main_download(source_langs_tatoeba)
    
    # Step 2: Load and create parallel corpus
    corpus_objects = main_corpus(source_langs_tatoeba, source_langs_nllb)
    
    # Step 3: Train the model
    main_train(corpus_objects)
    
    # Step 3.5: Try out the model! Evaluation by means of vibes
    main_tryout(MODEL_SAVE_PATH, new_lang_nllb)
    
    # Step 4: Evaluate the model
    # main_evaluate(corpus_objects, MODEL_SAVE_PATH, new_lang_nllb)

if __name__ == "__main__":
    main()