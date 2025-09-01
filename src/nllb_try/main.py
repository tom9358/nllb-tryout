from .config import config
from .downloadtatoeba import main_download
from .corpus import main_corpus
from .train import main_train
from .evaluate import main_evaluate
from .tryout import main_tryout


def main():
    # Step 1: Download data
    main_download(config["source_langs_tatoeba"])
    
    # Step 2: Load and create parallel corpus
    corpus_objects = main_corpus(config["source_langs_tatoeba"], config["source_langs_nllb"])
    
    # Step 3: Train the model
    main_train(corpus_objects)
    
    # Step 3.5: Try out the model! Evaluation by means of vibes
    main_tryout(config["MODEL_SAVE_PATH"], config["new_lang_nllb"])
    
    # Step 4: Evaluate the model properly
    main_evaluate(corpus_objects, config["MODEL_SAVE_PATH"], config["new_lang_nllb"])

if __name__ == "__main__":
    main()