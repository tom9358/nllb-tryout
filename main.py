from config import source_langs, MODEL_SAVE_PATH
from downloadtatoeba import main_download
from corpus import main_corpus
from train import main_train
from evaluate import main_evaluate

def main():
    # Step 1: Download data
    main_download(source_langs)
    
    # Step 2: Load and create parallel corpus
    corpus_objects = main_corpus(source_langs)
    
    # Step 3: Train the model
    main_train(corpus_objects)
    
    # # Step 4: Evaluate the model
    # main_evaluate(corpus_objects, MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()