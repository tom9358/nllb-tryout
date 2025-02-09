import torch
from transformers import Adafactor, get_constant_schedule_with_warmup
import numpy as np
from tokenizer_and_model_setup import setup_model_and_tokenizer 
import config
import unicodedata

def preproc(text, mpn, replace_nonprint):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean

def add_data_variations(xx, yy, source_lang_long, target_lang_long, batch_size):
    # Add your implementation for data variations here
    pass

def get_batch_pairs(batch_size, corpus_objects):
    # Implement the function to get batch pairs (similar to your existing logic)
    pass

def train_model(model, tokenizer, corpus_objects):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = Adafactor(model.parameters(), lr=1e-4)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=100)
    losses = []
    
    for step in range(config.training_steps):
        xx, yy, lang1, lang2 = get_batch_pairs(config.batch_size, corpus_objects)
        
        tokenizer.src_lang = lang1
        x = tokenizer(xx, return_tensors='pt', padding='longest').to(device)
        
        tokenizer.src_lang = lang2
        y = tokenizer(yy, return_tensors='pt', padding='longest').to(device)
        
        y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
        
        loss = model(**x, labels=y.input_ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        losses.append(loss.item())
        
        if step % 25 == 0:
            print(f"Step {step}, average loss: {np.mean(losses[-25:])}")
        
        if step > config.training_steps / 4:
            model.save_pretrained(config.MODEL_SAVE_PATH + f"_{step}")
            tokenizer.save_pretrained(config.MODEL_SAVE_PATH + f"_{step}")

def main_train(corpus_objects):
    model, tokenizer = setup_model_and_tokenizer(config.modelname, config.modelpath)
    train_model(model, tokenizer, corpus_objects)