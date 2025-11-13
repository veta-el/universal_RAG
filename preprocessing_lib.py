from config import Config

import nltk
nltk.data.path.append(Config.FOLDER_PATH+'\\nltk_data')

import pymorphy2
import re

import transformers
import numpy as np
from spellchecker import SpellChecker


# Setting NLP processes
morph = pymorphy2.MorphAnalyzer()
lang = 'english'

# Setting embeddings processes
emb_model_name = Config.FOLDER_PATH+'\\'+Config.EMBEDDINGS_MODEL_DIR
emb_model = transformers.AutoModel.from_pretrained(emb_model_name)
emb_tokenizer = transformers.AutoTokenizer.from_pretrained(emb_model_name)

def text_analyze (text: str): # Get linguistic analyze
    global morph, lang

    tokens = nltk.tokenize.sent_tokenize (text.lower(), language=lang) # Tokenize

    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens] # Remove symbols
    tokens = [token for token in tokens if token not in set (nltk.corpus.stopwords.words(lang))] # Remove stopwords

    return ' '.join([morph.parse(token)[0].normal_form for token in tokens]) # Get normal forms

def get_embeddings (text: str): # Get embeddings
    tokenized = emb_tokenizer.tokenize(text)
    text = text.split (' ')

    if len (tokenized) > 512: # If too many tokens - remove them
        fin_index = int((len (text)*512)/len (tokenized))
        text = ' '.join(text [:fin_index])
    else:
        text = ' '.join (text)

    tokenized = emb_tokenizer(text, return_tensors='pt')
    embeddings = (emb_model(**tokenized, output_hidden_states=True).hidden_states[0] [0]).tolist()
    return np.mean(embeddings, axis=0) # Get mean embedding based on each word embedding

def transform_unknown (text: str): # Transform user words into familiar ones (to avoid misspellings) based on spellchecker
    spell = SpellChecker(language=lang [:2])
    text = text.split (' ')
    new_text = []

    for word in text:
        corrected = spell.correction(word)
        if corrected:
            new_text.append (corrected)
        else:
            new_text.append (word)
            
    return ' '.join (new_text)