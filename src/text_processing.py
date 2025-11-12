# src/text_processing.py

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from typing import List
import unicodedata 

STEMMER = SnowballStemmer('spanish')
STOP_WORDS = set(stopwords.words('spanish'))


def preprocess_text(text: str) -> List[str]:

    text = text.lower()
    
    try:
        nfkd_form = unicodedata.normalize('NFKD', text)
        text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    except Exception as e:
        print(f"Advertencia: Falló la normalización de unicodedata: {e}")
        pass 
    tokens = word_tokenize(text, language='spanish')
    
    processed_tokens = []
    for token in tokens:
        if token.isalpha() and token not in STOP_WORDS:
            stemmed_token = STEMMER.stem(token)
            processed_tokens.append(stemmed_token)
            
    return processed_tokens