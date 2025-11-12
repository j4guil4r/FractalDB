# src/text_processing.py

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from typing import List

STEMMER = SnowballStemmer('spanish')
STOP_WORDS = set(stopwords.words('spanish'))


def preprocess_text(text: str) -> List[str]:
    
    # 1. Convertir a minúsculas
    text = text.lower()
    
    tokens = word_tokenize(text, language='spanish')
    
    processed_tokens = []
    for token in tokens:
        if token.isalpha() and token not in STOP_WORDS:
            stemmed_token = STEMMER.stem(token)
            processed_tokens.append(stemmed_token)
            
    return processed_tokens
