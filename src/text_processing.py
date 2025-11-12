# src/text_processing.py

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from typing import List

# --- Variables globales cacheadas ---
# Inicializamos el stemmer y las stopwords una sola vez para eficiencia.
# Usamos español como idioma.
try:
    STEMMER = SnowballStemmer('spanish')
    STOP_WORDS = set(stopwords.words('spanish'))
except LookupError:
    print("NLTK data not found. Running download step...")
    nltk.download('punkt')
    nltk.download('stopwords')
    STEMMER = SnowballStemmer('spanish')
    STOP_WORDS = set(stopwords.words('spanish'))

def preprocess_text(text: str) -> List[str]:
    """
    Toma un bloque de texto y aplica el pipeline de procesamiento completo:
    1. Lowercasing
    2. Tokenization
    3. Eliminación de stopwords
    4. Eliminación de signos de puntuación y números
    5. Stemming
    """
    
    # 1. Convertir a minúsculas
    text = text.lower()
    
    # 2. Tokenization [cite: 17]
    tokens = word_tokenize(text, language='spanish')
    
    processed_tokens = []
    for token in tokens:
        # 3. Eliminar stopwords [cite: 18] y 
        # 4. Eliminar signos/números (quedarse solo con alfabéticos) [cite: 19]
        if token.isalpha() and token not in STOP_WORDS:
            # 5. Reducción de palabras (Stemming) [cite: 20]
            stemmed_token = STEMMER.stem(token)
            processed_tokens.append(stemmed_token)
            
    return processed_tokens

# --- Bloque de configuración inicial ---

def setup_nltk():
    """
    Función auxiliar para descargar los modelos necesarios de NLTK
    la primera vez que se use el módulo.
    """
    print("Verificando paquetes de NLTK...")
    try:
        # Intenta cargar los recursos para ver si ya existen
        stopwords.words('spanish')
        word_tokenize("test")
        print("Paquetes 'stopwords' y 'punkt' (spanish) ya están disponibles.")
    except LookupError:
        print("Descargando paquetes 'punkt' y 'stopwords' de NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("Descarga completada.")

if __name__ == "__main__":
    # Este bloque permite ejecutar el archivo directamente para configurar NLTK
    setup_nltk()
    
    # Prueba rápida
    test_text = "¡Hola! Este es un texto de prueba, corriendo para el Proyecto 2. ¿Funcionará?"
    print(f"\nTexto Original: {test_text}")
    processed = preprocess_text(test_text)
    print(f"Tokens Procesados: {processed}")
    # Esperado: ['hol', 'text', 'prueb', 'corr', 'proyect', 'funcion']