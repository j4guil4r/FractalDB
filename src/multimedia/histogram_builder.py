# src/multimedia/histogram_builder.py

import os
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from typing import Optional, Union

# Importamos nuestros otros módulos multimedia
from src.multimedia.feature_extractor import SIFTExtractor
from src.multimedia.codebook_builder import CodebookBuilder

class BoVWHistogramBuilder:
    """
    Convierte imágenes en histogramas de "Bag of Visual Words" (BoVW)
    utilizando un codebook K-Means pre-entrenado.
    """
    
    def __init__(self, k: int, data_dir: str = 'data'):
        """
        Args:
            k: El tamaño del vocabulario (el mismo K usado para K-Means).
            data_dir: Directorio donde está guardado el codebook.
        """
        self.k = k
        self.data_dir = data_dir
        
        # 1. Cargar el Codebook (el modelo K-Means entrenado)
        self.kmeans: Optional[MiniBatchKMeans] = CodebookBuilder.load_codebook(k, data_dir)
        if self.kmeans is None:
            raise FileNotFoundError(
                f"No se pudo cargar el codebook para K={k}. "
                "¿Ejecutaste 'codebook_builder.py' primero?"
            )
            
        # 2. Inicializar el extractor SIFT
        self.extractor = SIFTExtractor()
        
        print(f"Generador de histogramas (K={k}) listo.")

    def _create_histogram(self, descriptors: Optional[np.ndarray]) -> np.ndarray:
        """
        Función interna para convertir descriptores SIFT en un histograma TF.
        """
        # Si no hay descriptores, devuelve un vector de ceros
        if descriptors is None:
            return np.zeros(self.k, dtype=np.float32)
            
        # 3. Asignar cada descriptor al "visual word" (cluster) más cercano
        # self.kmeans.predict() devuelve un array [N] con los IDs de los clusters
        visual_words = self.kmeans.predict(descriptors)
        
        # 4. Contar las ocurrencias de cada "visual word"
        # np.bincount es perfecto para esto. Crea el histograma de frecuencias.
        hist = np.bincount(visual_words, minlength=self.k)
        
        # Devolvemos el histograma como TF (Term Frequency) crudo
        return hist.astype(np.float32)

    def create_histogram_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        Genera un histograma BoVW (vector TF de tamaño K) para una
        imagen dada su ruta.
        """
        if self.kmeans is None:
            return None
            
        # 1. Extraer descriptores SIFT
        descriptors = self.extractor.extract_from_path(image_path)
        
        # 2. Convertir descriptores a histograma
        return self._create_histogram(descriptors)

    def create_histogram_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Genera un histograma BoVW (vector TF de tamaño K) para una
        imagen dada en bytes.
        """
        if self.kmeans is None:
            return None
            
        # 1. Extraer descriptores SIFT
        descriptors = self.extractor.extract_from_bytes(image_bytes)
        
        # 2. Convertir descriptores a histograma
        return self._create_histogram(descriptors)