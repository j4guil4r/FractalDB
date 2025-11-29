# src/multimedia/histogram_builder.py

import os
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from typing import Optional, Union
from sklearn.neighbors import KDTree

# Importamos nuestros otros módulos multimedia
from src.multimedia.feature_extractor import SIFTExtractor
from src.multimedia.codebook_builder import CodebookBuilder

class BoVWHistogramBuilder:
    
    def __init__(self, k: int, data_dir: str = 'data'):
        self.k = k
        self.data_dir = data_dir
        
        # Cargar el Codebook (el modelo K-Means entrenado)
        self.kmeans: Optional[MiniBatchKMeans] = CodebookBuilder.load_codebook(k, data_dir)
        if self.kmeans is None:
            raise FileNotFoundError(
                f"No se pudo cargar el codebook para K={k}. "
                "¿Ejecutaste 'codebook_builder.py' primero?"
            )
        
        if self.kmeans:
            print("Construyendo KD-Tree para búsqueda rápida...")
            self.tree = KDTree(self.kmeans.cluster_centers_, leaf_size=40)
            
        # Inicializar el extractor SIFT
        self.extractor = SIFTExtractor()
        
        print(f"Generador de histogramas (K={k}) listo.")

    def _create_histogram(self, descriptors: Optional[np.ndarray]) -> np.ndarray:
        # Función interna para convertir descriptores SIFT en un histograma TF.

        # Si no hay descriptores, devuelve un vector de ceros
        if descriptors is None:
            return np.zeros(self.k, dtype=np.float32)
            
        # Asignar cada descriptor al "visual word" (cluster) más cercano
        _dist, visual_words = self.tree.query(descriptors, k=1)
        visual_words = visual_words.flatten()
        
        # Contar las ocurrencias de cada "visual word"
        hist = np.bincount(visual_words, minlength=self.k)
        
        # Devolvemos el histograma como TF (Term Frequency) crudo
        return hist.astype(np.float32)

    def create_histogram_from_path(self, image_path: str) -> Optional[np.ndarray]:
        # Genera un histograma BoVW (vector TF de tamaño K) para una

        if self.kmeans is None:
            return None
            
        # Extraer descriptores SIFT
        descriptors = self.extractor.extract_from_path(image_path)
        
        # Convertir descriptores a histograma
        return self._create_histogram(descriptors)

    def create_histogram_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        # Genera un histograma BoVW (vector TF de tamaño K) 

        if self.kmeans is None:
            return None
            
        # Extraer descriptores SIFT
        descriptors = self.extractor.extract_from_bytes(image_bytes)
        
        # Convertir descriptores a histograma
        return self._create_histogram(descriptors)