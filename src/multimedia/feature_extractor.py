# src/multimedia/feature_extractor.py

import cv2
import numpy as np
from typing import Optional

class SIFTExtractor:
    
    def __init__(self):
        try:
            # Inicializa el detector y extractor SIFT
            self.sift = cv2.SIFT_create()
            print("Extractor SIFT inicializado.")
        except cv2.error as e:
            print(f"Error al inicializar SIFT. ¿Está 'opencv-contrib-python' instalado?")
            print("Prueba con: pip install opencv-contrib-python")
            raise e

    def extract_from_path(self, image_path: str) -> Optional[np.ndarray]:
        # Extrae descriptores SIFT de una imagen dada su ruta.
        # Leer la imagen
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
            
        # Detectar keypoints y calcular descriptores

        _kp, des = self.sift.detectAndCompute(img, None)
        
        if des is None:
            return None
            
        # Retornar los descriptores (un array de N x 128)
        return des

    def extract_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        # Extrae descriptores SIFT de una imagen en formato de bytes.

        try:
            # Decodificar bytes a un array de numpy
            np_array = np.frombuffer(image_bytes, np.uint8)
            # Convertir array a imagen (escala de grises)
            img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print("Advertencia: No se pudieron decodificar los bytes de la imagen.")
                return None
                
            # Detectar y calcular
            _kp, des = self.sift.detectAndCompute(img, None)
            
            if des is None:
                return None
                
            return des
            
        except Exception as e:
            print(f"Error procesando bytes de imagen: {e}")
            return None
