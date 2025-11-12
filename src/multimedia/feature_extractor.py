# src/multimedia/feature_extractor.py

import cv2
import numpy as np
from typing import Optional

class SIFTExtractor:
    """
    Clase para inicializar el extractor SIFT y extraer descriptores.
    
    Un objeto SIFT es relativamente "pesado" de crear, por lo que 
    lo inicializamos una vez y lo reutilizamos.
    """
    
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
        """
        Extrae descriptores SIFT de una imagen dada su ruta.
        
        Args:
            image_path: La ruta al archivo de imagen.
            
        Returns:
            Un array de numpy de [N, 128], donde N es el número de
            keypoints encontrados, y 128 es la dimensionalidad de SIFT.
            Retorna None si la imagen no se puede leer o no se encuentran keypoints.
        """
        # 1. Leer la imagen
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # print(f"Advertencia: No se pudo leer la imagen en {image_path}")
            return None
            
        # 2. Detectar keypoints y calcular descriptores
        # keypoints, descriptors
        _kp, des = self.sift.detectAndCompute(img, None)
        
        if des is None:
            # print(f"Advertencia: No se encontraron descriptores SIFT en {image_path}")
            return None
            
        # 3. Retornar los descriptores (un array de N x 128)
        return des

    def extract_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extrae descriptores SIFT de una imagen en formato de bytes.
        
        Args:
            image_bytes: Los bytes crudos de la imagen.
            
        Returns:
            Un array de numpy de [N, 128] o None.
        """
        try:
            # 1. Decodificar bytes a un array de numpy
            np_array = np.frombuffer(image_bytes, np.uint8)
            # 2. Convertir array a imagen (escala de grises)
            img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print("Advertencia: No se pudieron decodificar los bytes de la imagen.")
                return None
                
            # 3. Detectar y calcular
            _kp, des = self.sift.detectAndCompute(img, None)
            
            if des is None:
                return None
                
            return des
            
        except Exception as e:
            print(f"Error procesando bytes de imagen: {e}")
            return None

# --- Para pruebas rápidas ---
if __name__ == "__main__":
    # Necesitarás una imagen de prueba.
    # Por ejemplo, descarga una imagen y guárdala como "test_image.jpg"
    
    # Crear un archivo de imagen de prueba falso si no existe
    test_image_path = "test_image.jpg"
    if not cv2.os.path.exists(test_image_path):
        print("Creando imagen de prueba 'test_image.jpg'...")
        img_data = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        # Añadir un círculo para que SIFT pueda encontrar algo
        cv2.circle(img_data, (320, 240), 100, (255, 255, 255), 5)
        cv2.imwrite(test_image_path, img_data)
        
    print(f"Probando SIFTExtractor con '{test_image_path}'...")
    
    extractor = SIFTExtractor()
    descriptors = extractor.extract_from_path(test_image_path)
    
    if descriptors is not None:
        print(f"¡Éxito! Se encontraron {descriptors.shape[0]} descriptores.")
        print(f"Dimensiones del array: {descriptors.shape}") # Debería ser (N, 128)
    else:
        print("No se encontraron descriptores en la imagen de prueba.")