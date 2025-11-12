# src/multimedia/codebook_builder.py

import os
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from typing import List, Iterable, Optional
from .feature_extractor import SIFTExtractor

class CodebookBuilder:
    """
    Construye el "Diccionario de Palabras Visuales" (Codebook) usando
    K-Means sobre un gran conjunto de descriptores SIFT.
    """
    
    def __init__(self, k: int, data_dir: str = 'data'):
        """
        Args:
            k: El número de "palabras visuales" (centroides) a generar.
            data_dir: El directorio donde se guardará el codebook.
        """
        self.k = k
        self.codebook_path = os.path.join(data_dir, f"mm_codebook_k{k}.pkl")
        self.extractor = SIFTExtractor()
        
        # Usamos MiniBatchKMeans para escalabilidad.
        # No entrenamos aquí, solo lo inicializamos.
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.k,
            verbose=True,
            batch_size=256,
            n_init=3, # Iniciar 3 veces con diferentes semillas
            max_iter=100,
            random_state=42
        )
        print(f"Constructor de Codebook (K={k}) inicializado.")

    def _get_descriptor_batches(self, image_paths: Iterable[str], 
                                batch_size: int = 100) -> Iterable[np.ndarray]:
        """
        Un generador que produce lotes de descriptores SIFT desde las rutas
        de las imágenes, para no agotar la RAM.
        """
        descriptors_batch = []
        count = 0
        
        for path in image_paths:
            des = self.extractor.extract_from_path(path)
            if des is not None:
                descriptors_batch.append(des)
                count += des.shape[0]
            
            # Si hemos acumulado suficientes descriptores, producirlos
            if count >= batch_size:
                yield np.vstack(descriptors_batch)
                descriptors_batch = []
                count = 0
        
        # Producir el último lote restante
        if descriptors_batch:
            yield np.vstack(descriptors_batch)

    def build_from_paths(self, image_paths: Iterable[str], sample_limit: int = 500_000):
        """
        Entrena el modelo K-Means usando descriptores de las imágenes
        proporcionadas.
        
        Args:
            image_paths: Un iterable (ej. lista) de rutas a las imágenes de entrenamiento.
            sample_limit: Nro. máximo de descriptores a usar para entrenar K-Means
                          (para evitar entrenar con millones si el dataset es gigante).
        """
        print(f"Iniciando construcción de Codebook (K={self.k})...")
        print(f"Usando un límite de {sample_limit} descriptores para el entrenamiento.")

        # Recolectar un subconjunto de descriptores para entrenar
        # Esto es más rápido que usar _get_descriptor_batches si cabe en RAM
        
        all_descriptors = []
        total_des_count = 0
        
        for path in image_paths:
            if total_des_count >= sample_limit:
                print(f"Alcanzado el límite de {sample_limit} descriptores para muestreo.")
                break
                
            des = self.extractor.extract_from_path(path)
            if des is not None:
                all_descriptors.append(des)
                total_des_count += des.shape[0]

        if not all_descriptors:
            print("Error: No se pudieron extraer descriptores de las imágenes proporcionadas.")
            return

        print(f"Recolectados {total_des_count} descriptores. Apilando en un solo array...")
        training_data = np.vstack(all_descriptors)
        
        # Asegurarnos de no exceder el límite (por si el último lote fue grande)
        if total_des_count > sample_limit:
            indices = np.random.permutation(total_des_count)[:sample_limit]
            training_data = training_data[indices]
            
        print(f"Array de entrenamiento final: {training_data.shape}")

        # Entrenar K-Means
        print("Iniciando entrenamiento de MiniBatchKMeans...")
        self.kmeans.fit(training_data)
        
        print("Entrenamiento de K-Means completado.")
        
        # Guardar el modelo K-Means (que contiene los centroides)
        self.save_codebook()

    def save_codebook(self):
        """
        Guarda el modelo K-Means entrenado (que contiene los centroides) 
        en un archivo pickle.
        """
        if self.kmeans.cluster_centers_ is None:
            print("Error: K-Means no ha sido entrenado. No se puede guardar.")
            return
            
        print(f"Guardando codebook en {self.codebook_path}...")
        # Guardamos el objeto K-Means completo, ya que lo necesitaremos
        # para asignar nuevos descriptores a los clusters (palabras).
        with open(self.codebook_path, 'wb') as f:
            pickle.dump(self.kmeans, f)
        print("Codebook guardado.")

    @staticmethod
    def load_codebook(k: int, data_dir: str = 'data') -> Optional[MiniBatchKMeans]:
        """
        Función estática para cargar un codebook K-Means guardado.
        """
        codebook_path = os.path.join(data_dir, f"mm_codebook_k{k}.pkl")
        if not os.path.exists(codebook_path):
            print(f"Error: No se encontró el codebook en {codebook_path}")
            return None
            
        try:
            with open(codebook_path, 'rb') as f:
                kmeans_model = pickle.load(f)
            print(f"Codebook (K={k}) cargado desde {codebook_path}")
            return kmeans_model
        except Exception as e:
            print(f"Error al cargar el codebook: {e}")
            return None