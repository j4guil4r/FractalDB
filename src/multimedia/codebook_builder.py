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
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.k,
            verbose=True,
            batch_size=256 * 4, # Batch size más grande para partial_fit
            n_init=3, 
            max_iter=100,
            random_state=42
        )
        print(f"Constructor de Codebook (K={k}) inicializado.")

    def _get_descriptor_batches(self, image_paths: Iterable[str], 
                                batch_size: int = 1024) -> Iterable[np.ndarray]:
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

    # --- INICIO DE LA SOLUCIÓN 2 (Optimización RAM) ---
    def build_from_paths(self, image_paths: Iterable[str], sample_limit: int = 500_000):
        """
        Entrena el modelo K-Means usando descriptores de las imágenes
        proporcionadas, usando partial_fit para no agotar la RAM.
        
        Args:
            image_paths: Un iterable (ej. lista) de rutas a las imágenes de entrenamiento.
            sample_limit: Nro. máximo de descriptores a usar para entrenar K-Means
        """
        print(f"Iniciando construcción de Codebook (K={self.k}) con partial_fit...")
        print(f"Usando un límite de ~{sample_limit} descriptores para el entrenamiento.")

        total_des_count = 0
        
        # Usamos el generador de lotes para entrenar K-Means en partes (partial_fit)
        # Esto mantiene el uso de RAM bajo y constante.
        batch_generator = self._get_descriptor_batches(image_paths, batch_size=1024 * 2)

        for i, batch in enumerate(batch_generator):
            if total_des_count >= sample_limit:
                print(f"Alcanzado el límite de {sample_limit} descriptores. Deteniendo entrenamiento.")
                break
                
            if batch is not None and len(batch) > self.k:
                # Muestrear el lote si es muy grande, para no sesgar el modelo
                indices = np.random.permutation(len(batch))[:1024]
                sample = batch[indices]
                
                # Entrenar en el lote
                self.kmeans.partial_fit(sample)
                total_des_count += len(sample)
                print(f"  -> Lote {i+1} entrenado. Total descriptores: {total_des_count}")

        if total_des_count == 0:
            print("Error: No se pudieron extraer descriptores de las imágenes proporcionadas.")
            # Intentar un 'fit' vacío para evitar errores, aunque el modelo no será útil
            self.kmeans.fit(np.zeros((self.k, 128), dtype=np.float32))
            
        print("Entrenamiento de K-Means (partial_fit) completado.")
        
        # Guardar el modelo K-Means
        self.save_codebook()
    # --- FIN DE LA SOLUCIÓN 2 ---

    def save_codebook(self):
        """
        Guarda el modelo K-Means entrenado (que contiene los centroides) 
        en un archivo pickle.
        """
        if self.kmeans.cluster_centers_ is None:
            print("Error: K-Means no ha sido entrenado. No se puede guardar.")
            return
            
        print(f"Guardando codebook en {self.codebook_path}...")
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