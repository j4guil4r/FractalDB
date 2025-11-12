# src/multimedia/knn_search.py

import os
import pickle
import numpy as np
import heapq
import math
from typing import List, Tuple, Iterable, Dict, Any

from src.multimedia.histogram_builder import BoVWHistogramBuilder

class KNNSearch:
    """
    Implementa el "KNN Secuencial" (Paso 1 del Proyecto 2 Multimedia).
    
    1.  Construye una base de datos de vectores TF-IDF + Normas
        para todas las imágenes de la colección.
    2.  Realiza búsquedas secuenciales (fuerza bruta) usando
        Similitud Coseno y un heap para el Top-K.
    """
    
    def __init__(self, k_clusters: int, data_dir: str = 'data'):
        """
        Args:
            k_clusters: El K del K-Means (tamaño del vocabulario).
        """
        self.k = k_clusters
        self.data_dir = data_dir
        
        # 1. Inicializar el generador de histogramas (Paso 3 anterior)
        # Esto cargará el codebook (K-Means entrenado)
        try:
            self.hist_builder = BoVWHistogramBuilder(k_clusters, data_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Asegúrate de haber construido el codebook primero.")
            raise
            
        # Ruta al archivo de la base de datos KNN (Paso 4)
        self.db_path = os.path.join(data_dir, f"mm_knn_seq_db_k{self.k}.pkl")
        
        # Estos se llenarán al construir o cargar la BD
        self.idf_vector: Optional[np.ndarray] = None
        self.image_ids: List[Any] = []
        self.tfidf_vectors: Optional[np.ndarray] = None
        self.vector_norms: Optional[np.ndarray] = None

    def build_database(self, image_id_path_tuples: Iterable[Tuple[Any, str]]):
        """
        Construye la base de datos TF-IDF y la guarda en disco.
        
        Args:
            image_id_path_tuples: Un iterable de (id_imagen, ruta_imagen)
        """
        print("Iniciando construcción de BD KNN Secuencial (TF-IDF)...")
        
        raw_histograms: Dict[Any, np.ndarray] = {}
        doc_count = 0
        # Doc Frequency (df): cuántos documentos contienen cada palabra visual
        df = np.zeros(self.k, dtype=np.int32) 
        
        # --- Fase 1: Calcular TF y DF ---
        print("Fase 1/3: Generando histogramas (TF) y calculando DF...")
        for img_id, img_path in image_id_path_tuples:
            hist_tf = self.hist_builder.create_histogram_from_path(img_path)
            
            if hist_tf is not None:
                raw_histograms[img_id] = hist_tf
                # Actualizar DF: +1 para cada palabra visual presente
                df[hist_tf > 0] += 1
                doc_count += 1
                
            if doc_count % 500 == 0:
                print(f"  ... {doc_count} histogramas procesados.")

        if doc_count == 0:
            print("Error: No se procesaron imágenes.")
            return

        # --- Fase 2: Calcular IDF ---
        print("Fase 2/3: Calculando vector IDF...")
        # IDF Suavizado: log( (N+1) / (df+1) ) + 1
        # Se suma 1 para evitar división por cero si una palabra visual
        # nunca aparece (df=0) o si aparece en todas (N=df).
        N = doc_count
        self.idf_vector = np.log((N + 1) / (df + 1)) + 1.0
        
        # --- Fase 3: Calcular TF-IDF y Normas ---
        print("Fase 3/3: Calculando vectores TF-IDF y Normas...")
        image_ids_list = []
        tfidf_vectors_list = []
        vector_norms_list = []
        
        for img_id, hist_tf in raw_histograms.items():
            # Ponderación TF-IDF 
            tfidf_vec = hist_tf * self.idf_vector
            
            # Calcular Norma Euclidiana (L2-norm)
            norm = np.linalg.norm(tfidf_vec)
            
            if norm > 0:
                image_ids_list.append(img_id)
                tfidf_vectors_list.append(tfidf_vec)
                vector_norms_list.append(norm)

        # Convertir listas a arrays de numpy eficientes
        self.image_ids = image_ids_list
        self.tfidf_vectors = np.array(tfidf_vectors_list, dtype=np.float32)
        self.vector_norms = np.array(vector_norms_list, dtype=np.float32)

        # Guardar en disco
        db_data = {
            'ids': self.image_ids,
            'tfidf_vectors': self.tfidf_vectors,
            'vector_norms': self.vector_norms,
            'idf_vector': self.idf_vector
        }
        
        with open(self.db_path, 'wb') as f:
            pickle.dump(db_data, f)
            
        print(f"¡Construcción completada! BD guardada en {self.db_path}")
        print(f"  -> Vectores TF-IDF: {self.tfidf_vectors.shape}")

    def load_database(self):
        """Carga la base de datos TF-IDF desde el disco a la RAM."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Archivo de BD no encontrado: {self.db_path}")
            
        print(f"Cargando BD KNN Secuencial desde {self.db_path}...")
        with open(self.db_path, 'rb') as f:
            db_data = pickle.load(f)
            
        self.image_ids = db_data['ids']
        self.tfidf_vectors = db_data['tfidf_vectors']
        self.vector_norms = db_data['vector_norms']
        self.idf_vector = db_data['idf_vector']
        print(f"  -> BD cargada. {len(self.image_ids)} vectores en RAM.")

    def search_by_path(self, query_image_path: str, top_k: int = 10) -> List[Tuple[float, Any]]:
        """
        Busca las Top-K imágenes más similares a una imagen de consulta.
        
        Args:
            query_image_path: Ruta a la imagen de consulta.
            top_k: Número de resultados a devolver.
            
        Returns:
            Lista de tuplas (score_similitud, id_imagen)
        """
        # 1. Asegurarse de que la BD esté en RAM
        if self.tfidf_vectors is None:
            self.load_database()
            
        # 2. Generar Histograma (TF) para la consulta [cite: 125]
        q_hist_tf = self.hist_builder.create_histogram_from_path(query_image_path)
        if q_hist_tf is None:
            print("Error: No se pudo procesar la imagen de consulta.")
            return []
            
        # 3. Ponderar (TF-IDF) y normalizar la consulta
        q_tfidf_vec = q_hist_tf * self.idf_vector
        q_norm = np.linalg.norm(q_tfidf_vec)
        
        if q_norm == 0:
            print("Advertencia: El vector de consulta es cero (sin características).")
            return []
            
        # 4. Calcular Similitud Coseno (Sequential Scan) [cite: 127]
        # SimCos(Q, D) = (Q · D) / (|Q| * |D|)
        
        # (Q · D) -> Producto punto de Q con TODOS los vectores de la BD
        # np.dot es mucho más rápido que un bucle en Python
        dot_products = np.dot(self.tfidf_vectors, q_tfidf_vec)
        
        # (|Q| * |D|) -> Producto de normas
        norm_products = self.vector_norms * q_norm
        
        # División elemento a elemento
        similarities = dot_products / norm_products

        # 5. Usar un heap para mantener el Top-K 
        top_k_heap: List[Tuple[float, Any]] = []
        
        for i, score in enumerate(similarities):
            img_id = self.image_ids[i]
            
            # heapq es un min-heap, así que guardamos (score, id)
            if len(top_k_heap) < top_k:
                heapq.heappush(top_k_heap, (score, img_id))
            else:
                # Reemplaza el más pequeño si el actual es más grande
                heapq.heappushpop(top_k_heap, (score, img_id))
                
        # 6. Devolver resultados ordenados de mayor a menor
        return sorted(top_k_heap, reverse=True)

    def search_by_bytes(self, query_image_bytes: bytes, top_k: int = 10) -> List[Tuple[float, Any]]:
        """
        Busca las Top-K imágenes más similares a unos bytes de imagen.
        (La lógica es idéntica a search_by_path)
        """
        if self.tfidf_vectors is None:
            self.load_database()
            
        q_hist_tf = self.hist_builder.create_histogram_from_bytes(query_image_bytes)
        if q_hist_tf is None:
            return []
            
        q_tfidf_vec = q_hist_tf * self.idf_vector
        q_norm = np.linalg.norm(q_tfidf_vec)
        
        if q_norm == 0:
            return []
            
        dot_products = np.dot(self.tfidf_vectors, q_tfidf_vec)
        norm_products = self.vector_norms * q_norm
        similarities = dot_products / norm_products

        top_k_heap: List[Tuple[float, Any]] = []
        for i, score in enumerate(similarities):
            img_id = self.image_ids[i]
            if len(top_k_heap) < top_k:
                heapq.heappush(top_k_heap, (score, img_id))
            else:
                heapq.heappushpop(top_k_heap, (score, img_id))
                
        return sorted(top_k_heap, reverse=True)