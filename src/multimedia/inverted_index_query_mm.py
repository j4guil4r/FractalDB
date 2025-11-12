# src/multimedia/inverted_index_query_mm.py

import os
import pickle
import numpy as np
import heapq
import math
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

# Importamos los módulos multimedia necesarios
from src.multimedia.histogram_builder import BoVWHistogramBuilder

class MMInvertedIndexQuery:
    """
    Realiza búsquedas KNN rápidas (Top-K) utilizando un índice
    invertido multimedia pre-construido.
    
    Usa el mismo 'scoring' de similitud coseno que el índice textual.
    """
    
    def __init__(self, k_clusters: int, data_dir: str = 'data'):
        self.k = k_clusters
        self.data_dir = data_dir

        self.index_path = os.path.join(data_dir, f"mm_inverted_index_k{self.k}.dat")
        self.meta_path = os.path.join(data_dir, f"mm_inverted_index_k{self.k}.meta")

        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Índice MM (k={self.k}) no encontrado. "
                                    "Asegúrate de construirlo primero.")

        # 1. Cargar metadatos (Lexicón, Normas, IDF) en RAM
        self._load_metadata()

        # 2. Inicializar el generador de histogramas
        # (Esto cargará el codebook K-Means)
        try:
            self.hist_builder = BoVWHistogramBuilder(k_clusters, data_dir)
        except FileNotFoundError as e:
            print(f"Error fatal: {e}")
            raise

        # 3. Abrir el archivo de postings (.dat)
        try:
            self.index_file = open(self.index_path, 'rb')
        except IOError as e:
            print(f"Error al abrir el archivo de índice MM: {e}")
            raise

        print(f"Módulo de consulta KNN Indexado (K={self.k}) listo.")

    def _load_metadata(self):
        """Carga el lexicón, metadatos de documentos, K e IDF."""
        with open(self.meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        self.k: int = meta['k']
        self.total_docs: int = meta['total_docs']
        # {img_id: (length, norm)}
        self.doc_metadata: Dict[Any, Tuple[float, float]] = meta['doc_metadata']
        # {term_id (int): (offset, length_bytes)}
        self.lexicon: Dict[int, Tuple[int, int]] = meta['lexicon']
        # Vector IDF precalculado
        self.idf_vector: np.ndarray = meta['idf_vector']
        print(f"Metadatos MM (K={self.k}) cargados. {self.total_docs} imágenes.")

    def _get_postings(self, term_id: int) -> List[Tuple[Any, float]]:
        """
        Obtiene la lista de postings para una palabra visual (term_id)
        leyendo desde el disco (memoria secundaria).
        """
        lookup = self.lexicon.get(term_id)
        if not lookup:
            return []
            
        offset, length = lookup
        
        try:
            self.index_file.seek(offset)
            data = self.index_file.read(length)
            # Retorna [(img_id, tfidf_weight), ...]
            return pickle.loads(data)
        except Exception as e:
            print(f"Error al leer postings para term_id '{term_id}': {e}")
            return []

    def close(self):
        """Cierra el archivo de índice."""
        if hasattr(self, 'index_file') and self.index_file:
            self.index_file.close()
            print("Módulo de consulta MM: Archivo de índice cerrado.")

    def __del__(self):
        self.close()

    def _calculate_similarity(self, q_hist_tf: np.ndarray, top_k: int) -> List[Tuple[float, Any]]:
        """
        Función interna para calcular la similitud coseno usando
        el índice invertido.
        """
        if q_hist_tf is None or np.sum(q_hist_tf) == 0:
            return []

        # 1. Calcular el vector TF-IDF de la consulta
        q_tfidf_vec = q_hist_tf * self.idf_vector
        q_norm = np.linalg.norm(q_tfidf_vec)

        if q_norm == 0:
            return []

        # 2. Calcular Scores (Similitud Coseno)
        # Usamos acumuladores para el producto punto (V(q) . V(d))
        scores = defaultdict(float) # img_id -> score (acumulado)
        
        # Iterar solo sobre las "palabras visuales" que SÍ están en la consulta
        # (Esto es lo que hace que sea rápido)
        query_word_ids = np.nonzero(q_hist_tf)[0]
        
        for term_id in query_word_ids:
            q_weight = q_tfidf_vec[term_id]
            
            # Traer postings desde el disco
            # posting = (img_id, d_weight)
            postings = self._get_postings(int(term_id)) # Asegurar que es int
            
            for img_id, d_weight in postings:
                # Acumular el producto punto
                scores[img_id] += q_weight * d_weight
                
        # 3. Normalizar y obtener Top-K
        # Usamos un min-heap
        top_k_heap = [] # (score, img_id)
        
        for img_id, dot_product in scores.items():
            # Obtener la norma pre-calculada de la imagen
            d_norm = self.doc_metadata[img_id][1]
            
            if d_norm > 0:
                final_score = dot_product / (q_norm * d_norm)
                
                if len(top_k_heap) < top_k:
                    heapq.heappush(top_k_heap, (final_score, img_id))
                else:
                    heapq.heappushpop(top_k_heap, (final_score, img_id))

        # 4. Ordenar los K resultados finales
        return sorted(top_k_heap, reverse=True)

    def query_by_path(self, query_image_path: str, top_k: int = 10) -> List[Tuple[float, Any]]:
        """
        Función pública para buscar por similitud usando la ruta de una imagen.
        """
        # 1. Convertir imagen de consulta en histograma (TF)
        q_hist_tf = self.hist_builder.create_histogram_from_path(query_image_path)
        
        # 2. Calcular similitud
        return self._calculate_similarity(q_hist_tf, top_k)

    def query_by_bytes(self, query_image_bytes: bytes, top_k: int = 10) -> List[Tuple[float, Any]]:
        """
        Función pública para buscar por similitud usando los bytes de una imagen.
        """
        # 1. Convertir imagen de consulta en histograma (TF)
        q_hist_tf = self.hist_builder.create_histogram_from_bytes(query_image_bytes)
        
        # 2. Calcular similitud
        return self._calculate_similarity(q_hist_tf, top_k)