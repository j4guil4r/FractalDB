# src/indices/inverted_index/query.py

import os
import pickle
import math
import heapq
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from src.text_processing import preprocess_text

class InvertedIndexQuery:
    
    def __init__(self, data_dir: str = 'data'):
        self.index_file = None
        
        self.final_index_path = os.path.join(data_dir, "inverted_index.dat")
        self.final_meta_path = os.path.join(data_dir, "inverted_index.meta")
        
        if not os.path.exists(self.final_index_path) or not os.path.exists(self.final_meta_path):
            raise FileNotFoundError("No se encontró el índice. Asegúrese de construirlo primero.")
        
        # Cargar los metadatos a la memoria 
        self._load_metadata()
        
        # abrir el archivo de índice y mantenerlo abierto
        try:
            self.index_file = open(self.final_index_path, 'rb')
        except IOError as e:
            print(f"Error al abrir el archivo de índice: {e}")
            if hasattr(self, 'index_file') and self.index_file:
                self.index_file.close()
            raise

    def _load_metadata(self):
        # Carga el lexicón, metadatos de documentos y N

        with open(self.final_meta_path, 'rb') as f:
            final_metadata = pickle.load(f)
        
        self.total_docs: int = final_metadata['total_docs']
        self.doc_metadata: Dict[Any, Tuple[int, float]] = final_metadata['doc_metadata']
        self.lexicon: Dict[str, Tuple[int, int]] = final_metadata['lexicon']
        print("Módulo de consulta: Metadatos cargados.")

    def _get_postings(self, term: str) -> List[Tuple[Any, float]]:
        # Obtiene la lista de postings para un término desde el disco.

        lookup = self.lexicon.get(term)
        if not lookup:
            return []
            
        offset, length = lookup
        
        try:
            self.index_file.seek(offset)
            data = self.index_file.read(length)
            return pickle.loads(data)
        except (IOError, pickle.UnpicklingError) as e:
            print(f"Error al leer postings para el término '{term}': {e}")
            return []

    def close(self):
        if hasattr(self, 'index_file') and self.index_file:
            self.index_file.close()
            print("Módulo de consulta: Archivo de índice cerrado.")

    def __del__(self):
        # Asegurarnos de que el archivo se cierre si el objeto es destruido
        self.close()

    def _calculate_cosine_similarity(self, query_text: str, k: int) -> List[Tuple[float, Any]]:
        """
        Calcula el score de similitud coseno y devuelve el Top-K.
        """
        
        # Procesar la consulta 
        query_terms = preprocess_text(query_text)
        if not query_terms:
            return []

        # Calcular el vector TF-IDF de la consulta
        
        # TF (Logarítmico) de la consulta
        query_tf_map = defaultdict(int)
        for term in query_terms:
            query_tf_map[term] += 1
            
        query_vector = {} 
        query_norm_squared = 0.0
        
        for term, tf in query_tf_map.items():
            if term in self.lexicon:
                q_tf_weight = 1 + math.log10(tf)
                
                postings_count = len(self._get_postings(term)) 
                if postings_count == 0:
                    continue 
                    
                q_idf = math.log10(self.total_docs / postings_count)
                
                q_weight = q_tf_weight * q_idf
                query_vector[term] = q_weight
                
                query_norm_squared += q_weight**2
        
        query_norm = math.sqrt(query_norm_squared)
        
        if query_norm == 0:
            return [] # La consulta no tiene términos relevantes

        # Calcular el Score (Similitud Coseno)
        scores = defaultdict(float)
        
        for term, q_weight in query_vector.items():
            postings = self._get_postings(term)
            
            for docID, d_weight in postings:
                scores[docID] += q_weight * d_weight
                
        # 4. Normalizar y obtener Top-K. Usamos un min-heap para mantener el Top-K eficientemente
        top_k_heap = [] 
        
        for docID, dot_product in scores.items():
            doc_norm = self.doc_metadata[docID][1]
            
            if doc_norm > 0:
                # Sim(q, d) = (V(q).V(d)) / (|V(q)| * |V(d)|)
                final_score = dot_product / (query_norm * doc_norm)
                
                # Lógica del Min-Heap para Top-K
                if len(top_k_heap) < k:
                    heapq.heappush(top_k_heap, (final_score, docID))
                else:
                    # Si es mejor que el peor del heap, reemplazarlo
                    heapq.heappushpop(top_k_heap, (final_score, docID))

        # Ordenar los K resultados finales de mayor a menor
        return sorted(top_k_heap, reverse=True)

    def query(self, query_text: str, k: int = 10) -> List[Tuple[float, Any]]:
        # Función pública para ejecutar una consulta Top-K.

        return self._calculate_cosine_similarity(query_text, k)