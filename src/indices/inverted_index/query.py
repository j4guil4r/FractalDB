# src/indices/inverted_index/query.py

import os
import pickle
import math
import heapq
from collections import defaultdict
from typing import List, Tuple, Dict, Any
# Importamos nuestro módulo de preprocesamiento
from src.text_processing import preprocess_text

class InvertedIndexQuery:
    
    def __init__(self, data_dir: str = 'data'):
        # --- INICIO DE LA SOLUCIÓN ---
        # Pre-declarar index_file como None para que __del__ siempre funcione
        self.index_file = None
        # --- FIN DE LA SOLUCIÓN ---
        
        self.final_index_path = os.path.join(data_dir, "inverted_index.dat")
        self.final_meta_path = os.path.join(data_dir, "inverted_index.meta")
        
        if not os.path.exists(self.final_index_path) or not os.path.exists(self.final_meta_path):
            raise FileNotFoundError("No se encontró el índice. Asegúrese de construirlo primero.")
        
        # 1. Cargar los metadatos a la memoria (esto es pequeño)
        self._load_metadata()
        
        # 2. Abrir el archivo de índice (postings) y mantenerlo abierto
        #    No leemos nada, solo tenemos el puntero al archivo.
        try:
            self.index_file = open(self.final_index_path, 'rb')
        except IOError as e:
            print(f"Error al abrir el archivo de índice: {e}")
            if hasattr(self, 'index_file') and self.index_file:
                self.index_file.close()
            raise

    def _load_metadata(self):
        """Carga el lexicón, metadatos de documentos y N."""
        with open(self.final_meta_path, 'rb') as f:
            final_metadata = pickle.load(f)
        
        self.total_docs: int = final_metadata['total_docs']
        # docID -> (length, norm)
        self.doc_metadata: Dict[Any, Tuple[int, float]] = final_metadata['doc_metadata']
        # term -> (offset, length_bytes)
        self.lexicon: Dict[str, Tuple[int, int]] = final_metadata['lexicon']
        print("Módulo de consulta: Metadatos cargados.")

    def _get_postings(self, term: str) -> List[Tuple[Any, float]]:
        """
        Obtiene la lista de postings para un término desde el disco.
        Esto cumple el requisito de "memoria secundaria".
        """
        lookup = self.lexicon.get(term)
        if not lookup:
            return []
            
        offset, length = lookup
        
        try:
            # Mover el puntero del archivo
            self.index_file.seek(offset)
            # Leer solo los bytes necesarios
            data = self.index_file.read(length)
            # Deserializar
            return pickle.loads(data)
        except (IOError, pickle.UnpicklingError) as e:
            print(f"Error al leer postings para el término '{term}': {e}")
            return []

    def close(self):
        """Cierra el archivo de índice."""
        # --- MODIFICADO ---
        # Chequeo más robusto
        if hasattr(self, 'index_file') and self.index_file:
        # --- FIN MODIFICADO ---
            self.index_file.close()
            print("Módulo de consulta: Archivo de índice cerrado.")

    def __del__(self):
        # Asegurarnos de que el archivo se cierre si el objeto es destruido
        self.close()

    def _calculate_cosine_similarity(self, query_text: str, k: int) -> List[Tuple[float, Any]]:
        """
        Calcula el score de similitud coseno y devuelve el Top-K.
        """
        
        # 1. Procesar la consulta (igual que los documentos)
        query_terms = preprocess_text(query_text)
        if not query_terms:
            return []

        # 2. Calcular el vector TF-IDF de la consulta
        
        # TF (Logarítmico) de la consulta
        query_tf_map = defaultdict(int)
        for term in query_terms:
            query_tf_map[term] += 1
            
        query_vector = {} # term -> tf_idf_weight
        query_norm_squared = 0.0
        
        for term, tf in query_tf_map.items():
            # Ver si el término existe en nuestra colección (en el lexicón)
            if term in self.lexicon:
                # TF-peso de la consulta
                q_tf_weight = 1 + math.log10(tf)
                
                # IDF del término (lo calculamos al vuelo)
                postings_count = len(self._get_postings(term)) # df
                if postings_count == 0:
                    continue # Debería estar en el lexicón, pero por seguridad
                    
                q_idf = math.log10(self.total_docs / postings_count)
                
                # Peso final del término en la consulta
                q_weight = q_tf_weight * q_idf
                query_vector[term] = q_weight
                
                query_norm_squared += q_weight**2
        
        query_norm = math.sqrt(query_norm_squared)
        
        if query_norm == 0:
            return [] # La consulta no tiene términos relevantes

        # 3. Calcular el Score (Similitud Coseno)
        # Usamos acumuladores para el producto punto (V(q) . V(d))
        scores = defaultdict(float) # docID -> score (acumulado)
        
        for term, q_weight in query_vector.items():
            # Traer postings desde el disco
            # posting = (docID, d_weight)
            postings = self._get_postings(term)
            
            for docID, d_weight in postings:
                # Acumular el producto punto
                scores[docID] += q_weight * d_weight
                
        # 4. Normalizar y obtener Top-K
        # Usamos un min-heap para mantener el Top-K eficientemente
        top_k_heap = [] # (score, docID)
        
        for docID, dot_product in scores.items():
            # Obtener la norma pre-calculada del documento
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

        # 7. Ordenar los K resultados finales de mayor a menor
        return sorted(top_k_heap, reverse=True)

    def query(self, query_text: str, k: int = 10) -> List[Tuple[float, Any]]:
        """
        Función pública para ejecutar una consulta Top-K.
        """
        return self._calculate_cosine_similarity(query_text, k)