# src/multimedia/inverted_index_builder_mm.py

import os
import pickle
import numpy as np
import math
import heapq
import shutil
from collections import defaultdict
from typing import Iterator, Tuple, Dict, List, Any

class MMInvertedIndexBuilder:
    """
    Construye un Índice Invertido para vectores de histogramas multimedia (BoVW).
    
    Usa la misma lógica SPIMI que el índice de texto, pero los "términos"
    son los IDs de los clusters (palabras visuales, 0 a K-1) y los
    "documentos" son las imágenes (img_id).
    
    El peso TF-IDF se calcula y almacena en las listas de postings.
    """
    
    def __init__(self, data_dir: str = 'data', k_clusters: int = 0):
        self.data_dir = data_dir
        
        # Necesitamos K para saber el IDF
        if k_clusters == 0:
            raise ValueError("Se debe especificar k_clusters.")
        self.k = k_clusters
        
        # 1. Archivos temporales (SPIMI)
        self.temp_block_dir = os.path.join(data_dir, f"spimi_blocks_mm_k{k_clusters}")
        os.makedirs(self.temp_block_dir, exist_ok=True)
        
        # 2. Archivos finales
        self.final_index_path = os.path.join(data_dir, f"mm_inverted_index_k{k_clusters}.dat")
        self.final_meta_path = os.path.join(data_dir, f"mm_inverted_index_k{k_clusters}.meta")

        # 3. Metadatos
        self.doc_metadata: Dict[Any, Tuple[int, float]] = {} # img_id -> (length, norm)
        self.total_docs = 0
        self.idf_vector: Optional[np.ndarray] = None
        self.block_file_paths: List[str] = []

    def _write_block_to_disk(self, in_memory_index: Dict[int, list], block_num: int) -> str:
        """ Escribe un bloque temporal en disco. """
        block_path = os.path.join(self.temp_block_dir, f"block_{block_num:04d}.pkl")
        
        # Ordenar los "términos" (IDs de cluster) numéricamente
        sorted_terms = sorted(in_memory_index.keys())
        
        with open(block_path, 'wb') as f:
            for term_id in sorted_terms:
                postings = in_memory_index[term_id]
                # Escribimos (id_palabra_visual, lista_de_postings)
                # Formato posting: (img_id, term_frequency)
                pickle.dump((term_id, postings), f)
                
        self.block_file_paths.append(block_path)
        return block_path

    def build(self, hist_iterator: Iterator[Tuple[Any, np.ndarray]]):
        """
        Construye el índice completo (Fase 1 y 2) desde un iterador
        de histogramas (TF crudos).
        
        Args:
            hist_iterator: Un generador que produce (img_id, hist_tf)
        """
        
        # --- Fase 1: Generar Bloques SPIMI y Calcular DF ---
        print("MM-Index (Fase 1): Generando bloques y calculando DF...")
        block_num = 0
        in_memory_index = defaultdict(list)
        df = np.zeros(self.k, dtype=np.int32)
        
        temp_hist_data = {} # Guardamos temporalmente para Fase 2

        for img_id, hist_tf in hist_iterator:
            self.total_docs += 1
            if self.total_docs % 1000 == 0:
                print(f"  ... {self.total_docs} histogramas procesados.")

            if hist_tf is None or np.sum(hist_tf) == 0:
                continue

            # Guardamos el TF para la Fase 2
            temp_hist_data[img_id] = hist_tf
            
            # Iterar solo sobre las "palabras" que SÍ aparecen (eficiencia)
            # np.nonzero() devuelve los índices donde el histograma no es cero
            for term_id in np.nonzero(hist_tf)[0]:
                tf = int(hist_tf[term_id])
                
                # Actualizar DF
                df[term_id] += 1
                
                # Añadir al índice en memoria (Término = ID del cluster)
                in_memory_index[int(term_id)].append((img_id, tf))

            # Lógica de volcado de SPIMI (simplificada, no chequea memoria)
            # (En un caso real, chequearíamos sys.getsizeof)
            if self.total_docs % 5000 == 0:
                self._write_block_to_disk(in_memory_index, block_num)
                block_num += 1
                in_memory_index.clear()
        
        # Volcar el último bloque
        if in_memory_index:
            self._write_block_to_disk(in_memory_index, block_num)

        print(f"MM-Index (Fase 1): Completada. {self.total_docs} imágenes.")
        
        # --- Fase 2: Calcular IDF y Normas (pre-fusión) ---
        print("MM-Index (Fase 2a): Calculando IDF y Normas...")
        N = self.total_docs
        # IDF Suavizado (mismo que en KNN Secuencial)
        self.idf_vector = np.log((N + 1) / (df + 1)) + 1.0

        # Calcular la Norma (L2-norm) para CADA documento (imagen)
        for img_id, hist_tf in temp_hist_data.items():
            tfidf_vec = hist_tf * self.idf_vector
            norm = np.linalg.norm(tfidf_vec)
            # (length, norm) - 'length' no es tan relevante aquí
            self.doc_metadata[img_id] = (np.sum(hist_tf), norm)

        # Liberar memoria
        del temp_hist_data 
        
        # --- Fase 3: Fusión de Bloques (Merge) ---
        print("MM-Index (Fase 2b): Iniciando Fusión de Bloques...")
        self._merge_blocks()
        
        print(f"¡Construcción de Índice Invertido Multimedia completada!")
        print(f"Índice: {self.final_index_path}")
        print(f"Metadatos: {self.final_meta_path}")

    def _merge_blocks(self):
        """
        Fase 2 de SPIMI: Fusiona k-bloques usando un min-heap.
        Calcula TF-IDF y escribe el índice y lexicón finales.
        """
        lexicon = {} # El diccionario final: {'term_id': (offset, length)}
        heap = []
        block_files = []
        
        if not self.block_file_paths:
            print("No se encontraron bloques para fusionar.")
            return

        try:
            # 1. Abrir todos los archivos de bloque y "cebar" el heap
            for i, block_path in enumerate(self.block_file_paths):
                f = open(block_path, 'rb')
                block_files.append(f)
                try:
                    term_id, postings = pickle.load(f)
                    heapq.heappush(heap, (term_id, i, postings))
                except EOFError:
                    f.close()

            # 2. Abrir el archivo de índice final
            with open(self.final_index_path, 'wb') as f_out:
                
                while heap:
                    # 3. Tomar el término (palabra visual) más pequeño
                    current_term_id, block_idx, first_postings = heapq.heappop(heap)
                    merged_postings = first_postings
                    
                    # 4. Combinar todos los postings para este término
                    while heap and heap[0][0] == current_term_id:
                        _, next_block_idx, next_postings = heapq.heappop(heap)
                        merged_postings.extend(next_postings)
                        # Recargar el heap desde el bloque que acabamos de usar
                        try:
                            term_id, postings = pickle.load(block_files[next_block_idx])
                            heapq.heappush(heap, (term_id, next_block_idx, postings))
                        except EOFError:
                            block_files[next_block_idx].close()
                    
                    # 5. Recargar el heap desde el primer bloque
                    try:
                        term_id, postings = pickle.load(block_files[block_idx])
                        heapq.heappush(heap, (term_id, block_idx, postings))
                    except EOFError:
                        block_files[block_idx].close()
                        
                    # 6. Ponderar (TF-IDF) y Escribir
                    idf = self.idf_vector[current_term_id]
                    
                    weighted_postings = []
                    for img_id, tf in merged_postings:
                        final_weight = tf * idf # TF ya es lineal, no logarítmico
                        weighted_postings.append((img_id, final_weight))

                    # 7. Escribir los postings finales y guardar la posición
                    current_offset = f_out.tell()
                    data_to_write = pickle.dumps(weighted_postings)
                    f_out.write(data_to_write)
                    
                    # Guardar en el lexicón (la clave es el ID numérico)
                    lexicon[current_term_id] = (current_offset, len(data_to_write))

            print("MM-Index (Fase 2b): Fusión completada.")
            
            # 8. Guardar metadatos finales
            self._save_final_metadata(lexicon)

        finally:
            # 9. Limpieza
            for f in block_files:
                if not f.closed:
                    f.close()
            if os.path.exists(self.temp_block_dir):
                shutil.rmtree(self.temp_block_dir)
            print("Directorio temporal de bloques MM eliminado.")

    def _save_final_metadata(self, lexicon: Dict):
        """
        Guarda el lexicón final, los metadatos de documentos (normas) y N.
        """
        final_metadata = {
            'k': self.k,
            'total_docs': self.total_docs,
            'doc_metadata': self.doc_metadata, # {img_id: (length, norm)}
            'idf_vector': self.idf_vector,
            'lexicon': lexicon # {term_id: (offset, length_bytes)}
        }
        with open(self.final_meta_path, 'wb') as f:
            pickle.dump(final_metadata, f)
        print(f"Metadatos finales (MM) y Lexicón guardados en {self.final_meta_path}")