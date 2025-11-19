import os
import pickle
import numpy as np
import math
import heapq
import shutil
from collections import defaultdict
from typing import Iterator, Tuple, Dict, List, Any, Callable

class MMInvertedIndexBuilder:
    #Construye un Índice Invertido para vectores de histogramas multimedia (BoVW).
    #Usa una lógica SPIMI optimizada en 2 pases para no agotar la RAM.
    
    
    def __init__(self, data_dir: str = 'data', k_clusters: int = 0):
        self.data_dir = data_dir
        
        if k_clusters == 0:
            raise ValueError("Se debe especificar k_clusters.")
        self.k = k_clusters
        
        self.temp_block_dir = os.path.join(data_dir, f"spimi_blocks_mm_k{k_clusters}")
        os.makedirs(self.temp_block_dir, exist_ok=True)
        
        self.final_index_path = os.path.join(data_dir, f"mm_inverted_index_k{k_clusters}.dat")
        self.final_meta_path = os.path.join(data_dir, f"mm_inverted_index_k{k_clusters}.meta")

        self.doc_metadata: Dict[Any, Tuple[int, float]] = {} 
        self.total_docs = 0
        self.idf_vector: Optional[np.ndarray] = None
        self.block_file_paths: List[str] = []

    def _write_block_to_disk(self, in_memory_index: Dict[int, list], block_num: int) -> str:
        # Escribe un bloque temporal en disco.
        block_path = os.path.join(self.temp_block_dir, f"block_{block_num:04d}.pkl")
        sorted_terms = sorted(in_memory_index.keys())
        with open(block_path, 'wb') as f:
            for term_id in sorted_terms:
                postings = in_memory_index[term_id]
                pickle.dump((term_id, postings), f)
        self.block_file_paths.append(block_path)
        return block_path

    def build(self, hist_iterator_factory: Callable[[], Iterator[Tuple[Any, np.ndarray]]]):
        #Construye el índice en 2 pases + merge, sin cargar todo a RAM.
        
        # --- Fase 1: Calcular DF (Document Frequency) ---
        print("MM-Index (Fase 1/3): Calculando DF...")
        df = np.zeros(self.k, dtype=np.int32)
        self.total_docs = 0
        
        for img_id, hist_tf in hist_iterator_factory():
            self.total_docs += 1
            if hist_tf is not None and np.sum(hist_tf) > 0:
                # Actualizar DF: +1 para cada palabra visual presente
                df[hist_tf > 0] += 1
        
        if self.total_docs == 0:
            print("Error: No se encontraron histogramas. Abortando.")
            return

        print(f"MM-Index (Fase 1/3): Completada. {self.total_docs} imágenes.")
        
        # Calcular IDF
        N = self.total_docs
        self.idf_vector = np.log((N + 1) / (df + 1)) + 1.0

        # --- Fase 2: Calcular Normas y Construir Bloques SPIMI ---
        print("MM-Index (Fase 2/3): Calculando Normas y generando bloques SPIMI...")
        block_num = 0
        in_memory_index = defaultdict(list)
        
        for img_id, hist_tf in hist_iterator_factory():
            if hist_tf is None or np.sum(hist_tf) == 0:
                self.doc_metadata[img_id] = (0, 0.0)
                continue

            # 2a. Calcular y guardar la norma (L2-norm)
            tfidf_vec = hist_tf * self.idf_vector
            norm = np.linalg.norm(tfidf_vec)
            self.doc_metadata[img_id] = (np.sum(hist_tf), norm)

            # 2b. Añadir a los bloques SPIMI (solo el TF crudo)
            for term_id in np.nonzero(hist_tf)[0]:
                tf = int(hist_tf[term_id])
                in_memory_index[int(term_id)].append((img_id, tf))

            # (Lógica de volcado de SPIMI simplificada)
            if self.total_docs % 5000 == 0: # Volcar cada 5000 docs
                self._write_block_to_disk(in_memory_index, block_num)
                block_num += 1
                in_memory_index.clear()
        
        if in_memory_index:
            self._write_block_to_disk(in_memory_index, block_num)
        
        print("MM-Index (Fase 2/3): Completada.")
        
        # --- Fase 3: Fusión de Bloques (Merge) ---
        print("MM-Index (Fase 3/3): Iniciando Fusión de Bloques...")
        self._merge_blocks()
        
        print(f"¡Construcción de Índice Invertido Multimedia completada!")
        print(f"Índice: {self.final_index_path}")
        print(f"Metadatos: {self.final_meta_path}")

    def _merge_blocks(self):
        #Fase 2 de SPIMI: Fusiona k-bloques usando un min-heap. Calcula TF-IDF y escribe el índice y lexicón finales.

        lexicon = {} 
        heap = []
        block_files = []
        
        if not self.block_file_paths:
            print("No se encontraron bloques para fusionar.")
            self._save_final_metadata(lexicon)
            return

        try:
            for i, block_path in enumerate(self.block_file_paths):
                f = open(block_path, 'rb')
                block_files.append(f)
                try:
                    term_id, postings = pickle.load(f)
                    heapq.heappush(heap, (term_id, i, postings))
                except EOFError:
                    f.close()

            with open(self.final_index_path, 'wb') as f_out:
                
                while heap:
                    current_term_id, block_idx, first_postings = heapq.heappop(heap)
                    merged_postings = first_postings
                    
                    while heap and heap[0][0] == current_term_id:
                        _, next_block_idx, next_postings = heapq.heappop(heap)
                        merged_postings.extend(next_postings)
                        try:
                            term_id, postings = pickle.load(block_files[next_block_idx])
                            heapq.heappush(heap, (term_id, next_block_idx, postings))
                        except EOFError:
                            block_files[next_block_idx].close()
                    
                    try:
                        term_id, postings = pickle.load(block_files[block_idx])
                        heapq.heappush(heap, (term_id, block_idx, postings))
                    except EOFError:
                        block_files[block_idx].close()
                        
                    idf = self.idf_vector[current_term_id]
                    
                    weighted_postings = []
                    for img_id, tf in merged_postings:
                        final_weight = tf * idf 
                        weighted_postings.append((img_id, final_weight))

                    current_offset = f_out.tell()
                    data_to_write = pickle.dumps(weighted_postings)
                    f_out.write(data_to_write)
                    
                    lexicon[current_term_id] = (current_offset, len(data_to_write))

            print("MM-Index (Fase 3/3): Fusión completada.")
            
            self._save_final_metadata(lexicon)

        finally:
            for f in block_files:
                if not f.closed:
                    f.close()
            if os.path.exists(self.temp_block_dir):
                shutil.rmtree(self.temp_block_dir)
            print("Directorio temporal de bloques MM eliminado.")

    def _save_final_metadata(self, lexicon: Dict):
        #Guarda el lexicón final, los metadatos de documentos (normas) y N.

        final_metadata = {
            'k': self.k,
            'total_docs': self.total_docs,
            'doc_metadata': self.doc_metadata, 
            'idf_vector': self.idf_vector,
            'lexicon': lexicon 
        }
        with open(self.final_meta_path, 'wb') as f:
            pickle.dump(final_metadata, f)
        print(f"Metadatos finales (MM) y Lexicón guardados en {self.final_meta_path}")