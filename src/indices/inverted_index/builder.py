# src/indices/inverted_index/builder.py

import os
import pickle
import sys
import math
import heapq 
import shutil 
from collections import defaultdict
from typing import Iterator, Tuple, Dict, List, Any
from src.text_processing import preprocess_text
import psutil

class InvertedIndexBuilder:
    
    def __init__(self, data_dir: str, index_name: str = "inverted_index", block_size_limit_mb: int = 128):
        self.data_dir = data_dir
        
        # Directorio temporal para SPIMI
        self.temp_block_dir = os.path.join(data_dir, "spimi_blocks")
        os.makedirs(self.temp_block_dir, exist_ok=True)
        
        # Archivo de metadatos temporal
        self.temp_meta_path = os.path.join(self.temp_block_dir, "spimi_build.meta")
        
        # Archivos del índice final
        self.final_index_path = os.path.join(data_dir, f"{index_name}.dat")
        self.final_meta_path = os.path.join(data_dir, f"{index_name}.meta")

        self.block_size_limit = block_size_limit_mb * 1024 * 1024
        
        self.block_file_paths: List[str] = []
        self.doc_metadata: Dict[Any, Tuple[int, float]] = {}
        self.total_docs = 0

    def _write_block_to_disk(self, in_memory_index: Dict[str, list], block_num: int) -> str:
        # Escribe un bloque y devuelve la ruta del archivo.
        if not in_memory_index:
            return ""

        block_path = os.path.join(self.temp_block_dir, f"block_{block_num:04d}.pkl")
        
        sorted_terms = sorted(in_memory_index.keys())
        
        with open(block_path, 'wb') as f:
            for term in sorted_terms:
                postings = in_memory_index[term]
                pickle.dump((term, postings), f)
                
        print(f"SPIMI (Fase 1): Bloque {block_num} escrito en {block_path}")
        return block_path

    def build_blocks(self, document_iterator: Iterator[Tuple[Any, str]]):
        # Fase 1 de SPIMI: Genera bloques temporales.
        self.total_docs = 0
        block_num = 0
        in_memory_index = defaultdict(list)
        generated_block_paths = []
        
        print("Iniciando SPIMI (Fase 1): Generación de Bloques...")

        for docID, text_content in document_iterator:
            self.total_docs += 1
            if self.total_docs % 1000 == 0:
                print(f"SPIMI (Fase 1): Procesando documento {self.total_docs}...")

            terms = preprocess_text(text_content)
            
            if not terms:
                self.doc_metadata[docID] = (0, 0.0)
                continue

            term_freqs = defaultdict(int)
            for term in terms:
                term_freqs[term] += 1
            
            # Usamos log-tf weighting: (1 + log(tf))
            doc_norm_squared = 0.0
            for term, tf in term_freqs.items():
                # Almacenamos el tf simple por ahora
                in_memory_index[term].append((docID, tf))
                
                # Calculamos la norma usando el peso logarítmico
                tf_weight = 1 + math.log10(tf)
                doc_norm_squared += tf_weight**2
            
            self.doc_metadata[docID] = (len(terms), math.sqrt(doc_norm_squared))

            current_size_bytes = sys.getsizeof(in_memory_index)
            if current_size_bytes > self.block_size_limit:
                path = self._write_block_to_disk(in_memory_index, block_num)
                if path: generated_block_paths.append(path)
                block_num += 1
                in_memory_index.clear()
        
        if in_memory_index:
            path = self._write_block_to_disk(in_memory_index, block_num)
            if path: generated_block_paths.append(path)
            
        print(f"SPIMI (Fase 1): Generación de bloques completada.")
        print(f"  -> Documentos totales: {self.total_docs}")
        print(f"  -> Bloques temporales creados: {len(generated_block_paths)}")
        
        self.block_file_paths = generated_block_paths
        self._save_temp_metadata()

    def _save_temp_metadata(self):
        # Guarda los metadatos temporales necesarios para la fase de merge. 
        metadata = {
            'total_docs': self.total_docs,
            'doc_metadata': self.doc_metadata,
            'block_file_paths': self.block_file_paths
        }
        with open(self.temp_meta_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadatos temporales de construcción guardados.")

    def _load_temp_metadata(self):
        # Carga los metadatos de la Fase 1 para iniciar la Fase 2.
        with open(self.temp_meta_path, 'rb') as f:
            metadata = pickle.load(f)
        self.total_docs = metadata['total_docs']
        self.doc_metadata = metadata['doc_metadata']
        self.block_file_paths = metadata['block_file_paths']
        print("Metadatos temporales cargados para la Fase 2.")

    # Merge
    def _merge_blocks(self):
        print("Iniciando SPIMI (Fase 2): Fusión de Bloques...")
        self._load_temp_metadata()
        
        if not self.block_file_paths:
            print("No se encontraron bloques para fusionar. Abortando.")
            return
        
        try:
            # Obtiene la RAM disponible real del sistema
            available_ram = psutil.virtual_memory().available
            TOTAL_RAM_TO_USE = int(available_ram * 0.90)
            print(f"Detectada RAM disponible: {available_ram/1024**3:.2f} GB. Usando 90%: {TOTAL_RAM_TO_USE/1024**3:.2f} GB")
        except ImportError:
            print("Librería 'psutil' no encontrada. Usando configuración manual de memoria.")
            TOTAL_RAM_TO_USE = 2 * 1024 * 1024 * 1024

        num_blocks = len(self.block_file_paths)
        
        # Calculamos el buffer por archivo
        buffer_per_file = int(TOTAL_RAM_TO_USE / num_blocks)
        
        # Límite de seguridad inferior (para no romper open() con 0 bytes)
        if buffer_per_file < 32768: buffer_per_file = 32768

        MAX_C_INT = 2147483647
        if buffer_per_file >= MAX_C_INT:
            print(f"Advertencia: El buffer calculado excede el límite de Python. Ajustando a 2GB por archivo.")
            buffer_per_file = MAX_C_INT - 1024

        lexicon = {} # El diccionario final: {'term': (offset, length)}
        heap = []
        block_files = []

        try:
            # Abrir todos los archivos de bloque
            for i, block_path in enumerate(self.block_file_paths):
                f = open(block_path, 'rb', buffering=buffer_per_file)
                block_files.append(f)
                
                # "Priming the heap"
                try:
                    term, postings = pickle.load(f)
                    heapq.heappush(heap, (term, i, postings))
                except EOFError:
                    f.close()

            write_buffer = min(buffer_per_file, 50 * 1024 * 1024)

            # Abrir el archivo de índice final (donde irán los postings)
            with open(self.final_index_path, 'wb', buffering=write_buffer) as f_out:
                counter = 0
                process = psutil.Process()
                # Iniciar el K-way merge
                while heap:
                    # --- MONITOREO DE RAM ---
                    if counter % 1000 == 0:
                        mem_info = process.memory_info()
                        # RSS: Resident Set Size (Memoria física real usada)
                        print(f"Procesando término {counter}... RAM usada por Python: {mem_info.rss / 1024 / 1024:.2f} MB")
                    counter += 1
                    # ------------------------
                    
                    # Tomar el término alfabéticamente menor
                    current_term, block_idx, first_postings = heapq.heappop(heap)
                    merged_postings = first_postings
                    
                    # Combinar todos los postings para este término
                    while heap and heap[0][0] == current_term:
                        _, next_block_idx, next_postings = heapq.heappop(heap)
                        merged_postings.extend(next_postings)
                        
                        # Recargar el heap desde el bloque que acabamos de usar
                        try:
                            term, postings = pickle.load(block_files[next_block_idx])
                            heapq.heappush(heap, (term, next_block_idx, postings))
                        except EOFError:
                            block_files[next_block_idx].close()
                    
                    # Recargar el heap desde el primer bloque
                    try:
                        term, postings = pickle.load(block_files[block_idx])
                        heapq.heappush(heap, (term, block_idx, postings))
                    except EOFError:
                        block_files[block_idx].close()
                        
                    # Calcular TF-IDF
                    
                    df = len(merged_postings)
                    idf = math.log10(self.total_docs / df)
                    
                    weighted_postings = []
                    for docID, tf in merged_postings:
                        tf_weight = 1 + math.log10(tf)
                        final_weight = tf_weight * idf
                        weighted_postings.append((docID, final_weight))

                    # Escribir los postings finales y guardar la posición
                    current_offset = f_out.tell()
                    data_to_write = pickle.dumps(weighted_postings)
                    f_out.write(data_to_write)
                    
                    lexicon[current_term] = (current_offset, len(data_to_write))

            print("SPIMI (Fase 2): Fusión completada.")
            
            self._save_final_metadata(lexicon)

        finally:
            # Limpieza
            for f in block_files:
                if not f.closed:
                    f.close()
            
            if os.path.exists(self.temp_block_dir):
                shutil.rmtree(self.temp_block_dir)
            print("Directorio temporal de bloques eliminado.")

    def _save_final_metadata(self, lexicon: Dict):
        # Guarda el lexicón final, los metadatos de documentos (normas) y N

        final_metadata = {
            'total_docs': self.total_docs,
            'doc_metadata': self.doc_metadata,
            'lexicon': lexicon
        }
        with open(self.final_meta_path, 'wb') as f:
            pickle.dump(final_metadata, f)
        print(f"Metadatos finales y Lexicón guardados en {self.final_meta_path}")

    def build(self, document_iterator: Iterator[Tuple[Any, str]]):
        # Función principal para construir el índice completo
        
        # Generar bloques
        self.build_blocks(document_iterator)
        
        # Fusionar bloques
        self._merge_blocks()
        
        print(f"¡Construcción de Índice Invertido completada!")
        print(f"Índice: {self.final_index_path}")
        print(f"Metadatos: {self.final_meta_path}")