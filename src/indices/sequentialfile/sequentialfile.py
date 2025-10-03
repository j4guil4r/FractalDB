# src/indices/sequentialfile/sequentialfile.py

import os
import pickle
import bisect
from typing import List, Any, Generator

class SequentialFile:
    def __init__(self, file_path_prefix: str, record_manager, key_column_index: int, aux_capacity: int = 10):
        self.prefix = file_path_prefix
        self.main_path = f"{self.prefix}.dat"
        self.aux_path = f"{self.prefix}.aux"
        
        self.record_manager = record_manager
        self.key_col_idx = key_column_index
        self.aux_capacity = aux_capacity
        
        if not os.path.exists(self.main_path):
            open(self.main_path, 'w').close()
        if not os.path.exists(self.aux_path):
            open(self.aux_path, 'w').close()

    def _get_aux_count(self) -> int:
        if os.path.getsize(self.aux_path) == 0:
            return 0
        return os.path.getsize(self.aux_path) // self.record_manager.record_size

    def _read_records_from_file(self, file_path: str) -> Generator[list, None, None]:
        record_size = self.record_manager.record_size
        if not os.path.exists(file_path):
            return
            
        with open(file_path, 'rb') as f:
            while True:
                packed_data = f.read(record_size)
                if not packed_data or len(packed_data) < record_size:
                    break
                yield list(self.record_manager.unpack(packed_data))

    def add(self, record_values: list):
        packed_record = self.record_manager.pack(record_values)
        with open(self.aux_path, 'ab') as f:
            f.write(packed_record)

        if self._get_aux_count() >= self.aux_capacity:
            self.reconstruct()

    def reconstruct(self):
        print("-> Capacidad del archivo auxiliar alcanzada. Iniciando reconstrucción...")
        
        main_records = list(self._read_records_from_file(self.main_path))
        aux_records = list(self._read_records_from_file(self.aux_path))
        
        all_records = main_records + aux_records
        
        all_records.sort(key=lambda r: r[self.key_col_idx])
        
        temp_main_path = self.main_path + '.tmp'
        with open(temp_main_path, 'wb') as f:
            for record in all_records:
                f.write(self.record_manager.pack(record))
        
        os.remove(self.main_path)
        os.rename(temp_main_path, self.main_path)
        
        open(self.aux_path, 'w').close()
        print("-> Reconstrucción completada.")

    def search(self, key: Any) -> List[list]:
        results = []
        
        for record in self._read_records_from_file(self.aux_path):
            if record[self.key_col_idx] == key:
                results.append(record)

        main_records = list(self._read_records_from_file(self.main_path))
        keys = [r[self.key_col_idx] for r in main_records]
        
        i = bisect.bisect_left(keys, key)
        while i < len(keys) and keys[i] == key:
            results.append(main_records[i])
            i += 1
            
        return results

    def range_search(self, start_key: Any, end_key: Any) -> List[list]:
        results = []
        
        for record in self._read_records_from_file(self.aux_path):
            if start_key <= record[self.key_col_idx] <= end_key:
                results.append(record)

        main_records = list(self._read_records_from_file(self.main_path))
        keys = [r[self.key_col_idx] for r in main_records]

        start_idx = bisect.bisect_left(keys, start_key)
        for i in range(start_idx, len(keys)):
            if keys[i] <= end_key:
                results.append(main_records[i])
            else:
                break

        results.sort(key=lambda r: r[self.key_col_idx])
        return results