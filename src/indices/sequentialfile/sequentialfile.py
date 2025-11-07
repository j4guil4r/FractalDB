# src/indices/sequentialfile/sequentialfile.py

import os
from typing import List, Any, Generator, Optional, Tuple

class SequentialFile:
    def __init__(self, file_path_prefix: str, record_manager, key_column_index: int, aux_capacity: int = 10):
        self.prefix = file_path_prefix
        self.main_path = f"{self.prefix}.dat"
        self.aux_path = f"{self.prefix}.aux"
        
        self.record_manager = record_manager
        self.key_col_idx = key_column_index
        self.aux_capacity = aux_capacity
        
        if not os.path.exists(self.main_path):
            open(self.main_path, 'wb').close()
        if not os.path.exists(self.aux_path):
            open(self.aux_path, 'wb').close()

    def _get_record_count(self, file_path: str) -> int:
        record_size = self.record_manager.record_size
        if record_size == 0: return 0
        
        try:
            file_size = os.path.getsize(file_path)
            return file_size // record_size
        except FileNotFoundError:
            return 0

    def _get_aux_count(self) -> int:
        return self._get_record_count(self.aux_path)

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

    def _read_record_at_index(self, file_path: str, index: int) -> Optional[Tuple]:
        record_size = self.record_manager.record_size
        offset = index * record_size
        
        try:
            with open(file_path, 'rb') as f:
                f.seek(offset)
                packed_data = f.read(record_size)
                if packed_data and len(packed_data) == record_size:
                    return self.record_manager.unpack(packed_data)
        except (IOError, FileNotFoundError):
            pass
        return None

    def _find_first_record_gte(self, key: Any) -> int:
        low = 0
        high = self._get_record_count(self.main_path)
        
        while low < high:
            mid = (low + high) // 2
            record = self._read_record_at_index(self.main_path, mid)
            
            if record is None:
                high = mid
                continue

            record_key = record[self.key_col_idx]
            
            if record_key < key:
                low = mid + 1
            else:
                high = mid
                
        return low

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
        
        os.replace(temp_main_path, self.main_path)
        
        open(self.aux_path, 'wb').close()
        print("-> Reconstrucción completada.")

    def search(self, key: Any) -> List[list]:
        results = []
        
        for record in self._read_records_from_file(self.aux_path):
            if record[self.key_col_idx] == key:
                results.append(record)

        start_idx = self._find_first_record_gte(key)
        
        record_count = self._get_record_count(self.main_path)
        for i in range(start_idx, record_count):
            record = self._read_record_at_index(self.main_path, i)
            if record is None:
                break
            
            record_key = record[self.key_col_idx]
            
            if record_key == key:
                results.append(list(record))
            elif record_key > key:
                break
            
        return results

    def range_search(self, start_key: Any, end_key: Any) -> List[list]:
        results = []
        
        for record in self._read_records_from_file(self.aux_path):
            if start_key <= record[self.key_col_idx] <= end_key:
                results.append(record)

        start_idx = self._find_first_record_gte(start_key)
        
        record_count = self._get_record_count(self.main_path)
        for i in range(start_idx, record_count):
            record = self._read_record_at_index(self.main_path, i)
            if record is None:
                break

            record_key = record[self.key_col_idx]
            
            if record_key <= end_key:
                results.append(list(record))
            else:
                break

        results.sort(key=lambda r: r[self.key_col_idx])
        return results