# src/indices/sequentialfile/sequentialfileindex.py

import os
from typing import List, Any
from ..base_index import BaseIndex
from .sequentialfile import SequentialFile

class SequentialFileIndex(BaseIndex):
    def __init__(self, table_name: str, column_name: str,
                 record_manager, key_column_index: int,
                 data_dir: str = 'data', aux_capacity: int = 10):

        self.table_name = table_name
        self.column_name = column_name

        file_prefix = os.path.join(data_dir, f"{table_name}_{column_name}_seq")
        os.makedirs(data_dir, exist_ok=True)

        main_path = f"{file_prefix}.dat"
        aux_path = f"{file_prefix}.aux"
        
        open(main_path, 'wb').close()
        open(aux_path, 'wb').close()

        self.engine = SequentialFile(
            file_path_prefix=file_prefix,
            record_manager=record_manager,
            key_column_index=key_column_index,
            aux_capacity=aux_capacity
        )

    def add(self, key: Any, value: Any):
        if not isinstance(value, list):
            raise TypeError("SequentialFileIndex.add espera un 'value' de tipo list (registro completo).")
        self.engine.add(value)

    def search(self, key: Any) -> List[Any]:
        return self.engine.search(key)

    def remove(self, key: Any, value: Any = None):
        print("WARN: delete en SEQ implica reconstrucción completa.")
        main_records = list(self.engine._read_records_from_file(self.engine.main_path))
        aux_records = list(self.engine._read_records_from_file(self.engine.aux_path))
        kept = [r for r in (main_records + aux_records) if r[self.engine.key_col_idx] != key]
        kept.sort(key=lambda r: r[self.engine.key_col_idx])

        with open(self.engine.main_path, 'wb') as f:
            for rec in kept:
                f.write(self.engine.record_manager.pack(rec))

        open(self.engine.aux_path, 'wb').close()

    def rangeSearch(self, start_key: Any, end_key: Any) -> List[Any]:
        return self.engine.range_search(start_key, end_key)
