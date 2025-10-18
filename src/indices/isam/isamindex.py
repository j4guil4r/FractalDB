# src/indices/isam/isamindex.py

import os
from typing import List, Any
from ..base_index import BaseIndex
from .isam import ISAM

class ISAMIndex(BaseIndex):
    def __init__(self, table_name: str, column_name: str, data_dir: str = 'data'):
        os.makedirs(data_dir, exist_ok=True)
        
        file_prefix = os.path.join(data_dir, f"{table_name}_{column_name}")
        self.engine = ISAM(file_prefix)
        
        if os.path.exists(self.engine.paths['meta']):
            self.engine.load_metadata()

    @staticmethod
    def build_from_table(table, column_name: str, index_capacity=4, data_capacity=4):
        print(f"Construyendo índice ISAM para '{table.name}.{column_name}'...")
        
        index = ISAMIndex(table.name, column_name, data_dir=table.data_dir)
        
        index.engine.index_capacity = index_capacity
        index.engine.data_capacity = data_capacity
        
        col_idx = [s[0] for s in table.schema].index(column_name)
        all_records = []
        for rid, values in table.scan():
            key = values[col_idx]
            all_records.append((key, rid))
            
        all_records.sort(key=lambda x: x[0])
        
        index.engine.build(all_records)
        print("Construcción del índice ISAM finalizada.")
        return index

    def add(self, key: Any, value: Any):
        if not isinstance(value, int):
             raise TypeError("ISAMIndex.add espera un 'value' de tipo int (RID).")
        self.engine.add(key, value)

    def search(self, key: Any) -> List[Any]:
        return self.engine.search(key)

    def remove(self, key: Any, value: Any = None):
        print(f"Iniciando eliminación lógica para la clave: {key}")
        self.engine.remove(key)
        print(f"Eliminación lógica completada para la clave: {key}")

    def rangeSearch(self, start_key: Any, end_key: Any) -> List[Any]:
        return self.engine.rangeSearch(start_key, end_key)