# src/indices/isam/isamindex.py

import os
from typing import List, Any
from ..base_index import BaseIndex
from .isam import ISAM
from src.core.table import Table

class ISAMIndex(BaseIndex):
    """
    Wrapper para el motor ISAM con tolerancia a metadatos/archivos faltantes.
    """
    def __init__(self, table_name: str, column_name: str, data_dir: str = 'data'):
        os.makedirs(data_dir, exist_ok=True)
        file_prefix = os.path.join(data_dir, f"{table_name}_{column_name}")
        self.table_name = table_name
        self.column_name = column_name
        self.engine = ISAM(file_prefix)

        meta_path = self.engine.paths['meta']
        if os.path.exists(meta_path):
            try:
                self.engine.load_metadata()
                for key in ('idx2', 'idx1', 'dat'):
                    path = self.engine.paths[key]
                    if not os.path.exists(path):
                        raise FileNotFoundError(path)
            except Exception:
                for k in ('meta','idx2','idx1','dat','ovf'):
                    p = self.engine.paths[k]
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except:
                        pass

    @staticmethod
    def build_from_table(table: Table, column_name: str, index_capacity=4, data_capacity=4):
        """
        Construye el índice desde la tabla (scan + sort).
        Siempre crea idx2/idx1/dat incluso si no hay registros.
        """
        index = ISAMIndex(table.name, column_name, data_dir=table.data_dir)
        index.engine.index_capacity = index_capacity
        index.engine.data_capacity = data_capacity

        col_idx = [s[0] for s in table.schema].index(column_name)
        all_records = []
        for rid, values in (table.scan() or []):
            all_records.append((values[col_idx], rid))
        all_records.sort(key=lambda x: x[0])

        index.engine.build(all_records)
        return index

    # ---------- interfaz BaseIndex ----------

    def add(self, key: Any, value: Any):
        if not isinstance(value, int):
            raise TypeError("ISAMIndex.add espera un 'value' int (RID).")
        self.engine.add(key, value)

    def search(self, key: Any) -> List[Any]:
        return self.engine.search(key)

    def remove(self, key: Any, value: Any = None):
        self.engine.remove(key)

    def rangeSearch(self, start_key: Any, end_key: Any) -> List[Any]:
        return self.engine.rangeSearch(start_key, end_key)
