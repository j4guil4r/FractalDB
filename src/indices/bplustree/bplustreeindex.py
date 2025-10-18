# src/indices/bplustree/bplustreeindex.py

import os
from typing import List, Any
from ..base_index import BaseIndex
from .bplustree import BPlusTree

class BPlusTreeIndex(BaseIndex):
    def __init__(self, table_name: str, column_name: str, data_dir: str = 'data', order: int = 3):
        os.makedirs(data_dir, exist_ok=True)
        
        file_prefix = f"{table_name}_{column_name}_bpt"
        self.file_path_prefix = os.path.join(data_dir, file_prefix)
        
        self.tree = BPlusTree.load(self.file_path_prefix, order)

    def add(self, key: Any, value: Any):
        if not isinstance(value, int):
             raise TypeError("BPlusTreeIndex.add espera un 'value' de tipo int (RID).")
        self.tree.insert(key, value)
        self.tree.save_meta()

    def search(self, key: Any) -> List[Any]:
        return self.tree.search(key)

    def remove(self, key: Any, value: Any = None):
        self.tree.remove(key, value)
        self.tree.save_meta()
        
    def rangeSearch(self, start_key: Any, end_key: Any) -> List[Any]:
        return self.tree.range_search(start_key, end_key)