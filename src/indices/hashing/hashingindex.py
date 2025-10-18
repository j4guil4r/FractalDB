# src/indices/hashing/hashingindex.py

import os
from typing import List, Any
from ..base_index import BaseIndex
from .hashing import Directory

class HashIndex(BaseIndex):
    def __init__(self, table_name: str, column_name: str, data_dir: str = 'data', bucket_size: int = 4):
        os.makedirs(data_dir, exist_ok=True)
        
        file_prefix = f"{table_name}_{column_name}_hash"
        self.file_path_prefix = os.path.join(data_dir, file_prefix)
        
        self.directory = Directory.load(self.file_path_prefix, bucket_size_if_new=bucket_size)

    def add(self, key: Any, value: Any):
        if not isinstance(value, int):
             raise TypeError("HashIndex.add espera un 'value' de tipo int (RID).")
        self.directory.add(key, value)
        self.directory.save()

    def search(self, key: Any) -> List[Any]:
        return self.directory.search(key)

    def remove(self, key: Any, value: Any = None):
        bucket_index = self.directory._get_bucket_index(key)
        bucket_offset = self.directory.bucket_pointers[bucket_index]
        bucket = self.directory._read_bucket(bucket_offset)
        
        original_count = len(bucket.values)
        
        if value is not None:
            bucket.values = [v for v in bucket.values if v != (key, value)]
        else:
            bucket.values = [v for v in bucket.values if v[0] != key]
        
        if len(bucket.values) != original_count:
            self.directory._write_bucket(bucket, bucket_offset)

    def rangeSearch(self, start_key: Any, end_key: Any) -> List[Any]:
        raise NotImplementedError("Hashing no soporta búsqueda por rango.")