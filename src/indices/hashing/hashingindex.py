# src/indices/hashing/hashingindex.py

import os
from typing import List, Any

from ..base_index import BaseIndex
from .hashing import Directory

class HashIndex(BaseIndex):
    def __init__(self, table_name: str, column_name: str, data_dir: str = 'data'):
        index_filename = f"{table_name}_{column_name}.hidx"
        self.file_path = os.path.join(data_dir, index_filename)
        
        self.directory = Directory.load(self.file_path)
        if not self.directory:
            self.directory = Directory(self.file_path)

    def add(self, key: Any, rid: int):
        self.directory.add(key, rid)
        self.directory.save()

    def search(self, key: Any) -> List[int]:
        return self.directory.search(key)

    def remove(self, key: Any, rid: int = None):
        bucket_index = self.directory._get_bucket_index(key)
        bucket = self.directory.buckets[bucket_index]
        
        if rid:
            bucket.values = [v for v in bucket.values if v != (key, rid)]
        else:
            bucket.values = [v for v in bucket.values if v[0] != key]
        
        self.directory.save()