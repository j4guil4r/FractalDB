# src/indices/hashing/hashing.py

import pickle
from typing import List, Tuple, Any

class Bucket:
    def __init__(self, depth: int, size: int):
        self.local_depth = depth
        self.size = size
        self.values: List[Tuple[Any, int]] = []

    def is_full(self) -> bool:
        return len(self.values) >= self.size

    def add(self, key: Any, rid: int):
        if not self.is_full():
            self.values.append((key, rid))
            return True
        return False

    def get_records(self, key: Any) -> List[int]:
        return [r for k, r in self.values if k == key]

class Directory:
    def __init__(self, file_path: str, bucket_size: int = 4):
        self.file_path = file_path
        self.bucket_size = bucket_size
        self.global_depth = 1
        self.buckets = [Bucket(1, self.bucket_size) for _ in range(2)]

    def _hash(self, key: Any) -> int:
        return hash(key)

    def _get_bucket_index(self, key: Any) -> int:
        h = self._hash(key)
        return h & ((1 << self.global_depth) - 1)

    def search(self, key: Any) -> List[int]:
        bucket_index = self._get_bucket_index(key)
        bucket = self.buckets[bucket_index]
        return bucket.get_records(key)

    def add(self, key: Any, rid: int):
        bucket_index = self._get_bucket_index(key)
        bucket = self.buckets[bucket_index]

        if bucket.add(key, rid):
            return

        self._split_bucket(bucket_index)
        self.add(key, rid)

    def _split_bucket(self, bucket_index: int):
        old_bucket = self.buckets[bucket_index]
        old_bucket.local_depth += 1
        
        if old_bucket.local_depth > self.global_depth:
            self._grow_directory()

        new_bucket = Bucket(old_bucket.local_depth, self.bucket_size)
        
        current_values = old_bucket.values
        old_bucket.values = []

        split_index_pattern = 1 << (old_bucket.local_depth - 1)
        
        for i in range(len(self.buckets)):
            if self.buckets[i] == old_bucket and (i & split_index_pattern):
                self.buckets[i] = new_bucket

        for k, r in current_values:
            self.add(k, r)

    def _grow_directory(self):
        self.buckets.extend(self.buckets)
        self.global_depth += 1
        
    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return None