# src/indices/rtree/rtreeindex.py

import os
from typing import List, Any, Tuple
from ..base_index import BaseIndex

try:
    from rtree import index
except ImportError:
    raise ImportError("La librería 'Rtree' no está instalada...")

class RTreeIndex(BaseIndex):
    def __init__(self, table_name: str, column_name: str, data_dir: str = 'data'):
        index_basename = os.path.join(data_dir, f"{table_name}_{column_name}_rtree")
        os.makedirs(data_dir, exist_ok=True)
        p = index.Property()
        self.idx = index.Index(index_basename, properties=p)

    def add(self, key: Tuple[float, ...], rid: int):
        bounding_box = tuple(list(key) * 2)
        self.idx.insert(rid, bounding_box)

    def remove(self, key: Tuple[float, ...], rid: int):
        bounding_box = tuple(list(key) * 2)
        self.idx.delete(rid, bounding_box)

    def radius_search(self, point: Tuple[float, ...], radius: float) -> List[int]:
        min_coords = [c - radius for c in point]
        max_coords = [c + radius for c in point]
        
        query_box = tuple(min_coords + max_coords)
        
        return list(self.idx.intersection(query_box))

    def knn_search(self, point: Tuple[float, ...], k: int) -> List[int]:
        point_coords = tuple(point)
        return list(self.idx.nearest(point_coords, num_results=k))

    def search(self, key: Any) -> List[int]:
        raise NotImplementedError("Para R-Tree, use 'radius_search' o 'knn_search'.")

    def rangeSearch(self, start_key: Any, end_key: Any) -> List[int]:
        raise NotImplementedError("Para R-Tree, use 'radius_search' o 'knn_search'.")