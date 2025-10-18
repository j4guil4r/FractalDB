# src/indices/base_index.py

from abc import ABC, abstractmethod
from typing import List, Any

class BaseIndex(ABC):
    
    @abstractmethod
    def add(self, key: Any, value: Any):
        pass

    @abstractmethod
    def search(self, key: Any) -> List[Any]:
        pass

    @abstractmethod
    def remove(self, key: Any, value: Any = None):
        pass

    def rangeSearch(self, start_key: Any, end_key: Any) -> List[Any]:
        raise NotImplementedError("La búsqueda por rango no está soportada por este tipo de índice.")