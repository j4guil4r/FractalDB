from bisect import bisect_left, bisect_right
from typing import Any, List, Tuple

class Record:
    def __init__(self, key: Any, value: Any):
        self.key = key
        self.value = value

class SequentialFile:
    """ en main están las llaves ordenadas, en aux no ordenado add = O(1) """
    def __init__(self, rebuild_threshold: int = 64, index_block_size: int = 128):
        self.main: List[Record] = []
        self.aux:  List[Record] = []
        self.rebuild_threshold = rebuild_threshold
        self.index_block_size = index_block_size
        self.sparse_index: List[Tuple[Any, int]] = []

    # índice disperso (nos ayudara para busquedas rapidas)
    def _build_sparse_index(self) -> None:
        self.sparse_index.clear()
        if not self.main:
            return
        for i in range(0, len(self.main), self.index_block_size):
            self.sparse_index.append((self.main[i].key, i))

    def _block_start_for(self, key: Any) -> int:
        """ salta al posible bloque segun el indice disperso """
        if not self.sparse_index:
            return 0
        keys = [k for (k, _) in self.sparse_index]
        pos = bisect_right(keys, key) - 1
        return self.sparse_index[pos][1] if pos >= 0 else 0

    # funciones para hacer el merge y corroborar
    def _merge_rebuild(self) -> None:
        """ fusiona main y aux, ordena por key y regenera índice """
        if not self.aux:
            return
        aux_sorted = sorted(self.aux, key=lambda r: r.key)
        merged: List[Record] = []
        i = j = 0
        while i < len(self.main) and j < len(aux_sorted):
            if self.main[i].key <= aux_sorted[j].key:
                merged.append(self.main[i]); i += 1
            else:
                merged.append(aux_sorted[j]); j += 1
        if i < len(self.main): merged.extend(self.main[i:])
        if j < len(aux_sorted): merged.extend(aux_sorted[j:])
        self.main = merged
        self.aux.clear()
        self._build_sparse_index()

    def _check_rebuild(self) -> None:
        """ su función es ver si ya se llego al limite """
        if len(self.aux) >= self.rebuild_threshold:
            self._merge_rebuild()

    # estas dos funciones es para la busqueda binaria, derecha y izquierda
    def _left_binary(self, key: Any) -> int:
        if not self.main:
            return 0
        start = self._block_start_for(key)
        while start > 0 and self.main[start - 1].key == key:
            start = max(0, start - self.index_block_size)

        keys = [r.key for r in self.main[start:]]
        return start + bisect_left(keys, key)

    def _right_binary(self, key: Any) -> int:
        if not self.main:
            return 0
        start = self._block_start_for(key)
        keys = [r.key for r in self.main[start:]]
        return start + bisect_right(keys, key)

    # funciones principales
    def add(self, record: Record) -> None:
        """ inserta en aux y cehca si estamos en el limite """
        self.aux.append(record)
        self._check_rebuild()

    def search(self, key: Any) -> List[Record]:
        """ devuelve todos los elementos con la llave, por eso usamos una lista """
        out: List[Record] = []
        if self.main:
            lo = self._left_binary(key)
            hi = self._right_binary(key)
            if lo < hi:
                out.extend(self.main[lo:hi])
        if self.aux:
            # aux es chica es pequeña = 64 así que aplicamos O(n)
            out.extend([r for r in self.aux if r.key == key])
        return out

    def range_search(self, start_key: Any, end_key: Any) -> List[Record]:
        """ devuelve registros que este dentro del rango start y end key """
        out: List[Record] = []
        if self.main:
            lo = self._left_binary(start_key)
            hi = self._right_binary(end_key)
            if lo < hi:
                out.extend(self.main[lo:hi])
        if self.aux:
            out.extend([r for r in self.aux if start_key <= r.key <= end_key])
        # ordena por key
        out.sort(key=lambda r: r.key)
        return out

    def remove(self, key: Any) -> bool:
        """ borra todos los registros con esa key en main y aux"""
        removed = False
        if self.main:
            lo = self._left_binary(key)
            hi = self._right_binary(key)
            if lo < hi:
                del self.main[lo:hi]
                removed = True
        if self.aux:
            new_aux = [r for r in self.aux if r.key != key]
            if len(new_aux) != len(self.aux):
                self.aux = new_aux
                removed = True
        # si hubo muchas inserciones recientes, igual se fusionará por umbral más adelante
        # reconstruimos índice por si la ventana de bloques cambió
        self._build_sparse_index()
        return removed
