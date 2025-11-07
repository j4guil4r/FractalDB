import os
from typing import List, Any, Tuple
from ..base_index import BaseIndex

try:
    from rtree import index
except ImportError:
    index = None  # usamos shadow aunque no haya rtree nativo


class RTreeIndex(BaseIndex):
    """
    Índice espacial 2D con shadow en memoria para debug:
      - add(): guarda (rid, (x,y)) en _shadow y, si existe, en rtree.Index
      - radius_search():
          1) intenta con rtree (si está disponible)
          2) siempre revisa _shadow y aplica la condición de radio exacta
    """
    def __init__(self, table_name: str, column_name: str, data_dir: str = 'data'):
        os.makedirs(data_dir, exist_ok=True)
        self.table_name = table_name
        self.column_name = column_name

        self._shadow: List[Tuple[int, Tuple[float, float]]] = []

        self._has_rtree = index is not None
        if self._has_rtree:
            index_basename = os.path.join(data_dir, f"{table_name}_{column_name}_rtree")
            p = index.Property()
            p.dimension = 2
            self.idx = index.Index(index_basename, properties=p)
        else:
            self.idx = None

    def add(self, key: Any, value: Any):
        if not isinstance(value, int):
            raise TypeError("RTreeIndex.add espera un 'value' de tipo int (RID).")
        if not isinstance(key, (tuple, list)):
            raise TypeError("RTreeIndex.add espera una 'key' de tipo tuple o list (coordenadas).")

        try:
            coords = tuple(float(x) for x in key)
        except Exception as e:
            raise TypeError(f"Coordenadas inválidas para RTREE: {key} ({e})")

        if len(coords) != 2:
            raise ValueError("Este RTreeIndex está implementado para 2D (x, y).")

        x, y = coords
        self._shadow.append((int(value), (x, y)))

        print(f"[RTREE add] table={self.table_name} col={self.column_name} rid={int(value)} coords=({x}, {y})")

        if self._has_rtree and self.idx is not None:
            bbox = (x, y, x, y)
            try:
                self.idx.insert(int(value), bbox)
            except Exception as e:
                print(f"[RTREE add] error insertando en rtree nativo: {e}")

    def remove(self, key: Any, value: Any = None):
        if value is None:
            raise ValueError("RTreeIndex.remove requiere 'value' (RID).")
        if not isinstance(key, (tuple, list)):
            raise TypeError("RTreeIndex.remove espera 'key' tuple/list.")

        try:
            coords = tuple(float(x) for x in key)
        except Exception as e:
            raise TypeError(f"Coordenadas inválidas para RTREE: {key} ({e})")

        if len(coords) != 2:
            raise ValueError("Este RTreeIndex está implementado para 2D (x, y).")

        x, y = coords
        before = len(self._shadow)
        self._shadow = [
            (rid, c) for (rid, c) in self._shadow
            if not (rid == int(value) and c == (x, y))
        ]
        after = len(self._shadow)
        print(f"[RTREE remove] rid={int(value)} coords=({x},{y}) shadow_size {before}->{after}")

        if self._has_rtree and self.idx is not None:
            bbox = (x, y, x, y)
            try:
                self.idx.delete(int(value), bbox)
            except Exception as e:
                print(f"[RTREE remove] error borrando en rtree nativo: {e}")

    def radius_search(self, point: Tuple[float, ...], radius: float) -> List[int]:
        # Normaliza punto
        try:
            px, py = float(point[0]), float(point[1])
        except Exception:
            print(f"[RTREE radius_search] punto inválido: {point}")
            return []

        r2 = radius * radius

        print(f"[RTREE radius_search] point=({px},{py}) radius={radius} shadow_size={len(self._shadow)}")

        # 1) Intentar con rtree nativo (si está)
        cand_from_rtree: List[int] = []
        if self._has_rtree and self.idx is not None:
            try:
                minx, miny = px - radius, py - radius
                maxx, maxy = px + radius, py + radius
                bbox = (minx, miny, maxx, maxy)
                cand_from_rtree = list(self.idx.intersection(bbox))
                print(f"[RTREE radius_search] cand_from_rtree={cand_from_rtree}")
            except Exception as e:
                print(f"[RTREE radius_search] error en rtree.intersection: {e}")

        results: List[int] = []

        # 2) Filtro exacto usando shadow (siempre)
        #    - si hay cand_from_rtree, filtramos solo esos
        #    - si no hay, probamos contra todos en shadow
        if cand_from_rtree:
            cand_set = set(int(rid) for rid in cand_from_rtree)
            for rid, (cx, cy) in self._shadow:
                if rid in cand_set:
                    dx = cx - px
                    dy = cy - py
                    if dx*dx + dy*dy <= r2:
                        results.append(rid)
        else:
            for rid, (cx, cy) in self._shadow:
                dx = cx - px
                dy = cy - py
                if dx*dx + dy*dy <= r2:
                    results.append(rid)

        print(f"[RTREE radius_search] results={results}")
        return results

    def knn_search(self, point: Tuple[float, ...], k: int) -> List[int]:
        try:
            px, py = float(point[0]), float(point[1])
        except Exception:
            print(f"[RTREE knn_search] punto inválido: {point}")
            return []

        # Si hay rtree, intentamos usarlo
        if self._has_rtree and self.idx is not None:
            try:
                cand = list(self.idx.nearest((px, py, px, py), num_results=int(k)))
                print(f"[RTREE knn_search] cand_from_rtree={cand}")
                return [int(rid) for rid in cand]
            except Exception as e:
                print(f"[RTREE knn_search] error en rtree.nearest: {e}")

        # Fallback: ordenar shadow por distancia
        dists = []
        for rid, (cx, cy) in self._shadow:
            dx = cx - px
            dy = cy - py
            dists.append((dx*dx + dy*dy, rid))
        dists.sort()
        res = [rid for _, rid in dists[:int(k)]]
        print(f"[RTREE knn_search] fallback_res={res}")
        return res

    def search(self, key: Any) -> List[Any]:
        raise NotImplementedError("Para RTreeIndex usa radius_search o knn_search.")

    def rangeSearch(self, start_key: Any, end_key: Any) -> List[Any]:
        raise NotImplementedError("Para RTreeIndex usa radius_search o knn_search.")
