# src/indices/manager.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional

from .base_index import BaseIndex
from .bplustree.bplustreeindex import BPlusTreeIndex
from .hashing.hashingindex import HashIndex as EHHashIndex
from .isam.isamindex import ISAMIndex
from .rtree.rtreeindex import RTreeIndex
from .sequentialfile.sequentialfileindex import SequentialFileIndex


class IndexManager:
    def __init__(self):
        # índices vivos: {table: {column: BaseIndex}}
        self._indexes: Dict[str, Dict[str, BaseIndex]] = {}
        # specs para reconstrucción: {table: [(column, type_str)]}
        self._specs: Dict[str, List[Tuple[str, str]]] = {}

        # alias públicos -> internos
        self._ALIASES: Dict[str, str] = {
            'BTREE': 'BTREE',
            'B+TREE': 'BTREE',
            'BPLUSTREE': 'BTREE',
            'BPLUS': 'BTREE',

            'HASH': 'EHASH',       
            'EHASH': 'EHASH',

            'ISAM': 'ISAM',

            'SEQ': 'SEQ',
            'SEQUENTIAL': 'SEQ',
            'SEQUENTIALFILE': 'SEQ',

            'RTREE': 'RTREE',
        }

    # -------------------- utilidades internas --------------------

    def _normalize(self, idx_type: Optional[str]) -> str:
        key = (idx_type or '').upper()
        return self._ALIASES.get(key, key)

    def _name_to_pos(self, schema: List[Tuple[str, str, int]]) -> Dict[str, int]:
        return {n: i for i, (n, _t, _l) in enumerate(schema)}
    
    
    @staticmethod
    def _parse_coords(val: Any):
        """
        Intenta parsear coordenadas 2D desde:
          - tupla/lista: (x, y) o [x, y]
          - string con paréntesis: "(x, y)"
          - string con corchetes: "[x, y]"
          - string plano: "x, y"
        Devuelve tuple[float, float] o None.
        """
        # Ya viene como tupla/lista numérica
        if isinstance(val, (list, tuple)):
            try:
                coords = tuple(float(x) for x in val)
                return coords if len(coords) >= 2 else None
            except Exception:
                return None

        # Si es string, limpiamos
        if isinstance(val, str):
            s = val.strip()

            # Quitar paréntesis o corchetes exteriores
            if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
                s = s[1:-1]

            # Split por coma
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) < 2:
                return None

            try:
                coords = tuple(float(p) for p in parts)
                return coords if len(coords) >= 2 else None
            except Exception:
                return None

        return None




    # -------------------- API pública --------------------

    def list_for_table(self, table_name: str) -> Dict[str, BaseIndex]:
        return self._indexes.get(table_name, {})

    def declare(self, table_name: str, column: str, idx_type: str):
        typ = self._normalize(idx_type)
        specs = self._specs.setdefault(table_name, [])
        if (column, typ) not in specs:
            specs.append((column, typ))

    def create_index(self, table, columns_or_colname: Any, idx_type: str):
        """
        Crea un índice para una o más columnas.
        Para tu uso actual:
        - BTREE, EHASH(HASH), ISAM, SEQ: 1 columna.
        - RTREE: 1 columna con coords tipo "(x, y)" o "[x, y]" o "x,y".
        """
        idx_type = self._normalize(idx_type)

        # Normalizar a lista de columnas
        if isinstance(columns_or_colname, (list, tuple)):
            columns = list(columns_or_colname)
        else:
            columns = [columns_or_colname]

        if not columns:
            raise ValueError("No se especificaron columnas para el índice.")

        name_to_pos = self._name_to_pos(table.schema)

        # Clave canónica solo para specs (para reconstrucción futura)
        key_str = ",".join(columns)
        self.declare(table.name, key_str, idx_type)

        if table.name not in self._indexes:
            self._indexes[table.name] = {}

        # -------------------- EHASH (HASH) --------------------
        if idx_type == "EHASH":
            if len(columns) != 1:
                raise ValueError("HASH solo soporta 1 columna.")
            col = columns[0]
            pos = name_to_pos[col]
            idx = EHHashIndex(table.name, col, data_dir=table.data_dir)
            for rid, values in table.scan() or []:
                idx.add(values[pos], rid)
            # clave = nombre de columna (para probe_eq)
            self._indexes[table.name][col] = idx
            return

        # -------------------- BTREE --------------------
        if idx_type == "BTREE":
            if len(columns) != 1:
                raise ValueError("BTREE solo soporta 1 columna.")
            col = columns[0]
            pos = name_to_pos[col]
            idx = BPlusTreeIndex(table.name, col, data_dir=table.data_dir)
            for rid, values in table.scan() or []:
                idx.add(values[pos], rid)
            if hasattr(idx, "persist"):
                idx.persist()
            self._indexes[table.name][col] = idx
            return

        # -------------------- ISAM --------------------
        if idx_type == "ISAM":
            if len(columns) != 1:
                raise ValueError("ISAM solo soporta 1 columna.")
            col = columns[0]
            idx = ISAMIndex.build_from_table(table, col)
            self._indexes[table.name][col] = idx
            return

        # -------------------- SEQ --------------------
        if idx_type == "SEQ":
            if len(columns) != 1:
                raise ValueError("SEQ solo soporta 1 columna.")
            col = columns[0]
            pos = name_to_pos[col]
            idx = SequentialFileIndex(
                table_name=table.name,
                column_name=col,
                record_manager=table.record_manager,
                key_column_index=pos,
                data_dir=table.data_dir,
                aux_capacity=10,
            )
            for _rid, values in table.scan() or []:
                idx.add(values[pos], list(values))
            self._indexes[table.name][col] = idx
            return

        # -------------------- RTREE --------------------
        if idx_type == "RTREE":
            # key_str ya es "lat,lon" o "Coord", etc.
            idx = RTreeIndex(table.name, "_rtree_" + key_str, data_dir=table.data_dir)

            # Caso 1: una sola columna con "(x,y)" o "[x,y]"
            if len(columns) == 1:
                col = columns[0]
                pos = name_to_pos[col]
                count = 0
                for rid, values in table.scan() or []:
                    coords = self._parse_coords(values[pos])
                    if coords is not None:
                        idx.add(coords, rid)
                        count += 1
                # guardamos bajo el nombre de la columna
                self._indexes[table.name][key_str] = idx
                print(f"[IndexManager RTREE] construido índice para {table.name}.{col} con {count} entradas")
                return

            # Caso 2: dos columnas numéricas lat, lon
            if len(columns) == 2:
                c_lat, c_lon = columns
                p_lat, p_lon = name_to_pos[c_lat], name_to_pos[c_lon]
                count = 0
                for rid, values in table.scan() or []:
                    try:
                        lat = float(values[p_lat])
                        lon = float(values[p_lon])
                        idx.add((lat, lon), rid)
                        count += 1
                    except Exception:
                        continue
                # lo registramos con la clave "lat,lon"
                self._indexes[table.name][key_str] = idx
                print(f"[IndexManager RTREE] construido índice 2D para {table.name}.({c_lat},{c_lon}) con {count} entradas")
                return

            # Si no es 1 o 2, no soportamos
            raise ValueError("RTREE solo soporta 1 columna '(x,y)' o 2 columnas numéricas (lat, lon).")

        # -------------------- desconocido --------------------
        raise ValueError(f"Tipo de índice no soportado: {idx_type}")



    def drop_index(self, table, column: str):
        if table.name in self._indexes:
            self._indexes[table.name].pop(column, None)
            if not self._indexes[table.name]:
                self._indexes.pop(table.name, None)
        # quita spec para no reconstruir
        specs = self._specs.get(table.name, [])
        self._specs[table.name] = [(c, t) for (c, t) in specs if c != column]
        # TODO: borrar archivos persistidos del índice si aplica

    def rebuild_all(self, table) -> None:
        """Reconstruye todos los índices declarados para la tabla desde cero."""
        specs = self._specs.get(table.name) or getattr(table, "index_specs", [])
        if not specs:
            self._indexes.pop(table.name, None)
            return

        # limpia el mapa en memoria y re-crea cada índice
        self._indexes[table.name] = {}
        for col, typ in specs:
            cols = col.split(",") if isinstance(col, str) and "," in col else col
            self.create_index(table, cols, typ)

    def on_insert(self, table, rid: int, values: List[Any]) -> None:
        """Actualiza todos los índices con el nuevo registro."""
        idxs = self._indexes.get(table.name)
        if not idxs:
            return
        name_pos = self._name_to_pos(table.schema)
        
        for key_str, idx in idxs.items():
            cols = key_str.split(",")  # 1 o 2 columnas
            if isinstance(idx, SequentialFileIndex):
                p = name_pos[cols[0]]
                idx.add(values[p], list(values))
            elif isinstance(idx, RTreeIndex):
                if len(cols) == 1:
                    p = name_pos[cols[0]]
                    coords = self._parse_coords(values[p])
                    if coords is not None:
                        idx.add(coords, rid)
                else:
                    p0, p1 = name_pos[cols[0]], name_pos[cols[1]]
                    try:
                        lat = float(values[p0]); lon = float(values[p1])
                        idx.add((lat, lon), rid)
                    except Exception:
                        pass
            else:
                p = name_pos[cols[0]]
                idx.add(values[p], rid)
    # -------------------- probes para el planner --------------------

    def probe_eq(self, table_name: str, column: str, value: Any):
        idx = self._indexes.get(table_name, {}).get(column)
        if not idx:
            return None, None
        if isinstance(idx, SequentialFileIndex):
            return 'records', idx.search(value)
        return 'rids', idx.search(value)

    def probe_between(self, table_name: str, column: str, low: Any, high: Any):
        idx = self._indexes.get(table_name, {}).get(column)
        if not idx or not hasattr(idx, "rangeSearch"):
            return None, None
        try:
            res = idx.rangeSearch(low, high)
        except NotImplementedError:
            return None, None
        if res and isinstance(res[0], list):
            return 'records', res
        return 'rids', res

    def probe_rtree_radius(self, table_name: str, column: str, coords: List[float], radius: float):
        tmap = self._indexes.get(table_name, {})
        idx = tmap.get(column)

        # fallback: si no está con ese nombre, usar el primer RTREE disponible
        if not isinstance(idx, RTreeIndex):
            for key, cand in tmap.items():
                if isinstance(cand, RTreeIndex):
                    idx = cand
                    break

        if not isinstance(idx, RTreeIndex):
            return None, None

        rids = idx.radius_search(tuple(coords), float(radius))
        return 'rids', rids
