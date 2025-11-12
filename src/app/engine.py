# src/app/engine.py
from src.core.table import Table
from src.indices.manager import IndexManager

from typing import Any, Dict, List, Optional, Tuple, Callable, Iterator
import io
import csv
import os
import re

# --- NUEVOS IMPORTS ---
from src.indices.inverted_index.builder import InvertedIndexBuilder
from src.indices.inverted_index.query import InvertedIndexQuery
# --- FIN NUEVOS IMPORTS ---


def _parse_sql_type_to_core(sql_type: str) -> Tuple[str, int]:
    t = sql_type.upper()
    if t == "INT":
        return ("INT", 0)
    if t == "FLOAT":
        return ("FLOAT", 0)
    if t == "DATE":
        return ("VARCHAR", 10)
    if t.startswith("VARCHAR[") and t.endswith("]"):
        n = int(t[len("VARCHAR["):-1])
        return ("VARCHAR", n)
    if t == "ARRAY[FLOAT]":
        return ("VARCHAR", 512)
    return ("VARCHAR", 255)

def _cast_for_column(value: Any, col_type: str, length: int) -> Any:
    t = col_type.upper()
    if t == "INT":
        return int(value)
    if t == "FLOAT":
        return float(value)
    # VARCHAR base (incluye DATE y ARRAY[FLOAT] serializado)
    if isinstance(value, list):
        s = "[" + ", ".join(str(float(x)) for x in value) + "]"
    else:
        s = str(value)
    if length and len(s.encode("utf-8")) > length:
        s = s.encode("utf-8")[:length].decode("utf-8", errors="replace")
    return s

def _name_to_pos(schema: List[Tuple[str, str, int]]) -> Dict[str, int]:
    return {name: i for i, (name, _t, _l) in enumerate(schema)}

def _parse_coords_from_value(val: Any) -> Optional[Tuple[float, ...]]:
    if isinstance(val, (list, tuple)):
        try:
            return tuple(float(x) for x in val)
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if len(s) >= 2 and ((s[0] == '(' and s[-1] == ')') or (s[0] == '[' and s[-1] == ']')):
            s = s[1:-1].strip()
        parts = [p.strip() for p in s.split(',') if p.strip()]
        try:
            return tuple(float(p) for p in parts)
        except Exception:
            return None
    return None

# -------------------------
# Motor con IndexManager
# -------------------------

class Engine:
    """
    Conecta el AST del parser con tu core:
      - Catalog en memoria: nombre -> Table
      - Operaciones: create_table, create_table_from_file, insert, select, delete
      - Índices: delegados en IndexManager
    """
    def __init__(self, data_dir: str = "data"):
        self.catalog: Dict[str, Table] = {}
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.idx = IndexManager()
        
        # --- MODIFICADO: Cargar FTS ---
        self.fts_query: Optional[InvertedIndexQuery] = None
        self._load_fts_query_module()
            
    def _load_fts_query_module(self):
        """ Carga o recarga el módulo de consulta FTS. """
        try:
            # Cierra el archivo anterior si existe
            if self.fts_query:
                self.fts_query.close()
                
            self.fts_query = InvertedIndexQuery(data_dir=self.data_dir)
            print("Motor: Módulo de consulta FTS cargado/recargado.")
        except FileNotFoundError:
            print("Motor: Índice FTS no encontrado. Las consultas @@ fallarán.")
            self.fts_query = None
        except Exception as e:
            print(f"Motor: Error al cargar el módulo FTS: {e}")
            self.fts_query = None

    # ---------- API pública ----------
    def execute(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        act = stmt["action"]
        if act == "create_table":
            return self._create_table(stmt)
        if act == "create_table_from_file":
            return self._create_table_from_file(stmt)
        if act == "insert":
            return self._insert(stmt)
        if act == "delete":
            return self._delete(stmt)
        if act == "select":
            return self._select(stmt)
        if act == "create_index":
            return self._create_index_action(stmt)
        
        # --- NUEVO: Hook para CREATE_FTS_INDEX ---
        if act == "create_fts_index":
            return self._create_fts_index(stmt)
            
        if act == "drop_index":
            return self._drop_index_action(stmt)
        raise ValueError(f"Acción no soportada: {act}")


    def load_csv_bytes(self, table: str, blob: bytes, has_header: bool = True) -> int:
        header, rows = self._read_csv_bytes(blob, has_header)

        # Crear tabla con inferencia si no existe
        if table not in self.catalog:
            schema = self._infer_schema_from_rows(header, rows)
            t = Table(table, schema=schema, data_dir=self.data_dir)
            self.catalog[table] = t
        else:
            t = self.catalog[table]

        inserted = 0
        for r in rows:
            # INSERT REAL (lo tenías comentado)
            values = self._cast_row_to_schema(r, t.schema)
            rid = t.insert_record(values)
            # actualiza índices en línea (si prefieres rendimiento, reconstruye al final)
            self.idx.on_insert(t, rid, values)
            inserted += 1

        return inserted
    
    def load_csv_path(self, table: str, csv_path: str, has_header: bool = True):
        import csv as _csv
        import os

        if not os.path.exists(csv_path):
            raise ValueError(f"Archivo no encontrado: {csv_path}")

        # 1) Leer muestra para inferir esquema (siempre tolerante)
        sample_rows = []
        columns = []
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = _csv.reader(f)
            header = next(reader, None) if has_header else None
            if has_header and not header:
                raise ValueError("Se esperaba header en el CSV")
            columns = header if has_header else []
            for i, row in enumerate(reader, start=1):
                sample_rows.append(row)
                if i >= 50_000:
                    break

        schema = self._infer_schema_from_rows(columns if has_header else None, sample_rows)

        # 2) Crear o reutilizar tabla
        if table not in self.catalog:
            t = Table(table, schema=schema, data_dir=self.data_dir)
            self.catalog[table] = t
        else:
            t = self.catalog[table]

        casters = self._compile_casters(schema)
        rm = t.record_manager

        # 3) Cargar todo el CSV (fallback único, robusto)
        inserted = 0
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f, \
            open(t.dat_path, "ab", buffering=2**20) as fout:
            reader = _csv.reader(f)
            if has_header:
                _ = next(reader, None)  # saltar header real

            page_size = 4096
            buf = bytearray(page_size)
            pos = 0

            for row in reader:
                if not row:
                    continue
                rec_vals = self._cast_row_fast(row, casters)
                rec = rm.pack(rec_vals)
                if pos + len(rec) > page_size:
                    fout.write(memoryview(buf)[:pos])
                    pos = 0
                buf[pos:pos+len(rec)] = rec
                pos += len(rec)
                inserted += 1

            if pos:
                fout.write(memoryview(buf)[:pos])

        # 4) Reconstruir índices
        try:
            self.idx.rebuild_all(t)
        except Exception:
            pass

        if not columns:
            columns = [name for name, _t, _l in t.schema]

        return inserted, columns

    
    def _compile_casters(self, schema):
        casters = []
        for _name, base_type, length in schema:
            # Cuidado: usar default bindings para evitar cierre lento
            if base_type == "INT":
                casters.append(lambda v, _int=int: _int(v) if v != "" else 0)
            elif base_type == "FLOAT":
                casters.append(lambda v, _float=float: _float(v) if v != "" else 0.0)
            else:
                if length and length < 255:
                    L = int(length)
                    casters.append(lambda v, _str=str, _L=L: (_str(v)[:_L]))
                else:
                    casters.append(lambda v, _str=str: _str(v))
        return casters

    def _cast_row_fast(self, row_tuple, casters):
        # Funciona con tuplas de pandas y con listas de csv.reader
        # No ramifica por columna; aplica el caster ya vinculado
        return [f(v) for f, v in zip(casters, row_tuple)]


    # ---------- Implementaciones por acción ----------
    def _create_table(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        name = stmt["table"]
        if name in self.catalog:
            raise ValueError(f"La tabla '{name}' ya existe")

        # stmt["columns"] = [{"name","type","key","index"}]
        schema_triples: List[Tuple[str, str, int]] = []
        declared_indexes: List[Tuple[str, str]] = []  # (col_name, index_type)

        for col in stmt["columns"]:
            base_type, length = _parse_sql_type_to_core(col["type"])
            schema_triples.append((col["name"], base_type, length))
            if col.get("index"):
                declared_indexes.append((col["name"], col["index"].upper()))

        t = Table(name, schema=schema_triples, data_dir=self.data_dir)
        self.catalog[name] = t

        # Construir índices declarados
        for col_name, idx_type in declared_indexes:
            self.idx.create_index(t, col_name, idx_type)

            if not hasattr(t, "index_specs"):
                t.index_specs = []
            if (col_name, idx_type) not in t.index_specs:
                t.index_specs.append((col_name, idx_type))
                t._save_metadata()

        return {"ok": True}

    def _create_table_from_file(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        """
        CREATE TABLE X FROM FILE "ruta.csv" USING INDEX BTREE("col");
        - Infiero tipos a partir de una muestra
        - Inserto en streaming (sin on_insert por fila)
        - Creo el índice al final
        """
        name = stmt["table"]
        file_path = stmt["file"]
        index_type = stmt["index_type"].upper()
        index_col = stmt["index_column"]

        if not os.path.exists(file_path):
            raise ValueError(f"Archivo no encontrado: {file_path}")
        if name in self.catalog or os.path.exists(os.path.join(self.data_dir, f"{name}.meta")):
            raise ValueError(f"La tabla '{name}' ya existe. Borra los archivos en data/ o usa otro nombre.")

        # 1) Primera pasada: leo header + muestra para inferir
        sample_rows = []
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                raise ValueError("Se espera header en el CSV para crear el esquema")
            for i, row in enumerate(reader, start=1):
                sample_rows.append(row)
                if i >= 50000:   # cap de inferencia
                    break

        schema = self._infer_schema_from_rows(header, sample_rows)

        # 2) Creo tabla con el esquema inferido
        t = Table(name, schema=schema, data_dir=self.data_dir)
        self.catalog[name] = t

        # 3) Segunda pasada: inserción en streaming (sin actualizar índices por fila)
        inserted = 0
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # saltar header
            for row in reader:
                if not row:
                    continue
                values = self._cast_row_to_schema(row, t.schema)
                t.insert_record(values)
                inserted += 1

        # 4) Construyo el índice solicitado una sola vez al final
        self.idx.create_index(t, index_col, index_type)

        # 5) Guardo spec en metadata para reconstrucción futura
        if not hasattr(t, "index_specs"):
            t.index_specs = []
        if (index_col, index_type) not in t.index_specs:
            t.index_specs.append((index_col, index_type))
            t._save_metadata()

        return {"ok": True, "table": name, "inserted": inserted, "indexed": {"column": index_col, "type": index_type}}

    def _insert(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        t = self._get_table(stmt["table"])
        values = self._cast_row_to_schema(stmt["values"], t.schema)
        rid = t.insert_record(values)
        self.idx.on_insert(t, rid, values)
        return {"ok": True}

    def _delete(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        t = self._get_table(stmt["table"])
        pred = self._make_predicate(t.schema, stmt.get("condition"))

        kept: List[List[Any]] = []
        deleted = 0
        for _rid, row_tuple in t.scan() or []:
            row = list(row_tuple)
            if pred(row):
                deleted += 1
            else:
                kept.append(row)

        # Reescribir archivo y reconstruir índices
        self._rewrite_table_file(t, kept)
        self.idx.rebuild_all(t)
        return {"ok": True, "deleted": deleted}

    # --- NUEVA FUNCIÓN: Generador de documentos para FTS ---
    def _make_doc_iterator(self, t: Table, text_columns: List[str]) -> Iterator[Tuple[int, str]]:
        """
        Crea un generador que escanea una tabla y concatena las columnas
        de texto especificadas para el constructor FTS.
        Produce (rid, full_text).
        """
        name_pos = _name_to_pos(t.schema)
        pos_to_concat = []
        for col_name in text_columns:
            if col_name not in name_pos:
                raise ValueError(f"La columna '{col_name}' no existe en la tabla '{t.name}'.")
            pos_to_concat.append(name_pos[col_name])
            
        print(f"Iterador FTS: Concatenando columnas en posiciones: {pos_to_concat}")

        for rid, row_tuple in t.scan() or []:
            # Concatena todos los campos de texto en un solo bloque
            full_text = " ".join(str(row_tuple[p]) for p in pos_to_concat)
            yield (rid, full_text)

    # --- NUEVA FUNCIÓN: Acción para crear FTS ---
    def _create_fts_index(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        table_name = stmt["table"]
        columns = stmt["columns"]
        
        print(f"Iniciando construcción de FTS INDEX para {table_name} en columnas: {columns}")
        
        t = self._get_table(table_name)
        
        # 1. Crear el iterador de documentos
        doc_iterator = self._make_doc_iterator(t, columns)
        
        # 2. Instanciar y ejecutar el constructor
        # Asumimos que el constructor FTS vive en el data_dir raíz
        builder = InvertedIndexBuilder(data_dir=self.data_dir)
        
        start_time = time.time()
        builder.build(doc_iterator)
        end_time = time.time()
        
        total_docs = builder.total_docs
        
        # 3. Recargar el módulo de consulta para que esté disponible
        self._load_fts_query_module()
        
        return {
            "ok": True,
            "message": "Índice FTS construido exitosamente.",
            "total_docs_indexed": total_docs,
            "time_taken_sec": (end_time - start_time)
        }

    # --- MODIFICADO: _select ---
    def _select(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        t = self._get_table(stmt["table"])
        cols = stmt.get("columns", ["*"])
        cond = stmt.get("condition")
        
        # --- MODIFICADO: Obtener K (limit) ---
        limit_k = stmt.get("limit", 100) # Default a 100 si no se especifica
        
        name_pos = _name_to_pos(t.schema)

        # Proyección
        if cols == ["*"]:
            proj_pos = list(range(len(t.schema)))
            proj_names = [n for n, _t, _l in t.schema]
        else:
            proj_pos = [name_pos[c] for c in cols]
            proj_names = cols

        # Planner ultra básico: intenta usar índice antes que scan
        if cond:
            op = cond["op"]

            # --- NUEVA LÓGICA FTS ---
            if op == "FTS":
                if not self.fts_query:
                    return {"ok": False, "rows": [], "columns": [], "error": "FTS index not found. Use CREATE FTS INDEX ON table(cols) first."}
                
                query_text = cond["query_text"]
                # Usamos el K del LIMIT
                # fts_query.query() devuelve: [(score, docID), ...]
                results = self.fts_query.query(query_text, k=limit_k)
                
                rows = []
                # Añadimos 'score' a las columnas
                final_columns = ["score"] + proj_names
                
                for score, rid in results:
                    try:
                        record_tuple = t.get_record(rid)
                        projected_row = [record_tuple[p] for p in proj_pos]
                        # Prepend el score formateado
                        rows.append(["{:.6f}".format(score)] + projected_row)
                    except (IOError, IndexError):
                        # El registro podría no existir si el índice está desactualizado
                        continue
                        
                return {"ok": True, "rows": rows, "columns": final_columns, "used_index": {"type": "FTS", "column": cond["field"]}}
            
            # ----- "=" y BETWEEN e IN 1D -----
            if op in ("=", "BETWEEN", "IN"):
                field = cond["field"]

                if op == "=":
                    kind, payload = self.idx.probe_eq(t.name, field, cond["value"])
                    if kind == 'records':
                        rows = [[rec[p] for p in proj_pos] for rec in payload]
                    elif kind == 'rids':
                        rows = [[list(t.get_record(rid))[p] for p in proj_pos] for rid in payload]
                    else:
                        rows = None

                    if rows is not None:
                        idx_inst = self.idx.list_for_table(t.name).get(field, None)
                        used = {"column": field, "type": idx_inst.__class__.__name__} if idx_inst else None
                        return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used} # Aplicar limit

                elif op == "BETWEEN":
                    kind, payload = self.idx.probe_between(t.name, field, cond["low"], cond["high"])
                    if kind == 'records':
                        rows = [[rec[p] for p in proj_pos] for rec in payload]
                    elif kind == 'rids':
                        rows = [[list(t.get_record(rid))[p] for p in proj_pos] for rid in payload]
                    else:
                        rows = None

                    if rows is not None:
                        idx_inst = self.idx.list_for_table(t.name).get(field, None)
                        used = {"column": field, "type": idx_inst.__class__.__name__} if idx_inst else None
                        return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used} # Aplicar limit

                elif op == "IN":
                    kind, payload = self.idx.probe_rtree_radius(
                        t.name, field, cond["coords"], cond["radius"]
                    )
                    if kind == 'rids':
                        rows = [[list(t.get_record(rid))[p] for p in proj_pos] for rid in payload]
                        idx_inst = self.idx.list_for_table(t.name).get(field, None)
                        used = {"column": field, "type": idx_inst.__class__.__name__} if idx_inst else None
                        return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used} # Aplicar limit

            # ----- IN2: RTREE 2D sobre (lat,lon) -----
            if op == "IN2":
                fields = cond["fields"]        # ['latitude', 'longitude']
                synthetic = ",".join(fields)   # "latitude,longitude"

                kind, payload = self.idx.probe_rtree_radius(
                    t.name,
                    synthetic,
                    cond["coords"],
                    cond["radius"],
                )
                if kind == 'rids':
                    rows = [[list(t.get_record(rid))[p] for p in proj_pos] for rid in payload]
                    idx_inst = self.idx.list_for_table(t.name).get(synthetic, None)
                    used = {"column": synthetic, "type": idx_inst.__class__.__name__} if idx_inst else None
                    return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used} # Aplicar limit

        # Fallback: scan con predicado (sin índice)
        pred = self._make_predicate(t.schema, cond)
        rows: List[List[Any]] = []
        count = 0
        for _rid, row_tuple in t.scan() or []:
            if count >= limit_k: # <-- MODIFICADO: Aplicar LIMIT también al scan
                break
            row = list(row_tuple)
            if pred(row):
                rows.append([row[p] for p in proj_pos])
                count += 1
                
        return {"ok": True, "rows": rows, "columns": proj_names}

    
    def _create_index_action(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        t = self._get_table(stmt["table"])
        # 👇 Normalizar a lista
        cols = stmt.get("columns")
        if cols is None:
            col = stmt.get("column")
            cols = [col] if isinstance(col, str) else (col or [])
        elif isinstance(cols, str):
            cols = [cols]

        if not cols:
            raise ValueError("CREATE INDEX: faltan columnas")

        typ = stmt["index_type"]

        # crea el índice (IndexManager ya acepta lista o 1 col)
        self.idx.create_index(t, cols, typ)

        # persistir spec SIEMPRE como ["col"] o ["lat,lon"]
        key_str = ",".join(cols)

        if not hasattr(t, "index_specs"):
            t.index_specs = []
        spec_item = [key_str, typ]
        if spec_item not in t.index_specs:
            t.index_specs.append(spec_item)
            t._save_metadata()

        return {"ok": True}


    def _drop_index_action(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        t = self._get_table(stmt["table"])
        col = stmt["column"]
        self.idx.drop_index(t, col)
        # actualiza metadata
        t.index_specs = [(c, ty) for (c, ty) in getattr(t, "index_specs", []) if c != col]
        t._save_metadata()
        return {"ok": True}


    # ---------- Utilidades ----------
    def _get_table(self, name: str) -> Table:
        if name not in self.catalog:
            meta_path = os.path.join(self.data_dir, f"{name}.meta")
            if os.path.exists(meta_path):
                t = Table(name, data_dir=self.data_dir)
                self.catalog[name] = t
                # reconstruir índices declarados en metadata
                for col, typ in getattr(t, "index_specs", []):
                    # Evitar reconstruir FTS aquí, se maneja globalmente
                    if typ.upper() == "FTS":
                        continue
                    cols_list = col.split(',')
                    self.idx.create_index(t, cols_list, typ)
            else:
                raise ValueError(f"Tabla no existe: {name}")
        return self.catalog[name]


    def _read_csv_bytes(self, blob: bytes, has_header: bool):
        f = io.StringIO(blob.decode("utf-8-sig", errors="replace"))
        reader = csv.reader(f)
        header = next(reader) if has_header else None
        rows = list(reader)
        return header, rows


    def _cast_row_to_schema(self, values: List[Any], schema: List[Tuple[str, str, int]]) -> List[Any]:
        if len(values) != len(schema):
            raise ValueError(f"Número de valores {len(values)} no coincide con columnas {len(schema)}")
        out: List[Any] = []
        for v, (_name, base_type, length) in zip(values, schema):
            out.append(_cast_for_column(v, base_type, length))
        return out

    def _make_predicate(self, schema: List[Tuple[str, str, int]], cond: Optional[Dict[str, Any]]) -> Callable[[List[Any]], bool]:
        if cond is None:
            return lambda row: True
        name_pos = _name_to_pos(schema)
        op = cond["op"]
        
        # FTS no se puede aplicar como predicado de scan, se maneja arriba
        if op == "FTS":
            return lambda row: True # (No debería llegar aquí)

        field = cond["field"]
        pos = name_pos[field]

        if op == "=":
            val = cond["value"]
            _t, _len = schema[pos][1], schema[pos][2]
            cmp_val = _cast_for_column(val, _t, _len)
            return lambda row: row[pos] == cmp_val

        if op == "BETWEEN":
            low = cond["low"]
            high = cond["high"]
            _t, _len = schema[pos][1], schema[pos][2]
            low_c = _cast_for_column(low, _t, _len)
            high_c = _cast_for_column(high, _t, _len)
            return lambda row: low_c <= row[pos] <= high_c

        if op == "IN":
            coords = tuple(cond["coords"])
            radius = float(cond["radius"])
            # Distancia euclídea con parseo de cadena "[x, y]".
            def _dist_ok(row):
                v = _parse_coords_from_value(row[pos])
                if v is None or len(v) != len(coords):
                    return False
                s = 0.0
                for a, b in zip(v, coords):
                    d = (a - b)
                    s += d * d
                return s <= radius * radius  # sin sqrt
            return _dist_ok

        return lambda row: True

    def _rewrite_table_file(self, t: Table, rows: List[List[Any]]) -> None:
        rm = t.record_manager
        with open(t.dat_path, "wb") as f:
            for r in rows:
                f.write(rm.pack(r))

# Singleton para integrar con la API
_engine_singleton: Optional[Engine] = None

def get_engine() -> Engine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = Engine()
    return _engine_singleton