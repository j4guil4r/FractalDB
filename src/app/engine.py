# src/app/engine.py
from src.core.table import Table
from src.indices.manager import IndexManager

from typing import Any, Dict, List, Optional, Tuple, Callable, Iterator
import io
import csv
import os
import re
import time
import glob 
import numpy as np 

from src.indices.inverted_index.builder import InvertedIndexBuilder
from src.indices.inverted_index.query import InvertedIndexQuery

from src.multimedia.codebook_builder import CodebookBuilder
from src.multimedia.histogram_builder import BoVWHistogramBuilder
from src.multimedia.knn_search import KNNSearch 
from src.multimedia.inverted_index_builder_mm import MMInvertedIndexBuilder
from src.multimedia.inverted_index_query_mm import MMInvertedIndexQuery

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

class Engine:
    def __init__(self, data_dir: str = "data"):
        self.catalog: Dict[str, Table] = {}
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.idx = IndexManager()
        
        self.fts_query: Optional[InvertedIndexQuery] = None
        self.mm_query_modules: Dict[int, MMInvertedIndexQuery] = {}
        self.knn_seq_search: Dict[int, KNNSearch] = {}
        
        self._load_query_modules()
            
    def _load_query_modules(self):
        """ Carga/Recarga todos los módulos de consulta FTS y MM disponibles. """
        
        try:
            if self.fts_query: self.fts_query.close()
            self.fts_query = InvertedIndexQuery(data_dir=self.data_dir)
            print("Motor: Módulo de consulta FTS cargado.")
        except FileNotFoundError:
            print("Motor: Índice FTS no encontrado. Las consultas @@ fallarán.")
            self.fts_query = None
        except Exception as e:
            print(f"Motor: Error al cargar módulo FTS: {e}")
            self.fts_query = None
        
        for mod in self.mm_query_modules.values(): mod.close()
        self.mm_query_modules.clear()
        
        meta_pattern = os.path.join(self.data_dir, "mm_inverted_index_k*.meta")
        for meta_path in glob.glob(meta_pattern):
            try:
                k_str = re.search(r"_k(\d+)\.meta$", meta_path)
                if k_str:
                    k = int(k_str.group(1))
                    if k not in self.mm_query_modules:
                        print(f"Motor: Detectado índice MM K={k}. Cargando...")
                        self.mm_query_modules[k] = MMInvertedIndexQuery(k_clusters=k, data_dir=self.data_dir)
            except Exception as e:
                print(f"Motor: Error al cargar índice MM desde {meta_path}: {e}")

        self.knn_seq_search.clear()

    # --- Volvemos a tu _infer_schema_from_rows original (P1) ---
    def _infer_schema_from_rows(self, header, rows, scan_cap=50000):
        if header:
            names = header
            ncols = len(header)
        else:
            if not rows:
                raise ValueError("CSV sin header ni filas")
            ncols = len(rows[0])
            names = [f"col{i}" for i in range(ncols)]

        int_re = re.compile(r"^-?\d+$")
        float_re = re.compile(r"^-?\d+(?:\.\d+)?$")
        date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        flags = []
        max_len = [0] * ncols
        for _ in range(ncols):
            flags.append({
                "all_int": True,
                "all_numeric": True,
                "all_date_iso": True,
                "any_empty": False,
                "force_varchar": False,
            })

        for r_i, row in enumerate(rows):
            if r_i >= scan_cap:
                break
            for i in range(ncols):
                val = "" if i >= len(row) else (row[i] if row[i] is not None else "")
                s = str(val).strip()
                max_len[i] = max(max_len[i], len(s.encode("utf-8")))

                if s == "":
                    flags[i]["any_empty"] = True
                    flags[i]["all_int"] = False
                    flags[i]["all_numeric"] = False
                    flags[i]["all_date_iso"] = False
                    flags[i]["force_varchar"] = True
                    continue

                if int_re.match(s):
                    flags[i]["all_date_iso"] = False
                    continue

                if float_re.match(s):
                    flags[i]["all_int"] = False
                    flags[i]["all_date_iso"] = False
                    continue

                if date_re.match(s):
                    flags[i]["all_int"] = False
                    flags[i]["all_numeric"] = False
                    continue

                flags[i]["all_int"] = False
                flags[i]["all_numeric"] = False
                flags[i]["all_date_iso"] = False
                flags[i]["force_varchar"] = True

        schema = []
        for i, name in enumerate(names):
            f = flags[i]
            if not f["force_varchar"]:
                if f["all_int"]:
                    schema.append((name, "INT", 0))
                    continue
                if f["all_numeric"]:
                    schema.append((name, "FLOAT", 0))
                    continue
                if f["all_date_iso"]:
                    schema.append((name, "VARCHAR", 10))
                    continue
            length = min(max_len[i] if max_len[i] > 0 else 1, 255)
            schema.append((name, "VARCHAR", length))
        return schema
    # --- FIN _infer_schema_from_rows ---

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
        if act == "create_fts_index":
            return self._create_fts_index(stmt)
        if act == "create_mm_index":
            return self._create_mm_index(stmt)
        if act == "drop_index":
            return self._drop_index_action(stmt)
        raise ValueError(f"Acción no soportada: {act}")

    def load_csv_bytes(self, table: str, blob: bytes, has_header: bool = True) -> int:
        header, rows = self._read_csv_bytes(blob, has_header)

        if table not in self.catalog:
            schema = self._infer_schema_from_rows(header, rows)
            t = Table(table, schema=schema, data_dir=self.data_dir)
            self.catalog[table] = t
        else:
            t = self.catalog[table]

        inserted = 0
        for r in rows:
            values = self._cast_row_to_schema(r, t.schema)
            rid = t.insert_record(values)
            self.idx.on_insert(t, rid, values)
            inserted += 1

        return inserted
    
    def load_csv_path(self, table: str, csv_path: str, has_header: bool = True):
        import csv as _csv
        import os

        if not os.path.exists(csv_path):
            raise ValueError(f"Archivo no encontrado: {csv_path}")

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

        if table not in self.catalog:
            t = Table(table, schema=schema, data_dir=self.data_dir)
            self.catalog[table] = t
        else:
            t = self.catalog[table]

        casters = self._compile_casters(schema)
        rm = t.record_manager

        inserted = 0
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f, \
            open(t.dat_path, "ab", buffering=2**20) as fout:
            reader = _csv.reader(f)
            if has_header:
                _ = next(reader, None)  

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
        return [f(v) for f, v in zip(casters, row_tuple)]

    def _create_table(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        name = stmt["table"]
        if name in self.catalog:
            raise ValueError(f"La tabla '{name}' ya existe")

        schema_triples: List[Tuple[str, str, int]] = []
        declared_indexes: List[Tuple[str, str]] = []  

        for col in stmt["columns"]:
            base_type, length = _parse_sql_type_to_core(col["type"])
            schema_triples.append((col["name"], base_type, length))
            if col.get("index"):
                declared_indexes.append((col["name"], col["index"].upper()))

        t = Table(name, schema=schema_triples, data_dir=self.data_dir)
        self.catalog[name] = t

        for col_name, idx_type in declared_indexes:
            self.idx.create_index(t, col_name, idx_type)

            if not hasattr(t, "index_specs"):
                t.index_specs = []
            if (col_name, idx_type) not in t.index_specs:
                t.index_specs.append((col_name, idx_type))
                t._save_metadata()

        return {"ok": True}

    def _create_table_from_file(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        name = stmt["table"]
        file_path = stmt["file"]
        index_type = stmt["index_type"].upper()
        index_col = stmt["index_column"]

        if not os.path.exists(file_path):
            raise ValueError(f"Archivo no encontrado: {file_path}")
        if name in self.catalog or os.path.exists(os.path.join(self.data_dir, f"{name}.meta")):
            raise ValueError(f"La tabla '{name}' ya existe. Borra los archivos en data/ o usa otro nombre.")

        sample_rows = []
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                raise ValueError("Se espera header en el CSV para crear el esquema")
            for i, row in enumerate(reader, start=1):
                sample_rows.append(row)
                if i >= 50000:   
                    break

        schema = self._infer_schema_from_rows(header, sample_rows)
        t = Table(name, schema=schema, data_dir=self.data_dir)
        self.catalog[name] = t

        inserted = 0
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  
            for row in reader:
                if not row:
                    continue
                values = self._cast_row_to_schema(row, t.schema)
                t.insert_record(values)
                inserted += 1

        self.idx.create_index(t, index_col, index_type)

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

        self._rewrite_table_file(t, kept)
        
        self.idx.rebuild_all(t)
        
        print(f"DELETE: Reconstruyendo índices especiales para {t.name}...")
        specs_to_rebuild = getattr(t, "index_specs", [])
        
        for col_key, idx_type in specs_to_rebuild:
            
            if idx_type == "FTS":
                print(f"  -> Reconstruyendo índice FTS en columnas: {col_key}")
                columns = col_key.split(",")
                try:
                    doc_iterator = self._make_doc_iterator(t, columns)
                    builder = InvertedIndexBuilder(data_dir=self.data_dir)
                    builder.build(doc_iterator)
                except Exception as e:
                    print(f"  -> ERROR al reconstruir FTS: {e}")

            elif idx_type.startswith("MM_BOVW_K"):
                print(f"  -> Reconstruyendo índice {idx_type} en columna: {col_key}")
                try:
                    k = int(idx_type.split("=")[1])
                    column_name = col_key
                    self._rebuild_mm_index_internal(t, column_name, k)
                except Exception as e:
                    print(f"  -> ERROR al reconstruir MM: {e}")
        
        self._load_query_modules()

        return {"ok": True, "deleted": deleted}

    def _make_doc_iterator(self, t: Table, text_columns: List[str]) -> Iterator[Tuple[int, str]]:
        name_pos = _name_to_pos(t.schema)
        pos_to_concat = []
        for col_name in text_columns:
            if col_name not in name_pos:
                raise ValueError(f"La columna '{col_name}' no existe en la tabla '{t.name}'.")
            pos_to_concat.append(name_pos[col_name])
            
        print(f"Iterador FTS: Concatenando columnas en posiciones: {pos_to_concat}")

        for rid, row_tuple in t.scan() or []:
            full_text = " ".join(str(row_tuple[p]) for p in pos_to_concat)
            yield (rid, full_text)

    def _create_fts_index(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        table_name = stmt["table"]
        columns = stmt["columns"]
        
        print(f"Iniciando construcción de FTS INDEX para {table_name} en columnas: {columns}")
        t = self._get_table(table_name)
        
        doc_iterator = self._make_doc_iterator(t, columns)
        builder = InvertedIndexBuilder(data_dir=self.data_dir)
        
        start_time = time.time()
        builder.build(doc_iterator)
        end_time = time.time()
        
        total_docs = builder.total_docs
        
        spec_key = ",".join(columns)
        spec_type = "FTS"
        if not hasattr(t, "index_specs"): t.index_specs = []
        if (spec_key, spec_type) not in t.index_specs:
            t.index_specs.append((spec_key, spec_type))
            t._save_metadata()
        
        self._load_query_modules() 
        
        return {
            "ok": True,
            "message": "Índice FTS construido exitosamente.",
            "total_docs_indexed": total_docs,
            "time_taken_sec": (end_time - start_time)
        }

    def _make_image_iterator(self, t: Table, img_col_name: str) -> Iterator[Tuple[int, str]]:
        try:
            pos = _name_to_pos(t.schema)[img_col_name]
        except KeyError:
            raise ValueError(f"La columna de imagen '{img_col_name}' no existe en la tabla '{t.name}'.")
            
        print(f"Iterador MM: Leyendo rutas de imagen desde la columna '{img_col_name}' (pos {pos})")

        for rid, row_tuple in t.scan() or []:
            path = str(row_tuple[pos])
            if path and (path.endswith('.jpg') or path.endswith('.png') or path.endswith('.jpeg')):
                yield (rid, path)

    def _rebuild_mm_index_internal(self, t: Table, column_name: str, k: int):
        print(f"Paso 1/3 (Rebuild): Escaneando rutas de imágenes para K={k}...")
        image_paths_tuples = list(self._make_image_iterator(t, column_name))
        image_paths_only = [path for _rid, path in image_paths_tuples]
        
        if not image_paths_only:
            print(f"Advertencia: No se encontraron rutas de imagen válidas para K={k}. El índice estará vacío.")

        print(f"Paso 2/3 (Rebuild): Construyendo Codebook (K={k})...")
        cb = CodebookBuilder(k, self.data_dir)
        cb.build_from_paths(image_paths_only) 
        
        print(f"Paso 3/3 (Rebuild): Construyendo Índice Invertido MM (K={k})...")
        hist_builder = BoVWHistogramBuilder(k, self.data_dir)
        mm_idx_builder = MMInvertedIndexBuilder(self.data_dir, k_clusters=k)
        
        def hist_generator() -> Iterator[Tuple[int, np.ndarray]]:
            print("  -> Iniciando generador de histogramas...")
            for rid, path in image_paths_tuples:
                hist_tf = hist_builder.create_histogram_from_path(path)
                if hist_tf is not None:
                    yield (rid, hist_tf)
        
        mm_idx_builder.build(hist_generator())
        return mm_idx_builder.total_docs

    def _create_mm_index(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        table_name = stmt["table"]
        column_name = stmt["column"]
        k = stmt["k"]
        
        print(f"Iniciando construcción de MM INDEX (BoVW K={k}) para {table_name} en columna: {column_name}")
        t = self._get_table(table_name)
        start_time = time.time()

        total_docs = self._rebuild_mm_index_internal(t, column_name, k)
        
        spec_key = column_name
        spec_type = f"MM_BOVW_K={k}"
        if not hasattr(t, "index_specs"): t.index_specs = []
        if (spec_key, spec_type) not in t.index_specs:
            t.index_specs.append((spec_key, spec_type))
            t._save_metadata()
        
        print("Paso 4/4: Recargando módulos de consulta.")
        self._load_query_modules()
        
        end_time = time.time()
        
        return {
            "ok": True,
            "message": f"Índice Multimedia (BoVW K={k}) construido exitosamente.",
            "total_images_indexed": total_docs,
            "codebook_size": k,
            "time_taken_sec": (end_time - start_time)
        }

    def _select(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        t = self._get_table(stmt["table"])
        cols = stmt.get("columns", ["*"])
        cond = stmt.get("condition")
        
        # --- ESTA ES LA CORRECCIÓN CLAVE #2 ---
        limit_k = stmt.get("limit") or 100 # <--- 'or 100' maneja el 'None'
        # --- FIN DE LA CORRECCIÓN ---
        
        name_pos = _name_to_pos(t.schema)

        if cols == ["*"]:
            proj_pos = list(range(len(t.schema)))
            proj_names = [n for n, _t, _l in t.schema]
        else:
            proj_pos = [name_pos[c] for c in cols]
            proj_names = cols

        if cond:
            op = cond["op"]

            if op == "MM_SIM":
                if not self.mm_query_modules:
                    return {"ok": False, "rows": [], "columns": [], "error": "Índice MM no encontrado. Use CREATE MM INDEX ON ..."}
                
                k_from_query = cond.get("k") 
                query_module = None
                
                if k_from_query:
                    query_module = self.mm_query_modules.get(k_from_query)
                    if not query_module:
                        return {"ok": False, "error": f"Índice MM con K={k_from_query} no encontrado o no cargado."}
                else:
                    if not self.mm_query_modules:
                         return {"ok": False, "error": "No hay índices MM cargados."}
                    query_module = list(self.mm_query_modules.values())[0]
                
                k_used = query_module.k
                query_path = cond["query_path"]
                
                if not os.path.exists(query_path):
                     return {"ok": False, "error": f"Archivo de consulta no encontrado: {query_path}"}

                results = query_module.query_by_path(query_path, k=limit_k)
                
                rows = []
                final_columns = ["score"] + proj_names
                
                for score, rid in results:
                    try:
                        record_tuple = t.get_record(rid)
                        projected_row = [record_tuple[p] for p in proj_pos]
                        rows.append(["{:.6f}".format(score)] + projected_row)
                    except (IOError, IndexError):
                        continue
                        
                return {"ok": True, "rows": rows, "columns": final_columns, "used_index": {"type": f"MM_BOVW_INV (K={k_used})", "column": cond["field"]}}

            if op == "FTS":
                if not self.fts_query:
                    return {"ok": False, "rows": [], "columns": [], "error": "FTS index not found."}
                query_text = cond["query_text"]
                results = self.fts_query.query(query_text, k=limit_k)
                rows = []
                final_columns = ["score"] + proj_names
                for score, rid in results:
                    try:
                        record_tuple = t.get_record(rid)
                        projected_row = [record_tuple[p] for p in proj_pos]
                        rows.append(["{:.6f}".format(score)] + projected_row)
                    except (IOError, IndexError):
                        continue
                return {"ok": True, "rows": rows, "columns": final_columns, "used_index": {"type": "FTS", "column": cond["field"]}}
            
            if op in ("=", "BETWEEN", "IN"):
                field = cond["field"]
                if op == "=":
                    kind, payload = self.idx.probe_eq(t.name, field, cond["value"])
                    if kind == 'records': rows = [[rec[p] for p in proj_pos] for rec in payload]
                    elif kind == 'rids': rows = [[list(t.get_record(rid))[p] for p in proj_pos] for rid in payload]
                    else: rows = None
                    if rows is not None:
                        idx_inst = self.idx.list_for_table(t.name).get(field, None)
                        used = {"column": field, "type": idx_inst.__class__.__name__} if idx_inst else None
                        return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used}
                elif op == "BETWEEN":
                    kind, payload = self.idx.probe_between(t.name, field, cond["low"], cond["high"])
                    if kind == 'records': rows = [[rec[p] for p in proj_pos] for rec in payload]
                    elif kind == 'rids': rows = [[list(t.get_record(rid))[p] for p in proj_pos] for rid in payload]
                    else: rows = None
                    if rows is not None:
                        idx_inst = self.idx.list_for_table(t.name).get(field, None)
                        used = {"column": field, "type": idx_inst.__class__.__name__} if idx_inst else None
                        return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used}
                elif op == "IN":
                    kind, payload = self.idx.probe_rtree_radius(t.name, field, cond["coords"], cond["radius"])
                    if kind == 'rids':
                        rows = [[list(t.get_record(rid))[p] for p in proj_pos] for rid in payload]
                        idx_inst = self.idx.list_for_table(t.name).get(field, None)
                        used = {"column": field, "type": idx_inst.__class__.__name__} if idx_inst else None
                        return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used}
            
            if op == "IN2":
                fields = cond["fields"]; synthetic = ",".join(fields)
                kind, payload = self.idx.probe_rtree_radius(t.name, synthetic, cond["coords"], cond["radius"])
                if kind == 'rids':
                    rows = [[list(t.get_record(rid))[p] for p in proj_pos] for rid in payload]
                    idx_inst = self.idx.list_for_table(t.name).get(synthetic, None)
                    used = {"column": synthetic, "type": idx_inst.__class__.__name__} if idx_inst else None
                    return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used}

        # Fallback: scan
        pred = self._make_predicate(t.schema, cond)
        rows: List[List[Any]] = []
        count = 0
        for _rid, row_tuple in t.scan() or []:
            if count >= limit_k: # <--- El bug 'int' >= 'NoneType' ocurría aquí si limit_k era None
                break
            row = list(row_tuple)
            if pred(row):
                rows.append([row[p] for p in proj_pos])
                count += 1
        return {"ok": True, "rows": rows, "columns": proj_names}

    
    def _create_index_action(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        t = self._get_table(stmt["table"])
        cols = stmt.get("columns")
        if cols is None:
            col = stmt.get("column")
            cols = [col] if isinstance(col, str) else (col or [])
        elif isinstance(cols, str):
            cols = [cols]
        if not cols: raise ValueError("CREATE INDEX: faltan columnas")
        typ = stmt["index_type"]
        self.idx.create_index(t, cols, typ)
        key_str = ",".join(cols)
        if not hasattr(t, "index_specs"): t.index_specs = []
        spec_item = [key_str, typ]
        if spec_item not in t.index_specs:
            t.index_specs.append(spec_item)
            t._save_metadata()
        return {"ok": True}


    def _drop_index_action(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        t = self._get_table(stmt["table"])
        col = stmt["column"]
        self.idx.drop_index(t, col)
        t.index_specs = [(c, ty) for (c, ty) in getattr(t, "index_specs", []) if c != col]
        t._save_metadata()
        return {"ok": True}


    def _get_table(self, name: str) -> Table:
        if name not in self.catalog:
            meta_path = os.path.join(self.data_dir, f"{name}.meta")
            if os.path.exists(meta_path):
                t = Table(name, data_dir=self.data_dir)
                self.catalog[name] = t
                # --- MODIFICADO (P2): No reconstruir FTS/MM al cargar ---
                for col, typ in getattr(t, "index_specs", []):
                    if typ.upper() == "FTS" or typ.upper().startswith("MM_BOVW"):
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
        
        # --- MODIFICADO (P2): Mover esto *después* del check de None ---
        name_pos = _name_to_pos(schema)
        op = cond["op"]
        
        if op == "FTS" or op == "MM_SIM":
            return lambda row: True 

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
            def _dist_ok(row):
                v = _parse_coords_from_value(row[pos])
                if v is None or len(v) != len(coords):
                    return False
                s = 0.0
                for a, b in zip(v, coords):
                    d = (a - b)
                    s += d * d
                return s <= radius * radius
            return _dist_ok

        return lambda row: True
    # --- FIN MODIFICADO (P2) ---

    def _rewrite_table_file(self, t: Table, rows: List[List[Any]]) -> None:
        rm = t.record_manager
        with open(t.dat_path, "wb") as f:
            for r in rows:
                f.write(rm.pack(r))

_engine_singleton: Optional[Engine] = None

def get_engine() -> Engine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = Engine()
    return _engine_singleton