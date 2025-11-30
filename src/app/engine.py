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
import concurrent.futures
import cv2
from sklearn.neighbors import KDTree

from src.indices.inverted_index.builder import InvertedIndexBuilder
from src.indices.inverted_index.query import InvertedIndexQuery

from src.multimedia.codebook_builder import CodebookBuilder, AudioCodebookBuilder
from src.multimedia.knn_search import KNNSearch, AudioKNNSearch
from src.multimedia.inverted_index_builder_mm import MMInvertedIndexBuilder,AudioMMInvertedIndexBuilder
from src.multimedia.inverted_index_query_mm import MMInvertedIndexQuery,AudioMMInvertedIndexQuery


from src.multimedia.histogram_builder import AudioBoVWHistogramBuilder,BoVWHistogramBuilder


_worker_sift = None
_worker_tree = None
_worker_k = 0

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

def _init_worker(centers, k):
    """Inicializa SIFT y KDTree en cada proceso worker."""
    global _worker_sift, _worker_tree, _worker_k
    _worker_sift = cv2.SIFT_create()
    _worker_tree = KDTree(centers)
    _worker_k = k


def _process_image_task(path):
    """Extrae histograma BoVW para una imagen (worker)."""
    global _worker_sift, _worker_tree, _worker_k
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        MAX_DIM = 300
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        _, des = _worker_sift.detectAndCompute(img, None)
        if des is None:
            return np.zeros(_worker_k, dtype=np.float32)

        _, visual_words = _worker_tree.query(des, k=1)
        visual_words = visual_words.flatten()
        hist = np.bincount(visual_words, minlength=_worker_k)
        return hist.astype(np.float32)
    except Exception:
        return None


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


def _detect_modality_from_paths(paths: List[str]) -> str:
    """Intenta decidir si son rutas de imagen o de audio según la extensión."""
    exts = {os.path.splitext(p)[1].lower() for p in paths if p}
    has_audio = any(e in AUDIO_EXTS for e in exts)
    has_image = any(e in IMAGE_EXTS for e in exts)
    if has_audio and not has_image:
        print("Motor MM: columna detectada como AUDIO.")
        return "audio"
    print("Motor MM: columna detectada como IMAGEN.")
    return "image"


def _detect_modality_from_query_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in AUDIO_EXTS:
        return "audio"
    return "image"


class Engine:
    def __init__(self, data_dir: str = "data"):
        self.catalog: Dict[str, Table] = {}
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.idx = IndexManager()

        self.fts_query: Dict[str, InvertedIndexQuery] = {}
        self.mm_query_modules: Dict[int, MMInvertedIndexQuery] = {}          # imagen
        self.mm_audio_query_modules: Dict[int, AudioMMInvertedIndexQuery] = {}  # audio
        self.knn_seq_search: Dict[int, KNNSearch] = {}                       # imagen
        self.audio_knn_seq_search: Dict[int, AudioKNNSearch] = {}            # audio

        self._load_query_modules()
    
    @staticmethod
    def _is_image_path(path: Any) -> bool:
        if not isinstance(path, str):
            return False
        p = path.lower()
        return p.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"))

    @staticmethod
    def _is_audio_path(path: Any) -> bool:
        if not isinstance(path, str):
            return False
        p = path.lower()
        return p.endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"))

    def _load_query_modules(self):
        """Carga/recarga módulos de consulta FTS y MM (imagen/audio)."""

        for q in self.fts_query.values():
            q.close()
        self.fts_query.clear()

        meta_files = [f[:-5] for f in os.listdir(self.data_dir)
                      if f.endswith(".meta") and "inverted_index" not in f]

        for table_name in meta_files:
            try:
                t = self._get_table(table_name)
                for col_spec, idx_type in getattr(t, "index_specs", []):
                    if idx_type == "FTS":
                        safe_cols = col_spec.replace(",", "_")
                        idx_name = f"fts_{table_name}_{safe_cols}"
                        try:
                            key = f"{table_name}.{col_spec}"
                            self.fts_query[key] = InvertedIndexQuery(self.data_dir, index_name=idx_name)
                            print(f"Motor: Índice FTS cargado para '{key}' -> {idx_name}.dat")
                        except Exception as e:
                            print(f"Motor: Error cargando FTS '{idx_name}': {e}")
            except Exception:
                pass

        for mod in self.mm_query_modules.values():
            mod.close()
        for mod in self.mm_audio_query_modules.values():
            mod.close()
        self.mm_query_modules.clear()
        self.mm_audio_query_modules.clear()

        meta_pattern_img = os.path.join(self.data_dir, "mm_inverted_index_k*.meta")
        for meta_path in glob.glob(meta_pattern_img):
            try:
                k_str = re.search(r"_k(\d+)\.meta$", meta_path)
                if k_str:
                    k = int(k_str.group(1))
                    if k not in self.mm_query_modules:
                        print(f"Motor: Detectado índice MM IMAGEN K={k}. Cargando...")
                        self.mm_query_modules[k] = MMInvertedIndexQuery(k_clusters=k, data_dir=self.data_dir)
            except Exception as e:
                print(f"Motor: Error al cargar índice MM imagen desde {meta_path}: {e}")

        meta_pattern_audio = os.path.join(self.data_dir, "mm_audio_inverted_index_k*.meta")
        for meta_path in glob.glob(meta_pattern_audio):
            try:
                k_str = re.search(r"_k(\d+)\.meta$", meta_path)
                if k_str:
                    k = int(k_str.group(1))
                    if k not in self.mm_audio_query_modules:
                        print(f"Motor: Detectado índice MM AUDIO K={k}. Cargando...")
                        self.mm_audio_query_modules[k] = AudioMMInvertedIndexQuery(k_clusters=k,
                                                                                   data_dir=self.data_dir)
            except Exception as e:
                print(f"Motor: Error al cargar índice MM audio desde {meta_path}: {e}")

        self.knn_seq_search.clear()
        self.audio_knn_seq_search.clear()

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
        import os as _os

        _csv.field_size_limit(10_000_000)

        if not _os.path.exists(csv_path):
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
                open(t.dat_path, "ab", buffering=2 ** 20) as fout:
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
                buf[pos:pos + len(rec)] = rec
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

        return {"ok": True, "table": name, "inserted": inserted,
                "indexed": {"column": index_col, "type": index_type}}

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

        spec_key = ",".join(columns)
        safe_cols = spec_key.replace(",", "_")
        unique_index_name = f"fts_{table_name}_{safe_cols}"

        doc_iterator = self._make_doc_iterator(t, columns)
        builder = InvertedIndexBuilder(data_dir=self.data_dir, index_name=unique_index_name)

        start_time = time.time()
        builder.build(doc_iterator)
        end_time = time.time()

        total_docs = builder.total_docs

        spec_type = "FTS"
        if not hasattr(t, "index_specs"):
            t.index_specs = []

        if (spec_key, spec_type) not in t.index_specs:
            t.index_specs.append((spec_key, spec_type))
            t._save_metadata()

        self._load_query_modules()

        return {
            "ok": True,
            "message": f"Índice FTS '{unique_index_name}' construido exitosamente.",
            "total_docs_indexed": total_docs,
            "time_taken_sec": (end_time - start_time)
        }

    def _make_mm_iterator(self, t: Table, col_name: str) -> Iterator[Tuple[int, str]]:
        """Lee rutas multimedia (imagen o audio) de una columna."""
        try:
            pos = _name_to_pos(t.schema)[col_name]
        except KeyError:
            raise ValueError(f"La columna multimedia '{col_name}' no existe en la tabla '{t.name}'.")

        print(f"Iterador MM: leyendo rutas desde la columna '{col_name}' (pos {pos})")

        for rid, row_tuple in t.scan() or []:
            path = str(row_tuple[pos])
            if path:
                yield (rid, path)

    def _rebuild_mm_index_internal(self, t: Table, column_name: str, k: int):
        print(f"Paso 1/3 (Rebuild): Escaneando rutas multimedia para K={k}...")
        mm_paths_tuples = list(self._make_mm_iterator(t, column_name))
        mm_paths_only = [path for _, path in mm_paths_tuples]

        if not mm_paths_only:
            print(f"Advertencia: No se encontraron rutas multimedia válidas para K={k}. El índice estará vacío.")
            return 0

        modality = _detect_modality_from_paths(mm_paths_only)

        if modality == "audio":
            print(f"Paso 2/3 (Rebuild AUDIO): Construyendo codebook de audio (K={k})...")
            cb = AudioCodebookBuilder(k, self.data_dir)
            cb.build_from_paths(mm_paths_only)

            print(f"Paso 3/3 (Rebuild AUDIO): Construyendo índice invertido MM audio (K={k})...")
            audio_hist_builder = AudioBoVWHistogramBuilder(k, self.data_dir)
            mm_idx_builder = AudioMMInvertedIndexBuilder(self.data_dir, k_clusters=k)

            def hist_generator_factory() -> Iterator[Tuple[int, np.ndarray]]:
                for rid, path in mm_paths_tuples:
                    hist = audio_hist_builder.create_histogram_from_path(path)
                    if hist is not None:
                        yield (rid, hist)

            mm_idx_builder.build(hist_generator_factory)

            print("Generando base de datos para KNN Secuencial de audio...")
            try:
                seq_builder = AudioKNNSearch(k, self.data_dir)
                seq_builder.build_database(mm_paths_tuples)
            except Exception as e:
                print(f"Advertencia: No se pudo construir la BD Secuencial de audio: {e}")

            return mm_idx_builder.total_docs

        print(f"Paso 2/3 (Rebuild IMAGEN): Construyendo Codebook (K={k})...")
        cb = CodebookBuilder(k, self.data_dir)
        cb.build_from_paths(mm_paths_only)

        print(f"Paso 3/3 (Rebuild IMAGEN): Construyendo Índice Invertido MM (K={k})...")

        if cb.kmeans is None:
            cb.kmeans = CodebookBuilder.load_codebook(k, self.data_dir)

        centers = cb.kmeans.cluster_centers_

        mm_idx_builder = MMInvertedIndexBuilder(self.data_dir, k_clusters=k)
        rids = [rid for rid, _ in mm_paths_tuples]

        def hist_generator_factory() -> Iterator[Tuple[int, np.ndarray]]:
            cpu_count = os.cpu_count() or 4
            print(f"  -> Iniciando generador paralelo con {cpu_count} núcleos...")

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cpu_count,
                initializer=_init_worker,
                initargs=(centers, k)
            ) as executor:
                results = executor.map(_process_image_task, mm_paths_only, chunksize=20)
                for rid, hist in zip(rids, results):
                    if hist is not None:
                        yield (rid, hist)

        mm_idx_builder.build(hist_generator_factory)

        print(f"Generando base de datos para KNN Secuencial de imágenes...")
        try:
            seq_builder = KNNSearch(k, self.data_dir)
            seq_builder.build_database(mm_paths_tuples)
        except Exception as e:
            print(f"Advertencia: No se pudo construir la BD Secuencial de imágenes: {e}")

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
        if not hasattr(t, "index_specs"):
            t.index_specs = []
        if (spec_key, spec_type) not in t.index_specs:
            t.index_specs.append((spec_key, spec_type))
            t._save_metadata()

        print("Paso 4/4: Recargando módulos de consulta.")
        self._load_query_modules()

        end_time = time.time()
        sample_path = None
        name_pos = _name_to_pos(t.schema)
        for rid, row in t.scan() or []:
            path = str(row[name_pos[column_name]])
            if path.strip():
                sample_path = path
                break

        mode = "BoAW" if (sample_path and self._is_audio_path(sample_path)) else "BoVW"

        return {
            "ok": True,
            "message": f"Índice Multimedia ({mode} K={k}) construido exitosamente.",
            "total_items_indexed": total_docs,
            "codebook_size": k,
            "time_taken_sec": (end_time - start_time)
        }

    def _select(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        t = self._get_table(stmt["table"])
        cols = stmt.get("columns", ["*"])
        cond = stmt.get("condition")

        limit_k = stmt.get("limit") or 100

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
                query_path = cond["query_path"]
                limit_k = cond.get("k") or stmt.get("limit") or 10

                if not os.path.exists(query_path):
                    return {"ok": False, "error": f"Archivo de consulta no encontrado: {query_path}"}

                modality = _detect_modality_from_query_path(query_path)

                # Normalizamos modo a MAYÚSCULAS por si acaso
                mode = (cond.get("mode") or "INDEX").upper()

                k_clusters_needed = cond.get("k")
                if not k_clusters_needed:
                    if modality == "audio" and self.mm_audio_query_modules:
                        k_clusters_needed = list(self.mm_audio_query_modules.keys())[0]
                    elif modality == "image" and self.mm_query_modules:
                        k_clusters_needed = list(self.mm_query_modules.keys())[0]
                if not k_clusters_needed:
                    k_clusters_needed = 20

                results = []
                used_method = ""

                if mode == 'SEQ':
                    # -------- KNN SECUENCIAL --------
                    if modality == "audio":
                        if k_clusters_needed not in self.audio_knn_seq_search:
                            try:
                                print(f"Cargando motor KNN Secuencial AUDIO para K={k_clusters_needed}...")
                                self.audio_knn_seq_search[k_clusters_needed] = AudioKNNSearch(
                                    k_clusters=k_clusters_needed,
                                    data_dir=self.data_dir
                                )
                                self.audio_knn_seq_search[k_clusters_needed].load_database()
                            except Exception as e:
                                return {
                                    "ok": False,
                                    "error": (
                                        f"No se pudo cargar BD Secuencial de audio (K={k_clusters_needed}). "
                                        f"¿Ejecutaste el benchmark primero? Error: {e}"
                                    )
                                }
                        used_method = f"SEQUENTIAL_SCAN_AUDIO (TF-IDF Matrix K={k_clusters_needed})"
                        results = self.audio_knn_seq_search[k_clusters_needed].search_by_path(
                            query_path, top_k=limit_k
                        )
                    else:
                        if k_clusters_needed not in self.knn_seq_search:
                            try:
                                print(f"Cargando motor KNN Secuencial IMAGEN para K={k_clusters_needed}...")
                                self.knn_seq_search[k_clusters_needed] = KNNSearch(
                                    k_clusters=k_clusters_needed,
                                    data_dir=self.data_dir
                                )
                                self.knn_seq_search[k_clusters_needed].load_database()
                            except Exception as e:
                                return {
                                    "ok": False,
                                    "error": (
                                        f"No se pudo cargar BD Secuencial (K={k_clusters_needed}). "
                                        f"¿Ejecutaste el benchmark primero? Error: {e}"
                                    )
                                }
                        used_method = f"SEQUENTIAL_SCAN (TF-IDF Matrix K={k_clusters_needed})"
                        results = self.knn_seq_search[k_clusters_needed].search_by_path(
                            query_path, top_k=limit_k
                        )
                else:
                    # -------- ÍNDICE INVERTIDO MM --------
                    if modality == "audio":
                        query_module = self.mm_audio_query_modules.get(k_clusters_needed)
                        if not query_module:
                            return {
                                "ok": False,
                                "error": (
                                    f"Índice MM Audio no encontrado para K={k_clusters_needed}. "
                                    f"Use CREATE MM INDEX..."
                                )
                            }
                        used_method = f"MM_AUDIO_INVERTED_INDEX (BoAW K={query_module.k})"
                        results = query_module.query_by_path(query_path, top_k=limit_k)
                    else:
                        query_module = self.mm_query_modules.get(k_clusters_needed)
                        if not query_module:
                            return {
                                "ok": False,
                                "error": (
                                    f"Índice MM Invertido no encontrado para K={k_clusters_needed}. "
                                    f"Use CREATE MM INDEX..."
                                )
                            }
                        used_method = f"MM_INVERTED_INDEX (BoVW K={query_module.k})"
                        results = query_module.query_by_path(query_path, top_k=limit_k)

                # Construimos filas resultado
                rows = []
                final_columns = ["score"] + proj_names
                for score, rid in results:
                    try:
                        record_tuple = t.get_record(rid)
                        projected_row = [record_tuple[p] for p in proj_pos]
                        rows.append(["{:.6f}".format(score)] + projected_row)
                    except (IOError, IndexError):
                        continue

                # Solo marcamos used_index si de verdad se usó índice invertido
                used_index_info = None
                if mode != "SEQ":
                    used_index_info = {
                        "type": used_method,
                        "column": cond.get("column", cond.get("field", "?"))
                    }

                return {
                    "ok": True,
                    "rows": rows,
                    "columns": final_columns,
                    "used_index": used_index_info,
                    "used_method": used_method  # para que el front pueda mostrar el método aunque no sea índice
                }

            if op == "FTS":
                target_col = cond["field"]
                fts_module = None
                found_key = None

                for key, module in self.fts_query.items():
                    if key.startswith(f"{t.name}."):
                        cols_in_index = key.split(".")[1].split(",")
                        if target_col in cols_in_index:
                            fts_module = module
                            found_key = key
                            break

                if not fts_module:
                    return {"ok": False, "rows": [], "columns": [],
                            "error": f"No se encontró un índice FTS que cubra la columna '{target_col}' "
                                     f"en la tabla '{t.name}'."}

                query_text = cond["query_text"]
                results = fts_module.query(query_text, k=limit_k)

                rows = []
                final_columns = ["score"] + proj_names
                for score, rid in results:
                    try:
                        record_tuple = t.get_record(rid)
                        projected_row = [record_tuple[p] for p in proj_pos]
                        rows.append(["{:.6f}".format(score)] + projected_row)
                    except (IOError, IndexError):
                        continue
                return {"ok": True, "rows": rows, "columns": final_columns,
                        "used_index": {"type": "FTS", "column": found_key}}

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
                        return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used}
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
                        return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used}
                elif op == "IN":
                    kind, payload = self.idx.probe_rtree_radius(t.name, field, cond["coords"], cond["radius"])
                    if kind == 'rids':
                        rows = [[list(t.get_record(rid))[p] for p in proj_pos] for rid in payload]
                        idx_inst = self.idx.list_for_table(t.name).get(field, None)
                        used = {"column": field, "type": idx_inst.__class__.__name__} if idx_inst else None
                        return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used}

            if op == "IN2":
                fields = cond["fields"]
                synthetic = ",".join(fields)
                kind, payload = self.idx.probe_rtree_radius(t.name, synthetic, cond["coords"], cond["radius"])
                if kind == 'rids':
                    rows = [[list(t.get_record(rid))[p] for p in proj_pos] for rid in payload]
                    idx_inst = self.idx.list_for_table(t.name).get(synthetic, None)
                    used = {"column": synthetic, "type": idx_inst.__class__.__name__} if idx_inst else None
                    return {"ok": True, "rows": rows[:limit_k], "columns": proj_names, "used_index": used}

        pred = self._make_predicate(t.schema, cond)
        rows: List[List[Any]] = []
        count = 0
        for _rid, row_tuple in t.scan() or []:
            if count >= limit_k:
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
        if not cols:
            raise ValueError("CREATE INDEX: faltan columnas")
        typ = stmt["index_type"]
        self.idx.create_index(t, cols, typ)
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
        t.index_specs = [(c, ty) for (c, ty) in getattr(t, "index_specs", []) if c != col]
        t._save_metadata()
        return {"ok": True}

    def _get_table(self, name: str) -> Table:
        if name not in self.catalog:
            meta_path = os.path.join(self.data_dir, f"{name}.meta")
            if os.path.exists(meta_path):
                t = Table(name, data_dir=self.data_dir)
                self.catalog[name] = t
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

    def _make_predicate(self, schema: List[Tuple[str, str, int]],
                        cond: Optional[Dict[str, Any]]) -> Callable[[List[Any]], bool]:
        if cond is None:
            return lambda row: True

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
