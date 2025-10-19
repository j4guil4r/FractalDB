# src/indices/isam/isam.py

import pickle
import os
from typing import List, Any, Tuple, Optional

class IndexNode:
    def __init__(self, capacity: int):
        self.keys: List[Any] = []
        self.pointers: List[int] = []
        self.capacity = capacity

    def is_full(self) -> bool:
        return len(self.keys) >= self.capacity


class DataPage:
    def __init__(self, capacity: int):
        # (key, rid, is_active)
        self.records: List[Tuple[Any, int, bool]] = []
        self.capacity = capacity
        self.next_overflow_offset = -1

    def is_full(self) -> bool:
        return len(self.records) >= self.capacity


class ISAM:
    BLOCK_SIZE = 4096

    def __init__(self, file_path_prefix: str, index_capacity=4, data_capacity=4):
        self.prefix = file_path_prefix
        self.paths = {
            'meta': f"{self.prefix}.meta",
            'idx2': f"{self.prefix}.idx2",
            'idx1': f"{self.prefix}.idx1",
            'dat': f"{self.prefix}.dat",
            'ovf': f"{self.prefix}.ovf"
        }
        self.index_capacity = index_capacity
        self.data_capacity = data_capacity
        self.root_offset_l2 = 0
        self.data_page_offsets: List[int] = []

    # ---------- IO helpers ----------

    def _ensure_dir(self):
        d = os.path.dirname(self.prefix)
        if d:
            os.makedirs(d, exist_ok=True)

    def _write_block(self, file_key: str, block: Any, offset: Optional[int] = None) -> int:
        """Escribe un bloque pickled de tamaño fijo BLOCK_SIZE; si offset=None, append."""
        self._ensure_dir()
        path = self.paths[file_key]
        data = pickle.dumps(block)
        if len(data) > self.BLOCK_SIZE:
            raise ValueError(
                f"Bloque {type(block)} demasiado grande ({len(data)} bytes) para BLOCK_SIZE ({self.BLOCK_SIZE})."
            )
        padded = data.ljust(self.BLOCK_SIZE, b'\0')

        mode = 'r+b' if os.path.exists(path) else 'w+b'
        with open(path, mode) as f:
            if offset is not None:
                f.seek(offset)
            else:
                f.seek(0, 2)
            pos = f.tell()
            f.write(padded)
            return pos

    def _read_block(self, file_key: str, offset: int) -> Any:
        """Lee un bloque fijo desde offset; errores claros si no existe."""
        path = self.paths[file_key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Falta el archivo de índice requerido: {path}")
        with open(path, 'rb') as f:
            f.seek(offset)
            padded = f.read(self.BLOCK_SIZE)
            if not padded:
                raise EOFError(f"Error al leer bloque en offset {offset} de {path}")
            data = padded.rstrip(b'\0')
            return pickle.loads(data)

    # ---------- Build / persist ----------

    def build(self, sorted_records: List[Tuple[Any, int]]):
        """
        Construye todos los niveles (L2, L1) y páginas de datos.
        AUN CON TABLA VACÍA crea idx2/idx1/dat/ovf y escribe raíz vacía.
        """
        self._ensure_dir()
        for key in ('idx2', 'idx1', 'dat', 'ovf'):
            with open(self.paths[key], 'wb') as f:
                f.truncate(0)

        self.data_page_offsets = []

        # Caso sin registros: raíz L2 vacía en offset 0
        if not sorted_records:
            root = IndexNode(self.index_capacity)
            self.root_offset_l2 = self._write_block('idx2', root, 0)
            self.save_metadata()
            return

        # 1) Empaquetar registros en páginas de datos
        data_pages: List[DataPage] = []
        current = DataPage(self.data_capacity)
        for key, rid in sorted_records:
            if current.is_full():
                data_pages.append(current)
                current = DataPage(self.data_capacity)
            current.records.append((key, rid, True))
        if current.records:
            data_pages.append(current)

        # 2) Escribir páginas y formar entradas L1
        index1_entries: List[Tuple[Any, int]] = []
        for page in data_pages:
            off = self._write_block('dat', page)
            self.data_page_offsets.append(off)
            highest = page.records[-1][0]
            index1_entries.append((highest, off))

        # 3) Nodos L1
        index1_nodes: List[IndexNode] = []
        node = IndexNode(self.index_capacity)
        for key, ptr in index1_entries:
            if node.is_full():
                index1_nodes.append(node)
                node = IndexNode(self.index_capacity)
            node.keys.append(key)
            node.pointers.append(ptr)
        if node.keys:
            index1_nodes.append(node)

        # 4) Escribir L1 y formar entradas L2
        index2_entries: List[Tuple[Any, int]] = []
        for n in index1_nodes:
            off = self._write_block('idx1', n)
            highest = n.keys[-1]
            index2_entries.append((highest, off))

        # 5) Raíz L2
        root = IndexNode(self.index_capacity)
        if len(index2_entries) <= self.index_capacity:
            for key, ptr in index2_entries:
                root.keys.append(key)
                root.pointers.append(ptr)
            self.root_offset_l2 = self._write_block('idx2', root, 0)
        else:
            for key, ptr in index2_entries[: self.index_capacity]:
                root.keys.append(key)
                root.pointers.append(ptr)
            self.root_offset_l2 = self._write_block('idx2', root, 0)

        self.save_metadata()

    def save_metadata(self):
        meta = {
            'root_offset_l2': self.root_offset_l2,
            'index_capacity': self.index_capacity,
            'data_capacity': self.data_capacity,
            'data_page_offsets': self.data_page_offsets,
        }
        self._ensure_dir()
        with open(self.paths['meta'], 'wb') as f:
            pickle.dump(meta, f)

    def load_metadata(self):
        with open(self.paths['meta'], 'rb') as f:
            meta = pickle.load(f)
        self.root_offset_l2 = meta['root_offset_l2']
        self.index_capacity = meta['index_capacity']
        self.data_capacity = meta['data_capacity']
        self.data_page_offsets = meta.get('data_page_offsets', [])

    # ---------- Búsqueda / modificación ----------

    def _find_pointer(self, node: IndexNode, key: Any) -> Optional[int]:
        # primera clave >= key
        for i, k in enumerate(node.keys):
            if key <= k:
                return node.pointers[i]
        if node.pointers:
            return node.pointers[-1]
        return None

    def search(self, key: Any) -> List[int]:
        root = self._read_block('idx2', self.root_offset_l2)
        l1_offset = self._find_pointer(root, key)
        if l1_offset is None:
            return []

        l1_node = self._read_block('idx1', l1_offset)
        data_page_offset = self._find_pointer(l1_node, key)
        if data_page_offset is None:
            return []

        results: List[int] = []
        current_offset = data_page_offset
        current_file = 'dat'

        while current_offset != -1:
            page: DataPage = self._read_block(current_file, current_offset)
            for k, rid, active in page.records:
                if active and k == key:
                    results.append(rid)
            current_offset = page.next_overflow_offset
            current_file = 'ovf'
        return results

    def add(self, key: Any, rid: int):
        # leer raíz L2
        root: IndexNode = self._read_block('idx2', self.root_offset_l2)

        # Bootstrap: si L2 no tiene punteros, crea L1 y la primera página de datos
        if not root.pointers:
            # 1) crear página de datos con el primer registro
            first_page = DataPage(self.data_capacity)
            first_page.records.append((key, rid, True))
            dp_off = self._write_block('dat', first_page, None)

            # guarda offsets de páginas de datos para rangeSearch
            if dp_off not in self.data_page_offsets:
                self.data_page_offsets.append(dp_off)
                self.save_metadata()

            # 2) crear nodo L1 apuntando a esa página
            l1 = IndexNode(self.index_capacity)
            l1.keys.append(key)      # tope de la página
            l1.pointers.append(dp_off)
            l1_off = self._write_block('idx1', l1, None)

            # 3) actualizar raíz L2
            root.keys = [key]
            root.pointers = [l1_off]
            self._write_block('idx2', root, self.root_offset_l2)
            self.save_metadata()
            return

        # Camino normal: L2 → L1 → página de datos
        l1_offset = self._find_pointer(root, key)
        if l1_offset is None:
            raise IndexError("ISAM.add: no hay puntero L1 para la clave.")

        l1_node: IndexNode = self._read_block('idx1', l1_offset)

        # Defensa: si el L1 está vacío (inconsistencia), bootstrap local
        if not l1_node.pointers:
            first_page = DataPage(self.data_capacity)
            first_page.records.append((key, rid, True))
            dp_off = self._write_block('dat', first_page, None)
            if dp_off not in self.data_page_offsets:
                self.data_page_offsets.append(dp_off)
                self.save_metadata()

            l1_node.keys = [key]
            l1_node.pointers = [dp_off]
            self._write_block('idx1', l1_node, l1_offset)

            # actualiza tope en raíz
            root_ptr_idx = root.pointers.index(l1_offset)
            if key > (root.keys[root_ptr_idx] if root.keys else key):
                if len(root.keys) <= root_ptr_idx:
                    root.keys += [key]
                else:
                    root.keys[root_ptr_idx] = key
            self._write_block('idx2', root, self.root_offset_l2)
            self.save_metadata()
            return

        data_page_offset = self._find_pointer(l1_node, key)
        if data_page_offset is None:
            data_page_offset = l1_node.pointers[0]

        # Ir al final de la cadena de overflow
        last_file = 'dat'
        last_off = data_page_offset
        page: DataPage = self._read_block(last_file, last_off)
        while page.next_overflow_offset != -1:
            last_file = 'ovf'
            last_off = page.next_overflow_offset
            page = self._read_block(last_file, last_off)

        # Insertar
        if not page.is_full():
            page.records.append((key, rid, True))
            page.records.sort(key=lambda x: x[0])
            self._write_block(last_file, page, last_off)
        else:
            new_ovf = DataPage(self.data_capacity)
            new_ovf.records.append((key, rid, True))
            new_off = self._write_block('ovf', new_ovf, None)
            page.next_overflow_offset = new_off
            self._write_block(last_file, page, last_off)

        # Actualizar tope en L1 y L2 si la clave nueva es mayor
        base_page: DataPage = self._read_block('dat', data_page_offset)
        new_max_key = base_page.records[-1][0] if base_page.records else key

        # actualizar L1
        def _find_ptr_idx(node: IndexNode, ptr: int) -> Optional[int]:
            for i, p in enumerate(node.pointers):
                if p == ptr:
                    return i
            return None

        idx_ptr = _find_ptr_idx(l1_node, data_page_offset)
        if idx_ptr is not None and new_max_key > l1_node.keys[idx_ptr]:
            l1_node.keys[idx_ptr] = new_max_key
            self._write_block('idx1', l1_node, l1_offset)

            # actualizar L2
            root_idx = root.pointers.index(l1_offset)
            if new_max_key > root.keys[root_idx]:
                root.keys[root_idx] = new_max_key
                self._write_block('idx2', root, self.root_offset_l2)

    def remove(self, key: Any):
        root = self._read_block('idx2', self.root_offset_l2)
        l1_offset = self._find_pointer(root, key)
        if l1_offset is None:
            return

        l1_node = self._read_block('idx1', l1_offset)
        data_page_offset = self._find_pointer(l1_node, key)
        if data_page_offset is None:
            return

        current_offset = data_page_offset
        current_file = 'dat'
        while current_offset != -1:
            page: DataPage = self._read_block(current_file, current_offset)
            modified = False
            for i, (k, rid, active) in enumerate(page.records):
                if active and k == key:
                    page.records[i] = (k, rid, False)
                    modified = True
            if modified:
                self._write_block(current_file, page, current_offset)

            if page.records and page.records[-1][0] < key:
                break

            current_offset = page.next_overflow_offset
            current_file = 'ovf'

    def rangeSearch(self, start_key: Any, end_key: Any) -> List[int]:
        results: List[int] = []
        if start_key > end_key:
            return results

        root = self._read_block('idx2', self.root_offset_l2)
        l1_offset = self._find_pointer(root, start_key)
        if l1_offset is None:
            return results

        l1_node = self._read_block('idx1', l1_offset)
        start_page_offset = self._find_pointer(l1_node, start_key)
        if start_page_offset is None:
            return results

        try:
            start_idx = self.data_page_offsets.index(start_page_offset)
        except ValueError:
            return results

        stop_main = False
        for i in range(start_idx, len(self.data_page_offsets)):
            if stop_main:
                break
            current_offset = self.data_page_offsets[i]
            current_file = 'dat'
            while current_offset != -1:
                page: DataPage = self._read_block(current_file, current_offset)
                for k, rid, active in page.records:
                    if not active:
                        continue
                    if start_key <= k <= end_key:
                        results.append(rid)
                    if k > end_key:
                        if current_file == 'dat':
                            stop_main = True
                            break
                        else:
                            current_offset = -1
                            break
                if current_offset != -1:
                    current_offset = page.next_overflow_offset
                    current_file = 'ovf'

        return results
