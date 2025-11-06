# src/core/table.py
import os
import json
from typing import List, Any, Dict, Tuple, Generator
from .record import RecordManager

class Table:
    def __init__(self, table_name: str, schema: List[Tuple[str, str, int]] = None, data_dir: str = 'data'):
        self.name = table_name
        self.data_dir = data_dir
        self.dat_path = os.path.join(data_dir, f"{table_name}.dat")
        self.meta_path = os.path.join(data_dir, f"{table_name}.meta")

        os.makedirs(self.data_dir, exist_ok=True)

        if os.path.exists(self.meta_path):
            self._load_metadata()
        elif schema:
            self.schema = schema
            self.index_definitions: Dict[str, str] = {}
            self.index_specs: List[Tuple[str, str]] = []
            self._save_metadata()
        else:
            raise ValueError("Se debe proporcionar un esquema para una tabla nueva.")

        self.record_manager = RecordManager(self.schema)
        self.indexes: Dict[str, Any] = {}

    def _save_metadata(self):
        metadata = {
            'schema': self.schema,
            'index_definitions': self.index_definitions,
            'index_specs': self.index_specs
        }
        with open(self.meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def _load_metadata(self):
        with open(self.meta_path, 'r') as f:
            metadata = json.load(f)
        self.schema = metadata['schema']
        self.index_definitions = metadata.get('index_definitions', {})
        self.index_specs = metadata.get('index_specs', [])
    
    def insert_record(self, values: List[Any]) -> int:
        packed_data = self.record_manager.pack(values)
        with open(self.dat_path, 'ab') as f:
            rid = f.tell()
            f.write(packed_data)
        return rid

    def get_record(self, rid: int) -> Tuple[Any, ...]:
        with open(self.dat_path, 'rb') as f:
            f.seek(rid)
            packed_data = f.read(self.record_manager.record_size)
            if len(packed_data) != self.record_manager.record_size:
                raise IndexError(f"No se pudo leer un registro completo en el RID {rid}.")
        return self.record_manager.unpack(packed_data)

    def scan(self) -> Generator[Tuple[int, Tuple], None, None]:
        if not os.path.exists(self.dat_path):
            return
        with open(self.dat_path, 'rb') as f:
            rid = 0
            while True:
                packed_data = f.read(self.record_manager.record_size)
                if not packed_data:
                    break
                yield rid, self.record_manager.unpack(packed_data)
                rid += self.record_manager.record_size