# src/core/record.py

import struct
from typing import List, Any, Tuple

class RecordManager:
    def __init__(self, schema: List[Tuple[str, str, int]]):
        self.schema = schema
        self.format_string = self._build_format_string()
        self.record_size = struct.calcsize(self.format_string)

    def _build_format_string(self) -> str:
        format_parts = []
        for _, col_type, length in self.schema:
            if col_type.upper() == 'INT':
                format_parts.append('i')
            elif col_type.upper() == 'FLOAT':
                format_parts.append('d')
            elif col_type.upper() == 'VARCHAR':
                format_parts.append(f'{length}s')
        
        return '<' + ''.join(format_parts)

    def pack(self, values: List[Any]) -> bytes:
        packed_values = []
        for i, value in enumerate(values):
            col_type = self.schema[i][1].upper()
            
            if col_type == 'VARCHAR':
                length = self.schema[i][2]
                encoded_value = str(value).encode('utf-8')
                packed_values.append(encoded_value.ljust(length, b'\0'))
            else:
                packed_values.append(value)
        
        try:
            return struct.pack(self.format_string, *packed_values)
        except struct.error as e:
            print(f"Error al empaquetar: {e}")
            print(f"  - Formato: {self.format_string}")
            print(f"  - Valores: {packed_values}")
            raise

    def unpack(self, data: bytes) -> Tuple[Any, ...]:
        unpacked_values = list(struct.unpack(self.format_string, data))

        for i, value in enumerate(unpacked_values):
            col_type = self.schema[i][1].upper()
            
            if col_type == 'VARCHAR':
                unpacked_values[i] = value.strip(b'\0').decode('utf-8')
                
        return tuple(unpacked_values)