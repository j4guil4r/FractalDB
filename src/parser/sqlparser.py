# src/parser/sqlparser.py

import re
import shlex
from typing import Dict, Any

class SQLParser:
    def __init__(self):
        
        self.re_create = re.compile(
            r"CREATE TABLE (\w+)\s*\((.*?)\)", 
            re.IGNORECASE | re.DOTALL
        )
        self.re_create_from_file = re.compile(
            r"CREATE TABLE (\w+) FROM FILE \"(.*?)\"",
            re.IGNORECASE
        )
        
        
        self.re_insert = re.compile(
            r"INSERT INTO (\w+) VALUES\s*\((.*?)\)", 
            re.IGNORECASE | re.DOTALL
        )
        
        
        self.re_select = re.compile(
            r"SELECT \* FROM (\w+)(?:\s+WHERE\s+(.*))?", 
            re.IGNORECASE
        )
        
        
        self.re_delete = re.compile(
            r"DELETE FROM (\w+) WHERE (.*)", 
            re.IGNORECASE
        )

    def parse(self, sql: str) -> Dict[str, Any]:
        
        sql = sql.strip().rstrip(';')
        
        
        match = self.re_create.fullmatch(sql)
        if match:
            return self._parse_create(match.group(1), match.group(2))
            
        match = self.re_create_from_file.fullmatch(sql)
        if match:
            
            return {
                'command': 'CREATE_TABLE_FROM_FILE',
                'table_name': match.group(1),
                'from_file': match.group(2)
            }

        
        match = self.re_insert.fullmatch(sql)
        if match:
            return self._parse_insert(match.group(1), match.group(2))

        
        match = self.re_select.fullmatch(sql)
        if match:
            return self._parse_select(match.group(1), match.group(2))
            
        
        match = self.re_delete.fullmatch(sql)
        if match:
            return self._parse_delete(match.group(1), match.group(2))

        raise ValueError(f"Consulta SQL no válida o no soportada: {sql}")

    def _parse_create(self, table_name: str, schema_str: str) -> Dict[str, Any]:
        
        plan = {
            'command': 'CREATE_TABLE',
            'table_name': table_name,
            'schema': [],
            'index_definitions': {}
        }
        
        col_defs = [col.strip() for col in schema_str.split(',')]
        
        for col_def in col_defs:
            if not col_def: continue
            
            # Formato: nombre TIPO[longitud] [INDEX tipo_indice]
            parts = col_def.split()
            col_name = parts[0]
            col_type_full = parts[1]
            length = 0

            if '[' in col_type_full:
                col_type, length_str = col_type_full.replace(']', '').split('[')
                length = int(length_str) if col_type.upper() != 'FLOAT' else 8 # Asumir 8 para ARRAY[FLOAT]
            else:
                col_type = col_type_full
                if col_type.upper() == 'INT': length = 4
                elif col_type.upper() == 'FLOAT': length = 8
            
            plan['schema'].append((col_name, col_type.upper(), length))
            
            if 'INDEX' in col_def.upper():
                index_type = parts[-1]
                # Mapear nombres de SQL a nombres de clase
                type_map = {
                    'BTREE': 'BPlusTreeIndex',
                    'HASH': 'HashIndex',
                    'ISAM': 'ISAMIndex',
                    'SEQ': 'SequentialFileIndex',
                    'RTREE': 'RTreeIndex'
                }
                plan['index_definitions'][col_name] = type_map.get(index_type.upper(), index_type)

        return plan

    def _parse_insert(self, table_name: str, values_str: str) -> Dict[str, Any]:
        """Parsea los valores en INSERT."""
        plan = {
            'command': 'INSERT',
            'table_name': table_name,
            'values': []
        }
        
        # shlex.split maneja comillas correctamente
        try:
            values = shlex.split(values_str, posix=False)
        except ValueError:
             # Caso simple si shlex falla (ej. tuplas)
             values = [v.strip() for v in values_str.split(',')]

        plan['values'] = [self._cast_value(v) for v in values]
        return plan

    def _parse_select(self, table_name: str, where_str: str) -> Dict[str, Any]:
        """Parsea la cláusula WHERE."""
        plan = {
            'command': 'SELECT',
            'table_name': table_name,
            'where': None
        }
        if not where_str:
            return plan # SELECT * FROM table (sin where)

        # 1. BETWEEN 
        match = re.match(r"(\w+) BETWEEN (.*) AND (.*)", where_str, re.IGNORECASE)
        if match:
            plan['where'] = {
                'column': match.group(1),
                'op': 'BETWEEN',
                'value1': self._cast_value(match.group(2)),
                'value2': self._cast_value(match.group(3))
            }
            return plan
            
        # 2. R-Tree IN (point, radius) 
        match = re.match(r"(\w+) IN \(\((.*?)\),\s*(.*?)\)", where_str, re.IGNORECASE)
        if match:
            point = tuple(float(p) for p in match.group(2).split(','))
            plan['where'] = {
                'column': match.group(1),
                'op': 'IN',
                'point': point,
                'radius': float(match.group(3))
            }
            return plan

        # 3. Igualdad (=) 
        match = re.match(r"(\w+)\s*=\s*(.*)", where_str, re.IGNORECASE)
        if match:
            plan['where'] = {
                'column': match.group(1),
                'op': '=',
                'value': self._cast_value(match.group(2))
            }
            return plan

        raise ValueError(f"Cláusula WHERE no soportada: {where_str}")

    def _parse_delete(self, table_name: str, where_str: str) -> Dict[str, Any]:
        """Parsea el DELETE (solo soporta igualdad)[cite: 45]."""
        match = re.match(r"(\w+)\s*=\s*(.*)", where_str, re.IGNORECASE)
        if not match:
             raise ValueError(f"Cláusula WHERE para DELETE no soportada: {where_str}")
        
        return {
            'command': 'DELETE',
            'table_name': table_name,
            'where': {
                'column': match.group(1),
                'op': '=',
                'value': self._cast_value(match.group(2))
            }
        }
        
    def _cast_value(self, value: str) -> Any:
        """Intenta convertir un valor de SQL a un tipo de Python."""
        value = value.strip()
        # 1. String (entre comillas)
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            return value[1:-1]
        
        # 2. Tupla (para RTree)
        if value.startswith('(') and value.endswith(')'):
            try:
                return tuple(float(p) for p in value.strip('()').split(','))
            except:
                pass # Devolver como string si falla

        # 3. Float
        try:
            return float(value)
        except ValueError:
            pass
            
        # 4. Int
        try:
            return int(value)
        except ValueError:
            pass
            
        # 5. Devolver como string si todo falla
        return value