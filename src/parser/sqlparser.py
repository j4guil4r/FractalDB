# src/parser/sqlparser.py

import re
from typing import Dict, Any, List

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

    # ----------------- helpers internos -----------------

    def _split_args(self, s: str) -> List[str]:
        """
        Divide 'a, "b, c", 'd', 12' por comas SOLO cuando están fuera de comillas.
        Soporta comillas simples y dobles; preserva el contenido interno tal cual.
        """
        out, buf = [], []
        q = None  # comilla abierta: "'" o '"'
        i, n = 0, len(s)
        while i < n:
            ch = s[i]
            if q is not None:
                # estamos dentro de comillas
                if ch == q:
                    # soporta comillas duplicadas '' o ""
                    if i + 1 < n and s[i + 1] == q:
                        buf.append(q)
                        i += 2
                        continue
                    q = None
                    i += 1
                    continue
                buf.append(ch)
                i += 1
                continue

            # fuera de comillas
            if ch in ("'", '"'):
                q = ch
                i += 1
                continue
            if ch == ",":
                token = "".join(buf).strip()
                if token:
                    out.append(token)
                buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1

        token = "".join(buf).strip()
        if token:
            out.append(token)
        return out

    # ----------------- parseos por comando -----------------

    def _parse_create(self, table_name: str, schema_str: str) -> Dict[str, Any]:
        plan = {
            'command': 'CREATE_TABLE',
            'table_name': table_name,
            'schema': [],
            'index_definitions': {}
        }

        col_defs = [col.strip() for col in schema_str.split(',')]
        for col_def in col_defs:
            if not col_def:
                continue

            parts = col_def.split()
            if len(parts) < 2:
                raise ValueError(f"Definición de columna inválida: {col_def}")

            col_name = parts[0]
            col_type_full = parts[1]
            length = 0

            if '[' in col_type_full:
                col_type, length_str = col_type_full.replace(']', '').split('[')
                length = int(length_str) if col_type.upper() != 'FLOAT' else 8
            else:
                col_type = col_type_full
                if col_type.upper() == 'INT':
                    length = 4
                elif col_type.upper() == 'FLOAT':
                    length = 8

            plan['schema'].append((col_name, col_type.upper(), length))

            if 'INDEX' in col_def.upper():
                index_type = parts[-1]
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
        """
        Parsea INSERT INTO t VALUES (...), respetando comillas y comas.
        """
        plan = {
            'command': 'INSERT',
            'table_name': table_name,
            'values': []
        }

        parts = self._split_args(values_str.strip())
        # Limpieza por si llega "12," o " 'foo', "
        parts = [p[:-1] if p.endswith(",") else p for p in parts]

        plan['values'] = [self._cast_value(p) for p in parts]
        return plan

    def _parse_select(self, table_name: str, where_str: str) -> Dict[str, Any]:
        plan = {
            'command': 'SELECT',
            'table_name': table_name,
            'where': None
        }
        if not where_str:
            return plan

        match = re.match(r"(\w+) BETWEEN (.*) AND (.*)", where_str, re.IGNORECASE)
        if match:
            plan['where'] = {
                'column': match.group(1),
                'op': 'BETWEEN',
                'value1': self._cast_value(match.group(2)),
                'value2': self._cast_value(match.group(3))
            }
            return plan

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
        value = value.strip()

        # String entre comillas
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            return value[1:-1]

        # Tupla para RTree
        if value.startswith('(') and value.endswith(')'):
            try:
                return tuple(float(p) for p in value.strip('()').split(','))
            except Exception:
                pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # Int
        try:
            return int(value)
        except ValueError:
            pass

        # Fallback string crudo
        return value
