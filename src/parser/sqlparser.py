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
        
        # --- MODIFICADO (P2): Añadido LIMIT opcional ---
        self.re_select = re.compile(
            r"SELECT\s+(.*?)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.*?))?(?:\s+LIMIT\s+(\d+))?",
            re.IGNORECASE | re.DOTALL
        )
        
        self.re_delete = re.compile(
            r"DELETE FROM (\w+) WHERE (.*)",
            re.IGNORECASE
        )
        self.re_create_index = re.compile(
            r"CREATE INDEX (\w+) ON (\w+)\((\w+)\) TYPE (\w+)",
            re.IGNORECASE
        )
        self.re_create_index_rtree = re.compile(
            r"""CREATE\s+INDEX\s+(\w+)\s+ON\s+(\w+)\s*
                \(\s*(\w+)\s*,\s*(\w+)\s*\)\s+
                TYPE\s+RTREE\s*""",
            re.IGNORECASE | re.VERBOSE
        )
        
        # --- NUEVO (P2): Regex para FTS y MM ---
        self.re_create_fts_index = re.compile(
            r"CREATE\s+FTS\s+INDEX\s+ON\s+(\w+)\s*\((.*?)\)",
            re.IGNORECASE | re.DOTALL
        )

        # MODIFICADO: ahora acepta TYPE BOVW o TYPE BOAW para imagen/audio
        self.re_create_mm_index = re.compile(
            r"CREATE\s+MM\s+INDEX\s+ON\s+(\w+)\s*\((\w+)\)\s+TYPE\s+(BOVW|BOAW)\s+K=(\d+)",
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
        
        match = self.re_create_index_rtree.fullmatch(sql)
        if match:
            return {
                'command': 'CREATE_INDEX',
                'index_name': match.group(1),
                'table_name': match.group(2),
                'column_name': [match.group(3), match.group(4)],
                'index_type': 'RTREE'
            }
        
        match = self.re_create_index.fullmatch(sql)
        if match:
            return self._parse_create_index(match.group(1), match.group(2), match.group(3), match.group(4))

        # --- NUEVO (P2): Parseo de FTS y MM ---
        match = self.re_create_fts_index.fullmatch(sql)
        if match:
            return self._parse_create_fts_index(match.group(1), match.group(2))

        match = self.re_create_mm_index.fullmatch(sql)
        if match:
            # grupos:
            # 1: tabla
            # 2: columna
            # 3: tipo índice multimedia (BOVW | BOAW)
            # 4: K
            return self._parse_create_mm_index(
                match.group(1),
                match.group(2),
                match.group(3),
                match.group(4),
            )

        match = self.re_insert.fullmatch(sql)
        if match:
            return self._parse_insert(match.group(1), match.group(2))

        match = self.re_select.fullmatch(sql)
        if match:
            return self._parse_select(match.group(1), match.group(2), match.group(3), match.group(4))

        match = self.re_delete.fullmatch(sql)
        if match:
            return self._parse_delete(match.group(1), match.group(2))

        raise ValueError(f"Consulta SQL no válida o no soportada: {sql}")

    # ----------------- helpers internos -----------------

    def _split_args(self, s: str) -> List[str]:
        out, buf = [], []
        q = None  
        i, n = 0, len(s)
        while i < n:
            ch = s[i]
            if q is not None:
                if ch == q:
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
        plan = {
            'command': 'INSERT',
            'table_name': table_name,
            'values': []
        }

        parts = self._split_args(values_str.strip())
        parts = [p[:-1] if p.endswith(",") else p for p in parts]
        plan['values'] = [self._cast_value(p) for p in parts]
        return plan

    # (P2): Aceptar limit_str y parsear FTS/MM 
    def _parse_select(self, cols_str: str, table_name: str, where_str: str, limit_str: str) -> Dict[str, Any]:
        columns = [c.strip() for c in cols_str.split(',')]
        if "*" in columns: 
            columns = ["*"]
        plan = {
            'command': 'SELECT',
            'table_name': table_name,
            'columns': columns,
            'where': None,
            'limit': int(limit_str) if limit_str else None 
        }
        if not where_str:
            return plan

        # --- NUEVO: Soporte para MODE='SEQ' ---
        # Captura: ... <-> 'path' USING K=6 MODE='SEQ'
        match_mm = re.match(
            r"(\w+)\s*<->\s*'(.*?)'(?:\s+USING\s+K=(\d+))?(?:\s+MODE='(\w+)')?",
            where_str,
            re.IGNORECASE
        )
        if match_mm:
            k_value = match_mm.group(3)
            mode_value = match_mm.group(4)  # 'SEQ' o 'INDEX'
            plan['where'] = {
                'column': match_mm.group(1),
                'op': 'MM_SIM',
                'query_path': match_mm.group(2),
                'k': int(k_value) if k_value else None,
                'mode': mode_value.upper() if mode_value else 'INDEX'
            }
            return plan

        match_fts = re.match(r"(\w+)\s*@@\s*'(.*?)'", where_str, re.IGNORECASE | re.DOTALL)
        if match_fts:
            plan['where'] = {
                'column': match_fts.group(1),
                'op': 'FTS',
                'query_text': match_fts.group(2)
            }
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
        
        m = re.match(r"(\w+)\s*,\s*(\w+)\s+IN\s*\(\((.*?)\)\s*,\s*(.*?)\)", where_str, re.IGNORECASE)
        if m:
            point = tuple(float(p) for p in m.group(3).split(','))
            plan['where'] = {
                'op': 'IN2',
                'columns': [m.group(1), m.group(2)],
                'point': point,
                'radius': float(m.group(4))
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
            column = match.group(1)
            value = match.group(2).strip()
            if not (value.startswith("'") and value.endswith("'")):
                value = f"'{value}'"
            plan['where'] = {
                'column': column,
                'op': '=',
                'value': self._cast_value(value)
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

    def _parse_create_index(self, index_name: str, table_name: str, column_name: str, index_type: str) -> Dict[str, Any]:
        return {
            'command': 'CREATE_INDEX',
            'index_name': index_name,
            'table_name': table_name,
            'column_name': column_name,
            'index_type': index_type
        }
    
    # --- NUEVO (P2): Parseo de FTS y MM ---
    def _parse_create_fts_index(self, table_name: str, cols_str: str) -> Dict[str, Any]:
        columns = [col.strip() for col in cols_str.split(',')]
        if not columns:
            raise ValueError("CREATE FTS INDEX debe especificar al menos una columna.")
        
        return {
            'command': 'CREATE_FTS_INDEX',
            'table_name': table_name,
            'columns': columns
        }

    def _parse_create_mm_index(self, table_name: str, col_name: str, mm_type: str, k_str: str) -> Dict[str, Any]:
        """
        Soporta:
          CREATE MM INDEX ON tabla(columna) TYPE BOVW K=32;  -- típico para imágenes
          CREATE MM INDEX ON tabla(columna) TYPE BOAW K=32;  -- típico para audio
        """
        return {
            'command': 'CREATE_MM_INDEX',
            'table_name': table_name,
            'column': col_name.strip(),
            'mm_type': mm_type.upper(),   # 'BOVW' o 'BOAW'
            'k': int(k_str)
        }
    # --- FIN NUEVO (P2) ---

    def _cast_value(self, value: str) -> Any:
        value = value.strip()

        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            return value[1:-1]

        if value.startswith('(') and value.endswith(')'):
            try:
                return tuple(float(p) for p in value.strip('()').split(','))
            except Exception:
                pass

        try:
            return float(value)
        except ValueError:
            pass

        try:
            return int(value)
        except ValueError:
            pass

        return value
