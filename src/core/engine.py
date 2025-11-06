# src/core/engine.py

import os
import json
import csv
from typing import List, Dict, Any, Generator, Tuple

from src.core.table import Table
from src.core.record import RecordManager

from src.indices.base_index import BaseIndex
from src.indices.bplustree.bplustreeindex import BPlusTreeIndex
from src.indices.hashing.hashingindex import HashIndex
from src.indices.isam.isamindex import ISAMIndex
from src.indices.rtree.rtreeindex import RTreeIndex
from src.indices.sequentialfile.sequentialfileindex import SequentialFileIndex

from src.parser.sqlparser import SQLParser

INDEX_CLASS_MAP = {
    "BPlusTreeIndex": BPlusTreeIndex,
    "HashIndex": HashIndex,
    "ISAMIndex": ISAMIndex,
    "RTreeIndex": RTreeIndex,
    "SequentialFileIndex": SequentialFileIndex,
}

class DatabaseEngine:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.tables: Dict[str, Table] = {}
        self.parser = SQLParser()
        os.makedirs(self.data_dir, exist_ok=True)
        self._load_all_tables()

    def _load_all_tables(self):
        print("Cargando tablas existentes...")
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".meta") and "_bpt" not in filename and \
               "_hash" not in filename and "_seq" not in filename and \
               "_rtree" not in filename:
                
                table_name = filename.replace(".meta", "")
                try:
                    table = Table(table_name, data_dir=self.data_dir)
                    self.tables[table_name] = table
                    print(f"  - Tabla '{table_name}' cargada.")
                    self._load_indexes_for_table(table)
                except Exception as e:
                    print(f"Error al cargar la tabla '{table_name}': {e}")
        print("Carga de tablas finalizada.")

    def _load_indexes_for_table(self, table: Table):
        for col_name, index_type in table.index_definitions.items():
            if col_name in table.indexes: continue
            
            try:
                print(f"    - Cargando índice {index_type} en '{table.name}.{col_name}'...")
                IndexClass = INDEX_CLASS_MAP.get(index_type)
                
                if IndexClass is None:
                    print(f"      ERROR: Tipo de índice desconocido '{index_type}'")
                    continue
                
                if index_type == "SequentialFileIndex":
                    col_idx = [s[0] for s in table.schema].index(col_name)
                    idx = SequentialFileIndex(
                        table_name=table.name,
                        column_name=col_name,
                        record_manager=table.record_manager,
                        key_column_index=col_idx,
                        data_dir=self.data_dir
                    )
                elif index_type == "ISAMIndex":
                    idx = ISAMIndex(
                        table_name=table.name, 
                        column_name=col_name, 
                        data_dir=self.data_dir
                    )
                else:
                    idx = IndexClass(
                        table_name=table.name, 
                        column_name=col_name, 
                        data_dir=self.data_dir
                    )
                
                table.indexes[col_name] = idx
                
            except Exception as e:
                print(f"      ERROR al cargar el índice '{col_name}': {e}")

    def execute(self, sql_string: str) -> (List[Tuple] | str):
        try:
            plan = self.parser.parse(sql_string)
            command = plan['command']
            
            if command == 'CREATE_TABLE':
                return self._handle_create_table(plan)
            
            if command == 'CREATE_TABLE_FROM_FILE':
                return self._handle_create_from_file(plan)

            if command == 'INSERT':
                return self._handle_insert(plan)

            if command == 'SELECT':
                return self._handle_select(plan)
                
            if command == 'DELETE':
                return self._handle_delete(plan)
            
            if command == 'CREATE_INDEX':
                return self._handle_create_index(plan) 

            return f"Comando '{command}' no reconocido."

        except Exception as e:
            print(f"Error de ejecución: {e}")
            return f"Error: {e}"

    def _handle_create_table(self, plan: Dict) -> str:
        table_name = plan['table_name']
        if table_name in self.tables:
            raise ValueError(f"La tabla '{table_name}' ya existe.")
            
        table = Table(table_name, schema=plan['schema'], data_dir=self.data_dir)
        table.index_definitions = plan['index_definitions']
        table._save_metadata()
        
        self.tables[table_name] = table
        self._load_indexes_for_table(table)
        
        return f"Tabla '{table_name}' creada exitosamente."

    def _handle_create_from_file(self, plan: Dict) -> str:
        table_name = plan['table_name']
        if table_name not in self.tables:
            raise ValueError(f"Tabla '{table_name}' no existe. Defina el esquema primero con CREATE TABLE (...).")

        table = self.tables[table_name]
        
        isam_col = None
        for col, idx_type in table.index_definitions.items():
            if idx_type == "ISAMIndex":
                isam_col = col
                break
        
        if isam_col:
            print(f"Construyendo índice ISAM estático en '{isam_col}'...")
            idx = ISAMIndex.build_from_table(table, isam_col)
            table.indexes[isam_col] = idx
        
        print(f"Cargando datos desde '{plan['from_file']}' en '{table_name}'...")
        count = 0
        with open(plan['from_file'], 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            for row in reader:
                if not row: continue
                values = self._cast_row_values(row, table.schema)
                self._insert_record_into_table(table, values)
                count += 1
        
        return f"{count} registros insertados en '{table_name}'."

    def _handle_insert(self, plan: Dict) -> str:
        table_name = plan['table_name']
        table = self.tables.get(table_name)
        if not table:
            raise ValueError(f"Tabla '{table_name}' no encontrada.")
            
        values = self._cast_row_values(plan['values'], table.schema)
        
        self._insert_record_into_table(table, values)
        
        return "1 registro insertado."

    def _insert_record_into_table(self, table: Table, values: List[Any]):
        rid = table.insert_record(values)
        
        for col_name, index_obj in table.indexes.items():
            col_idx = [s[0] for s in table.schema].index(col_name)
            key = values[col_idx]
            
            if isinstance(index_obj, SequentialFileIndex):
                index_obj.add(key, values)
            else:
                index_obj.add(key, rid)

    def _handle_select(self, plan: Dict) -> List[Tuple]:
        table_name = plan['table_name']
        table = self.tables.get(table_name)
        if not table:
            raise ValueError(f"Tabla '{table_name}' no encontrada.")
        
        where = plan['where']
        
        if not where:
            print(f"Ejecutando Full Table Scan en '{table_name}'...")
            return [record for rid, record in table.scan()]

        col = where['column']
        
        if col not in table.indexes:
            print(f"Ejecutando Full Table Scan (filtro en '{col}') en '{table_name}'...")
            results = []
            col_idx = [s[0] for s in table.schema].index(col)
            op = where['op']
            
            for rid, record in table.scan():
                if op == '=' and record[col_idx] == where['value']:
                    results.append(record)
                elif op == 'BETWEEN' and where['value1'] <= record[col_idx] <= where['value2']:
                    results.append(record)
            return results

        print(f"Ejecutando Index Scan en '{table_name}.{col}'...")
        index_obj = table.indexes[col]
        op = where['op']
        rids_or_records = []
        
        if op == '=':
            rids_or_records = index_obj.search(where['value'])
            
        elif op == 'BETWEEN':
            rids_or_records = index_obj.rangeSearch(where['value1'], where['value2'])
            
        elif op == 'IN' and isinstance(index_obj, RTreeIndex):
            rids_or_records = index_obj.radius_search(where['point'], where['radius'])
            
        else:
             raise ValueError(f"Operación '{op}' no soportada por el índice en '{col}'.")
        
        if isinstance(index_obj, SequentialFileIndex):
            return rids_or_records
        else:
            return [table.get_record(rid) for rid in rids_or_records]
            
    def _handle_delete(self, plan: Dict) -> str:
        table_name = plan['table_name']
        table = self.tables.get(table_name)
        if not table:
            raise ValueError(f"Tabla '{table_name}' no encontrada.")
        
        where = plan['where']
        col = where['column']
        
        if col not in table.indexes:
            raise ValueError(f"DELETE requiere un índice en la columna '{col}'.")
            
        index_to_use = table.indexes[col]
        rids_to_delete = index_to_use.search(where['value'])
        
        if isinstance(index_to_use, SequentialFileIndex):
            index_to_use.remove(where['value'], None)
            
            print("Reconstruyendo todos los otros índices...")
            for col_name, idx in table.indexes.items():
                if col_name == col: continue
                
                if isinstance(idx, ISAMIndex):
                    new_idx = ISAMIndex.build_from_table(table, col_name)
                    table.indexes[col_name] = new_idx
                else:
                    new_idx = INDEX_CLASS_MAP[table.index_definitions[col_name]](
                        table.name, col_name, self.data_dir
                    )
                    for rid, record in table.scan():
                        key = record[[s[0] for s in table.schema].index(col_name)]
                        new_idx.add(key, rid)
                    table.indexes[col_name] = new_idx
            
            return f"{len(rids_to_delete)} registros eliminados (reconstrucción completa)."
        
        schema_cols = [s[0] for s in table.schema]
        
        for rid in rids_to_delete:
            record = table.get_record(rid)
            if not record: continue
            
            for col_name, index_obj in table.indexes.items():
                col_idx = schema_cols.index(col_name)
                key = record[col_idx]
                index_obj.remove(key, rid)
        
        return f"{len(rids_to_delete)} registros eliminados."
    
    def _handle_create_index(self, plan: Dict) -> str:
        table_name = plan['table_name']
        column_name = plan['column_name']
        index_name = plan['index_name']
        index_type = plan['index_type']  # Como 'BTREE' o cualquier otro tipo de índice que necesites

        # Verificar si la tabla existe
        if table_name not in self.tables:
            raise ValueError(f"Tabla '{table_name}' no encontrada.")
        
        table = self.tables[table_name]

        # Llamar al método para crear el índice
        self.idx.create_index(table, column_name, index_type)

        # Registrar el índice en las definiciones de la tabla
        table.index_definitions[column_name] = index_type
        table._save_metadata()  # Guardar metadatos de la tabla con la definición del índice

        return f"Índice '{index_name}' creado exitosamente en la columna '{column_name}' de la tabla '{table_name}'."

    def _cast_row_values(self, row: List[str], schema: List[Tuple]) -> List[Any]:
        if len(row) != len(schema):
            raise ValueError(f"Conteo de columnas incorrecto. Se esperaban {len(schema)} pero se recibieron {len(row)}.")
        
        casted_values = []
        for i, (col_name, col_type, length) in enumerate(schema):
            value = row[i]
            try:
                if col_type == 'INT':
                    casted_values.append(int(value))
                elif col_type == 'FLOAT':
                    casted_values.append(float(value))
                elif col_type == 'ARRAY[FLOAT]':
                    coords = value.strip('()').split(',')
                    casted_values.append(tuple(float(c) for c in coords))
                else:
                    casted_values.append(str(value))
            except Exception as e:
                raise ValueError(f"Error al convertir valor '{value}' para columna '{col_name}': {e}")
        
        return casted_values