# tests/test_isam.py

import unittest
import os
import shutil
import random

from src.core.table import Table
from src.indices.isam.isamindex import ISAMIndex

class TestISAMIndex(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_data_isam'
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.schema = [('id', 'INT', 4), ('name', 'VARCHAR', 20)]
        self.table = Table('test_table', self.schema, data_dir=self.test_dir)
        
        self.records = [
            (5, 'cherry'), (2, 'banana'), (8, 'elderberry'), (1, 'apple'),
            (10, 'fig'), (12, 'grape'), (3, 'date')
        ]
        self.rids = {record[0]: self.table.insert_record(record) for record in self.records}

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_build_and_search(self):
        print("\nEjecutando: test_build_and_search")
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=2, data_capacity=2)
        
        for key, rid in self.rids.items():
            print(f"  -> Buscando clave: {key}")
            result = index.search(key)
            self.assertEqual(result, [rid], f"Falló la búsqueda para la clave {key}")
        print(" -> Pasó")
        
    def test_search_nonexistent_key(self):
        print("\nEjecutando: test_search_nonexistent_key")
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=2, data_capacity=2)
        
        result = index.search(999)
        self.assertEqual(result, [])
        print(" -> Pasó")

    def test_add_and_search_overflow(self):
        print("\nEjecutando: test_add_and_search_overflow")
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=2, data_capacity=2)
        
        new_key, new_rid = 4, 1000
        print(f"  -> Añadiendo nueva clave (en overflow): {new_key}")
        index.add(new_key, new_rid)
        
        result_new = index.search(new_key)
        self.assertEqual(result_new, [new_rid])
        
        original_key = 5
        result_original = index.search(original_key)
        self.assertEqual(result_original, [self.rids[original_key]])
        print(" -> Pasó")

    def test_overflow_chain(self):
        print("\nEjecutando: test_overflow_chain")
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=2, data_capacity=1)
        
        overflow_keys = [(2.1, 2001), (2.2, 2002), (2.3, 2003)]
        
        for key, rid in overflow_keys:
            print(f"  -> Añadiendo en cadena de overflow: {key}")
            index.add(key, rid)
            
        for key, rid in overflow_keys:
            self.assertEqual(index.search(key), [rid])
            
        self.assertEqual(index.search(2), [self.rids[2]])
        print(" -> Pasó")

    def test_persistence(self):
        print("\nEjecutando: test_persistence")
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=2, data_capacity=2)
        index.add(99, 9999)

        del index
        
        print("  -> Recargando índice desde disco...")
        reloaded_index = ISAMIndex('test_table', 'id', data_dir=self.test_dir)
        
        self.assertEqual(reloaded_index.search(8), [self.rids[8]])
        self.assertEqual(reloaded_index.search(99), [9999])
        print(" -> Pasó")

if __name__ == '__main__':
    unittest.main()