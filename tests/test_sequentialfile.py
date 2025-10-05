# tests/test_sequentialfile.py

import unittest
import os
import shutil

from src.core.record import RecordManager
from src.indices.sequentialfile.sequentialfileindex import SequentialFileIndex

class TestSequentialFileIndex(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_data_seq'
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.schema = [('id', 'INT', 4), ('name', 'VARCHAR', 20), ('value', 'FLOAT', 8)]
        self.record_manager = RecordManager(self.schema)
        
        self.key_column_index = 0
        
        self.index = SequentialFileIndex(
            table_name='test_table',
            column_name='id',
            record_manager=self.record_manager,
            key_column_index=self.key_column_index,
            data_dir=self.test_dir,
            aux_capacity=3 
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_add_to_aux_file(self):
        print("\nEjecutando: test_add_to_aux_file")
        record1 = [10, 'record_diez', 10.1]
        self.index.add(None, record1)

        self.assertEqual(os.path.getsize(self.index.engine.main_path), 0)
        self.assertGreater(os.path.getsize(self.index.engine.aux_path), 0)

        results = self.index.search(10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 'record_diez')
        print(" -> Pasó")
        
    def test_reconstruction_trigger(self):
        print("\nEjecutando: test_reconstruction_trigger")
        self.index.add(None, [20, 'record_veinte', 20.2])
        self.index.add(None, [5, 'record_cinco', 5.5])
        
        print("  -> Añadiendo 3er registro para disparar la reconstrucción...")
        self.index.add(None, [15, 'record_quince', 15.1])

        self.assertEqual(os.path.getsize(self.index.engine.aux_path), 0)
        
        main_records = list(self.index.engine._read_records_from_file(self.index.engine.main_path))
        self.assertEqual(len(main_records), 3)
        main_keys = [r[self.key_column_index] for r in main_records]
        self.assertEqual(main_keys, [5, 15, 20])
        print(" -> Pasó")

    def test_search_after_reconstruction(self):
        print("\nEjecutando: test_search_after_reconstruction")
        self.index.add(None, [20, 'main_veinte', 20.2])
        self.index.add(None, [5, 'main_cinco', 5.5])
        self.index.add(None, [15, 'main_quince', 15.1])
        
        self.index.add(None, [50, 'aux_cincuenta', 50.5])
        
        result_main = self.index.search(5)
        self.assertEqual(result_main[0][1], 'main_cinco')
        
        result_aux = self.index.search(50)
        self.assertEqual(result_aux[0][1], 'aux_cincuenta')
        print(" -> Pasó")

    def test_range_search(self):
        print("\nEjecutando: test_range_search")
        self.index.add(None, [10, 'diez', 10.0])
        self.index.add(None, [30, 'treinta', 30.0])
        self.index.add(None, [20, 'veinte', 20.0])
        
        self.index.add(None, [45, 'cuarentaycinco', 45.0])
        self.index.add(None, [1, 'uno', 1.0])
        
        results = self.index.rangeSearch(15, 50)
        result_keys = [r[self.key_column_index] for r in results]
        
        self.assertCountEqual(result_keys, [20, 30, 45])
        self.assertEqual(result_keys, [20, 30, 45])
        print(" -> Pasó")

    def test_remove(self):
        print("\nEjecutando: test_remove")
        self.index.add(None, [10, 'diez', 10.0])
        self.index.add(None, [20, 'veinte', 20.0])
        
        self.index.remove(10)
        
        self.assertEqual(self.index.search(10), [])
        self.assertNotEqual(self.index.search(20), [])
        self.assertEqual(self.index.search(20)[0][1], 'veinte')
        print(" -> Pasó")

if __name__ == '__main__':
    unittest.main()