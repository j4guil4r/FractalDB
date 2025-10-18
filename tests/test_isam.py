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
        
        # --- MODIFICADO ---
        # Asegura una limpieza total antes de cada prueba
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        # --- FIN MODIFICADO ---
        
        self.schema = [('id', 'INT', 4), ('name', 'VARCHAR', 20)]
        self.table = Table('test_table', self.schema, data_dir=self.test_dir)
        
        # Claves ordenadas: 1, 2, 3, 5, 8, 10, 12
        self.records = [
            (5, 'cherry'), (2, 'banana'), (8, 'elderberry'), (1, 'apple'),
            (10, 'fig'), (12, 'grape'), (3, 'date')
        ]
        self.rids = {}
        for record in self.records:
             # Usamos list() para ser explícitos, aunque tupla también funciona
            self.rids[record[0]] = self.table.insert_record(list(record))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_build_and_search(self):
        print("\nEjecutando: test_build_and_search")
        # .build() escribe bloques con padding
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=2, data_capacity=2)
        
        for key, rid in self.rids.items():
            print(f"   -> Buscando clave: {key}")
            # .search() lee bloques con padding
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
        print(f"   -> Añadiendo nueva clave (en overflow): {new_key}")
        
        # .add() escribe un bloque nuevo en .ovf y SOBRESCRIBE un bloque en .dat
        # Esta es la prueba clave para el padding.
        index.add(new_key, new_rid)
        
        result_new = index.search(new_key)
        self.assertEqual(result_new, [new_rid])
        
        original_key = 5
        result_original = index.search(original_key)
        self.assertEqual(result_original, [self.rids[original_key]])
        print(" -> Pasó")

    def test_overflow_chain(self):
        print("\nEjecutando: test_overflow_chain")
        
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=4, data_capacity=1)
        
        overflow_keys = [(2.1, 2001), (2.2, 2002), (2.3, 2003)]
        
        for key, rid in overflow_keys:
            print(f"   -> Añadiendo en cadena de overflow: {key}")
            # Esto prueba la sobrescritura de bloques en la cadena de .ovf
            index.add(key, rid)
            
        print("   -> Verificando cadena de overflow...")
        for key, rid in overflow_keys:
            self.assertEqual(index.search(key), [rid])
            
        self.assertEqual(index.search(2), [self.rids[2]])
        print(" -> Pasó")

    def test_persistence(self):
        print("\nEjecutando: test_persistence")
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=2, data_capacity=2)
        index.add(99, 9999)

        meta_path = index.engine.paths['meta']
        dat_path = index.engine.paths['dat']
        
        self.assertTrue(os.path.exists(meta_path))
        self.assertTrue(os.path.exists(dat_path))

        del index
        
        print("   -> Recargando índice desde disco...")
        # El constructor carga el .meta
        reloaded_index = ISAMIndex('test_table', 'id', data_dir=self.test_dir)
        
        # .search() usa el .meta recargado para leer los archivos .dat/.ovf con padding
        self.assertEqual(reloaded_index.search(8), [self.rids[8]])
        self.assertEqual(reloaded_index.search(99), [9999])
        print(" -> Pasó")

    def test_remove_logical(self):
        print("\nEjecutando: test_remove_logical")
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=2, data_capacity=2)
        
        key_to_remove = 5
        rid_to_remove = self.rids[key_to_remove]
        
        print(f"   -> Buscando clave antes de eliminar: {key_to_remove}")
        result_before = index.search(key_to_remove)
        self.assertEqual(result_before, [rid_to_remove])
        
        print(f"   -> Eliminando lógicamente la clave: {key_to_remove}")
        # .remove() SOBRESCRIBE el bloque .dat donde está la clave 5
        # Esta es la segunda prueba clave para el padding.
        index.remove(key_to_remove)
        
        print(f"   -> Buscando clave después de eliminar: {key_to_remove}")
        # .search() lee el bloque sobrescrito
        result_after = index.search(key_to_remove)
        self.assertEqual(result_after, [])
        
        self.assertEqual(index.search(2), [self.rids[2]])
        print(" -> Pasó")

    def test_range_search(self):
        print("\nEjecutando: test_range_search")
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=2, data_capacity=2)
        
        # Claves ordenadas: 1, 2, 3, 5, 8, 10, 12
        print("   -> Buscando rango [3, 9]")
        # .rangeSearch() lee secuencialmente los bloques .dat
        results = index.rangeSearch(3, 9)
        expected_rids = {self.rids[3], self.rids[5], self.rids[8]}
        self.assertEqual(set(results), expected_rids)
        
        print("   -> Buscando rango [1, 12]")
        results_all = index.rangeSearch(1, 12)
        self.assertEqual(set(results_all), set(self.rids.values()))
        
        print("   -> Buscando rango [100, 200] (vacío)")
        results_empty = index.rangeSearch(100, 200)
        self.assertEqual(results_empty, [])
        print(" -> Pasó")

    def test_range_search_with_overflow_and_deletes(self):
        print("\nEjecutando: test_range_search_with_overflow_and_deletes")
        index = ISAMIndex.build_from_table(self.table, 'id', index_capacity=2, data_capacity=2)
        
        # .add() prueba la escritura en .ovf
        index.add(4, 1004) 
        index.add(9, 1009) 
        
        # .remove() prueba la sobrescritura en .dat
        index.remove(5)
        
        print("   -> Buscando rango [3, 9] con overflow y eliminados")
        # .rangeSearch() prueba la lectura de .dat, .ovf y flags de borrado
        results = index.rangeSearch(3, 9)
        expected_rids = {self.rids[3], 1004, self.rids[8], 1009}
        self.assertEqual(set(results), expected_rids)
        print(" -> Pasó")


if __name__ == '__main__':
    unittest.main()