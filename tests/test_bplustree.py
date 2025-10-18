# tests/test_bplustree.py

import unittest
import os
import shutil
import random

from src.indices.bplustree.bplustreeindex import BPlusTreeIndex

class TestBPlusTreeIndex(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_data_bpt'
        
        # MODIFICADO: Asegura una limpieza total antes de cada prueba
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        self.index = BPlusTreeIndex(
            table_name='test_table', 
            column_name='id', 
            data_dir=self.test_dir,
            order=3 # Con order=3, los nodos se llenan con 2 claves
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_add_and_search_simple(self):
        print("\nEjecutando: test_add_and_search_simple")
        self.index.add(10, 100)
        self.index.add(20, 200)
        self.assertEqual(self.index.search(10), [100])
        self.assertEqual(self.index.search(20), [200])
        print(" -> Pasó")

    def test_search_nonexistent_key(self):
        print("\nEjecutando: test_search_nonexistent_key")
        self.assertEqual(self.index.search(999), [])
        print(" -> Pasó")

    def test_add_duplicate_keys(self):
        print("\nEjecutando: test_add_duplicate_keys")
        self.index.add(15, 150)
        self.index.add(15, 151)
        self.index.add(15, 152)
        self.assertCountEqual(self.index.search(15), [150, 151, 152])
        print(" -> Pasó")

    def test_node_splits(self):
        print("\nEjecutando: test_node_splits")
        # order=3 significa que los nodos se llenan con 2 claves
        # Esto forzará múltiples divisiones
        keys_to_insert = {i: i * 10 for i in range(1, 11)}
        
        for key, rid in keys_to_insert.items():
            print(f"   -> Añadiendo ({key}, {rid}). Root offset: {self.index.tree.root_offset}")
            self.index.add(key, rid)
        
        print("   -> Verificando todas las claves...")
        for key, rid in keys_to_insert.items():
            self.assertEqual(self.index.search(key), [rid])
        print(" -> Pasó")

    def test_range_search(self):
        print("\nEjecutando: test_range_search")
        for i in range(1, 11):
            self.index.add(i, i * 10)
        
        print("   -> Buscando rango [4, 7]")
        results = self.index.rangeSearch(4, 7)
        expected_rids = [40, 50, 60, 70]
        self.assertCountEqual(results, expected_rids)
        
        print("   -> Buscando rango [8, 12] (hasta el final)")
        results_at_end = self.index.rangeSearch(8, 12)
        expected_at_end = [80, 90, 100]
        self.assertCountEqual(results_at_end, expected_at_end)
        
        print("   -> Buscando rango [1, 3] (desde el inicio)")
        results_at_start = self.index.rangeSearch(1, 3)
        expected_at_start = [10, 20, 30]
        self.assertCountEqual(results_at_start, expected_at_start)
        print(" -> Pasó")

    def test_persistence(self):
        print("\nEjecutando: test_persistence")
        self.index.add(50, 500)
        self.index.add(60, 600)

        # Guardamos el offset de la raíz para verificar que se recarga
        original_root_offset = self.index.tree.root_offset
        self.assertTrue(os.path.exists(self.index.tree.meta_path))
        self.assertTrue(os.path.exists(self.index.tree.dat_path))

        del self.index
        
        print("   -> Recargando índice desde disco...")
        reloaded_index = BPlusTreeIndex(
            table_name='test_table', 
            column_name='id', 
            data_dir=self.test_dir,
            order=3 # El 'order' debe coincidir
        )
        
        # Verificar que se cargó el estado (order, offsets)
        self.assertEqual(reloaded_index.tree.order, 3)
        self.assertEqual(reloaded_index.tree.root_offset, original_root_offset)
        
        # Verificar que los datos están allí
        self.assertEqual(reloaded_index.search(50), [500])
        self.assertEqual(reloaded_index.search(60), [600])
        print(" -> Pasó")
        
    def test_remove(self):
        print("\nEjecutando: test_remove")
        self.index.add(10, 100)
        self.index.add(20, 200)
        self.index.add(10, 101)

        print("   -> Eliminando RID específico (10, 100)")
        self.index.remove(10, 100) # Eliminar solo el RID 100
        self.assertEqual(self.index.search(10), [101]) # 101 debe quedar

        print("   -> Eliminando clave completa (20)")
        self.index.remove(20) # Eliminar toda la clave 20
        self.assertEqual(self.index.search(20), [])
        
        # Verificar que la otra clave no se vio afectada
        self.assertEqual(self.index.search(10), [101])
        print(" -> Pasó")

if __name__ == '__main__':
    unittest.main()