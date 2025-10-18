# tests/test_hash_index.py

import unittest
import os
import shutil

from src.indices.hashing.hashingindex import HashIndex

class TestHashIndex(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_data_hash'
        # Aseguramos que el directorio esté limpio antes de cada test
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        # --- MODIFICADO ---
        # Pasamos el bucket_size=2 en el constructor.
        # Esto asegura que el índice se *crea* en disco con el tamaño correcto.
        self.index = HashIndex(
            table_name='test_table', 
            column_name='name', 
            data_dir=self.test_dir,
            bucket_size=2  # <-- ¡AQUÍ ESTÁ EL CAMBIO!
        )
        # Esta línea ya no es necesaria (y no funcionaría)
        # self.index.directory.bucket_size = 2 

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_add_and_search_simple(self):
        print("\nEjecutando: test_add_and_search_simple")
        self.index.add('apple', 100)
        results = self.index.search('apple')
        self.assertEqual(results, [100])
        print(" -> Pasó")

    def test_search_nonexistent_key(self):
        print("\nEjecutando: test_search_nonexistent_key")
        results = self.index.search('banana')
        self.assertEqual(results, [])
        print(" -> Pasó")

    def test_add_duplicate_keys(self):
        print("\nEjecutando: test_add_duplicate_keys")
        self.index.add('cherry', 200)
        self.index.add('cherry', 201)
        results = self.index.search('cherry')
        # assertCountEqual es mejor para listas donde no importa el orden
        self.assertCountEqual(results, [200, 201])
        print(" -> Pasó")

    def test_bucket_split_and_directory_grow(self):
        print("\nEjecutando: test_bucket_split_and_directory_grow")
        # Con bucket_size=2, esto forzará varios splits y crecimientos
        keys_to_insert = {
            'apple': 100, 'banana': 200, 'cherry': 300, 'date': 400,
            'elderberry': 500, 'fig': 600, 'grape': 700
        }
        
        for key, rid in keys_to_insert.items():
            self.index.add(key, rid)
            print(f"   -> Añadido ('{key}', {rid}). Profundidad Global: {self.index.directory.global_depth}")
        
        print("   -> Verificando todas las claves...")
        for key, rid in keys_to_insert.items():
            self.assertEqual(self.index.search(key), [rid], f"Falló la búsqueda para {key}")
        print(" -> Pasó")

    def test_persistence(self):
        print("\nEjecutando: test_persistence")
        self.index.add('persistent_key', 999)
        
        # --- MODIFICADO ---
        # self.index.file_path ya no existe, es self.index.file_path_prefix
        # De todos modos, la línea no era necesaria para el test.
        # index_path = self.index.file_path 
        
        del self.index
        
        print("   -> Recargando índice desde disco...")
        new_index = HashIndex(
            table_name='test_table', 
            column_name='name', 
            data_dir=self.test_dir
        )
        
        # El new_index cargará el bucket_size=2 desde el archivo .meta
        self.assertEqual(new_index.directory.bucket_size, 2)
        
        results = new_index.search('persistent_key')
        self.assertEqual(results, [999])
        print(" -> Pasó")

    def test_remove_key(self):
        print("\nEjecutando: test_remove_key")
        self.index.add('key_to_keep', 500)
        self.index.add('key_to_delete', 501)
        self.index.add('key_to_delete', 502)

        self.index.remove('key_to_delete') # Elimina todas las 'key_to_delete'

        self.assertEqual(self.index.search('key_to_delete'), [])
        self.assertEqual(self.index.search('key_to_keep'), [500])
        print(" -> Pasó")

    def test_remove_specific_rid(self):
        print("\nEjecutando: test_remove_specific_rid")
        self.index.add('key', 601)
        self.index.add('key', 602)
        
        self.index.remove('key', rid=601) # Elimina solo el (key, 601)
        
        self.assertEqual(self.index.search('key'), [602])
        print(" -> Pasó")

    def test_range_search_raises_error(self):
        print("\nEjecutando: test_range_search_raises_error")
        with self.assertRaises(NotImplementedError):
            self.index.rangeSearch('a', 'z')
        print(" -> Pasó")

if __name__ == '__main__':
    unittest.main()