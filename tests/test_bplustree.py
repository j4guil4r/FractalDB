# tests/test_bplustree.py

import unittest
import os
import shutil
import random

from src.indices.bplustree.bplustreeindex import BPlusTreeIndex

class TestBPlusTreeIndex(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_data_bpt'
        os.makedirs(self.test_dir, exist_ok=True)
        self.index = BPlusTreeIndex(
            table_name='test_table', 
            column_name='id', 
            data_dir=self.test_dir,
            order=3
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
        keys_to_insert = {i: i * 10 for i in range(1, 11)}
        
        for key, rid in keys_to_insert.items():
            self.index.add(key, rid)
        
        for key, rid in keys_to_insert.items():
            self.assertEqual(self.index.search(key), [rid])
        print(" -> Pasó")

    def test_range_search(self):
        print("\nEjecutando: test_range_search")
        for i in range(1, 11):
            self.index.add(i, i * 10)
        
        results = self.index.rangeSearch(4, 7)
        expected_rids = [40, 50, 60, 70]
        self.assertCountEqual(results, expected_rids)
        
        results_at_end = self.index.rangeSearch(8, 12)
        expected_at_end = [80, 90, 100]
        self.assertCountEqual(results_at_end, expected_at_end)
        print(" -> Pasó")

    def test_persistence(self):
        print("\nEjecutando: test_persistence")
        self.index.add(50, 500)
        self.index.add(60, 600)

        del self.index
        
        reloaded_index = BPlusTreeIndex(
            table_name='test_table', 
            column_name='id', 
            data_dir=self.test_dir,
            order=3
        )
        self.assertEqual(reloaded_index.search(50), [500])
        self.assertEqual(reloaded_index.search(60), [600])
        print(" -> Pasó")
        
    def test_remove(self):
        print("\nEjecutando: test_remove")
        self.index.add(10, 100)
        self.index.add(20, 200)
        self.index.add(10, 101)

        self.index.remove(10, 100)
        self.assertEqual(self.index.search(10), [101])

        self.index.remove(20)
        self.assertEqual(self.index.search(20), [])
        
        self.assertEqual(self.index.search(10), [101])
        print(" -> Pasó")

if __name__ == '__main__':
    unittest.main()