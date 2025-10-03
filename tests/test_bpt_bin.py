import unittest
from src.core.storagebtree import Storage
from src.indices.bplustree.bplustreeindex import BPlusTreeIndex
import os

class TestBPlusTreeComprehensive(unittest.TestCase):
    STORAGE_FILE = "test_storage_full.dat"

    def setUp(self):
        # Limpiar archivo de storage antes de cada test
        if os.path.exists(self.STORAGE_FILE):
            os.remove(self.STORAGE_FILE)
        self.storage = Storage(self.STORAGE_FILE)
        self.bptree = BPlusTreeIndex(order=4, storage=self.storage)

    def test_insert_and_search(self):
        rows = [[1, "A", "2025-01-01"], [2, "B", "2025-01-02"], [3, "C", "2025-01-03"]]
        for row in rows:
            self.bptree.add_record(row[2], row)
        # Search exact
        self.assertEqual(self.bptree.search("2025-01-02"), [2, "B", "2025-01-02"])
        self.assertIsNone(self.bptree.search("2025-01-10"))  # No existe

    def test_range_search_basic(self):
        rows = [[1, "A", "2025-01-01"], [2, "B", "2025-01-02"], [3, "C", "2025-01-03"]]
        for row in rows:
            self.bptree.add_record(row[2], row)
        # Range
        result = self.bptree.range_search("2025-01-01", "2025-01-03")
        self.assertEqual(len(result), 3)
        result = self.bptree.range_search("2025-01-02", "2025-01-02")
        self.assertEqual(result, [[2, "B", "2025-01-02"]])
        result = self.bptree.range_search("2025-02-01", "2025-03-01")
        self.assertEqual(result, [])  # Fuera de rango

    def test_remove_key(self):
        rows = [[1, "A", "2025-01-01"], [2, "B", "2025-01-02"], [3, "C", "2025-01-03"]]
        for row in rows:
            self.bptree.add_record(row[2], row)
        # Remove middle key
        self.assertTrue(self.bptree.remove_record("2025-01-02"))
        self.assertIsNone(self.bptree.search("2025-01-02"))
        # Remove non-existent
        self.assertFalse(self.bptree.remove_record("2025-01-10"))
        # Remove first
        self.assertTrue(self.bptree.remove_record("2025-01-01"))
        # Remove last
        self.assertTrue(self.bptree.remove_record("2025-01-03"))
        self.assertEqual(self.bptree.range_search("2025-01-01", "2025-01-03"), [])

    def test_duplicate_keys(self):
        row = [1, "A", "2025-01-01"]
        self.bptree.add_record(row[2], row)
        self.bptree.add_record(row[2], row)  # duplicado
        results = self.bptree.range_search("2025-01-01", "2025-01-01")
        self.assertEqual(len(results), 2)
        # Remove one duplicate
        self.assertTrue(self.bptree.remove_record("2025-01-01"))
        results = self.bptree.range_search("2025-01-01", "2025-01-01")
        self.assertEqual(len(results), 1)

    def test_sequential_and_reverse_inserts(self):
        # Secuenciales
        for i in range(1, 11):
            self.bptree.add_record(f"2025-01-{i:02}", [i, f"Name{i}", f"2025-01-{i:02}"])
        for i in range(1, 11):
            self.assertEqual(self.bptree.search(f"2025-01-{i:02}")[0], i)
        # Limpiar
        self.setUp()
        # Reversa
        for i in range(10, 0, -1):
            self.bptree.add_record(f"2025-01-{i:02}", [i, f"Name{i}", f"2025-01-{i:02}"])
        for i in range(1, 11):
            self.assertEqual(self.bptree.search(f"2025-01-{i:02}")[0], i)

    def test_edge_cases_empty_tree(self):
        self.assertIsNone(self.bptree.search("2025-01-01"))
        self.assertEqual(self.bptree.range_search("2025-01-01", "2025-12-31"), [])
        self.assertFalse(self.bptree.remove_record("2025-01-01"))

    def test_range_after_removals(self):
        # Insertar varias claves
        for i in range(1, 11):
            self.bptree.add_record(f"2025-01-{i:02}", [i, f"Name{i}", f"2025-01-{i:02}"])
        # Remover algunos
        for i in [3, 5, 7]:
            self.bptree.remove_record(f"2025-01-{i:02}")
        results = self.bptree.range_search("2025-01-01", "2025-01-10")
        expected = [[i, f"Name{i}", f"2025-01-{i:02}"] for i in range(1, 11) if i not in [3,5,7]]
        self.assertEqual(results, expected)

    def test_large_range_search(self):
        for i in range(1, 21):
            self.bptree.add_record(f"2025-01-{i:02}", [i, f"Name{i}", f"2025-01-{i:02}"])
        results = self.bptree.range_search("2025-01-05", "2025-01-15")
        expected = [[i, f"Name{i}", f"2025-01-{i:02}"] for i in range(5,16)]
        self.assertEqual(results, expected)

    def test_stress_insert_remove(self):
        # Insertar 50 elementos
        for i in range(50):
            self.bptree.add_record(f"2025-01-{i:02}", [i, f"Name{i}", f"2025-01-{i:02}"])
        # Eliminar algunos aleatorios
        for i in range(0,50,5):
            self.bptree.remove_record(f"2025-01-{i:02}")
        # Verificar todos los demás
        for i in range(50):
            expected = None if i % 5 == 0 else [i, f"Name{i}", f"2025-01-{i:02}"]
            self.assertEqual(self.bptree.search(f"2025-01-{i:02}"), expected)

if __name__ == "__main__":
    unittest.main(verbosity=2)
