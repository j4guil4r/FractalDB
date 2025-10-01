# tests/test_sequentialfile.py
import unittest
import random

# Intenta varias rutas de import según tu estructura real
try:
    from src.indices.sequentialfile.sequentialfile_min import SequentialFile, Record
    from src.indices.sequentialfile.sequentialfileindex_min import SequentialFileIndex
except ImportError:
    try:
        from src.indices.sequentialfile.sequentialfile import SequentialFile, Record
        from src.indices.sequentialfile.sequentialfileindex import SequentialFileIndex
    except ImportError:
        # fallback si lo corres local sin paquete
        from sequentialfile_min import SequentialFile, Record
        from sequentialfileindex_min import SequentialFileIndex


class TestSequentialFileCore(unittest.TestCase):
    def setUp(self):
        # Umbral pequeño para forzar rebuilds durante las pruebas
        self.sf = SequentialFile(rebuild_threshold=3, index_block_size=2)

    def test_basic_insert_search(self):
        self.sf.add(Record(10, "v10"))
        self.sf.add(Record(5, "v5"))
        res10 = self.sf.search(10)
        self.assertEqual(len(res10), 1)
        self.assertEqual(res10[0].key, 10)
        self.assertEqual(res10[0].value, "v10")
        self.assertEqual(self.sf.search(99), [])

    def test_rebuild_threshold_merge(self):
        # 3 inserts en aux disparan rebuild
        self.sf.add(Record(3, "v3"))
        self.sf.add(Record(1, "v1"))
        self.sf.add(Record(2, "v2"))  # aquí debe ocurrir _merge_rebuild()
        # Aux debe quedar vacía y main ordenado
        self.assertEqual(len(self.sf.aux), 0)
        keys = [r.key for r in self.sf.range_search(-999, 999)]
        self.assertEqual(keys, [1, 2, 3])

        # Luego una inserción extra queda en aux pero aún debe aparecer en search
        self.sf.add(Record(4, "v4"))
        vals = [r.value for r in self.sf.search(4)]
        self.assertEqual(vals, ["v4"])

    def test_range_search(self):
        for k in [1, 4, 5, 7, 9]:
            self.sf.add(Record(k, f"v{k}"))
        res_keys = [r.key for r in self.sf.range_search(4, 7)]
        self.assertEqual(res_keys, [4, 5, 7])

    def test_edge_cases_range(self):
        # vacío
        self.assertEqual(self.sf.range_search(10, 20), [])
        # un elemento
        self.sf.add(Record(15, "v15"))
        self.assertEqual([r.key for r in self.sf.range_search(10, 20)], [15])
        # start > end => vacío
        self.assertEqual(self.sf.range_search(20, 10), [])

    def test_duplicates(self):
        self.sf.add(Record(2, "a"))
        self.sf.add(Record(2, "b"))
        self.sf.add(Record(2, "c"))  # dispara rebuild
        self.sf.add(Record(2, "d"))  # queda en aux
        res = self.sf.search(2)
        self.assertEqual(sorted([r.value for r in res]), ["a", "b", "c", "d"])
        # range sobre el mismo valor debe traer los 4
        res2 = self.sf.range_search(2, 2)
        self.assertEqual(len(res2), 4)

    def test_remove(self):
        for k in [10, 20, 30]:
            self.sf.add(Record(k, f"v{k}"))
        removed = self.sf.remove(20)
        self.assertTrue(removed)
        self.assertEqual(self.sf.search(20), [])
        self.assertEqual([r.key for r in self.sf.search(10)], [10])
        self.assertEqual([r.key for r in self.sf.search(30)], [30])

    def test_remove_nonexistent(self):
        self.sf.add(Record(10, "v10"))
        self.assertFalse(self.sf.remove(99))
        self.assertEqual([r.key for r in self.sf.search(10)], [10])

    def test_remove_then_insert_again(self):
        self.sf.add(Record(10, "v10"))
        self.sf.remove(10)
        self.sf.add(Record(10, "v10b"))
        res = self.sf.search(10)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].value, "v10b")

    def test_negative_numbers(self):
        for k in [-10, -5, 0, 5, 10]:
            self.sf.add(Record(k, f"v{k}"))
        for k in [-10, -5, 0, 5, 10]:
            self.assertEqual([r.key for r in self.sf.search(k)], [k])
        res = self.sf.range_search(-7, 7)
        self.assertEqual([r.key for r in res], [-5, 0, 5])

    def test_sparse_index_exists_after_rebuild(self):
        for k in [5, 1, 3]:
            self.sf.add(Record(k, f"v{k}"))  # dispara rebuild
        self.assertTrue(len(self.sf.sparse_index) > 0)
        # primer marcador debe apuntar a la primera clave en main
        first_key_idx, pos = self.sf.sparse_index[0]
        self.assertEqual(first_key_idx, self.sf.main[pos].key)


class TestSequentialFileIndexWrapper(unittest.TestCase):
    def setUp(self):
        self.idx = SequentialFileIndex(rebuild_threshold=3, index_block_size=2)

    def test_wrapper_basic_insert_search(self):
        self.idx.add_record(10, "v10")
        self.idx.add_record(5, "v5")
        res10 = self.idx.search(10)
        self.assertEqual(len(res10), 1)
        self.assertEqual((res10[0].key, res10[0].value), (10, "v10"))
        self.assertEqual(self.idx.search(99), [])

    def test_wrapper_multiple_and_ranges(self):
        data = [50, 20, 80, 10, 30, 60, 90, 5, 15, 25, 35, 55, 65, 85, 95]
        for k in data:
            self.idx.add_record(k, f"v{k}")
        # rango existente
        res = self.idx.range_search(20, 35)
        self.assertEqual([r.key for r in res], [20, 25, 30, 35])
        # rango total
        res_all = self.idx.range_search(5, 95)
        self.assertEqual(sorted([r.key for r in res_all]), sorted(data))
        # rango vacío
        self.assertEqual(self.idx.range_search(26, 29), [])
        # límites exactos
        self.assertEqual([r.key for r in self.idx.range_search(50, 50)], [50])
        self.assertEqual([r.key for r in self.idx.range_search(95, 95)], [95])

    def test_wrapper_sequential_and_reverse_inserts(self):
        # secuencial
        self.idx = SequentialFileIndex(rebuild_threshold=3, index_block_size=2)
        for k in range(1, 51):
            self.idx.add_record(k, f"v{k}")
        for k in range(1, 51):
            self.assertEqual([r.key for r in self.idx.search(k)], [k])

        # reversa
        self.idx = SequentialFileIndex(rebuild_threshold=3, index_block_size=2)
        for k in range(50, 0, -1):
            self.idx.add_record(k, f"v{k}")
        for k in range(1, 51):
            self.assertEqual([r.key for r in self.idx.search(k)], [k])

    def test_wrapper_remove_and_large_range(self):
        for k in range(1, 21):
            self.idx.add_record(k, f"v{k}")
        for k in [5, 10, 15]:
            self.idx.remove_record(k)
        res = self.idx.range_search(1, 20)
        self.assertEqual([r.key for r in res], [k for k in range(1, 21) if k not in [5, 10, 15]])

    def test_wrapper_duplicates(self):
        self.idx.add_record(7, "a")
        self.idx.add_record(7, "b")
        self.idx.add_record(7, "c")
        vals = sorted([r.value for r in self.idx.search(7)])
        self.assertEqual(vals, ["a", "b", "c"])

    def test_wrapper_stress(self):
        big = SequentialFileIndex(rebuild_threshold=32, index_block_size=16)
        data = list(range(1, 1001))
        for k in data:
            big.add_record(k, f"v{k}")
        # muestras aleatorias
        for k in random.sample(data, 50):
            res = big.search(k)
            self.assertEqual(len(res), 1)
            self.assertEqual((res[0].key, res[0].value), (k, f"v{k}"))
        # rango grande
        res = big.range_search(100, 900)
        self.assertEqual(len(res), 801)
        self.assertEqual([r.key for r in res], list(range(100, 901)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
