# tests/test_rtree.py

import unittest
import os
import shutil

IMPORT_ERROR_MSG = ""
RTREE_INSTALLED = False
RTreeIndex = None
try:
    from src.indices.rtree.rtreeindex import RTreeIndex
    RTREE_INSTALLED = True
except ImportError as e:
    IMPORT_ERROR_MSG = str(e)


@unittest.skipIf(not RTREE_INSTALLED, f"Skipping R-Tree tests: {IMPORT_ERROR_MSG}")
class TestRTreeIndex(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_data_rtree'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.index = RTreeIndex(
            table_name='test_locations',
            column_name='coords',
            data_dir=self.test_dir
        )
        self.points = {
            100: (2.0, 2.0), 101: (3.0, 3.0), 102: (8.0, 8.0),
            103: (15.0, 15.0), 104: (2.5, 2.5)
        }
        for rid, coords in self.points.items():
            self.index.add(coords, rid)

    def tearDown(self):
        if hasattr(self.index.idx, 'close'):
            try:
                self.index.idx.close()
            except Exception:
                pass
        shutil.rmtree(self.test_dir)

    def test_radius_search(self):
        print("\nEjecutando: test_radius_search")
        center_point = (0.0, 0.0)
        radius = 4.0
        results = self.index.radius_search(center_point, radius)
        expected_rids = [100, 101, 104]
        self.assertCountEqual(results, expected_rids)
        print(" -> Pasó")

    def test_knn_search(self):
        print("\nEjecutando: test_knn_search")
        query_point = (2.2, 2.2)
        k = 3
        results = self.index.knn_search(query_point, k)
        expected_rids = [100, 104, 101]
        self.assertCountEqual(results, expected_rids)
        print(" -> Pasó")

    def test_remove(self):
        print("\nEjecutando: test_remove")
        coords_to_remove = self.points[101]
        self.index.remove(coords_to_remove, 101)
        results = self.index.radius_search((0.0, 0.0), 4.0)
        self.assertNotIn(101, results)
        self.assertIn(100, results)
        print(" -> Pasó")
        
    def test_persistence(self):
        print("\nEjecutando: test_persistence")
        test_point = self.points[103]
        test_rid = 103
        
        self.index.idx.close()

        reloaded_index = RTreeIndex(
            table_name='test_locations',
            column_name='coords',
            data_dir=self.test_dir
        )
        
        results = reloaded_index.knn_search(test_point, 1)
        self.assertEqual(results, [test_rid])
        
        reloaded_index.idx.close()
        print(" -> Pasó")

    def test_unsupported_methods_raise_error(self):
        print("\nEjecutando: test_unsupported_methods_raise_error")
        with self.assertRaises(NotImplementedError):
            self.index.search((2.0, 2.0))
        with self.assertRaises(NotImplementedError):
            self.index.rangeSearch(1, 10)
        print(" -> Pasó")

if __name__ == '__main__':
    unittest.main()