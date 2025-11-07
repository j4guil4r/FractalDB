import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parser.sqlparser import SQLParser


class TestSQLParserBasic(unittest.TestCase):
    def setUp(self):
        self.p = SQLParser()

    def test_create_table_simple(self):
        plan = self.p.parse("CREATE TABLE t (id INT, name VARCHAR[20]);")
        self.assertEqual(plan["command"], "CREATE_TABLE")
        self.assertEqual(plan["table_name"], "t")
        self.assertEqual(plan["schema"][0][0], "id")
        self.assertEqual(plan["schema"][0][1], "INT")

    def test_create_index_simple(self):
        plan = self.p.parse("CREATE INDEX idx_t_id ON t(id) TYPE BTREE;")
        self.assertEqual(plan["command"], "CREATE_INDEX")
        self.assertEqual(plan["index_name"], "idx_t_id")
        self.assertEqual(plan["table_name"], "t")
        self.assertEqual(plan["column_name"], "id")
        self.assertEqual(plan["index_type"].upper(), "BTREE")

    def test_create_index_rtree_2d(self):
        plan = self.p.parse("CREATE INDEX idx_geo ON cities(lat,lon) TYPE RTREE;")
        self.assertEqual(plan["command"], "CREATE_INDEX")
        self.assertEqual(plan["index_name"], "idx_geo")
        self.assertEqual(plan["table_name"], "cities")
        self.assertEqual(plan["column_name"], ["lat", "lon"])
        self.assertEqual(plan["index_type"].upper(), "RTREE")

    def test_select_eq(self):
        plan = self.p.parse("SELECT * FROM t WHERE id = 1;")
        self.assertEqual(plan["command"], "SELECT")
        self.assertEqual(plan["where"]["op"], "=")
        self.assertEqual(plan["where"]["column"], "id")
        self.assertEqual(plan["where"]["value"], 1)

    def test_select_between(self):
        plan = self.p.parse("SELECT * FROM t WHERE edad BETWEEN 18 AND 30;")
        self.assertEqual(plan["where"]["op"], "BETWEEN")
        self.assertEqual(plan["where"]["value1"], 18)
        self.assertEqual(plan["where"]["value2"], 30)

    def test_select_rtree_in(self):
        plan = self.p.parse(
            "SELECT * FROM t WHERE Coord IN ((10,20), 5);"
        )
        self.assertEqual(plan["where"]["op"], "IN")
        self.assertEqual(plan["where"]["column"], "Coord")
        self.assertEqual(plan["where"]["point"], (10.0, 20.0))
        self.assertEqual(plan["where"]["radius"], 5.0)

    def test_select_rtree_in2(self):
        plan = self.p.parse(
            "SELECT * FROM t WHERE lat, lon IN ((10,20), 5);"
        )
        self.assertEqual(plan["where"]["op"], "IN2")
        self.assertEqual(plan["where"]["columns"], ["lat", "lon"])
        self.assertEqual(plan["where"]["point"], (10.0, 20.0))
        self.assertEqual(plan["where"]["radius"], 5.0)

    def test_delete_eq(self):
        plan = self.p.parse("DELETE FROM t WHERE id = 10;")
        self.assertEqual(plan["command"], "DELETE")
        self.assertEqual(plan["where"]["column"], "id")
        self.assertEqual(plan["where"]["value"], 10)

    def test_invalid_sql(self):
        with self.assertRaises(ValueError):
            self.p.parse("UPDATE t SET x = 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)