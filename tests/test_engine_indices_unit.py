import sys, tempfile, shutil, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.engine import Engine


class TestEngineIndicesUnit(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.engine = Engine(data_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_btree_select_uses_index(self):
        # create table
        stmt = {
            "action": "create_table",
            "table": "t",
            "columns": [
                {"name": "Nombre", "type": "VARCHAR[20]"},
                {"name": "Edad", "type": "INT"},
            ],
        }
        self.engine.execute(stmt)

        # insert rows
        for nombre, edad in [("Ana", 30), ("Luis", 25), ("Ana", 40)]:
            self.engine.execute({
                "action": "insert",
                "table": "t",
                "values": [nombre, edad],
            })

        # create index
        self.engine.execute({
            "action": "create_index",
            "table": "t",
            "columns": ["Edad"],
            "index_type": "BTREE",
        })

        # select equality
        res = self.engine.execute({
            "action": "select",
            "table": "t",
            "columns": ["*"],
            "condition": {"op": "=", "field": "Edad", "value": 30},
        })
        self.assertTrue(res["ok"])
        self.assertEqual(len(res["rows"]), 1)
        self.assertEqual(res["rows"][0][0], "Ana")
        self.assertEqual(res["used_index"]["column"], "Edad")

    def test_hash_index(self):
        stmt = {
            "action": "create_table",
            "table": "t_hash",
            "columns": [
                {"name": "Nombre", "type": "VARCHAR[20]"},
                {"name": "Edad", "type": "INT"},
            ],
        }
        self.engine.execute(stmt)
        for nombre, edad in [("Ana", 30), ("Luis", 25), ("Ana", 40)]:
            self.engine.execute({
                "action": "insert",
                "table": "t_hash",
                "values": [nombre, edad],
            })
        self.engine.execute({
            "action": "create_index",
            "table": "t_hash",
            "columns": ["Nombre"],
            "index_type": "HASH",
        })
        res = self.engine.execute({
            "action": "select",
            "table": "t_hash",
            "columns": ["*"],
            "condition": {"op": "=", "field": "Nombre", "value": "Ana"},
        })
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(len(res["rows"]), 2)
        self.assertIn("used_index", res)

    def test_seq_index_delete_rebuild(self):
        stmt = {
            "action": "create_table",
            "table": "t_seq",
            "columns": [
                {"name": "Clave", "type": "INT"},
                {"name": "Valor", "type": "VARCHAR[10]"},
            ],
        }
        self.engine.execute(stmt)
        for clave, valor in [(1001, "X"), (1002, "Y"), (1003, "Z")]:
            self.engine.execute({
                "action": "insert",
                "table": "t_seq",
                "values": [clave, valor],
            })
        self.engine.execute({
            "action": "create_index",
            "table": "t_seq",
            "columns": ["Clave"],
            "index_type": "SEQ",
        })
        # delete uno
        self.engine.execute({
            "action": "delete",
            "table": "t_seq",
            "condition": {"op": "=", "field": "Clave", "value": 1002},
        })
        # debe quedar 1001 y 1003
        res = self.engine.execute({
            "action": "select",
            "table": "t_seq",
            "columns": ["*"],
            "condition": None,
        })
        claves = {row[0] for row in res["rows"]}
        self.assertEqual(claves, {1001, 1003})

    def test_rtree_1d(self):
        stmt = {
            "action": "create_table",
            "table": "t_r",
            "columns": [
                {"name": "Id", "type": "INT"},
                {"name": "Coord", "type": "VARCHAR[32]"},
            ],
        }
        self.engine.execute(stmt)

        rows = [
            (1, "(10,10)"),
            (2, "(15,15)"),
            (3, "(30,30)"),
        ]
        for r in rows:
            self.engine.execute({
                "action": "insert",
                "table": "t_r",
                "values": list(r),
            })

        self.engine.execute({
            "action": "create_index",
            "table": "t_r",
            "columns": ["Coord"],
            "index_type": "RTREE",
        })

        res = self.engine.execute({
            "action": "select",
            "table": "t_r",
            "columns": ["*"],
            "condition": {
                "op": "IN",
                "field": "Coord",
                "coords": (12.0, 12.0),
                "radius": 10.0,
            },
        })
        self.assertTrue(res["ok"])
        ids = {row[0] for row in res["rows"]}
        # esperamos 1 y 2 cerca de (12,12)
        self.assertEqual(ids, {1, 2})


if __name__ == "__main__":
    unittest.main(verbosity=2)
