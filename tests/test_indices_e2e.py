# tests/test_indices_e2e_unittest.py
import io
import sys
import tempfile
import unittest
from pathlib import Path

# --- asegurar imports "from src...." ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from src.app.main import app
from src.app.engine import Engine, _engine_singleton


def upload_csv(client: TestClient, table: str, csv_text: str, has_header=True):
    files = {"file": (f"{table}.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")}
    data = {"table": table, "has_header": "true" if has_header else "false"}
    r = client.post("/api/upload", files=files, data=data)
    assert r.status_code == 200, r.text
    return r.json()

def run_sql(client: TestClient, sql: str):
    return client.post("/api/sql", json={"query": sql})

def assert_schema_has_index(client: TestClient, table: str, col: str, typ: str):
    r = client.get(f"/api/tables/{table}/schema")
    assert r.status_code == 200, r.text
    meta = r.json()
    idxs = meta.get("indexes", [])
    exp = [col, typ]
    assert exp in idxs or (col, typ) in idxs, f"indexes={idxs} no contiene {exp}"


class IndicesE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # data_dir temporal por clase (rápido y aislado)
        cls._tmpdir = tempfile.TemporaryDirectory()
        global _engine_singleton
        _engine_singleton = Engine(data_dir=cls._tmpdir.name)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        # limpia singleton
        global _engine_singleton
        _engine_singleton = None
        cls._tmpdir.cleanup()

    def test_btree(self):
        csv_text = "Nombre,Edad\nAna,30\nLuis,25\nAna,40\n"
        up = upload_csv(self.client, "t_btree", csv_text)
        self.assertTrue(up["ok"]); self.assertEqual(up["inserted"], 3)

        r = run_sql(self.client, "CREATE INDEX idx_btree_edad ON t_btree(Edad) TYPE BTREE;")
        self.assertEqual(r.status_code, 200); self.assertTrue(r.json().get("ok", True))
        assert_schema_has_index(self.client, "t_btree", "Edad", "BTREE")

        r = run_sql(self.client, "SELECT * FROM t_btree WHERE Edad = 30;")
        self.assertEqual(r.status_code, 200)
        sel = r.json()
        self.assertTrue(sel["ok"])

        rows_str = [[str(x) for x in row] for row in sel["rows"]]
        self.assertIn(["Ana", "30"], rows_str)


        r = run_sql(self.client, "SELECT * FROM t_btree WHERE Edad BETWEEN 26 AND 39;")
        self.assertEqual(r.status_code, 200)
        rows = [[str(x) for x in row] for row in r.json()["rows"]]
        self.assertIn(["Ana", "30"], rows)
        self.assertNotIn(["Luis", "25"], rows)
        self.assertNotIn(["Ana", "40"], rows)

    def test_hash(self):
        csv_text = "Nombre,Edad\nAna,30\nLuis,25\nAna,40\n"
        up = upload_csv(self.client, "t_hash", csv_text)
        self.assertTrue(up["ok"]); self.assertEqual(up["inserted"], 3)

        r = run_sql(self.client, "CREATE INDEX idx_hash_nombre ON t_hash(Nombre) TYPE HASH;")
        self.assertEqual(r.status_code, 200); self.assertTrue(r.json().get("ok", True))
        assert_schema_has_index(self.client, "t_hash", "Nombre", "HASH")

        r = run_sql(self.client, "SELECT * FROM t_hash WHERE Nombre = 'Ana';")
        self.assertEqual(r.status_code, 200)
        rows = r.json()["rows"]
        self.assertTrue(any(row[0] == "Ana" for row in rows))

    def test_isam(self):
        csv_text = "Id,Edad\n1,21\n2,30\n3,33\n4,40\n5,45\n"
        up = upload_csv(self.client, "t_isam", csv_text)
        self.assertTrue(up["ok"]); self.assertEqual(up["inserted"], 5)

        r = run_sql(self.client, "CREATE INDEX idx_isam_edad ON t_isam(Edad) TYPE ISAM;")
        self.assertEqual(r.status_code, 200); self.assertTrue(r.json().get("ok", True))
        assert_schema_has_index(self.client, "t_isam", "Edad", "ISAM")

        r = run_sql(self.client, "SELECT * FROM t_isam WHERE Edad BETWEEN 30 AND 40;")
        self.assertEqual(r.status_code, 200)
        edades = {int(row[1]) for row in r.json()["rows"]}
        self.assertEqual(edades, {30, 33, 40})

    def test_seq(self):
        csv_text = "Clave,Valor\n1001,X\n1002,Y\n1003,Z\n"
        up = upload_csv(self.client, "t_seq", csv_text)
        self.assertTrue(up["ok"]); self.assertEqual(up["inserted"], 3)

        r = run_sql(self.client, "CREATE INDEX idx_seq_clave ON t_seq(Clave) TYPE SEQ;")
        self.assertEqual(r.status_code, 200); self.assertTrue(r.json().get("ok", True))
        assert_schema_has_index(self.client, "t_seq", "Clave", "SEQ")

        r = run_sql(self.client, "SELECT * FROM t_seq WHERE Clave = 1002;")
        self.assertEqual(r.status_code, 200)
        rows = [[str(x) for x in row] for row in r.json()["rows"]]
        self.assertIn(["1002", "Y"], rows)

    def test_rtree(self):
        # Usa tuplas "(x, y)" mejor que "[x, y]"
        csv_text = 'Id,Coord\n1,"(10, 10)"\n2,"(15, 15)"\n3,"(30, 30)"\n'
        up = upload_csv(self.client, "t_rtree", csv_text)
        self.assertTrue(up["ok"]); self.assertEqual(up["inserted"], 3)

        r = run_sql(self.client, "CREATE INDEX idx_rtree_coord ON t_rtree(Coord) TYPE RTREE;")
        self.assertEqual(r.status_code, 200); self.assertTrue(r.json().get("ok", True))
        assert_schema_has_index(self.client, "t_rtree", "Coord", "RTREE")

        # prueba con radio 10 y también 12 (por si la impl es estricta con bounding boxes)
        for radius in (10, 12):
            q = f"SELECT * FROM t_rtree WHERE Coord IN ((12, 12), {radius});"
            r = run_sql(self.client, q)
            self.assertEqual(r.status_code, 200, r.text)
            sel = r.json()
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", sel)
            self.assertTrue(sel["ok"], sel)

            ids = {int(row[0]) for row in sel["rows"]}
            if ids:  # si devolvió algo, validamos lo esperado y salimos
                self.assertEqual(ids, {1, 2}, f"radius={radius}, rows={sel['rows']}")
                break
            else:
                # Si ninguno de los radios devolvió filas, marca el test como fallido
                # (o cámbialo a un expected failure si sabes que radius_search falta)
                self.fail("RTREE radius_search no devolvió resultados; revisa formato de coords o impl de radius_search")



if __name__ == "__main__":
    unittest.main(verbosity=2)
