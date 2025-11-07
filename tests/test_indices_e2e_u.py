# tests/test_indices_e2e_unittest.py

import io
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

# --- asegurar imports "from src...." ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.app.main import app
from src.app.engine import Engine, _engine_singleton, get_engine
from src.indices.rtree.rtreeindex import RTreeIndex


# ------------------------ helpers HTTP ------------------------ #

def upload_csv(client: TestClient, table: str, csv_text: str, has_header=True):
    files = {
        "file": (f"{table}.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")
    }
    data = {"table": table, "has_header": "true" if has_header else "false"}
    r = client.post("/api/upload", files=files, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def run_sql(client: TestClient, sql: str):
    return client.post("/api/sql", json={"query": sql})


def assert_schema_has_index(client: TestClient, table: str, col_or_key: str, typ: str):
    """
    Espera que /api/tables/{table}/schema devuelva algo como:
      {"indexes": [[ "col" , "BTREE"], ["lat,lon","RTREE"], ...]}
    o tuplas equivalentes.
    """
    r = client.get(f"/api/tables/{table}/schema")
    assert r.status_code == 200, r.text
    meta = r.json()
    idxs = meta.get("indexes", [])
    exp = [col_or_key, typ]
    if exp in idxs:
        return
    if (col_or_key, typ) in idxs:
        return
    raise AssertionError(f"indexes={idxs} no contiene {exp}")


# =========================
#   E2E vía FastAPI + SQL
# =========================

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
        global _engine_singleton
        _engine_singleton = None
        cls._tmpdir.cleanup()

    # ---------- BTREE ----------

    def test_btree_eq_between_delete(self):
        csv_text = "Nombre,Edad\nAna,30\nLuis,25\nAna,40\n"
        up = upload_csv(self.client, "t_btree", csv_text)
        self.assertTrue(up["ok"])
        self.assertEqual(up["inserted"], 3)

        # create index
        r = run_sql(self.client,
                    "CREATE INDEX idx_btree_edad ON t_btree(Edad) TYPE BTREE;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json().get("ok", True))
        assert_schema_has_index(self.client, "t_btree", "Edad", "BTREE")

        # equality (usa índice)
        r = run_sql(self.client,
                    "SELECT * FROM t_btree WHERE Edad = 30;")
        self.assertEqual(r.status_code, 200)
        sel = r.json()
        self.assertTrue(sel["ok"])
        rows_str = [[str(x) for x in row] for row in sel["rows"]]
        self.assertIn(["Ana", "30"], rows_str)
        self.assertIn("used_index", sel)
        self.assertEqual(sel["used_index"]["column"], "Edad")

        # BETWEEN (usa rangeSearch / planner)
        r = run_sql(self.client,
                    "SELECT * FROM t_btree WHERE Edad BETWEEN 26 AND 39;")
        self.assertEqual(r.status_code, 200)
        rows = [[str(x) for x in row] for row in r.json()["rows"]]
        self.assertIn(["Ana", "30"], rows)
        self.assertNotIn(["Luis", "25"], rows)
        self.assertNotIn(["Ana", "40"], rows)

        # DELETE con condición (engine reescribe y reconstruye)
        r = run_sql(self.client,
                    "DELETE FROM t_btree WHERE Edad = 30;")
        self.assertEqual(r.status_code, 200, r.text)
        res = r.json()
        self.assertTrue(res["ok"])
        # Ahora SELECT BETWEEN ya no debe incluir Ana 30
        r = run_sql(self.client,
                    "SELECT * FROM t_btree WHERE Edad BETWEEN 26 AND 39;")
        rows = [[str(x) for x in row] for row in r.json()["rows"]]
        self.assertNotIn(["Ana", "30"], rows)

    # ---------- HASH ----------

    def test_hash_eq(self):
        csv_text = "Nombre,Edad\nAna,30\nLuis,25\nAna,40\n"
        up = upload_csv(self.client, "t_hash", csv_text)
        self.assertTrue(up["ok"])
        self.assertEqual(up["inserted"], 3)

        r = run_sql(self.client,
                    "CREATE INDEX idx_hash_nombre ON t_hash(Nombre) TYPE HASH;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json().get("ok", True))
        assert_schema_has_index(self.client, "t_hash", "Nombre", "HASH")

        r = run_sql(self.client,
                    "SELECT * FROM t_hash WHERE Nombre = 'Ana';")
        self.assertEqual(r.status_code, 200)
        sel = r.json()
        self.assertTrue(sel["ok"])
        rows = sel["rows"]
        self.assertTrue(any(row[0] == "Ana" for row in rows))
        self.assertIn("used_index", sel)
        self.assertEqual(sel["used_index"]["column"], "Nombre")

    # ---------- ISAM ----------

    def test_isam_between(self):
        csv_text = "Id,Edad\n1,21\n2,30\n3,33\n4,40\n5,45\n"
        up = upload_csv(self.client, "t_isam", csv_text)
        self.assertTrue(up["ok"])
        self.assertEqual(up["inserted"], 5)

        r = run_sql(self.client,
                    "CREATE INDEX idx_isam_edad ON t_isam(Edad) TYPE ISAM;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json().get("ok", True))
        assert_schema_has_index(self.client, "t_isam", "Edad", "ISAM")

        r = run_sql(self.client,
                    "SELECT * FROM t_isam WHERE Edad BETWEEN 30 AND 40;")
        self.assertEqual(r.status_code, 200)
        sel = r.json()
        self.assertTrue(sel["ok"])
        edades = {int(row[1]) for row in sel["rows"]}
        self.assertEqual(edades, {30, 33, 40})
        self.assertIn("used_index", sel)
        self.assertEqual(sel["used_index"]["column"], "Edad")

    # ---------- SEQ ----------

    def test_seq_eq_and_delete_rebuild(self):
        csv_text = "Clave,Valor\n1001,X\n1002,Y\n1003,Z\n"
        up = upload_csv(self.client, "t_seq", csv_text)
        self.assertTrue(up["ok"])
        self.assertEqual(up["inserted"], 3)

        r = run_sql(self.client,
                    "CREATE INDEX idx_seq_clave ON t_seq(Clave) TYPE SEQ;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json().get("ok", True))
        assert_schema_has_index(self.client, "t_seq", "Clave", "SEQ")

        # lookup
        r = run_sql(self.client,
                    "SELECT * FROM t_seq WHERE Clave = 1002;")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["ok"], data)

        rows = [[str(x) for x in row] for row in data["rows"]]
        self.assertIn(["1002", "Y"], rows)

        # delete
        r = run_sql(self.client,
                    "DELETE FROM t_seq WHERE Clave = 1002;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json().get("ok", False))

        # verificar que 1002 no está, pero 1001 y 1003 sí
        r = run_sql(self.client,
                    "SELECT * FROM t_seq;")
        self.assertEqual(r.status_code, 200, r.text)
        filas = r.json()["rows"]
        claves = {int(row[0]) for row in filas}
        self.assertEqual(claves, {1001, 1003})


    # ---------- RTREE 1D (col Coord con "(x,y)") ----------

    def test_rtree_coord_radius(self):
        # Coord como texto con tupla
        csv_text = 'Id,Coord\n1,"(10,10)"\n2,"(15,15)"\n3,"(30,30)"\n'
        up = upload_csv(self.client, "t_rtree", csv_text)
        self.assertTrue(up["ok"])
        self.assertEqual(up["inserted"], 3)

        r = run_sql(self.client,
                    "CREATE INDEX idx_rtree_coord ON t_rtree(Coord) TYPE RTREE;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json().get("ok", True))
        assert_schema_has_index(self.client, "t_rtree", "Coord", "RTREE")

        # radio alrededor de (12,12)
        q = "SELECT * FROM t_rtree WHERE Coord IN ((12, 12), 10);"
        r = run_sql(self.client, q)
        self.assertEqual(r.status_code, 200, r.text)
        sel = r.json()
        self.assertTrue(sel["ok"], sel)
        ids = {int(row[0]) for row in sel["rows"]}
        # 1 y 2 están cerca, 3 está lejos
        self.assertEqual(ids, {1, 2})
        self.assertIn("used_index", sel)
        self.assertEqual(sel["used_index"]["column"], "Coord")

    # ---------- RTREE 2D (lat,lon) ----------

    def test_rtree_cities_2d(self):
        # mini CSV con lat/lon reales
        csv_text = (
            "id,name,lat,lon\n"
            "1,A,36.68,71.53\n"
            "2,B,36.681,71.531\n"
            "3,C,40.0,10.0\n"
        )
        up = upload_csv(self.client, "cities", csv_text)
        self.assertTrue(up["ok"])

        # índice 2D
        r = run_sql(
            self.client,
            "CREATE INDEX idx_cities_geo ON cities(lat, lon) TYPE RTREE;"
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json().get("ok", False))
        # clave sintética "lat,lon"
        assert_schema_has_index(self.client, "cities", "lat,lon", "RTREE")

        # consulta cerca de A/B
        q = "SELECT * FROM cities WHERE lat, lon IN ((36.68,71.53), 0.01);"
        r = run_sql(self.client, q)
        self.assertEqual(r.status_code, 200, r.text)
        res = r.json()
        self.assertTrue(res["ok"], res)
        ids = {int(row[0]) for row in res["rows"]}
        # deberíamos ver 1 y 2, no 3
        self.assertEqual(ids, {1, 2})


# =========================
#   Unit tests RTreeIndex
# =========================

class RTreeIndexUnit(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.idx = RTreeIndex("t_rt", "Coord", data_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_radius_search(self):
        # puntos simples
        self.idx.add((0.0, 0.0), 1)
        self.idx.add((5.0, 0.0), 2)
        self.idx.add((100.0, 100.0), 3)

        # radio 10 alrededor de (0,0) debe traer 1 y 2
        r = self.idx.radius_search((0.0, 0.0), 10.0)
        self.assertSetEqual(set(r), {1, 2})

    def test_knn_search(self):
        self.idx.add((0.0, 0.0), 10)
        self.idx.add((1.0, 1.0), 20)
        self.idx.add((10.0, 10.0), 30)

        nn = self.idx.knn_search((0.0, 0.0), 2)
        # KNN de (0,0) con k=2 deben ser los dos más cercanos (10 y 20 en cualquier orden)
        self.assertEqual(set(nn), {10, 20})


if __name__ == "__main__":
    unittest.main(verbosity=2)
