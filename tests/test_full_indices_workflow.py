# tests/test_full_indices_workflow.py

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


# -------- helpers HTTP --------

def upload_csv(client: TestClient, table: str, csv_text: str, has_header=True):
    files = {"file": (f"{table}.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")}
    data = {"table": table, "has_header": "true" if has_header else "false"}
    r = client.post("/api/upload", files=files, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def run_sql(client: TestClient, sql: str):
    return client.post("/api/sql", json={"query": sql})


def get_schema(client: TestClient, table: str):
    r = client.get(f"/api/tables/{table}/schema")
    assert r.status_code == 200, r.text
    return r.json()


def assert_schema_has_index(client: TestClient, table: str, key_str: str, typ: str):
    """
    key_str:
      - "Edad"
      - "lat,lon"
      - "Coord"
    typ:
      - "BTREE", "HASH", "ISAM", "SEQ", "RTREE"
    """
    meta = get_schema(client, table)
    idxs = meta.get("indexes", [])
    exp = [key_str, typ]
    assert exp in idxs, f"indexes={idxs} no contiene {exp}"


# -------- suite principal --------

class FullIndicesWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # data_dir aislado por suite
        cls._tmpdir = tempfile.TemporaryDirectory()
        global _engine_singleton
        _engine_singleton = Engine(data_dir=cls._tmpdir.name)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        global _engine_singleton
        _engine_singleton = None
        cls._tmpdir.cleanup()

    # ---------- BTREE: eq, between, inserts, delete ----------

    def test_btree_full(self):
        csv_text = "Nombre,Edad\nAna,30\nLuis,25\nAna,40\n"
        up = upload_csv(self.client, "bt_personas", csv_text)
        self.assertTrue(up["ok"])
        self.assertEqual(up["inserted"], 3)

        # crear índice
        r = run_sql(self.client,
                    "CREATE INDEX idx_bt_edad ON bt_personas(Edad) TYPE BTREE;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["ok"])
        assert_schema_has_index(self.client, "bt_personas", "Edad", "BTREE")

        # igualdad usando índice
        r = run_sql(self.client,
                    "SELECT * FROM bt_personas WHERE Edad = 30;")
        data = r.json()
        self.assertTrue(data["ok"], data)
        rows = [[str(x) for x in row] for row in data["rows"]]
        self.assertIn(["Ana", "30"], rows)
        if data.get("used_index"):
            self.assertEqual(data["used_index"]["column"], "Edad")

        # rango usando índice
        r = run_sql(self.client,
                    "SELECT * FROM bt_personas WHERE Edad BETWEEN 26 AND 39;")
        data = r.json()
        rows = [[str(x) for x in row] for row in data["rows"]]
        self.assertIn(["Ana", "30"], rows)
        self.assertNotIn(["Luis", "25"], rows)

        # insertar con índice ya creado → debe actualizar on_insert
        r = run_sql(self.client,
                    "INSERT INTO bt_personas VALUES('Carla', 35);")
        self.assertEqual(r.status_code, 200, r.text)

        r = run_sql(self.client,
                    "SELECT * FROM bt_personas WHERE Edad = 35;")
        data = r.json()
        self.assertTrue(data["ok"], data)
        rows = [[str(x) for x in row] for row in data["rows"]]
        self.assertIn(["Carl", "35"], rows)

        # delete por columna indexada, engine reescribe archivo y rebuild_all
        r = run_sql(self.client,
                    "DELETE FROM bt_personas WHERE Edad = 30;")
        self.assertEqual(r.status_code, 200, r.text)

        r = run_sql(self.client, "SELECT * FROM bt_personas;")
        rows = [[str(x) for x in row] for row in r.json()["rows"]]
        # ya no 30
        self.assertNotIn(["Ana", "30"], rows)
        # pero siguen otros
        self.assertIn(["Luis", "25"], rows)
        self.assertIn(["Carl", "35"], rows)

    # ---------- HASH: eq, duplicados, no-hit ----------

    def test_hash_full(self):
        csv_text = "Usuario,Email\nu1,a@mail\nu2,b@mail\nu3,a@mail\n"
        up = upload_csv(self.client, "h_users", csv_text)
        self.assertTrue(up["ok"])

        r = run_sql(self.client,
                    "CREATE INDEX idx_h_email ON h_users(Email) TYPE HASH;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["ok"])
        assert_schema_has_index(self.client, "h_users", "Email", "HASH")

        # búsqueda por clave repetida
        r = run_sql(self.client,
                    "SELECT * FROM h_users WHERE Email = 'a@mail';")
        data = r.json()
        self.assertTrue(data["ok"], data)
        users = sorted(row[0] for row in data["rows"])
        self.assertEqual(users, ["u1", "u3"])

        # no-hit
        r = run_sql(self.client,
                    "SELECT * FROM h_users WHERE Email = 'nope';")
        data = r.json()
        self.assertTrue(data["ok"], data)
        self.assertEqual(data["rows"], [])

        # insertar con índice vigente
        r = run_sql(self.client,
                    "INSERT INTO h_users VALUES('u4','a@mail');")
        self.assertEqual(r.status_code, 200, r.text)

        r = run_sql(self.client,
                    "SELECT * FROM h_users WHERE Email = 'a@mail';")
        data = r.json()
        users = sorted(row[0] for row in data["rows"])
        self.assertEqual(users, ["u1", "u3", "u4"])

    # ---------- ISAM: estático + rango ----------

    def test_isam_full(self):
        csv_text = "Id,Score\n1,10\n2,20\n3,30\n4,40\n"
        up = upload_csv(self.client, "is_scores", csv_text)
        self.assertTrue(up["ok"])

        r = run_sql(self.client,
                    "CREATE INDEX idx_is_score ON is_scores(Score) TYPE ISAM;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["ok"])
        assert_schema_has_index(self.client, "is_scores", "Score", "ISAM")

        r = run_sql(self.client,
                    "SELECT * FROM is_scores WHERE Score BETWEEN 15 AND 35;")
        data = r.json()
        self.assertTrue(data["ok"], data)
        ids = sorted(int(row[0]) for row in data["rows"])
        self.assertEqual(ids, [2, 3])

        # Insertar después de ISAM (edge):
        # tu ISAM es estático, así que aquí *no* garantizamos que use índice.
        r = run_sql(self.client,
                    "INSERT INTO is_scores VALUES(5,25);")
        self.assertEqual(r.status_code, 200, r.text)

        # Pero el SELECT lógico debe ver el registro (aunque sea por scan)
        r = run_sql(self.client,
                    "SELECT * FROM is_scores WHERE Score = 25;")
        data = r.json()
        self.assertTrue(data["ok"], data)
        rows = data["rows"]
        self.assertTrue(any(int(r[0]) == 5 for r in rows))

    # ---------- SEQ: búsqueda, rango, delete + rebuild ----------

    def test_seq_full(self):
        csv_text = "Clave,Valor\n1001,A\n1002,B\n1003,C\n1004,D\n"
        up = upload_csv(self.client, "seq_kv", csv_text)
        self.assertTrue(up["ok"])

        r = run_sql(self.client,
                    "CREATE INDEX idx_seq ON seq_kv(Clave) TYPE SEQ;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["ok"])
        assert_schema_has_index(self.client, "seq_kv", "Clave", "SEQ")

        # igualdad
        r = run_sql(self.client,
                    "SELECT * FROM seq_kv WHERE Clave = 1002;")
        data = r.json()
        self.assertTrue(data["ok"], data)
        rows = [[str(x) for x in row] for row in data["rows"]]
        self.assertIn(["1002", "B"], rows)

        # rango
        r = run_sql(self.client,
                    "SELECT * FROM seq_kv WHERE Clave BETWEEN 1002 AND 1003;")
        data = r.json()
        self.assertTrue(data["ok"], data)
        claves = sorted(int(r[0]) for r in data["rows"])
        self.assertEqual(claves, [1002, 1003])

        # insertar varias para forzar aux_usage / reconstrucción
        for k in range(2000, 2015):
            sql = f"INSERT INTO seq_kv VALUES({k}, 'X{k}');"
            rr = run_sql(self.client, sql)
            self.assertEqual(rr.status_code, 200, rr.text)

        # buscamos algo insertado tarde
        r = run_sql(self.client,
                    "SELECT * FROM seq_kv WHERE Clave = 2005;")
        data = r.json()
        self.assertTrue(data["ok"], data)
        self.assertTrue(any(int(row[0]) == 2005 for row in data["rows"]))

        # delete via engine (reconstruye tabla + índices)
        r = run_sql(self.client,
                    "DELETE FROM seq_kv WHERE Clave = 1002;")
        self.assertEqual(r.status_code, 200, r.text)

        r = run_sql(self.client,
                    "SELECT * FROM seq_kv WHERE Clave = 1002;")
        data = r.json()
        self.assertTrue(data["ok"], data)
        self.assertEqual(data["rows"], [])

    # ---------- RTREE 1D: columna Coord "(x,y)" + radius ----------

    def test_rtree_coord_radius(self):
        csv_text = 'Id,Coord\n1,"(10,10)"\n2,"(15,15)"\n3,"(30,30)"\n'
        up = upload_csv(self.client, "rt_coord", csv_text)
        self.assertTrue(up["ok"])

        r = run_sql(self.client,
                    "CREATE INDEX idx_rt_coord ON rt_coord(Coord) TYPE RTREE;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["ok"])
        assert_schema_has_index(self.client, "rt_coord", "Coord", "RTREE")

        # punto cerca de 1 y 2
        q = "SELECT * FROM rt_coord WHERE Coord IN ((12,12), 10);"
        r = run_sql(self.client, q)
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["ok"], data)
        ids = {int(row[0]) for row in data["rows"]}
        self.assertEqual(ids, {1, 2})

        # no-hit
        q = "SELECT * FROM rt_coord WHERE Coord IN ((0,0), 1);"
        r = run_sql(self.client, q)
        data = r.json()
        self.assertTrue(data["ok"], data)
        self.assertEqual(data["rows"], [])

    # ---------- RTREE 2D: lat/lon sintético ----------

    def test_rtree_cities_2d(self):
        csv_text = """id,name,lat,lon
1,A,36.68,71.53
2,B,36.681,71.531
3,C,40.0,10.0
"""
        up = upload_csv(self.client, "rt_cities", csv_text)
        self.assertTrue(up["ok"])

        r = run_sql(self.client,
                    "CREATE INDEX idx_cities_geo ON rt_cities(lat, lon) TYPE RTREE;")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["ok"])
        assert_schema_has_index(self.client, "rt_cities", "lat,lon", "RTREE")

        # cerca de (36.68,71.53) con radio pequeño
        q = "SELECT * FROM rt_cities WHERE lat, lon IN ((36.68,71.53), 0.01);"
        r = run_sql(self.client, q)
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["ok"], data)
        ids = {int(row[0]) for row in data["rows"]}
        self.assertEqual(ids, {1, 2})

        # radio aún más chico → solo el 1
        q = "SELECT * FROM rt_cities WHERE lat, lon IN ((36.68,71.53), 0.0001);"
        r = run_sql(self.client, q)
        data = r.json()
        self.assertTrue(data["ok"], data)
        ids = {int(row[0]) for row in data["rows"]}
        self.assertEqual(ids, {1})

    # ---------- mezcla de índices en una misma tabla ----------

    def test_multi_index_same_table(self):
        csv_text = "Id,Nombre,Edad\n1,Ana,30\n2,Luis,25\n3,Ana,40\n"
        up = upload_csv(self.client, "multi_idx", csv_text)
        self.assertTrue(up["ok"])

        # BTREE en Edad + HASH en Nombre
        r1 = run_sql(self.client,
                     "CREATE INDEX idx_m_bt ON multi_idx(Edad) TYPE BTREE;")
        r2 = run_sql(self.client,
                     "CREATE INDEX idx_m_h ON multi_idx(Nombre) TYPE HASH;")
        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertTrue(r1.json()["ok"])
        self.assertTrue(r2.json()["ok"])

        meta = get_schema(self.client, "multi_idx")
        idxs = meta.get("indexes", [])
        self.assertIn(["Edad", "BTREE"], idxs)
        self.assertIn(["Nombre", "HASH"], idxs)

        # consulta por Nombre (HASH)
        r = run_sql(self.client,
                    "SELECT * FROM multi_idx WHERE Nombre = 'Ana';")
        data = r.json()
        self.assertTrue(data["ok"], data)
        self.assertTrue(any(row[0] == 1 for row in data["rows"]))
        self.assertTrue(any(row[0] == 3 for row in data["rows"]))

        # consulta por Edad (BTREE)
        r = run_sql(self.client,
                    "SELECT * FROM multi_idx WHERE Edad BETWEEN 26 AND 35;")
        data = r.json()
        self.assertTrue(data["ok"], data)
        ids = {int(row[0]) for row in data["rows"]}
        self.assertEqual(ids, {1})


# ---------- tests unitarios RTREE directo (opcional pero sano) ----------

from src.indices.rtree.rtreeindex import RTreeIndex


class RTreeIndexUnit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.idx = RTreeIndex("t_rt", "Coord", data_dir=self.tmp.name)

    def tearDown(self):
        try:
            if hasattr(self.idx, "idx"):
                self.idx.idx.close()
        except Exception:
            pass
        self.tmp.cleanup()

    def test_radius_search(self):
        self.idx.add((0.0, 0.0), 1)
        self.idx.add((5.0, 0.0), 2)
        self.idx.add((100.0, 100.0), 3)

        res = self.idx.radius_search((0.0, 0.0), 10.0)
        self.assertEqual(set(res), {1, 2})

    def test_knn_search(self):
        self.idx.add((0.0, 0.0), 10)
        self.idx.add((1.0, 1.0), 20)
        self.idx.add((10.0, 10.0), 30)

        res = self.idx.knn_search((0.0, 0.0), 2)
        self.assertEqual(set(res), {10, 20})


if __name__ == "__main__":
    unittest.main(verbosity=2)
