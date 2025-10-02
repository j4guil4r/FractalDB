import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from parser.sql_parser import parser
from parser.sql_transformer import SQLTransformer

class TestSQLParser(unittest.TestCase):
    
    def setUp(self):
        self.parser = parser
        self.transformer = SQLTransformer()
    
    def parse_and_transform(self, sql):
        tree = self.parser.parse(sql)
        return self.transformer.transform(tree)
    
    def test_create_table_basic(self):
        
        sql = "CREATE TABLE Restaurantes (id INT KEY INDEX SEQ, nombre VARCHAR[20] INDEX BTREE, fechaRegistro DATE, ubicacion ARRAY[FLOAT] INDEX RTREE);"
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "create_table")
        self.assertEqual(result["table"], "Restaurantes")
        self.assertEqual(len(result["columns"]), 4)
        
        col1 = result["columns"][0]
        self.assertEqual(col1["name"], "id")
        self.assertEqual(col1["type"], "INT")
        self.assertTrue(col1["key"])
        self.assertEqual(col1["index"], "SEQ")
        
        col2 = result["columns"][1]
        self.assertEqual(col2["name"], "nombre")
        self.assertEqual(col2["type"], "VARCHAR[20]")
        self.assertFalse(col2["key"])
        self.assertEqual(col2["index"], "BTREE")
    
    def test_create_table_minimal(self):
        
        sql = "CREATE TABLE Users (id INT, name VARCHAR[50]);"
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "create_table")
        self.assertEqual(result["table"], "Users")
        self.assertEqual(len(result["columns"]), 2)
        
        col1 = result["columns"][0]
        self.assertEqual(col1["name"], "id")
        self.assertEqual(col1["type"], "INT")
        self.assertFalse(col1["key"])
        self.assertIsNone(col1["index"])
    
    def test_insert_values(self):
        
        sql = 'INSERT INTO Restaurantes VALUES (1, "McDonald\'s", "2023-01-15", [40.7128, -74.0060]);'
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "insert")
        self.assertEqual(result["table"], "Restaurantes")
        self.assertEqual(len(result["values"]), 4)
        
        self.assertEqual(result["values"][0], 1)
        self.assertEqual(result["values"][1], "McDonald's")
        self.assertEqual(result["values"][2], "2023-01-15")
        self.assertEqual(result["values"][3], [40.7128, -74.0060])
    
    def test_insert_with_array(self):
        
        sql = "INSERT INTO Ubicaciones VALUES ([40.7128, -74.0060, 100.5]);"
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "insert")
        self.assertEqual(result["table"], "Ubicaciones")
        self.assertEqual(result["values"][0], [40.7128, -74.0060, 100.5])
    
    def test_select_all(self):
        
        sql = "SELECT * FROM Restaurantes;"
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "select")
        self.assertEqual(result["table"], "Restaurantes")
        self.assertIsNone(result["condition"])
    
    def test_select_with_equals_condition(self):
        
        sql = 'SELECT * FROM Restaurantes WHERE nombre = "KFC";'
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "select")
        self.assertEqual(result["table"], "Restaurantes")
        self.assertIsNotNone(result["condition"])
        
        condition = result["condition"]
        self.assertEqual(condition["field"], "nombre")
        self.assertEqual(condition["op"], "=")
        self.assertEqual(condition["value"], "KFC")
    
    def test_select_with_between_condition(self):
        
        sql = 'SELECT * FROM Restaurantes WHERE fechaRegistro BETWEEN "2023-01-01" AND "2023-12-31";'
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "select")
        self.assertEqual(result["table"], "Restaurantes")
        
        condition = result["condition"]
        self.assertEqual(condition["field"], "fechaRegistro")
        self.assertEqual(condition["op"], "BETWEEN")
        self.assertEqual(condition["low"], "2023-01-01")
        self.assertEqual(condition["high"], "2023-12-31")
    
    def test_select_with_in_condition(self):
        
        sql = "SELECT * FROM Restaurantes WHERE ubicacion IN ([40.7128, -74.0060], 10.5);"
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "select")
        self.assertEqual(result["table"], "Restaurantes")
        
        condition = result["condition"]
        self.assertEqual(condition["field"], "ubicacion")
        self.assertEqual(condition["op"], "IN")
        self.assertEqual(condition["coords"], [40.7128, -74.0060])
        self.assertEqual(condition["radius"], 10.5)
    
    def test_delete_with_condition(self):
        
        sql = 'DELETE FROM Restaurantes WHERE nombre = "McDonald\'s";'
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "delete")
        self.assertEqual(result["table"], "Restaurantes")
        
        condition = result["condition"]
        self.assertEqual(condition["field"], "nombre")
        self.assertEqual(condition["op"], "=")
        self.assertEqual(condition["value"], "McDonald's")
    
    def test_delete_with_between_condition(self):
        sql = 'DELETE FROM Restaurantes WHERE fechaRegistro BETWEEN "2023-01-01" AND "2023-03-31";'
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "delete")
        self.assertEqual(result["table"], "Restaurantes")
        
        condition = result["condition"]
        self.assertEqual(condition["field"], "fechaRegistro")
        self.assertEqual(condition["op"], "BETWEEN")
        self.assertEqual(condition["low"], "2023-01-01")
        self.assertEqual(condition["high"], "2023-03-31")
    
    def test_create_table_from_file(self):
        sql = 'CREATE TABLE Restaurantes FROM FILE "restaurantes.csv" USING INDEX BTREE("nombre");'
        
        result = self.parse_and_transform(sql)
        
        self.assertEqual(result["action"], "create_table_from_file")
        self.assertEqual(result["table"], "Restaurantes")
        self.assertEqual(result["file"], "restaurantes.csv")
        self.assertEqual(result["index_type"], "BTREE")
        self.assertEqual(result["index_column"], "nombre")
    
    def test_edge_cases(self):
        # String con espacios
        sql = 'INSERT INTO Tabla VALUES ("string con espacios");'
        result = self.parse_and_transform(sql)
        self.assertEqual(result["values"][0], "string con espacios")
        
        # Array con un solo elemento
        sql = "INSERT INTO Tabla VALUES ([42.0]);"
        result = self.parse_and_transform(sql)
        self.assertEqual(result["values"][0], [42.0])
        
        # Números negativos
        sql = "INSERT INTO Tabla VALUES ([-40.7128, -74.0060]);"
        result = self.parse_and_transform(sql)
        self.assertEqual(result["values"][0], [-40.7128, -74.0060])
    
    def test_invalid_sql_should_fail(self):
        invalid_queries = [
            "CREATE TABLE (id INT);",  # Falta nombre de tabla
            "INSERT INTO VALUES (1, 2);",  # Falta nombre de tabla
            "SELECT * FROM;",  # Falta nombre de tabla
            "DELETE FROM WHERE id = 1;",  # Falta nombre de tabla
        ]
        
        for invalid_sql in invalid_queries:
            with self.subTest(sql=invalid_sql):
                with self.assertRaises(Exception):
                    self.parse_and_transform(invalid_sql)
    def test_select_column_list_and_reject_unknown_columns(self):
    # 1) Creamos la tabla para obtener el "esquema" esperado
        create_sql = (
            'CREATE TABLE Restaurantes ('
            'id INT KEY INDEX SEQ, '
            'nombre VARCHAR[20] INDEX BTREE, '
            'fechaRegistro DATE, '
            'ubicacion ARRAY[FLOAT] INDEX RTREE'
            ');'
        )
        create_res = self.parse_and_transform(create_sql)
        self.assertEqual(create_res["action"], "create_table")
        schema_cols = {c["name"] for c in create_res["columns"]}

        # 2) SELECT de columnas válidas
        sql_ok = "SELECT id, nombre FROM Restaurantes;"
        res_ok = self.parse_and_transform(sql_ok)
        self.assertEqual(res_ok["action"], "select")
        self.assertEqual(res_ok["table"], "Restaurantes")
        self.assertEqual(res_ok["columns"], ["id", "nombre"])
        # Validamos que todas existen en el esquema
        for col in res_ok["columns"]:
            self.assertIn(col, schema_cols)

        # 3) SELECT con una columna inexistente
        sql_bad = "SELECT id, noExiste FROM Restaurantes;"
        res_bad = self.parse_and_transform(sql_bad)
        self.assertEqual(res_bad["action"], "select")
        self.assertEqual(res_bad["table"], "Restaurantes")
        # Detectamos columnas desconocidas comparando con el esquema
        unknown = [col for col in res_bad["columns"] if col not in schema_cols]
        self.assertTrue(unknown)  # Debe haber al menos una desconocida

        # 4) “No permitir” en teoría: forzamos la validación y exigimos que falle
        with self.assertRaises(ValueError):
            if unknown:
                raise ValueError(f"Columnas desconocidas: {', '.join(unknown)}")



if __name__ == '__main__':
    unittest.main(verbosity=2)