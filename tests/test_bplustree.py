import unittest
from src.indices.bplustree.bplustree import BPlusTree, Node
from src.indices.bplustree.bplustreeindex import BPlusTreeIndex

class TestBPlusTreeComprehensive(unittest.TestCase):
    
    def setUp(self):
        """Este método se ejecuta antes de cada prueba"""
        self.index = BPlusTreeIndex(order=4)
    
    def test_basic_insert_search(self):
        """Prueba básica de inserción y búsqueda"""
        self.index.add_record(10)
        self.assertEqual(self.index.search(10), 10)
        self.assertIsNone(self.index.search(99))

    def test_multiple_inserts_and_searches(self):
        """Prueba con múltiples inserciones y búsquedas"""
        test_data = [10, 20, 5, 15, 25, 3, 8, 12, 18, 22]
        for key in test_data:
            self.index.add_record(key)
        
        # Verificar que todos los datos insertados se pueden encontrar
        for key in test_data:
            self.assertEqual(self.index.search(key), key)
        
        # Verificar que claves no insertadas devuelven None
        self.assertIsNone(self.index.search(99))
        self.assertIsNone(self.index.search(1))

    def test_range_search_comprehensive(self):
        """Prueba exhaustiva de búsqueda por rango"""
        # Insertar datos en orden no secuencial
        data = [50, 20, 80, 10, 30, 60, 90, 5, 15, 25, 35, 55, 65, 85, 95]
        for key in data:
            self.index.add_record(key)
        
        # Test 1: Rango que existe completamente
        result = self.index.range_search(20, 35)
        expected = [20, 25, 30, 35]
        self.assertEqual(sorted(result), expected)
        
        # Test 2: Rango que incluye todos los elementos
        result = self.index.range_search(5, 95)
        self.assertEqual(sorted(result), sorted(data))
        
        # Test 3: Rango que no existe (entre elementos)
        result = self.index.range_search(26, 29)
        self.assertEqual(result, [])
        
        # Test 4: Rango que solo incluye el límite inferior
        result = self.index.range_search(50, 50)
        self.assertEqual(result, [50])
        
        # Test 5: Rango que solo incluye el límite superior
        result = self.index.range_search(95, 95)
        self.assertEqual(result, [95])

    def test_edge_cases_range_search(self):
        """Casos edge para búsqueda por rango"""
        # Test con árbol vacío
        result = self.index.range_search(10, 20)
        self.assertEqual(result, [])
        
        # Insertar solo un elemento
        self.index.add_record(15)
        result = self.index.range_search(10, 20)
        self.assertEqual(result, [15])
        
        # Rango donde start > end (debería devolver lista vacía)
        result = self.index.range_search(20, 10)
        self.assertEqual(result, [])

    def test_tree_structure_small_order(self):
        """Prueba con orden pequeño para verificar splits"""
        small_tree = BPlusTreeIndex(order=3)  # Solo 2 keys por nodo
        
        # Insertar datos que forzarán múltiples splits
        data = [10, 20, 5, 15, 25]
        for key in data:
            small_tree.add_record(key)
        
        # Verificar que todos los datos están accesibles
        for key in data:
            self.assertEqual(small_tree.search(key), key)

    def test_tree_structure_large_order(self):
        """Prueba con orden grande"""
        large_tree = BPlusTreeIndex(order=10)
        
        # Insertar muchos datos
        data = list(range(1, 101))  # 1 to 100
        for key in data:
            large_tree.add_record(key)
        
        # Verificar búsqueda de todos los elementos
        for key in data:
            self.assertEqual(large_tree.search(key), key)
        
        # Verificar rango grande
        result = large_tree.range_search(25, 75)
        self.assertEqual(result, list(range(25, 76)))

    def test_duplicate_handling(self):
        """Prueba el comportamiento con claves duplicadas"""
        # Insertar duplicados
        self.index.add_record(10)
        self.index.add_record(10)  # Duplicado
        
        # La búsqueda debería encontrar la clave (aunque esté duplicada en la estructura)
        self.assertEqual(self.index.search(10), 10)
        
        # Range search debería incluir la clave una vez (o múltiples dependiendo de la implementación)
        result = self.index.range_search(10, 10)
        # Esto depende de si tu implementación permite duplicados o no

    def test_sequential_and_reverse_inserts(self):
        """Prueba con inserciones secuenciales y en reversa"""
        # Inserción secuencial
        sequential_data = list(range(1, 51))
        for key in sequential_data:
            self.index.add_record(key)
        
        for key in sequential_data:
            self.assertEqual(self.index.search(key), key)
        
        # Limpiar el índice
        self.index = BPlusTreeIndex(order=4)
        
        # Inserción en reversa
        reverse_data = list(range(50, 0, -1))
        for key in reverse_data:
            self.index.add_record(key)
        
        for key in sequential_data:  # Verificar en orden normal
            self.assertEqual(self.index.search(key), key)

    def test_remove_basic(self):
        """Prueba básica de eliminación"""
        self.index.add_record(10)
        self.index.add_record(20)
        self.index.add_record(30)
        
        self.assertTrue(self.index.remove_record(20))
        self.assertIsNone(self.index.search(20))
        self.assertEqual(self.index.search(10), 10)
        self.assertEqual(self.index.search(30), 30)

    def test_remove_nonexistent(self):
        """Prueba eliminación de clave que no existe"""
        self.index.add_record(10)
        self.assertFalse(self.index.remove_record(99))
        self.assertEqual(self.index.search(10), 10)

    def test_remove_and_insert_again(self):
        """Prueba eliminar y luego insertar la misma clave"""
        self.index.add_record(10)
        self.index.remove_record(10)
        self.index.add_record(10)
        self.assertEqual(self.index.search(10), 10)

    def test_negative_numbers(self):
        """Prueba con números negativos"""
        data = [-10, -5, 0, 5, 10]
        for key in data:
            self.index.add_record(key)
        
        for key in data:
            self.assertEqual(self.index.search(key), key)
        
        result = self.index.range_search(-7, 7)
        self.assertEqual(sorted(result), [-5, 0, 5])

    def test_large_range_search_after_removals(self):
        """Prueba de rango grande después de eliminaciones"""
        # Insertar datos
        data = list(range(1, 21))
        for key in data:
            self.index.add_record(key)
        
        # Eliminar algunos
        removals = [5, 10, 15]
        for key in removals:
            self.index.remove_record(key)
        
        # Range search debería excluir los eliminados
        result = self.index.range_search(1, 20)
        expected = [key for key in data if key not in removals]
        self.assertEqual(sorted(result), expected)

    def test_stress_test(self):
        """Prueba de estrés con muchos elementos"""
        large_tree = BPlusTreeIndex(order=10)
        
        # Insertar 1000 elementos
        data = list(range(1, 1001))
        for key in data:
            large_tree.add_record(key)
        
        # Verificar muestras aleatorias
        import random
        samples = random.sample(data, 50)
        for key in samples:
            self.assertEqual(large_tree.search(key), key)
        
        # Verificar rangos grandes
        result = large_tree.range_search(100, 900)
        self.assertEqual(len(result), 801)  # 100 to 900 inclusive
        self.assertEqual(result, list(range(100, 901)))

if __name__ == '__main__':
    # Ejecutar todas las pruebas
    unittest.main(verbosity=2)