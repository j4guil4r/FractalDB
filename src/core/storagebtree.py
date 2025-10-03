import pickle
import struct
import os

class Storage:
    """
    Archivo de datos para almacenar tuplas completas.
    """
    def __init__(self, filename):
        self.filename = filename
        open(self.filename, 'ab').close()  # Crear si no existe

    def insert(self, row):
        """Guarda una tupla completa y devuelve el offset en el archivo"""
        with open(self.filename, 'ab') as f:
            offset = f.tell()
            data = pickle.dumps(row)
            f.write(struct.pack('I', len(data)))  # Tamaño en 4 bytes
            f.write(data)
            return offset

    def get(self, offset):
        """Recupera la tupla completa desde el offset"""
        with open(self.filename, 'rb') as f:
            f.seek(offset)
            size_bytes = f.read(4)
            if not size_bytes:
                return None
            size = struct.unpack('I', size_bytes)[0]
            data = f.read(size)
            return pickle.loads(data)
