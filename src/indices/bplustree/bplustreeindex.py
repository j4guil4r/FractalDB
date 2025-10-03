from .bplustree import BPlusTree

class BPlusTreeIndex:
    def __init__(self, order, storage):
        self.tree = BPlusTree(order)
        self.storage = storage

    def add_record(self, key, row):
        offset = self.storage.insert(row)
        self.tree.insert((key, offset))
        return offset

    def search(self, key):
        result = self.tree.search((key, 0))
        if result is None:
            return None
        _, offset = result
        return self.storage.get(offset)

    def range_search(self, start_key, end_key):
        leaves = self.tree.range_search((start_key, 0), (end_key, float('inf')))
        return [self.storage.get(offset) for key, offset in leaves]

    def remove_record(self, key):
        return self.tree.remove_key((key, 0))
