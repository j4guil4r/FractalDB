from .bplustree import BPlusTree

class BPlusTreeIndex:
    def __init__(self, order):
        self.tree = BPlusTree(order)
    
    def add_record(self, key):
        self.tree.insert(key)
    
    def search(self, key):
        return self.tree.search(key)
    
    def range_search(self, start_key, end_key):
        return self.tree.range_search(start_key, end_key)

    def remove_record(self, key):
        return self.tree.remove_key(key)