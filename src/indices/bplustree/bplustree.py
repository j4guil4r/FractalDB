# src/indices/bplustree/bplustree.py

import pickle
import bisect

class BPlusTreeNode:
    def __init__(self, parent=None, order=3):
        self.parent = parent
        self.keys = []
        self.order = order

    def is_full(self):
        return len(self.keys) >= self.order

class LeafNode(BPlusTreeNode):
    def __init__(self, parent=None, order=3):
        super().__init__(parent, order)
        self.values = []
        self.next_leaf = None

    def add(self, key, value):
        i = bisect.bisect_left(self.keys, key)
        
        if i < len(self.keys) and self.keys[i] == key:
            self.values[i].append(value)
        else:
            self.keys.insert(i, key)
            self.values.insert(i, [value])

class InternalNode(BPlusTreeNode):
    def __init__(self, parent=None, order=3):
        super().__init__(parent, order)
        self.children = []

    def find_child(self, key):
        i = bisect.bisect_right(self.keys, key)
        return self.children[i]

class BPlusTree:
    def __init__(self, file_path: str, order: int = 3):
        self.file_path = file_path
        self.order = order
        self.root = LeafNode(order=order)

    def _find_leaf(self, key):
        node = self.root
        while not isinstance(node, LeafNode):
            node = node.find_child(key)
        return node

    def insert(self, key, value):
        leaf = self._find_leaf(key)
        leaf.add(key, value)
        
        if leaf.is_full():
            self._split_leaf(leaf)

    def _split_leaf(self, leaf):
        mid = len(leaf.keys) // 2
        
        new_leaf = LeafNode(parent=leaf.parent, order=self.order)
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.values = leaf.values[mid:]
        
        new_leaf.next_leaf = leaf.next_leaf
        leaf.next_leaf = new_leaf

        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        
        self._insert_in_parent(leaf, new_leaf.keys[0], new_leaf)

    def _insert_in_parent(self, left_child, key, right_child):
        parent = left_child.parent
        if parent is None:
            self.root = InternalNode(order=self.order)
            self.root.keys.append(key)
            self.root.children.extend([left_child, right_child])
            left_child.parent = self.root
            right_child.parent = self.root
            return

        i = bisect.bisect_right(parent.keys, key)
        parent.keys.insert(i, key)
        parent.children.insert(i + 1, right_child)
        right_child.parent = parent
        
        if parent.is_full():
            self._split_internal_node(parent)

    def _split_internal_node(self, node):
        mid = len(node.keys) // 2
        
        promoted_key = node.keys[mid]

        new_node = InternalNode(parent=node.parent, order=self.order)
        new_node.keys = node.keys[mid+1:]
        new_node.children = node.children[mid+1:]
        
        for child in new_node.children:
            child.parent = new_node

        node.keys = node.keys[:mid]
        node.children = node.children[:mid+1]
        
        self._insert_in_parent(node, promoted_key, new_node)

    def search(self, key):
        leaf = self._find_leaf(key)
        try:
            i = leaf.keys.index(key)
            return leaf.values[i]
        except ValueError:
            return []

    def range_search(self, start_key, end_key):
        results = []
        leaf = self._find_leaf(start_key)
        
        while leaf:
            for i, key in enumerate(leaf.keys):
                if start_key <= key <= end_key:
                    results.extend(leaf.values[i])
                elif key > end_key:
                    return results
            leaf = leaf.next_leaf
        return results

    def remove(self, key, value=None):
        leaf = self._find_leaf(key)
        try:
            i = leaf.keys.index(key)
            if value:
                leaf.values[i].remove(value)
                if not leaf.values[i]:
                    leaf.keys.pop(i)
                    leaf.values.pop(i)
            else:
                leaf.keys.pop(i)
                leaf.values.pop(i)
        except (ValueError, IndexError):
            pass
            
    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str, order: int):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return BPlusTree(file_path, order)