# src/indices/bplustree/bplustree.py

import pickle
import bisect
import os
from typing import List

class BPlusTreeNode:
    def __init__(self, order: int, is_leaf: bool = False):
        self.order = order
        self.is_leaf = is_leaf
        self.keys = []
        self.parent: int = -1
        self.self_offset: int = -1

    def is_full(self) -> bool:
        return len(self.keys) >= self.order - 1

class LeafNode(BPlusTreeNode):
    def __init__(self, order: int):
        super().__init__(order, is_leaf=True)
        self.values = []
        self.next_leaf: int = -1

    def add(self, key, value):
        i = bisect.bisect_left(self.keys, key)
        
        if i < len(self.keys) and self.keys[i] == key:
            self.values[i].append(value)
        else:
            self.keys.insert(i, key)
            self.values.insert(i, [value])

class InternalNode(BPlusTreeNode):
    def __init__(self, order: int):
        super().__init__(order, is_leaf=False)
        self.children: List[int] = []

    def find_child_offset(self, key) -> int:
        i = bisect.bisect_right(self.keys, key)
        return self.children[i]

class BPlusTree:
    BLOCK_SIZE = 4096

    def __init__(self, file_path_prefix: str, order: int = 3):
        if order < 3:
            raise ValueError("Order must be at least 3")
            
        self.meta_path = f"{file_path_prefix}.meta"
        self.dat_path = f"{file_path_prefix}.dat"
        self.order = order
        self.root_offset: int = -1
        self.next_available_offset: int = 0

    def _read_node(self, offset: int) -> BPlusTreeNode:
        with open(self.dat_path, 'rb') as f:
            f.seek(offset)
            padded_data = f.read(self.BLOCK_SIZE)
        
        data = padded_data.rstrip(b'\0')
        if not data:
            raise IOError(f"No se pudo leer el nodo en el offset {offset}")
        
        return pickle.loads(data)

    def _write_node(self, node: BPlusTreeNode):
        if node.self_offset == -1:
            raise ValueError("No se puede escribir un nodo sin un 'self_offset' asignado.")
            
        data = pickle.dumps(node)
        if len(data) > self.BLOCK_SIZE:
            raise ValueError("El nodo es demasiado grande para el BLOCK_SIZE.")
        
        padded_data = data.ljust(self.BLOCK_SIZE, b'\0')
        
        with open(self.dat_path, 'r+b') as f:
            f.seek(node.self_offset)
            f.write(padded_data)

    def _get_new_offset(self) -> int:
        offset = self.next_available_offset
        self.next_available_offset += self.BLOCK_SIZE
        return offset

    def save_meta(self):
        metadata = {
            'root_offset': self.root_offset,
            'order': self.order,
            'next_available_offset': self.next_available_offset
        }
        with open(self.meta_path, 'wb') as f:
            pickle.dump(metadata, f)

    def _load_meta(self):
        with open(self.meta_path, 'rb') as f:
            metadata = pickle.load(f)
        self.root_offset = metadata['root_offset']
        self.order = metadata['order']
        self.next_available_offset = metadata['next_available_offset']

    def _initialize_new_tree(self):
        open(self.dat_path, 'wb').close()
        
        root = LeafNode(self.order)
        root.self_offset = self._get_new_offset()
        self.root_offset = root.self_offset
        
        self._write_node(root)
        self.save_meta()

    @staticmethod
    def load(file_path_prefix: str, order: int):
        tree = BPlusTree(file_path_prefix, order)
        meta_path = f"{file_path_prefix}.meta"
        
        if os.path.exists(meta_path):
            try:
                tree._load_meta()
            except (EOFError, pickle.UnpicklingError):
                tree._initialize_new_tree()
        else:
            tree._initialize_new_tree()
            
        return tree

    def _find_leaf(self, key) -> LeafNode:
        node_offset = self.root_offset
        node = self._read_node(node_offset)
        
        while not node.is_leaf:
            node_offset = node.find_child_offset(key)
            node = self._read_node(node_offset)
            
        return node

    def insert(self, key, value):
        leaf_node = self._find_leaf(key)
        leaf_node.add(key, value)
        
        if leaf_node.is_full():
            self._split_leaf(leaf_node)
        else:
            self._write_node(leaf_node)

    def _split_leaf(self, leaf: LeafNode):
        mid = len(leaf.keys) // 2
        
        new_leaf = LeafNode(self.order)
        new_leaf.self_offset = self._get_new_offset()
        new_leaf.parent = leaf.parent
        
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.values = leaf.values[mid:]
        
        new_leaf.next_leaf = leaf.next_leaf
        leaf.next_leaf = new_leaf.self_offset

        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        
        self._write_node(leaf)
        self._write_node(new_leaf)
        
        promoted_key = new_leaf.keys[0]
        self._insert_in_parent(leaf.parent, promoted_key, leaf.self_offset, new_leaf.self_offset)

    def _insert_in_parent(self, parent_offset: int, key, left_child_offset: int, right_child_offset: int):
        
        if parent_offset == -1:
            new_root = InternalNode(self.order)
            new_root.self_offset = self._get_new_offset()
            new_root.keys = [key]
            new_root.children = [left_child_offset, right_child_offset]
            
            self.root_offset = new_root.self_offset
            
            left_child = self._read_node(left_child_offset)
            right_child = self._read_node(right_child_offset)
            left_child.parent = new_root.self_offset
            right_child.parent = new_root.self_offset
            self._write_node(left_child)
            self._write_node(right_child)
            
            self._write_node(new_root)
            self.save_meta()
            return

        parent_node = self._read_node(parent_offset)
        
        i = bisect.bisect_right(parent_node.keys, key)
        parent_node.keys.insert(i, key)
        parent_node.children.insert(i + 1, right_child_offset)
        
        right_child = self._read_node(right_child_offset)
        right_child.parent = parent_offset
        self._write_node(right_child)
        
        if parent_node.is_full():
            self._split_internal_node(parent_node)
        else:
            self._write_node(parent_node)

    def _split_internal_node(self, node: InternalNode):
        mid = len(node.keys) // 2
        
        promoted_key = node.keys[mid]

        new_node = InternalNode(self.order)
        new_node.self_offset = self._get_new_offset()
        new_node.parent = node.parent

        new_node.keys = node.keys[mid+1:]
        new_node.children = node.children[mid+1:]
        
        node.keys = node.keys[:mid]
        node.children = node.children[:mid+1]
        
        for child_offset in new_node.children:
            child = self._read_node(child_offset)
            child.parent = new_node.self_offset
            self._write_node(child)

        self._write_node(node)
        self._write_node(new_node)
        
        self._insert_in_parent(node.parent, promoted_key, node.self_offset, new_node.self_offset)

    def search(self, key) -> list:
        leaf = self._find_leaf(key)
        try:
            i = leaf.keys.index(key)
            return leaf.values[i]
        except ValueError:
            return []

    def range_search(self, start_key, end_key) -> list:
        results = []
        leaf_node = self._find_leaf(start_key)
        
        while leaf_node:
            for i, key in enumerate(leaf_node.keys):
                if start_key <= key <= end_key:
                    results.extend(leaf_node.values[i])
                elif key > end_key:
                    return results
            
            if leaf_node.next_leaf != -1:
                leaf_node = self._read_node(leaf_node.next_leaf)
            else:
                leaf_node = None
                
        return results

    def remove(self, key, value=None):
        leaf = self._find_leaf(key)
        
        try:
            i = leaf.keys.index(key)
            if value:
                if value in leaf.values[i]:
                    leaf.values[i].remove(value)
                    if not leaf.values[i]:
                        leaf.keys.pop(i)
                        leaf.values.pop(i)
                else:
                    return
            else:
                leaf.keys.pop(i)
                leaf.values.pop(i)
            
            self._write_node(leaf)

        except (ValueError, IndexError):
            pass