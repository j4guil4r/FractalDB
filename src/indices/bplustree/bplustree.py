class Node:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []
    
    def add_key(self, key, child=None):
        self.keys.append(key)
        if not self.is_leaf:
            self.children.append(child)

    def remove_key(self, key):
        if key in self.keys:
            self.keys.remove(key)
            return True
        return False

class BPlusTree:
    def __init__(self, order):
        self.root = Node(True)
        self.order = order

    def search(self, key):
        return self._search(self.root, key)
    
    def _search(self, node, key):
        if node.is_leaf:
            for i, item in enumerate(node.keys):
                if item == key:
                    return item
            return None
        else:
            for i, item in enumerate(node.keys):
                if key < item:
                    return self._search(node.children[i], key)
            return self._search(node.children[-1], key)

    def range_search(self, start_key, end_key):
        return self._range_search(self.root, start_key, end_key)

    def _range_search(self, node, start_key, end_key):
        if node.is_leaf:
            results = []
            for key in node.keys:
                if start_key <= key <= end_key:
                    results.append(key)
            return results
        else:
            for i, item in enumerate(node.keys):
                if start_key < item:
                    return self._range_search(node.children[i], start_key, end_key)
            return self._range_search(node.children[-1], start_key, end_key)

    def insert(self, key):
        root = self.root
        if len(root.keys) == self.order - 1:
            new_node = Node(False)
            new_node.add_key(None, self.root)
            self._split(new_node, 0)
            self.root = new_node
        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        if node.is_leaf:
            node.add_key(key)
            node.keys.sort()
        else:
            for i, item in enumerate(node.keys):
                if key < item:
                    self._insert_non_full(node.children[i], key)
                    return
            self._insert_non_full(node.children[-1], key)

    def _split(self, parent, index):
        node = parent.children[index]
        new_node = Node(node.is_leaf)
        parent.add_key(node.keys[self.order // 2], new_node)
        new_node.keys = node.keys[self.order // 2:]
        node.keys = node.keys[:self.order // 2]
        if not node.is_leaf:
            new_node.children = node.children[self.order // 2:]
            node.children = node.children[:self.order // 2]

