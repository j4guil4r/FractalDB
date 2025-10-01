class Node:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []
        self.next = None
    
    def add_key(self, key, child=None):
        i = 0
        while i < len(self.keys) and self.keys[i] < key:
            i += 1
        self.keys.insert(i, key)
        
        if not self.is_leaf and child is not None:
            self.children.insert(i + 1, child)

    def remove_key(self, key):
        if key in self.keys:
            index = self.keys.index(key)
            self.keys.remove(key)
            if not self.is_leaf and index < len(self.children):
                self.children.pop(index + 1)
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

    def range_search(self, start_key, end_key):
        leaf = self._find_leaf(start_key)
        results = []
        current = leaf
        while current is not None:
            for key in current.keys:
                if start_key <= key <= end_key:
                    results.append(key)
                elif key > end_key:
                    return results
            current = current.next
        return results

    def _find_leaf(self, key):
        node = self.root
        while not node.is_leaf:
            # Encontrar el hijo
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        return node

    def insert(self, key):
        root = self.root
        # Si la raiz esta llena, hacer split
        if len(root.keys) == self.order - 1:
            new_root = Node(False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        if node.is_leaf:
            i = 0
            while i < len(node.keys) and node.keys[i] < key:
                i += 1
            node.keys.insert(i, key)
        else:
            # Encontrar el hijo por donde bajar
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            if len(node.children[i].keys) == self.order - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, child_index):
        child = parent.children[child_index]
        new_node = Node(child.is_leaf)
        
        mid = len(child.keys) // 2
        mid_key = child.keys[mid]
        
        if child.is_leaf:
            new_node.keys = child.keys[mid:]
            child.keys = child.keys[:mid]

            new_node.next = child.next
            child.next = new_node
        else:
            # si es interno, toma llaves e hijos de la derecha
            new_node.keys = child.keys[mid + 1:]
            new_node.children = child.children[mid + 1:]
            # el original se queda con los de la izq.
            child.keys = child.keys[:mid]
            child.children = child.children[:mid + 1]
        parent.add_key(mid_key, new_node)

    def remove_key(self, key):
        return self._remove(self.root, key)

    def _remove(self, node, key):
        if node.is_leaf:
            if key in node.keys:
                node.remove_key(key)
                return True
            return False
        else:
            for i, item in enumerate(node.keys):
                if key < item:
                    return self._remove(node.children[i], key)
            return self._remove(node.children[-1], key)