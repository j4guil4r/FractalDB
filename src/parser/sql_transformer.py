from lark import Transformer

class SQLTransformer(Transformer):
    def create_table(self, items):
        table_name = str(items[0])
        columns = items[1:]
        return {"action": "create_table", "table": table_name, "columns": columns}

    def column_def(self, items):
        col_name = str(items[0])
        type_spec = str(items[1])
        key = "KEY" in items
        index = None
        for i in items:
            if isinstance(i, str) and i in ["SEQ", "BTree", "RTree", "EHash", "ISAM"]:
                index = i
        return {"name": col_name, "type": type_spec, "key": key, "index": index}

    def insert(self, items):
        table_name = str(items[0])
        values = items[1:]
        return {"action": "insert", "table": table_name, "values": values}

    def delete(self, items):
        table_name = str(items[0])
        condition = items[1]
        return {"action": "delete", "table": table_name, "condition": condition}

    def select(self, items):
        table_name = str(items[0])
        condition = items[1] if len(items) > 1 else None
        return {"action": "select", "table": table_name, "condition": condition}

    def condition(self, items):
        if len(items) == 3 and str(items[1]) == "=":
            return {"field": str(items[0]), "op": "=", "value": items[2]}
        elif len(items) == 5 and str(items[1]).upper() == "BETWEEN":
            return {"field": str(items[0]), "op": "BETWEEN", "start": items[2], "end": items[4]}
        elif len(items) == 4 and str(items[1]).upper() == "IN":
            return {"field": str(items[0]), "op": "IN", "point": items[2], "radius": items[3]}

    def int(self, token):
        return int(token)

    def string(self, token):
        return token[1:-1]

    def array(self, items):
        return items

    def float_list(self, items):
        return [float(i) for i in items]