# parser_sql.py
from lark import Lark, Transformer

grammar = r"""
?start: statement

?statement: create_table | create_table_from_file | insert | delete | select

create_table: "CREATE" "TABLE" NAME "(" column_def ("," column_def)* ")" ";"
column_def: NAME type_spec ["KEY"] ["INDEX" index_type]
type_spec: "INT" | "DATE" | "VARCHAR" "[" INT "]" | "ARRAY" "[" "FLOAT" "]"
index_type: "SEQ" | "BTree" | "RTree" | "EHash" | "ISAM"

create_table_from_file: "CREATE" "TABLE" NAME "FROM" "FILE" ESCAPED_STRING "USING" "INDEX" NAME "(" ESCAPED_STRING ")" ";"

insert: "INSERT" "INTO" NAME "VALUES" "(" value_list ")" ";"
value_list: value ("," value)*
?value: INT -> int
      | ESCAPED_STRING -> string
      | array

array: "[" float_list "]"
float_list: FLOAT ("," FLOAT)*

delete: "DELETE" "FROM" NAME "WHERE" condition ";"
select: "SELECT" "*" "FROM" NAME ["WHERE" condition] ";"

condition: NAME "=" value
         | NAME "BETWEEN" value "AND" value
         | NAME "IN" "(" array "," FLOAT ")"  // Para RTree

%import common.CNAME -> NAME
%import common.INT
%import common.FLOAT
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', lexer='standard')