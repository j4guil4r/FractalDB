from lark import Lark

grammar = r"""
?start: statement
?statement: create_table | create_table_from_file | insert | delete | select
create_table: "CREATE" "TABLE" NAME "(" column_def ("," column_def)* ")" ";"
column_def: NAME type_spec [KEY] index_spec?
index_spec: "INDEX" (INDEX_TYPE | NAME) -> index_spec
type_spec: "INT" -> int_type
            | "DATE" -> date_type
            | "VARCHAR" "[" INT "]" -> varchar_type
            | "ARRAY" "[" "FLOAT" "]" -> array_type
INDEX_TYPE: "SEQ" | "BTREE" | "RTREE" | "EHASH" | "ISAM"
create_table_from_file: "CREATE" "TABLE" NAME "FROM" "FILE" STRING "USING" "INDEX" (INDEX_TYPE | NAME) "(" (STRING | NAME) ")" ";"
insert: "INSERT" "INTO" NAME "VALUES" "(" value_list ")" ";"
value_list: value ("," value)*
?value: INT -> int | FLOAT -> float | STRING -> string | array
array: "[" number_list "]"
number_list: number ("," number)*
?number: INT | FLOAT
delete: "DELETE" "FROM" NAME "WHERE" condition ";"
select: "SELECT" (select_all | column_list) "FROM" NAME ["WHERE" condition] ";"
select_all: "*"
column_list: NAME ("," NAME)*

condition: NAME "=" value | NAME "BETWEEN" value "AND" value | NAME "IN" "(" array "," FLOAT ")" // Para RTree
KEY: "KEY"
STRING: /"[^"]*"|'[^']*'/
INT: /-?\d+/
FLOAT: /-?\d+\.\d+/
%import common.CNAME -> NAME
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', lexer='standard')
