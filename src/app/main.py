# src/app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import tempfile, os
import re, time

from src.parser.sqlparser import SQLParser
from .engine import get_engine

app = FastAPI(title="FractalDB")
class SQLPayload(BaseModel):
    query: str

_parser = SQLParser()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _adapt_plan_for_engine(plan: dict) -> dict:
    cmd = plan.get("command", "").upper()
    if cmd == "CREATE_TABLE":
        table = plan["table_name"]
        schema = plan.get("schema", [])
        index_defs = {k: v for k, v in plan.get("index_definitions", {}).items()}
        columns = []
        for col_name, col_type, length in schema:
            t = col_type.upper()
            if t == "VARCHAR":
                type_str = f"VARCHAR[{length or 1}]"
            else:
                type_str = t
            col_def = {"name": col_name, "type": type_str}
            if col_name in index_defs:
                cls = index_defs[col_name].upper()
                if "BTREE" in cls or "BPLUSTREE" in cls: col_def["index"] = "BTREE"
                elif "HASH" in cls: col_def["index"] = "HASH"
                elif "ISAM" in cls: col_def["index"] = "ISAM"
                elif "SEQ" in cls: col_def["index"] = "SEQ"
                elif "RTREE" in cls: col_def["index"] = "RTREE"
            columns.append(col_def)
        return {"action": "create_table", "table": table, "columns": columns}

    if cmd == "CREATE_INDEX":
        index_name = plan.get("index_name")
        table_name = plan.get("table_name")
        column_name = plan.get("column_name")
        index_type = plan.get("index_type")

        if isinstance(column_name, list):
            columns = column_name
        else:
            columns = [column_name]
        return {
            "action": "create_index",
            "index_name": index_name,
            "table": table_name,
            "column": columns, # 'column' es ahora una lista
            "index_type": index_type
        }
    
    # --- NUEVO: Adaptador para CREATE_FTS_INDEX ---
    if cmd == "CREATE_FTS_INDEX":
        return {
            "action": "create_fts_index",
            "table": plan.get("table_name"),
            "columns": plan.get("columns", [])
        }
    
    if cmd == "INSERT":
        return {"action": "insert", "table": plan["table_name"], "values": plan.get("values", [])}

    if cmd == "SELECT":
        cond = None
        w = plan.get("where")
        if w:
            op = w.get("op")
            col = w.get("column")
            if op == "=":
                cond = {"op": "=", "field": col, "value": w.get("value")}
            elif op == "BETWEEN":
                cond = {"op": "BETWEEN", "field": col, "low": w.get("value1"), "high": w.get("value2")}
            elif op == "IN2":
                cols = w.get("columns", [])
                cond = {
                    "op": "IN2",
                    "fields": cols,
                    "coords": tuple(w.get("point", ())),
                    "radius": float(w.get("radius", 0)),
                }
            elif op == "IN":
                cond = {"op": "IN", "field": col, "coords": tuple(w.get("point", ())), "radius": float(w.get("radius", 0))}
            # --- NUEVO: Adaptador para FTS (@@) ---
            elif op == "FTS":
                cond = {"op": "FTS", "field": col, "query_text": w.get("query_text")}
                
        return {
            "action": "select", 
            "table": plan["table_name"], 
            "columns": ["*"], 
            "condition": cond,
            "limit": plan.get("limit", 100) # <-- MODIFICADO: Pasar el limit
        }

    if cmd == "DELETE":
        w = plan.get("where")
        if not w or w.get("op") != "=":
            raise ValueError("DELETE solo soporta igualdad.")
        return {"action": "delete", "table": plan["table_name"], "condition": {"op": "=", "field": w["column"], "value": w["value"]}}

    if cmd == "CREATE_TABLE_FROM_FILE":
        raise ValueError("CREATE TABLE ... FROM FILE no está cableado aún para el engine.")

    raise ValueError(f"Comando no soportado: {cmd}")

# --- Split por ';' ignorando lo que esté dentro de comillas ---
_STMT_SPLIT = re.compile(r';\s*(?=(?:[^"\']|"[^"]*"|\'[^\']*\')*$)')


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/sql")
async def run_sql(payload: SQLPayload):
    q = payload.query.strip()
    try:
        start_time = time.time()
        statements = [s.strip() for s in _STMT_SPLIT.split(q) if s.strip()]
        results = []
        for s in statements:
            plan_parser = _parser.parse(s)
            plan_engine = _adapt_plan_for_engine(plan_parser)
            results.append({"sql": s, "result": get_engine().execute(plan_engine)})

        end_time = time.time()
        execution_time = end_time - start_time

        response = {
            "results": results[0]["result"] if len(results) == 1 else {"ok": True, "results": results},
            "execution_time": execution_time
        }
        return response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Error al procesar SQL: {ve}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar SQL: {e}")

@app.post("/api/upload")
async def upload_csv(
    table: str = Form(...),
    file: UploadFile = File(...),
    has_header: bool = Form(True),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Solo se aceptan CSV")

    # 1) Guardar a un archivo temporal sin cargar todo a RAM
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            while True:
                chunk = await file.read(1_048_576)  # 1 MB
                if not chunk:
                    break
                tmp.write(chunk)
        tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(400, f"Error guardando archivo: {e}")

    # 2) Cargar por ruta con inserción en streaming y reconstrucción de índices al final
    try:
        inserted, columns = get_engine().load_csv_path(
            table=table,
            csv_path=tmp_path,
            has_header=has_header,
        )
        return {"ok": True, "table": table, "inserted": inserted, "columns": columns}
    except Exception as e:
        raise HTTPException(400, f"Error al cargar CSV: {e}")
    finally:
        try: os.remove(tmp_path)
        except: pass

@app.get("/api/tables")
async def list_tables():
    data_dir = get_engine().data_dir
    names = [fn[:-5] for fn in os.listdir(data_dir) if fn.endswith(".meta")]
    return {"tables": sorted(names)}

@app.get("/api/tables/{name}/schema")
async def table_schema(name: str):
    try:
        t = get_engine()._get_table(name)
        return {"table": name, "schema": t.schema, "indexes": getattr(t, "index_specs", [])}
    except Exception as e:
        raise HTTPException(404, str(e))