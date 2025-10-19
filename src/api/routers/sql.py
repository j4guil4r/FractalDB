# src/api/sql.py

from fastapi import APIRouter, HTTPException
from ..schemas import SQLRequest, SQLResponse
from ...parser.sqlparser import SQLParser
from ...app.engine import get_engine

router = APIRouter()
_parser = SQLParser()

@router.post("/sql", response_model=SQLResponse)
def run_sql(req: SQLRequest):
    try:
        plan = _parser.parse(req.sql)          
        result = get_engine().execute(plan)    
        return SQLResponse(**result, ok=True)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
