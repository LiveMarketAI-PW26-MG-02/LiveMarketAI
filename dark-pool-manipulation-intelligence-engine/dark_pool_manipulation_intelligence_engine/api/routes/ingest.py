from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.dark_pool_print import dark_pool_print_repository as repo
from ...schemas.dark_pool_print import DarkPoolPrintCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/dark_pool_print/batch")
def ingest_batch(rows: list[DarkPoolPrintCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "DarkPoolPrint"}
