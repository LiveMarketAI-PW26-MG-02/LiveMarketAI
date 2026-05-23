from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.e_t_f import e_t_f_repository as repo
from ...schemas.e_t_f import ETFCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/e_t_f/batch")
def ingest_batch(rows: list[ETFCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "ETF"}
