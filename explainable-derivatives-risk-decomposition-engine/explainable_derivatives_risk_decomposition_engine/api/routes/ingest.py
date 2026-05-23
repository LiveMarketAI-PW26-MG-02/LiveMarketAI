from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.derivative import derivative_repository as repo
from ...schemas.derivative import DerivativeCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/derivative/batch")
def ingest_batch(rows: list[DerivativeCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "Derivative"}
