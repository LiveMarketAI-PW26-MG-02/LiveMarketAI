from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.indicator import indicator_repository as repo
from ...schemas.indicator import IndicatorCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/indicator/batch")
def ingest_batch(rows: list[IndicatorCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "Indicator"}
