from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.sector import sector_repository as repo
from ...schemas.sector import SectorCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/sector/batch")
def ingest_batch(rows: list[SectorCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "Sector"}
