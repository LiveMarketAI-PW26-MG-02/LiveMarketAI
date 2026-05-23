from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.advisor import advisor_repository as repo
from ...schemas.advisor import AdvisorCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/advisor/batch")
def ingest_batch(rows: list[AdvisorCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "Advisor"}
