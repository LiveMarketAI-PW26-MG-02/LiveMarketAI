from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.claim import claim_repository as repo
from ...schemas.claim import ClaimCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/claim/batch")
def ingest_batch(rows: list[ClaimCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "Claim"}
