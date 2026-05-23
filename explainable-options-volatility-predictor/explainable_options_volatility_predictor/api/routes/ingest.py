from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.option import option_repository as repo
from ...schemas.option import OptionCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/option/batch")
def ingest_batch(rows: list[OptionCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "Option"}
