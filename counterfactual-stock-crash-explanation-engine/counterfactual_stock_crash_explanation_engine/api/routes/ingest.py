from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.crash_event import crash_event_repository as repo
from ...schemas.crash_event import CrashEventCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/crash_event/batch")
def ingest_batch(rows: list[CrashEventCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "CrashEvent"}
