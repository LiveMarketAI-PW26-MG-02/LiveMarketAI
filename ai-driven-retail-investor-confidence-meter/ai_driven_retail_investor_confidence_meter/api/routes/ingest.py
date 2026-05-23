from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.sentiment_signal import sentiment_signal_repository as repo
from ...schemas.sentiment_signal import SentimentSignalCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/sentiment_signal/batch")
def ingest_batch(rows: list[SentimentSignalCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "SentimentSignal"}
