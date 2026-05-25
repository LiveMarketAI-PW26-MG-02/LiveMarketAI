from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.emotion_signal import emotion_signal_repository as repo
from ...schemas.emotion_signal import EmotionSignalCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/emotion_signal/batch")
def ingest_batch(rows: list[EmotionSignalCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "EmotionSignal"}
