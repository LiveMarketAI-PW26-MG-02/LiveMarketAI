from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.media_clip import media_clip_repository as repo
from ...schemas.media_clip import MediaClipCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/media_clip/batch")
def ingest_batch(rows: list[MediaClipCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "MediaClip"}
