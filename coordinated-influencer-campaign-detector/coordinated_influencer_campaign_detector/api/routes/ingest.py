from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.influencer import influencer_repository as repo
from ...schemas.influencer import InfluencerCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/influencer/batch")
def ingest_batch(rows: list[InfluencerCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "Influencer"}
