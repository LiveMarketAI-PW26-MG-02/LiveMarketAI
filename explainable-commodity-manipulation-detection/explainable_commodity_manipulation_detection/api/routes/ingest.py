from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.commodity_tick import commodity_tick_repository as repo
from ...schemas.commodity_tick import CommodityTickCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/commodity_tick/batch")
def ingest_batch(rows: list[CommodityTickCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "CommodityTick"}
