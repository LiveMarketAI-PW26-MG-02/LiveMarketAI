from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.liquidity_metric import liquidity_metric_repository as repo
from ...schemas.liquidity_metric import LiquidityMetricCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/liquidity_metric/batch")
def ingest_batch(rows: list[LiquidityMetricCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "LiquidityMetric"}
