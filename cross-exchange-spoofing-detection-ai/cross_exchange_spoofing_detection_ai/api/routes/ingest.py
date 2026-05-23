from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.order_book_snapshot import order_book_snapshot_repository as repo
from ...schemas.order_book_snapshot import OrderBookSnapshotCreate

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/order_book_snapshot/batch")
def ingest_batch(rows: list[OrderBookSnapshotCreate], db: Session = Depends(get_db)):
    payload = [r.model_dump(exclude_none=True) for r in rows]
    inserted = repo.bulk_create(db, payload)
    return {"inserted": inserted, "entity": "OrderBookSnapshot"}
