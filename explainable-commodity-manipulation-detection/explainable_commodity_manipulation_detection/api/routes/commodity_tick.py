from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.commodity_tick import commodity_tick_repository as repo
from ...schemas.commodity_tick import CommodityTickCreate, CommodityTickRead

router = APIRouter(prefix="/commodity_ticks", tags=["CommodityTick"])


@router.get("", response_model=list[CommodityTickRead])
def list_items(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    return repo.list(db, limit=limit, offset=offset)


@router.get("/{item_id}", response_model=CommodityTickRead)
def get_item(item_id: int, db: Session = Depends(get_db)):
    obj = repo.get(db, item_id)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "CommodityTick not found")
    return obj


@router.post("", response_model=CommodityTickRead, status_code=status.HTTP_201_CREATED)
def create_item(payload: CommodityTickCreate, db: Session = Depends(get_db)):
    return repo.create(db, **payload.model_dump(exclude_none=True))


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    if not repo.delete(db, item_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "CommodityTick not found")
