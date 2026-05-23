from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.momentum_score import momentum_score_repository as repo
from ...schemas.momentum_score import MomentumScoreCreate, MomentumScoreRead

router = APIRouter(prefix="/momentum_scores", tags=["MomentumScore"])


@router.get("", response_model=list[MomentumScoreRead])
def list_items(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    return repo.list(db, limit=limit, offset=offset)


@router.get("/{item_id}", response_model=MomentumScoreRead)
def get_item(item_id: int, db: Session = Depends(get_db)):
    obj = repo.get(db, item_id)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "MomentumScore not found")
    return obj


@router.post("", response_model=MomentumScoreRead, status_code=status.HTTP_201_CREATED)
def create_item(payload: MomentumScoreCreate, db: Session = Depends(get_db)):
    return repo.create(db, **payload.model_dump(exclude_none=True))


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    if not repo.delete(db, item_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "MomentumScore not found")
