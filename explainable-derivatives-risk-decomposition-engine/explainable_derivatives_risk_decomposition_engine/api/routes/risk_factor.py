from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.risk_factor import risk_factor_repository as repo
from ...schemas.risk_factor import RiskFactorCreate, RiskFactorRead

router = APIRouter(prefix="/risk_factors", tags=["RiskFactor"])


@router.get("", response_model=list[RiskFactorRead])
def list_items(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    return repo.list(db, limit=limit, offset=offset)


@router.get("/{item_id}", response_model=RiskFactorRead)
def get_item(item_id: int, db: Session = Depends(get_db)):
    obj = repo.get(db, item_id)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "RiskFactor not found")
    return obj


@router.post("", response_model=RiskFactorRead, status_code=status.HTTP_201_CREATED)
def create_item(payload: RiskFactorCreate, db: Session = Depends(get_db)):
    return repo.create(db, **payload.model_dump(exclude_none=True))


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    if not repo.delete(db, item_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "RiskFactor not found")
