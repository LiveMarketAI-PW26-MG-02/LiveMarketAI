from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...repositories.crash_event import crash_event_repository as repo
from ...schemas.crash_event import CrashEventCreate, CrashEventRead

router = APIRouter(prefix="/crash_events", tags=["CrashEvent"])


@router.get("", response_model=list[CrashEventRead])
def list_items(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    return repo.list(db, limit=limit, offset=offset)


@router.get("/{item_id}", response_model=CrashEventRead)
def get_item(item_id: int, db: Session = Depends(get_db)):
    obj = repo.get(db, item_id)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "CrashEvent not found")
    return obj


@router.post("", response_model=CrashEventRead, status_code=status.HTTP_201_CREATED)
def create_item(payload: CrashEventCreate, db: Session = Depends(get_db)):
    return repo.create(db, **payload.model_dump(exclude_none=True))


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    if not repo.delete(db, item_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "CrashEvent not found")
