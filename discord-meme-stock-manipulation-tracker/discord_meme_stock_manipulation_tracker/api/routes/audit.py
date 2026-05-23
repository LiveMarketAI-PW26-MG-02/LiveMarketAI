from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...api.deps import get_db
from ...services.audit_service import list_audit, record_audit

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("")
def audit_log(limit: int = 100, db: Session = Depends(get_db)):
    return list_audit(db, limit=limit)


@router.post("")
def add_audit(action: str, detail: str = "", db: Session = Depends(get_db)):
    return record_audit(db, action=action, detail=detail)
