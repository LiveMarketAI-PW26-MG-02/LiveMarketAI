from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import Session

from ..db.base import Base


class AuditEntry(Base):
    __tablename__ = "audit_entries"
    id = Column(Integer, primary_key=True)
    action = Column(String(255), index=True)
    detail = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def record_audit(db: Session, action: str, detail: str = "") -> dict:
    entry = AuditEntry(action=action, detail=detail)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"id": entry.id, "action": entry.action, "created_at": str(entry.created_at)}


def list_audit(db: Session, limit: int = 100) -> list[dict]:
    rows = db.query(AuditEntry).order_by(AuditEntry.id.desc()).limit(limit).all()
    return [{"id": r.id, "action": r.action, "detail": r.detail,
             "created_at": str(r.created_at)} for r in rows]
