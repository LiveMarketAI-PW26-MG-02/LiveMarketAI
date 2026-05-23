from __future__ import annotations

import time

from fastapi import APIRouter

from ...config import get_settings

router = APIRouter(tags=["system"])
_START = time.time()


@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "service": s.title, "version": s.version}


@router.get("/ready")
def ready():
    return {"ready": True, "uptime_seconds": round(time.time() - _START, 2)}
