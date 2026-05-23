from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import order_book_snapshot as r_order_book_snapshot
from .routes import spoof_alert as r_spoof_alert
from .routes import exchange as r_exchange
from .routes import order_event as r_order_event
from .routes import explanation as r_explanation

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_order_book_snapshot.router)
api_router.include_router(r_spoof_alert.router)
api_router.include_router(r_exchange.router)
api_router.include_router(r_order_event.router)
api_router.include_router(r_explanation.router)
