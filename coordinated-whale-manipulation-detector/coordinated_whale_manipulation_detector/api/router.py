from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import wallet as r_wallet
from .routes import coordination_signal as r_coordination_signal
from .routes import whale_group as r_whale_group
from .routes import trade as r_trade
from .routes import edge as r_edge

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_wallet.router)
api_router.include_router(r_coordination_signal.router)
api_router.include_router(r_whale_group.router)
api_router.include_router(r_trade.router)
api_router.include_router(r_edge.router)
