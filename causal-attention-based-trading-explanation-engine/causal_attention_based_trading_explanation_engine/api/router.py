from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import trade as r_trade
from .routes import attention_weight as r_attention_weight
from .routes import causal_factor as r_causal_factor
from .routes import explanation as r_explanation
from .routes import strategy as r_strategy

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_trade.router)
api_router.include_router(r_attention_weight.router)
api_router.include_router(r_causal_factor.router)
api_router.include_router(r_explanation.router)
api_router.include_router(r_strategy.router)
