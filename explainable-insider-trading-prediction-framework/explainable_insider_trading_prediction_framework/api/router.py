from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import trade as r_trade
from .routes import insider as r_insider
from .routes import filing as r_filing
from .routes import prediction_signal as r_prediction_signal
from .routes import risk_score as r_risk_score

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_trade.router)
api_router.include_router(r_insider.router)
api_router.include_router(r_filing.router)
api_router.include_router(r_prediction_signal.router)
api_router.include_router(r_risk_score.router)
