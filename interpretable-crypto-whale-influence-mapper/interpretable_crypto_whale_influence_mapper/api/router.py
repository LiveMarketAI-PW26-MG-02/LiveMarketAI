from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import wallet as r_wallet
from .routes import transaction as r_transaction
from .routes import whale_cluster as r_whale_cluster
from .routes import influence_edge as r_influence_edge
from .routes import token as r_token

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_wallet.router)
api_router.include_router(r_transaction.router)
api_router.include_router(r_whale_cluster.router)
api_router.include_router(r_influence_edge.router)
api_router.include_router(r_token.router)
