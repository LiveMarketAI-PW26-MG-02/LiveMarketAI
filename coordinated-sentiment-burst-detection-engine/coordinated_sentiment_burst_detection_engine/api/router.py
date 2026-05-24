from __future__ import annotations

from fastapi import APIRouter

from .routes import auth, audit, explain, health, ingest
from .routes import post as r_post
from .routes import burst_signal as r_burst_signal
from .routes import source as r_source
from .routes import cluster as r_cluster
from .routes import detection_alert as r_detection_alert

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(audit.router)
api_router.include_router(ingest.router)
api_router.include_router(explain.router)
api_router.include_router(r_post.router)
api_router.include_router(r_burst_signal.router)
api_router.include_router(r_source.router)
api_router.include_router(r_cluster.router)
api_router.include_router(r_detection_alert.router)
