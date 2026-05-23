from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    title: str = "Causal Reasoning Engine for Macroeconomic Shocks"
    slug: str = "causal-reasoning-engine-macroeconomic-shocks"
    version: str = "1.0.0"
    secret_key: str = os.environ.get("SECRET_KEY", "dev-secret-rotate-me")
    token_ttl_seconds: int = int(os.environ.get("TOKEN_TTL", "3600"))
    database_url: str = os.environ.get("DATABASE_URL", "sqlite:///./causal_reasoning_engine_macroeconomic_shocks.db")
    audit_database_url: str = os.environ.get("AUDIT_DATABASE_URL", "sqlite:///./causal_reasoning_engine_macroeconomic_shocks`_audit.db")
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    mongo_url: str = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    clickhouse_url: str = os.environ.get("CLICKHOUSE_URL", "http://localhost:8123")
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    neo4j_url: str = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
    cors_origins: tuple = ("*",)


def get_settings() -> Settings:
    return Settings()
