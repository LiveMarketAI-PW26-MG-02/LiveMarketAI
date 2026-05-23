from __future__ import annotations

"""Eight logical datastores.

The two SQL stores are live (SQLAlchemy). The remaining clients are thin,
lazily-connected wrappers that degrade gracefully when the underlying service
or driver is not present, so the app imports and runs in a bare environment.
"""

from dataclasses import dataclass

from ..config import get_settings


@dataclass
class DatastoreInfo:
    name: str
    kind: str
    purpose: str
    dsn: str


def registry() -> list[DatastoreInfo]:
    s = get_settings()
    return [
        DatastoreInfo("postgres_core", "sql", "primary relational entities", s.database_url),
        DatastoreInfo("postgres_audit", "sql", "append-only audit trail", s.audit_database_url),
        DatastoreInfo("timescaledb", "sql", "time-series market data", s.database_url),
        DatastoreInfo("redis_cache", "redis", "cache / sessions / rate limits", s.redis_url),
        DatastoreInfo("mongo_documents", "mongo", "raw payloads & explanations", s.mongo_url),
        DatastoreInfo("clickhouse_analytics", "clickhouse", "OLAP analytics", s.clickhouse_url),
        DatastoreInfo("qdrant_vectors", "vector", "embedding similarity", s.qdrant_url),
        DatastoreInfo("neo4j_graph", "graph", "influence graph", s.neo4j_url),
    ]


class LazyClient:
    """Connects on first use; reports unavailable instead of crashing import."""

    def __init__(self, info: DatastoreInfo) -> None:
        self.info = info
        self._client = None

    def available(self) -> bool:
        try:
            self.connect()
            return self._client is not None
        except Exception:  # noqa: BLE001
            return False

    def connect(self):
        if self._client is not None:
            return self._client
        kind = self.info.kind
        if kind == "redis":
            import redis  # type: ignore
            self._client = redis.Redis.from_url(self.info.dsn)
        elif kind == "mongo":
            from pymongo import MongoClient  # type: ignore
            self._client = MongoClient(self.info.dsn, serverSelectionTimeoutMS=500)
        elif kind == "vector":
            from qdrant_client import QdrantClient  # type: ignore
            self._client = QdrantClient(url=self.info.dsn)
        elif kind == "graph":
            from neo4j import GraphDatabase  # type: ignore
            self._client = GraphDatabase.driver(self.info.dsn)
        else:
            self._client = {"dsn": self.info.dsn}  # clickhouse via http; placeholder handle
        return self._client


def client_for(name: str) -> LazyClient:
    for info in registry():
        if info.name == name:
            return LazyClient(info)
    raise KeyError(name)
