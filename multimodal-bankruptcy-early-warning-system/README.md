# Multimodal Bankruptcy Early-Warning System

Fuses financial ratios, filings text and news signals into an explainable bankruptcy-risk score with lead time.

This repository was produced by a project scaffolding generator. The structure
and code are real and coherent (working SQLAlchemy models, Pydantic schemas,
FastAPI routers, and a functional AI primitive) — it is a starting point to
extend, not a finished, deployed product.

## Architecture

- **8 logical datastores**: postgres_core, postgres_audit, timescaledb, redis_cache, mongo_documents, clickhouse_analytics, qdrant_vectors, neo4j_graph. The two SQL stores run live via SQLAlchemy
  (SQLite by default, Postgres in `docker-compose`); the rest are lazily-
  connected clients that degrade gracefully when their service is absent.
- **10 self-created APIs (routers)**: /health, /auth, /audit, /ingest, /explain, /companys, /financialss, /news_signals, /warning_scores, /filings
- **AI primitive**: `forecast` — see `multimodal_bankruptcy_early_warning_system/ai/engine.py`. Exposed at `POST /explain`.
- Clean layering: `models -> schemas -> repositories -> services -> api`, with
  JWT-style auth, request-context middleware, audit logging and structured errors.

## Run

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn multimodal_bankruptcy_early_warning_system.main:app --reload     # SQLite tables auto-created on boot
    # docs at http://localhost:8000/docs

    pytest                              # health, CRUD, auth and AI tests

    docker compose up --build           # full stack with Postgres/Redis/Mongo

## Layout

    multimodal_bankruptcy_early_warning_system/
      api/        routers (10) + dependencies
      ai/         the explainable AI engine
      core/       security, logging, error handling
      db/         Base, session, 8-datastore registry
      middleware/ request context / timing
      models/     SQLAlchemy models (5 domain entities + audit)
      repositories/ generic CRUD + per-entity
      schemas/    Pydantic v2 schemas
      services/   business logic + explanation service
    migrations/   create-all migration
    scripts/      seed + train helpers
    tests/        pytest suite
