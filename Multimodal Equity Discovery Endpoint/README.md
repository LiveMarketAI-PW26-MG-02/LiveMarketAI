# AAAI — Multimodal Equity Discovery Endpoint
### Module 1 | v1.0

---

## Overview

This module implements the **Multimodal Equity Discovery Endpoint** — an authoritative per-instrument profile system that deterministically retrieves and arithmetically assembles three structured numeric streams for every equity instrument:

| Dimension | Stream | Table |
|---|---|---|
| Dim-1 | Closing Price Observation Sequence | `price_observations` |
| Dim-2 | Chronologically Ordained Time Index Sequence | `time_indices` |
| Dim-3 | Activity Frequency Stream | `activity_frequencies` |

These three coexisting dimensional layers constitute the platform's **instrument browsing foundation**.

---

## Tech Stack

- **Backend**: FastAPI (Python 3.11)
- **Frontend**: React 18 + Recharts
- **Database**: PostgreSQL 16
- **Containerization**: Docker + Docker Compose

---

## API Integrations

| API | Purpose | Failsafe |
|---|---|---|
| AlphaVantage | Closing price sequences | Deterministic SHA-256 price series |
| ICICI Breeze | Activity frequency stream + market depth | Deterministic volume-derived stream |
| Mistral AI | Profile enrichment analysis | Deterministic sentiment mapping |
| YouTube Data API | Research video references | Deterministic video metadata |

---

## Prerequisites

- Docker Desktop (Windows): https://www.docker.com/products/docker-desktop
- Docker must be running before executing any batch files

---

## Quick Start (Windows)

### Step 1 — Setup
```
Double-click: setup.bat
```
This will:
- Copy `.env.example` → `.env`
- Build all Docker images

### Step 2 — Configure API Keys
Edit `backend/.env`:
```
ALPHAVANTAGE_API_KEY=your_key
MISTRAL_API_KEY=your_key
YOUTUBE_API_KEY=your_key
BREEZE_API_KEY=your_key
BREEZE_API_SECRET=your_secret
BREEZE_SESSION_TOKEN=your_token
```
> All keys are optional. The system operates fully without any keys using internal deterministic data generation.

### Step 3 — Run
```
Double-click: run.bat
```
This starts all services and opens the browser automatically.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/instruments/seed` | Seeds instruments + all 3 multimodal streams |
| GET | `/api/v1/instruments` | Paginated instrument browser list |
| GET | `/api/v1/instruments/{symbol}/profile` | Full multimodal profile (3 streams) |
| GET | `/api/v1/instruments/{symbol}/enriched` | Profile + AI analysis + research videos |
| GET | `/api/v1/instruments/{symbol}/depth` | Real-time market depth (bid/ask) |
| GET | `/health` | Health check |

---

## Database Schema

```
instruments          — master instrument registry
price_observations   — Dim-1: closing price sequence (sequence_ordinal indexed)
time_indices         — Dim-2: epoch time marker sequence (sequence_ordinal indexed)
activity_frequencies — Dim-3: update frequency count stream (sequence_ordinal indexed)
```

All three stream tables are aligned by `instrument_id` + `sequence_ordinal` for deterministic reconstruction.

---

## Access URLs

| Service | URL |
|---|---|
| Frontend UI | http://localhost:3000 |
| Backend API | http://localhost:8000/api/v1 |
| Swagger Docs | http://localhost:8000/docs |

---

## Stopping

```bat
docker-compose down
```

To remove all data:
```bat
docker-compose down -v
```
