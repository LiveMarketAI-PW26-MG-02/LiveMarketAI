# Stock Platform — All 6 Services

A complete stock trading platform backend with 6 microservices, built with Node.js, MySQL, Redis, and Elasticsearch.

---

## Architecture

```
stock-platform/
├── database/
│   └── schema.sql                     ← All DB tables + seed data
├── service1-catalog/                  ← Port 3001
├── service2-price-engine/             ← Port 3002
├── service3-price-cache/              ← Port 3003
├── service4-search-index/             ← Port 3004
├── service5-listing-api/              ← Port 3005
├── service6-availability-scheduler/  ← Port 3006
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Quick Start (One Command)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- That's it — nothing else needed!

### Steps

```bash
# 1. Clone / unzip the project
cd stock-platform

# 2. Start everything
docker-compose up --build

# 3. Wait ~60 seconds for all services to start
# You'll see: "🚀 Service X running on :300X" for each service
```

All 6 services + MySQL + Redis + Elasticsearch start automatically.

---

## Services & APIs

### Service 1 — Stock Catalog (`http://localhost:3001`)
Manages the master list of all stocks.

| Method | URL | Description |
|--------|-----|-------------|
| GET    | `/api/v1/health` | Health check |
| POST   | `/api/v1/stocks` | Create stock |
| GET    | `/api/v1/stocks` | List all stocks |
| GET    | `/api/v1/stocks/:stock_id` | Get by ID |
| GET    | `/api/v1/stocks/symbol/:symbol` | Get by symbol (e.g. AAPL) |
| PUT    | `/api/v1/stocks/:stock_id` | Update stock |
| DELETE | `/api/v1/stocks/:stock_id` | Delist stock |
| GET    | `/api/v1/catalog/stats` | Stats by sector/exchange |
| POST   | `/api/v1/categories` | Create category |
| GET    | `/api/v1/categories` | List categories |
| PUT    | `/api/v1/categories/:id` | Update category |
| DELETE | `/api/v1/categories/:id` | Delete category |

### Service 2 — Price Adjustment Engine (`http://localhost:3002`)
Manages base prices and applies discount/multiplier/override rules automatically every minute.

| Method | URL | Description |
|--------|-----|-------------|
| GET    | `/api/v1/health` | Health check |
| POST   | `/api/v1/prices` | Set base price for a stock |
| GET    | `/api/v1/prices` | List all prices |
| GET    | `/api/v1/prices/:stock_id` | Get price for a stock |
| POST   | `/api/v1/prices/:stock_id/apply-rules` | Apply rules to one stock |
| POST   | `/api/v1/prices/apply-rules/all` | Apply rules to all stocks |
| POST   | `/api/v1/rules` | Create adjustment rule |
| GET    | `/api/v1/rules` | List all rules |
| GET    | `/api/v1/rules/active` | List currently active rules |
| PUT    | `/api/v1/rules/:id` | Update rule |
| DELETE | `/api/v1/rules/:id` | Delete rule |

**Rule types:**
- `discount` — reduce price by X percent (e.g. 10 = 10% off)
- `multiplier` — multiply price by X (e.g. 1.05 = 5% increase)
- `override` — set price to exact value

### Service 3 — Price Cache (`http://localhost:3003`)
Stores live stock prices in Redis for fast access (60s TTL).

| Method | URL | Description |
|--------|-----|-------------|
| GET    | `/api/v1/health` | Health check |
| POST   | `/api/v1/cache/warmup` | Load all prices into cache |
| GET    | `/api/v1/cache` | Get all cached prices |
| GET    | `/api/v1/cache/:stock_id` | Get price from cache |
| PUT    | `/api/v1/cache/:stock_id` | Update price in cache |
| DELETE | `/api/v1/cache/:stock_id` | Invalidate cache entry |
| GET    | `/api/v1/cache/:stock_id/ttl` | Check TTL remaining |

### Service 4 — Search Index (`http://localhost:3004`)
Full-text search powered by Elasticsearch with fuzzy matching.

| Method | URL | Description |
|--------|-----|-------------|
| GET    | `/api/v1/health` | Health check |
| GET    | `/api/v1/search?q=apple` | Search stocks |
| POST   | `/api/v1/search/reindex` | Reindex all stocks |
| POST   | `/api/v1/search/index` | Index a single stock |
| DELETE | `/api/v1/search/:stock_id` | Remove from index |

**Search filters:** `q`, `sector`, `industry`, `exchange`, `country`, `category`, `status`, `page`, `limit`

### Service 5 — Stock Listing API (`http://localhost:3005`)
Paginated stock listing that combines catalog data + live prices from cache.

| Method | URL | Description |
|--------|-----|-------------|
| GET    | `/api/v1/health` | Health check |
| GET    | `/api/v1/listings` | Paginated stock list |
| GET    | `/api/v1/listings/:stock_id` | Full stock detail |
| GET    | `/api/v1/listings/meta/filters` | Filter options (sectors, exchanges, countries) |

**Listing filters:** `sector`, `exchange`, `country`, `category_id`, `status`, `sort_by`, `sort_order`, `page`, `limit`

### Service 6 — Availability Scheduler (`http://localhost:3006`)
Controls when stocks are available based on exchange hours and special events.

| Method | URL | Description |
|--------|-----|-------------|
| GET    | `/api/v1/health` | Health check |
| GET    | `/api/v1/exchanges` | List all exchanges |
| GET    | `/api/v1/exchanges/status` | Check all exchanges open/closed |
| GET    | `/api/v1/exchanges/:name/is-open` | Is a specific exchange open now? |
| POST   | `/api/v1/exchanges` | Add exchange hours |
| PUT    | `/api/v1/exchanges/:id` | Update exchange hours |
| POST   | `/api/v1/availability` | Create stock availability schedule |
| GET    | `/api/v1/availability` | List all schedules |
| GET    | `/api/v1/availability/stock/:stock_id/check` | Is this stock available now? |
| PUT    | `/api/v1/availability/:id` | Update schedule |
| DELETE | `/api/v1/availability/:id` | Delete schedule |

---

## Example API Calls

### Create a stock
```bash
curl -X POST http://localhost:3001/api/v1/stocks \
  -H "Content-Type: application/json" \
  -d '{
    "stock_symbol": "INFY",
    "stock_name": "Infosys Limited",
    "category_id": 1,
    "sector": "Technology",
    "industry": "IT Services",
    "exchange": "NSE",
    "currency": "INR",
    "isin": "INE009A01021",
    "market_cap": 600000000000,
    "ipo_date": "1993-02-09",
    "country": "India",
    "tags": ["tech", "nse", "it"],
    "status": "active"
  }'
```

### Search stocks
```bash
curl "http://localhost:3004/api/v1/search?q=tata&sector=Technology"
```

### Set a price rule (10% discount)
```bash
curl -X POST http://localhost:3002/api/v1/rules \
  -H "Content-Type: application/json" \
  -d '{
    "rule_name": "Festive Discount",
    "adjustment_type": "discount",
    "adjustment_value": 10,
    "valid_from": "2026-01-01T00:00:00",
    "valid_to": "2026-01-07T23:59:59",
    "status": "active"
  }'
```

### Check if NSE is open
```bash
curl http://localhost:3006/api/v1/exchanges/NSE/is-open
```

### Get stock listing (frontend call)
```bash
curl "http://localhost:3005/api/v1/listings?sector=Technology&page=1&limit=20"
```

---

## Database Tables

| Table | Service | Purpose |
|-------|---------|---------|
| `stock_categories` | Service 1 | Stock categories |
| `stocks` | Service 1 | Master stock catalog |
| `stock_prices` | Service 2 | Base + adjusted prices |
| `price_adjustment_rules` | Service 2 | Discount/multiplier rules |
| `exchange_hours` | Service 6 | Market open/close times |
| `stock_availability` | Service 6 | Special stock schedules |

Redis keys: `stock:price:{stock_id}` (60s TTL)
Elasticsearch index: `stock_search_index`

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Node.js 18 | Runtime |
| Express.js | REST API framework |
| MySQL 8.0 | Primary database |
| Redis 7 | Price cache |
| Elasticsearch 8 | Search engine |
| node-cron | Scheduled jobs |
| moment-timezone | Timezone handling |
| Docker Compose | Container orchestration |
