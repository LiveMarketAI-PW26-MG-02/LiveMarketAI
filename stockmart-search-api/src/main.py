from fastapi import FastAPI, Query
from typing import Optional
from .search_engine import search, autocomplete, related

app = FastAPI(title="StockMart Search API", version="1.0.0")

@app.get("/search")
def search_stocks(
    q: str = Query(default="", description="Search query"),
    sector: Optional[str] = None,
    exchange: Optional[str] = None,
    market_cap_tier: Optional[str] = None,
    limit: int = Query(default=10, le=50)
):
    return search(q, sector, exchange, market_cap_tier, limit)

@app.get("/autocomplete")
def suggest(prefix: str, limit: int = 5):
    return autocomplete(prefix, limit)

@app.get("/related/{symbol}")
def get_related(symbol: str, limit: int = 5):
    return related(symbol, limit)
