from typing import List, Optional
from rapidfuzz import fuzz, process
from .catalog import CATALOG, StockListing

def search(
    query: str,
    sector: Optional[str] = None,
    exchange: Optional[str] = None,
    market_cap_tier: Optional[str] = None,
    limit: int = 10
) -> List[StockListing]:
    candidates = CATALOG

    # Filter
    if sector:
        candidates = [s for s in candidates if s.sector.lower() == sector.lower()]
    if exchange:
        candidates = [s for s in candidates if s.exchange.upper() == exchange.upper()]
    if market_cap_tier:
        candidates = [s for s in candidates if s.market_cap_tier == market_cap_tier]

    if not query.strip():
        return candidates[:limit]

    # Score each listing
    scored = []
    q = query.upper()
    for stock in candidates:
        sym_score  = fuzz.ratio(q, stock.symbol.upper())
        name_score = fuzz.partial_ratio(query.lower(), stock.name.lower())
        tag_score  = max((fuzz.ratio(query.lower(), t) for t in stock.tags), default=0)
        best = max(sym_score, name_score * 0.8, tag_score * 0.6)
        if best > 30:
            scored.append((best, stock))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:limit]]

def autocomplete(prefix: str, limit: int = 5) -> List[dict]:
    prefix_up = prefix.upper()
    results = []
    for stock in CATALOG:
        if stock.symbol.startswith(prefix_up) or stock.name.upper().startswith(prefix_up):
            results.append({"symbol": stock.symbol, "name": stock.name})
    return results[:limit]

def related(symbol: str, limit: int = 5) -> List[StockListing]:
    target = next((s for s in CATALOG if s.symbol == symbol.upper()), None)
    if not target:
        return []
    same_sector = [s for s in CATALOG if s.sector == target.sector and s.symbol != symbol.upper()]
    tag_overlap = sorted(
        same_sector,
        key=lambda s: len(set(s.tags) & set(target.tags)),
        reverse=True
    )
    return tag_overlap[:limit]
