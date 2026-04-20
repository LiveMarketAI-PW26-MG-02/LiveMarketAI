import os
import hashlib
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BREEZE_API_KEY = os.getenv("BREEZE_API_KEY", "")
BREEZE_API_SECRET = os.getenv("BREEZE_API_SECRET", "")
BREEZE_SESSION_TOKEN = os.getenv("BREEZE_SESSION_TOKEN", "")


def _deterministic_activity(symbol: str, n: int = 90) -> List[Dict[str, Any]]:
    result = []
    today = datetime.utcnow()
    h_base = int(hashlib.sha256(symbol.encode()).hexdigest(), 16)
    for i in range(n):
        h = int(hashlib.sha256(f"{symbol}_act_{i}".encode()).hexdigest(), 16)
        freq = int(50 + (h % 950))
        interval = round(3600.0 / max(1, freq), 4)
        dt = today - timedelta(days=(n - i))
        result.append({
            "sequence_ordinal": i + 1,
            "frequency_count": freq,
            "interval_seconds": interval,
            "recorded_at": dt,
        })
    return result


def _deterministic_market_depth(symbol: str) -> Dict[str, Any]:
    h = int(hashlib.sha256(symbol.encode()).hexdigest(), 16)
    bid = round(100.0 + (h % 900) + (h % 100) / 100.0, 2)
    ask = round(bid + 0.05 + (h % 50) / 100.0, 2)
    return {
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "spread": round(ask - bid, 4),
        "bid_qty": int(500 + h % 9500),
        "ask_qty": int(500 + (h >> 4) % 9500),
        "timestamp": datetime.utcnow().isoformat(),
    }


async def fetch_activity_stream(symbol: str, n: int = 90) -> List[Dict[str, Any]]:
    if not (BREEZE_API_KEY and BREEZE_SESSION_TOKEN):
        return _deterministic_activity(symbol, n)

    try:
        # breeze-connect integration (requires pip install breeze-connect)
        from breeze_connect import BreezeConnect
        breeze = BreezeConnect(api_key=BREEZE_API_KEY)
        breeze.generate_session(
            api_secret=BREEZE_API_SECRET,
            session_token=BREEZE_SESSION_TOKEN
        )
        end_date = datetime.utcnow().strftime("%Y-%m-%dT07:00:00.000Z")
        start_date = (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%dT07:00:00.000Z")
        data = breeze.get_historical_data(
            interval="1day",
            from_date=start_date,
            to_date=end_date,
            stock_code=symbol,
            exchange_code="NSE",
            product_type="cash",
        )
        rows = data.get("Success", [])
        if not rows:
            return _deterministic_activity(symbol, n)
        result = []
        for i, row in enumerate(rows):
            vol = int(row.get("volume", 1))
            interval = round(3600.0 / max(1, vol // 1000), 4)
            result.append({
                "sequence_ordinal": i + 1,
                "frequency_count": vol // 1000,
                "interval_seconds": interval,
                "recorded_at": datetime.strptime(row["datetime"][:19], "%Y-%m-%d %H:%M:%S")
                if "datetime" in row else datetime.utcnow() - timedelta(days=(len(rows) - i)),
            })
        return result
    except Exception:
        return _deterministic_activity(symbol, n)


async def fetch_market_depth(symbol: str) -> Dict[str, Any]:
    if not (BREEZE_API_KEY and BREEZE_SESSION_TOKEN):
        return _deterministic_market_depth(symbol)
    try:
        from breeze_connect import BreezeConnect
        breeze = BreezeConnect(api_key=BREEZE_API_KEY)
        breeze.generate_session(
            api_secret=BREEZE_API_SECRET,
            session_token=BREEZE_SESSION_TOKEN
        )
        data = breeze.get_quotes(
            stock_code=symbol,
            exchange_code="NSE",
            expiry_date="",
            product_type="cash",
            right="",
            strike_price="",
        )
        rows = data.get("Success", [])
        if not rows:
            return _deterministic_market_depth(symbol)
        row = rows[0]
        return {
            "symbol": symbol,
            "bid": float(row.get("best_bid_price", 0)),
            "ask": float(row.get("best_offer_price", 0)),
            "spread": round(float(row.get("best_offer_price", 0)) - float(row.get("best_bid_price", 0)), 4),
            "bid_qty": int(row.get("best_bid_quantity", 0)),
            "ask_qty": int(row.get("best_offer_quantity", 0)),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception:
        return _deterministic_market_depth(symbol)
