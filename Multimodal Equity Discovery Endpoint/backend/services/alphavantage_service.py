import os
import httpx
import hashlib
import math
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"


def _deterministic_price_series(symbol: str, n: int = 90) -> List[Dict[str, Any]]:
    seed = int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % (10 ** 8)
    base = 100.0 + (seed % 900)
    series = []
    price = base
    today = datetime.utcnow()
    for i in range(n):
        h = int(hashlib.sha256(f"{symbol}{i}".encode()).hexdigest(), 16)
        delta = ((h % 1000) - 500) / 1000.0 * 2.5
        price = max(1.0, price + delta)
        dt = today - timedelta(days=(n - i))
        series.append({
            "date": dt.strftime("%Y-%m-%d"),
            "close": round(price, 4),
            "volume": int(100000 + (h % 900000)),
        })
    return series


async def fetch_daily_prices(symbol: str, n: int = 90) -> List[Dict[str, Any]]:
    if not ALPHAVANTAGE_API_KEY:
        return _deterministic_price_series(symbol, n)

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(ALPHAVANTAGE_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            ts = data.get("Time Series (Daily)", {})
            if not ts:
                return _deterministic_price_series(symbol, n)
            result = []
            for date_str, vals in sorted(ts.items())[-n:]:
                result.append({
                    "date": date_str,
                    "close": float(vals.get("5. adjusted close", vals.get("4. close", 0))),
                    "volume": int(vals.get("6. volume", 0)),
                })
            return result if result else _deterministic_price_series(symbol, n)
    except Exception:
        return _deterministic_price_series(symbol, n)


def _deterministic_instruments() -> List[Dict[str, str]]:
    return [
        {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "exchange": "NSE", "sector": "Energy"},
        {"symbol": "TCS", "name": "Tata Consultancy Services", "exchange": "NSE", "sector": "Technology"},
        {"symbol": "INFY", "name": "Infosys Ltd", "exchange": "NSE", "sector": "Technology"},
        {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "exchange": "NSE", "sector": "Financials"},
        {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "exchange": "NSE", "sector": "Financials"},
        {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd", "exchange": "NSE", "sector": "Consumer"},
        {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd", "exchange": "NSE", "sector": "Financials"},
        {"symbol": "WIPRO", "name": "Wipro Ltd", "exchange": "NSE", "sector": "Technology"},
        {"symbol": "SBIN", "name": "State Bank of India", "exchange": "NSE", "sector": "Financials"},
        {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "exchange": "NSE", "sector": "Telecom"},
        {"symbol": "ASIANPAINT", "name": "Asian Paints Ltd", "exchange": "NSE", "sector": "Materials"},
        {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "exchange": "NSE", "sector": "Financials"},
        {"symbol": "LT", "name": "Larsen & Toubro Ltd", "exchange": "NSE", "sector": "Industrials"},
        {"symbol": "AXISBANK", "name": "Axis Bank Ltd", "exchange": "NSE", "sector": "Financials"},
        {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries", "exchange": "NSE", "sector": "Healthcare"},
        {"symbol": "MARUTI", "name": "Maruti Suzuki India Ltd", "exchange": "NSE", "sector": "Consumer"},
        {"symbol": "TITAN", "name": "Titan Company Ltd", "exchange": "NSE", "sector": "Consumer"},
        {"symbol": "ULTRACEMCO", "name": "UltraTech Cement Ltd", "exchange": "NSE", "sector": "Materials"},
        {"symbol": "ONGC", "name": "Oil & Natural Gas Corp", "exchange": "NSE", "sector": "Energy"},
        {"symbol": "NTPC", "name": "NTPC Ltd", "exchange": "NSE", "sector": "Utilities"},
    ]


async def fetch_instruments_list() -> List[Dict[str, str]]:
    return _deterministic_instruments()
