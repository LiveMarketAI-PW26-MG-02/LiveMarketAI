"""data_fetcher.py — yfinance real-time data for regime detection"""
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Optional
from logger import get_logger

logger = get_logger("data_fetcher")


class DataFetcher:
    def __init__(self, symbols: list):
        self.symbols = symbols

    def fetch_history(self, period="1y", interval="1d") -> Optional[pd.DataFrame]:
        try:
            data = yf.download(self.symbols, period=period, interval=interval,
                               auto_adjust=True, progress=False)
            closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
            returns = closes.pct_change().dropna()
            logger.info(f"Historical data: {returns.shape}")
            return returns
        except Exception as e:
            logger.error(f"History fetch failed: {e}")
            return None

    def fetch_snapshot(self) -> Dict:
        result = {}
        for sym in self.symbols:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="3mo", interval="1d", auto_adjust=True)
                if hist.empty:
                    continue
                prices = hist["Close"].values
                returns = np.diff(np.log(prices))
                result[sym] = {
                    "price": float(prices[-1]),
                    "returns": returns,
                    "prices": prices,
                }
            except Exception as e:
                logger.warning(f"{sym}: {e}")
        return result
