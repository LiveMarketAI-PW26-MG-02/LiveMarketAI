"""In-memory and file-backed data store for inflation time series."""
import pandas as pd
import numpy as np
import json
from typing import Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InflationDataStore:
    """
    Stores and retrieves inflation indicator time series.
    Supports in-memory and optional file persistence.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._store: Dict[str, pd.DataFrame] = {}
        self.persist_path = persist_path

    def put(self, name: str, df: pd.DataFrame) -> None:
        self._store[name] = df.copy()
        logger.debug("Stored series: %s (%d rows)", name, len(df))

    def get(self, name: str) -> Optional[pd.DataFrame]:
        return self._store.get(name)

    def list_series(self) -> List[str]:
        return list(self._store.keys())

    def latest_value(self, name: str) -> Optional[float]:
        df = self._store.get(name)
        if df is None or df.empty:
            return None
        return float(df["value"].dropna().iloc[-1])

    def date_range(self, name: str):
        df = self._store.get(name)
        if df is None or df.empty:
            return None, None
        return df.index[0], df.index[-1]

    def merge(self, names: List[str]) -> pd.DataFrame:
        frames = {}
        for n in names:
            df = self._store.get(n)
            if df is not None:
                frames[n] = df["value"]
        return pd.DataFrame(frames)

    def statistics(self, name: str) -> Dict:
        df = self._store.get(name)
        if df is None or df.empty:
            return {}
        s = df["value"].dropna()
        return {
            "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "latest": float(s.iloc[-1]), "count": len(s),
        }

    def remove(self, name: str) -> None:
        self._store.pop(name, None)

    def clear(self) -> None:
        self._store.clear()
