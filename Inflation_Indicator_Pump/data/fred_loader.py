"""
FRED (Federal Reserve Economic Data) loader.
Fetches CPI, PPI, PCE, TIPS, and related series.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

FRED_SERIES = {
    "cpi":            "CPIAUCSL",
    "core_cpi":       "CPILFESL",
    "ppi":            "PPIACO",
    "pce":            "PCEPI",
    "core_pce":       "PCEPILFE",
    "tips_10y":       "DFII10",
    "nominal_10y":    "DGS10",
    "breakeven_10y":  "T10YIE",
    "unemployment":   "UNRATE",
    "m2":             "M2SL",
    "fed_funds_rate": "FEDFUNDS",
}


class FREDDataLoader:
    """
    Loads economic data from the FRED API.
    Falls back to synthetic data when API key is unavailable.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch(self, series_id: str, start: str = "2000-01-01",
              end: Optional[str] = None) -> pd.DataFrame:
        if series_id in self._cache:
            return self._cache[series_id]
        if self.api_key:
            try:
                from fredapi import Fred
                fred = Fred(api_key=self.api_key)
                s = fred.get_series(series_id, observation_start=start)
                df = s.to_frame(name="value")
                df.index = pd.to_datetime(df.index)
                self._cache[series_id] = df
                return df
            except Exception as exc:
                logger.warning("FRED fetch failed for %s: %s. Using synthetic.", series_id, exc)
        return self._synthetic(series_id)

    def fetch_multiple(self, names: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        result = {}
        for name in names:
            sid = FRED_SERIES.get(name, name)
            result[name] = self.fetch(sid, **kwargs)
        return result

    def _synthetic(self, series_id: str) -> pd.DataFrame:
        rng = np.random.default_rng(hash(series_id) % 10000)
        dates = pd.date_range("2000-01-01", periods=300, freq="MS")
        if "CPI" in series_id or "PCE" in series_id:
            base, drift = 150.0, 0.25
        elif "PPI" in series_id:
            base, drift = 120.0, 0.2
        elif "UNRATE" in series_id:
            base, drift = 5.0, 0.0
        else:
            base, drift = 100.0, 0.1
        values = base + np.cumsum(rng.normal(drift, 0.5, len(dates)))
        return pd.DataFrame({"value": values}, index=dates)

    def clear_cache(self) -> None:
        self._cache.clear()
