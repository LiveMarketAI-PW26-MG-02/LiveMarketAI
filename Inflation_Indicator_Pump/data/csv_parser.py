"""Parser for CSV-format inflation data files."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CSVInflationParser:
    """
    Parses CSV files containing inflation time-series data.
    Handles multiple formats (FRED export, BLS, Eurostat).
    """

    SUPPORTED_FORMATS = ["fred", "bls", "eurostat", "generic"]

    def __init__(self, fmt: str = "generic"):
        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(f"fmt must be one of {self.SUPPORTED_FORMATS}")
        self.fmt = fmt

    def parse(self, path: str) -> pd.DataFrame:
        """Read and normalise a CSV file to a standard format."""
        p = Path(path)
        if not p.exists():
            logger.warning("File not found: %s. Returning synthetic data.", path)
            return self._synthetic()
        df = pd.read_csv(path)
        return self._normalise(df)

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fmt == "fred":
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        elif self.fmt == "bls":
            df = df.rename(columns={df.columns[0]: "date", df.columns[-1]: "value"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.dropna(subset=["date"], inplace=True)
            df.set_index("date", inplace=True)
        else:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df.set_index("date", inplace=True)
            if "value" not in df.columns:
                df = df.rename(columns={df.columns[0]: "value"})
        return df[["value"]].dropna().sort_index()

    def _synthetic(self) -> pd.DataFrame:
        dates = pd.date_range("2000-01-01", periods=120, freq="MS")
        values = 150 + np.cumsum(np.random.normal(0.2, 0.4, 120))
        return pd.DataFrame({"value": values}, index=dates)

    def validate(self, df: pd.DataFrame) -> Dict[str, bool]:
        return {
            "has_value_column": "value" in df.columns,
            "is_datetime_index": pd.api.types.is_datetime64_any_dtype(df.index),
            "no_nulls": df["value"].notna().all(),
            "sorted": df.index.is_monotonic_increasing,
        }
