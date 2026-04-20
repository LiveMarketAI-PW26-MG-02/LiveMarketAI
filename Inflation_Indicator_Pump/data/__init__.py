"""Data loaders and parsers for inflation data sources."""
from .fred_loader import FREDDataLoader
from .csv_parser import CSVInflationParser
from .data_store import InflationDataStore
__all__ = ["FREDDataLoader","CSVInflationParser","InflationDataStore"]
