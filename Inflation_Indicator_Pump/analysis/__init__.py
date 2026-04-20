"""Inflation trend analysis and regime detection."""
from .trend_analysis import TrendAnalyser
from .regime_detector import RegimeDetector
from .decomposition import InflationDecomposer
__all__ = ["TrendAnalyser","RegimeDetector","InflationDecomposer"]
