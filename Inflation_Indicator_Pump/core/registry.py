"""Global registry for inflation indicators."""
from typing import Dict, List, Optional, Type
from .base_indicator import BaseIndicator


class IndicatorRegistry:
    """Singleton-style registry for all indicator instances."""

    _instance: Optional["IndicatorRegistry"] = None
    _indicators: Dict[str, BaseIndicator] = {}

    @classmethod
    def instance(cls) -> "IndicatorRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, indicator: BaseIndicator) -> None:
        self._indicators[indicator.name] = indicator

    def get(self, name: str) -> Optional[BaseIndicator]:
        return self._indicators.get(name)

    def all(self) -> Dict[str, BaseIndicator]:
        return dict(self._indicators)

    def names(self) -> List[str]:
        return list(self._indicators.keys())

    def remove(self, name: str) -> None:
        self._indicators.pop(name, None)

    def clear(self) -> None:
        self._indicators.clear()
