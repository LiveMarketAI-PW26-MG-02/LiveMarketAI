"""Inflation data pipeline — ingestion → transformation → storage."""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Callable, Optional

logger = logging.getLogger(__name__)


class PipelineStep:
    def __init__(self, name: str, fn: Callable[[pd.DataFrame], pd.DataFrame]):
        self.name = name
        self.fn = fn

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Running step: %s", self.name)
        return self.fn(df)


class InflationPipeline:
    """
    Sequential data pipeline for inflation indicator processing.
    Steps: load → clean → normalise → compute → store.
    """

    def __init__(self):
        self._steps: List[PipelineStep] = []
        self._results: Dict[str, pd.DataFrame] = {}

    def add_step(self, name: str, fn: Callable) -> "InflationPipeline":
        self._steps.append(PipelineStep(name, fn))
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        current = df.copy()
        for step in self._steps:
            try:
                current = step.run(current)
                self._results[step.name] = current.copy()
                logger.info("Step '%s' completed. Shape: %s", step.name, current.shape)
            except Exception as exc:
                logger.error("Step '%s' failed: %s", step.name, exc)
                raise
        return current

    @staticmethod
    def default_cleaning_steps() -> List[PipelineStep]:
        return [
            PipelineStep("drop_nulls", lambda df: df.dropna()),
            PipelineStep("sort_index", lambda df: df.sort_index()),
            PipelineStep("ffill", lambda df: df.ffill()),
        ]

    def get_step_result(self, name: str) -> Optional[pd.DataFrame]:
        return self._results.get(name)
