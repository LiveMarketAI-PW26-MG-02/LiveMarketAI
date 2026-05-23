from __future__ import annotations

from ..models.scenario import Scenario
from .base import CRUDRepository


class ScenarioRepository(CRUDRepository[Scenario]):
    def __init__(self) -> None:
        super().__init__(Scenario)


scenario_repository = ScenarioRepository()
