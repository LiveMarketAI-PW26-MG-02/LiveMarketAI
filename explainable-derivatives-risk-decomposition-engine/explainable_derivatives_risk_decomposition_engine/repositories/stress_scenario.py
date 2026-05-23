from __future__ import annotations

from ..models.stress_scenario import StressScenario
from .base import CRUDRepository


class StressScenarioRepository(CRUDRepository[StressScenario]):
    def __init__(self) -> None:
        super().__init__(StressScenario)


stress_scenario_repository = StressScenarioRepository()
