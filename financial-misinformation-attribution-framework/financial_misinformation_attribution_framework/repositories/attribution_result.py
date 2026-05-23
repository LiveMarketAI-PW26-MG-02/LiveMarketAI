from __future__ import annotations

from ..models.attribution_result import AttributionResult
from .base import CRUDRepository


class AttributionResultRepository(CRUDRepository[AttributionResult]):
    def __init__(self) -> None:
        super().__init__(AttributionResult)


attribution_result_repository = AttributionResultRepository()
