from __future__ import annotations

from ..models.company import Company
from .base import CRUDRepository


class CompanyRepository(CRUDRepository[Company]):
    def __init__(self) -> None:
        super().__init__(Company)


company_repository = CompanyRepository()
