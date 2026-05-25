from __future__ import annotations

from ..models.subject import Subject
from .base import CRUDRepository


class SubjectRepository(CRUDRepository[Subject]):
    def __init__(self) -> None:
        super().__init__(Subject)


subject_repository = SubjectRepository()
