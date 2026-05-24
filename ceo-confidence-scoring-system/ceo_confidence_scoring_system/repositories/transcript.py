from __future__ import annotations

from ..models.transcript import Transcript
from .base import CRUDRepository


class TranscriptRepository(CRUDRepository[Transcript]):
    def __init__(self) -> None:
        super().__init__(Transcript)


transcript_repository = TranscriptRepository()
