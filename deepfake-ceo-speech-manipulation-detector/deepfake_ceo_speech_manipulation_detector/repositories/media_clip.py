from __future__ import annotations

from ..models.media_clip import MediaClip
from .base import CRUDRepository


class MediaClipRepository(CRUDRepository[MediaClip]):
    def __init__(self) -> None:
        super().__init__(MediaClip)


media_clip_repository = MediaClipRepository()
