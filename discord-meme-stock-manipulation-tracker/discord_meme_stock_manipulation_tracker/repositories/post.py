from __future__ import annotations

from ..models.post import Post
from .base import CRUDRepository


class PostRepository(CRUDRepository[Post]):
    def __init__(self) -> None:
        super().__init__(Post)


post_repository = PostRepository()
