"""Initial schema migration.

This project uses SQLAlchemy ``create_all`` for the relational stores; this
script makes the bootstrap explicit and re-runnable for CI/CD.
"""

from __future__ import annotations

from dark_pool_manipulation_intelligence_engine.db.base import Base
from dark_pool_manipulation_intelligence_engine.db.session import engine
from dark_pool_manipulation_intelligence_engine import models  # noqa: F401  (register tables)
from dark_pool_manipulation_intelligence_engine.services import audit_service  # noqa: F401  (register audit table)


def upgrade() -> None:
    Base.metadata.create_all(bind=engine)


def downgrade() -> None:
    Base.metadata.drop_all(bind=engine)


if __name__ == "__main__":
    upgrade()
    print("schema created")
