"""Initial schema migration.

This project uses SQLAlchemy ``create_all`` for the relational stores; this
script makes the bootstrap explicit and re-runnable for CI/CD.
"""

from __future__ import annotations

from explainable_commodity_manipulation_detection.db.base import Base
from explainable_commodity_manipulation_detection.db.session import engine
from explainable_commodity_manipulation_detection import models  # noqa: F401  (register tables)
from explainable_commodity_manipulation_detection.services import audit_service  # noqa: F401  (register audit table)


def upgrade() -> None:
    Base.metadata.create_all(bind=engine)


def downgrade() -> None:
    Base.metadata.drop_all(bind=engine)


if __name__ == "__main__":
    upgrade()
    print("schema created")
