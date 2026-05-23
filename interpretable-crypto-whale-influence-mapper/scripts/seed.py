from __future__ import annotations

from datetime import datetime

from interpretable_crypto_whale_influence_mapper.db.session import SessionLocal, init_db
from interpretable_crypto_whale_influence_mapper.repositories.wallet import wallet_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, address="sample", label="sample", balance=1.0, first_seen=datetime.utcnow())
        print(f"seeded {n} wallet rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
