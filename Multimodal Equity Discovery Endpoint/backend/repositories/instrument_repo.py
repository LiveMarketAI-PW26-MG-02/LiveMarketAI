from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from models.equity_models import Instrument, PriceObservation, TimeIndex, ActivityFrequency
from models.schemas import InstrumentCreate
from typing import List, Optional, Tuple
from datetime import datetime


class InstrumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self, page: int = 1, page_size: int = 50) -> Tuple[List[Instrument], int]:
        offset = (page - 1) * page_size
        total = self.db.query(func.count(Instrument.id)).scalar()
        instruments = (
            self.db.query(Instrument)
            .order_by(Instrument.symbol)
            .offset(offset)
            .limit(page_size)
            .all()
        )
        return instruments, total

    def get_by_symbol(self, symbol: str) -> Optional[Instrument]:
        return self.db.query(Instrument).filter(Instrument.symbol == symbol.upper()).first()

    def get_by_id(self, instrument_id: int) -> Optional[Instrument]:
        return self.db.query(Instrument).filter(Instrument.id == instrument_id).first()

    def create(self, data: InstrumentCreate) -> Instrument:
        inst = Instrument(
            symbol=data.symbol.upper(),
            name=data.name,
            exchange=data.exchange,
            sector=data.sector,
        )
        self.db.add(inst)
        self.db.commit()
        self.db.refresh(inst)
        return inst

    def upsert(self, data: InstrumentCreate) -> Instrument:
        existing = self.get_by_symbol(data.symbol)
        if existing:
            existing.name = data.name
            existing.exchange = data.exchange
            existing.sector = data.sector
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            return existing
        return self.create(data)

    def get_latest_close(self, instrument_id: int) -> Optional[float]:
        obs = (
            self.db.query(PriceObservation)
            .filter(PriceObservation.instrument_id == instrument_id)
            .order_by(desc(PriceObservation.sequence_ordinal))
            .first()
        )
        return obs.closing_price if obs else None

    def count_observations(self, instrument_id: int) -> int:
        return (
            self.db.query(func.count(PriceObservation.id))
            .filter(PriceObservation.instrument_id == instrument_id)
            .scalar()
        )

    def count_activity(self, instrument_id: int) -> int:
        return (
            self.db.query(func.count(ActivityFrequency.id))
            .filter(ActivityFrequency.instrument_id == instrument_id)
            .scalar()
        )


class PriceObservationRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_for_instrument(self, instrument_id: int, limit: int = 200) -> List[PriceObservation]:
        return (
            self.db.query(PriceObservation)
            .filter(PriceObservation.instrument_id == instrument_id)
            .order_by(PriceObservation.sequence_ordinal)
            .limit(limit)
            .all()
        )

    def bulk_insert(self, records: List[PriceObservation]) -> None:
        self.db.bulk_save_objects(records)
        self.db.commit()

    def clear_for_instrument(self, instrument_id: int) -> None:
        self.db.query(PriceObservation).filter(
            PriceObservation.instrument_id == instrument_id
        ).delete()
        self.db.commit()


class TimeIndexRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_for_instrument(self, instrument_id: int, limit: int = 200) -> List[TimeIndex]:
        return (
            self.db.query(TimeIndex)
            .filter(TimeIndex.instrument_id == instrument_id)
            .order_by(TimeIndex.sequence_ordinal)
            .limit(limit)
            .all()
        )

    def bulk_insert(self, records: List[TimeIndex]) -> None:
        self.db.bulk_save_objects(records)
        self.db.commit()

    def clear_for_instrument(self, instrument_id: int) -> None:
        self.db.query(TimeIndex).filter(
            TimeIndex.instrument_id == instrument_id
        ).delete()
        self.db.commit()


class ActivityFrequencyRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_for_instrument(self, instrument_id: int, limit: int = 200) -> List[ActivityFrequency]:
        return (
            self.db.query(ActivityFrequency)
            .filter(ActivityFrequency.instrument_id == instrument_id)
            .order_by(ActivityFrequency.sequence_ordinal)
            .limit(limit)
            .all()
        )

    def bulk_insert(self, records: List[ActivityFrequency]) -> None:
        self.db.bulk_save_objects(records)
        self.db.commit()

    def clear_for_instrument(self, instrument_id: int) -> None:
        self.db.query(ActivityFrequency).filter(
            ActivityFrequency.instrument_id == instrument_id
        ).delete()
        self.db.commit()
