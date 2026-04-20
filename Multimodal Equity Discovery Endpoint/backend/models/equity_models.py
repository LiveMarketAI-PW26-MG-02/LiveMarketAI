from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger, ForeignKey, Index
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime


class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    exchange = Column(String(50), nullable=False, default="NSE")
    sector = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    price_observations = relationship("PriceObservation", back_populates="instrument", cascade="all, delete-orphan")
    time_indices = relationship("TimeIndex", back_populates="instrument", cascade="all, delete-orphan")
    activity_frequencies = relationship("ActivityFrequency", back_populates="instrument", cascade="all, delete-orphan")


class PriceObservation(Base):
    __tablename__ = "price_observations"

    id = Column(BigInteger, primary_key=True, index=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False, index=True)
    closing_price = Column(Float, nullable=False)
    observed_at = Column(DateTime, nullable=False, index=True)
    sequence_ordinal = Column(Integer, nullable=False)

    instrument = relationship("Instrument", back_populates="price_observations")

    __table_args__ = (
        Index("ix_price_obs_inst_seq", "instrument_id", "sequence_ordinal"),
    )


class TimeIndex(Base):
    __tablename__ = "time_indices"

    id = Column(BigInteger, primary_key=True, index=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False, index=True)
    time_marker = Column(Float, nullable=False)
    reference_at = Column(DateTime, nullable=False, index=True)
    sequence_ordinal = Column(Integer, nullable=False)

    instrument = relationship("Instrument", back_populates="time_indices")

    __table_args__ = (
        Index("ix_time_idx_inst_seq", "instrument_id", "sequence_ordinal"),
    )


class ActivityFrequency(Base):
    __tablename__ = "activity_frequencies"

    id = Column(BigInteger, primary_key=True, index=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False, index=True)
    frequency_count = Column(Integer, nullable=False)
    interval_seconds = Column(Float, nullable=False)
    recorded_at = Column(DateTime, nullable=False, index=True)
    sequence_ordinal = Column(Integer, nullable=False)

    instrument = relationship("Instrument", back_populates="activity_frequencies")

    __table_args__ = (
        Index("ix_act_freq_inst_seq", "instrument_id", "sequence_ordinal"),
    )
