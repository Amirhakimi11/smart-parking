from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Index,
                        Integer, String, create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATABASE_URL

_is_sqlite = DATABASE_URL.startswith("sqlite")
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


class Zone(Base):
    __tablename__ = "zones"
    id = Column(String(16), primary_key=True)        # e.g. "L1-A"
    floor = Column(String(8), nullable=False)        # e.g. "L1"
    floor_name = Column(String(32), nullable=False)  # e.g. "Level 1"
    name = Column(String(64), nullable=False)
    capacity = Column(Integer, nullable=False)
    is_open = Column(Boolean, default=True, nullable=False)
    source = Column(String(16), default="simulated", nullable=False)  # simulated | yolov8
    sort = Column(Integer, default=0)


class Reading(Base):
    __tablename__ = "readings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_id = Column(String(16), ForeignKey("zones.id"), nullable=False)
    ts = Column(DateTime, nullable=False)
    occupied = Column(Integer, nullable=False)
    __table_args__ = (Index("ix_readings_zone_ts", "zone_id", "ts"),)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_id = Column(String(16), ForeignKey("zones.id"), nullable=False)
    made_at = Column(DateTime, nullable=False)
    target_ts = Column(DateTime, nullable=False)
    horizon_min = Column(Integer, nullable=False)
    predicted_occupied = Column(Float, nullable=False)
    __table_args__ = (Index("ix_pred_target", "target_ts"),)


def init_db():
    Base.metadata.create_all(engine)
