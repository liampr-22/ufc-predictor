"""
SQLAlchemy ORM models.
Implemented in Phase 1 — Database Schema & Migrations.
"""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Fighter(Base):
    __tablename__ = "fighters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # inches
    reach: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # inches
    stance: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)   # Orthodox / Southpaw / Switch
    dob: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    weight_class: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    elo_rating: Mapped[float] = mapped_column(Float, nullable=False, default=1500.0)
    ufcstats_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, unique=True)

    # Relationships
    fights_as_a: Mapped[list["Fight"]] = relationship(
        "Fight", foreign_keys="Fight.fighter_a_id", back_populates="fighter_a"
    )
    fights_as_b: Mapped[list["Fight"]] = relationship(
        "Fight", foreign_keys="Fight.fighter_b_id", back_populates="fighter_b"
    )
    fights_won: Mapped[list["Fight"]] = relationship(
        "Fight", foreign_keys="Fight.winner_id", back_populates="winner"
    )
    fight_stats: Mapped[list["FightStats"]] = relationship(
        "FightStats", back_populates="fighter"
    )
    round_stats: Mapped[list["RoundStats"]] = relationship(
        "RoundStats", back_populates="fighter"
    )

    __table_args__ = (
        Index("ix_fighters_name", "name"),
        Index("ix_fighters_weight_class", "weight_class"),
        Index("ix_fighters_ufcstats_id", "ufcstats_id", unique=True),
    )


class Fight(Base):
    __tablename__ = "fights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    event: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    fighter_a_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("fighters.id", ondelete="RESTRICT"), nullable=False
    )
    fighter_b_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("fighters.id", ondelete="RESTRICT"), nullable=False
    )
    winner_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("fighters.id", ondelete="RESTRICT"), nullable=True
    )  # NULL = No Contest / Draw
    method: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # KO/TKO, SUB, DEC, NC
    round: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    time: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)     # "4:32" format
    ufcstats_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, unique=True)

    # Relationships
    fighter_a: Mapped["Fighter"] = relationship(
        "Fighter", foreign_keys=[fighter_a_id], back_populates="fights_as_a"
    )
    fighter_b: Mapped["Fighter"] = relationship(
        "Fighter", foreign_keys=[fighter_b_id], back_populates="fights_as_b"
    )
    winner: Mapped[Optional["Fighter"]] = relationship(
        "Fighter", foreign_keys=[winner_id], back_populates="fights_won"
    )
    fight_stats: Mapped[list["FightStats"]] = relationship(
        "FightStats", back_populates="fight", cascade="all, delete-orphan"
    )
    round_stats: Mapped[list["RoundStats"]] = relationship(
        "RoundStats", back_populates="fight", cascade="all, delete-orphan"
    )
    historical_odds: Mapped[list["HistoricalOdds"]] = relationship(
        "HistoricalOdds", back_populates="fight", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_fights_date", "date"),
        Index("ix_fights_fighter_a_id", "fighter_a_id"),
        Index("ix_fights_fighter_b_id", "fighter_b_id"),
        Index("ix_fights_ufcstats_id", "ufcstats_id", unique=True),
    )


class FightStats(Base):
    """
    One row per fighter per fight. Two rows exist per fight (one per corner).
    The feature pipeline joins both rows and computes differentials.
    """
    __tablename__ = "fight_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fight_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("fights.id", ondelete="CASCADE"), nullable=False
    )
    fighter_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("fighters.id", ondelete="CASCADE"), nullable=False
    )

    # Striking — volume
    knockdowns: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    significant_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    significant_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Striking — by target
    head_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    head_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    body_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    body_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    leg_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    leg_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Striking — by position
    distance_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    distance_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    clinch_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    clinch_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ground_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ground_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Grappling
    takedowns_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    takedowns_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    submission_attempts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reversals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    control_time_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    fight: Mapped["Fight"] = relationship("Fight", back_populates="fight_stats")
    fighter: Mapped["Fighter"] = relationship("Fighter", back_populates="fight_stats")

    __table_args__ = (
        Index("ix_fight_stats_fight_id", "fight_id"),
        Index("ix_fight_stats_fighter_id", "fighter_id"),
    )


class RoundStats(Base):
    """
    Same stat dimensions as FightStats, broken down by round.
    One row per fighter per round per fight.
    """
    __tablename__ = "round_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fight_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("fights.id", ondelete="CASCADE"), nullable=False
    )
    fighter_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("fighters.id", ondelete="CASCADE"), nullable=False
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Striking — volume
    knockdowns: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    significant_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    significant_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Striking — by target
    head_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    head_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    body_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    body_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    leg_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    leg_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Striking — by position
    distance_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    distance_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    clinch_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    clinch_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ground_strikes_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ground_strikes_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Grappling
    takedowns_landed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    takedowns_attempted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    submission_attempts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reversals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    control_time_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    fight: Mapped["Fight"] = relationship("Fight", back_populates="round_stats")
    fighter: Mapped["Fighter"] = relationship("Fighter", back_populates="round_stats")

    __table_args__ = (
        Index("ix_round_stats_fight_id", "fight_id"),
        Index("ix_round_stats_fighter_id", "fighter_id"),
    )


class ScrapeJob(Base):
    __tablename__ = "scrape_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    fights_added: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # "running" | "success" | "failed"
    error: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)

    __table_args__ = (
        Index("ix_scrape_jobs_started_at", "started_at"),
        Index("ix_scrape_jobs_status", "status"),
    )


class HistoricalOdds(Base):
    __tablename__ = "historical_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fight_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("fights.id", ondelete="CASCADE"), nullable=False
    )
    fighter_a_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # American odds e.g. -175
    fighter_b_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # American odds e.g. +145
    source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)      # e.g. "BestFightOdds"
    recorded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=text("CURRENT_TIMESTAMP")
    )

    # Relationship
    fight: Mapped["Fight"] = relationship("Fight", back_populates="historical_odds")

    __table_args__ = (
        Index("ix_historical_odds_fight_id", "fight_id"),
    )
