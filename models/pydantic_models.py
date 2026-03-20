"""
Pydantic request/response schemas.
Phase 1 provides base read schemas for seed validation.
Full API schemas implemented in Phase 8 — REST API.
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class FighterRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    height: Optional[float] = None
    reach: Optional[float] = None
    stance: Optional[str] = None
    dob: Optional[date] = None
    weight_class: Optional[str] = None
    elo_rating: float


class FightRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    date: date
    event: Optional[str] = None
    fighter_a_id: int
    fighter_b_id: int
    winner_id: Optional[int] = None
    method: Optional[str] = None
    round: Optional[int] = None
    time: Optional[str] = None


class FightStatsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    fight_id: int
    fighter_id: int
    knockdowns: Optional[int] = None
    significant_strikes_landed: Optional[int] = None
    significant_strikes_attempted: Optional[int] = None
    total_strikes_landed: Optional[int] = None
    total_strikes_attempted: Optional[int] = None
    takedowns_landed: Optional[int] = None
    takedowns_attempted: Optional[int] = None
    submission_attempts: Optional[int] = None
    control_time_seconds: Optional[int] = None


class HistoricalOddsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    fight_id: int
    fighter_a_odds: Optional[int] = None
    fighter_b_odds: Optional[int] = None
    source: Optional[str] = None
    recorded_at: Optional[datetime] = None
