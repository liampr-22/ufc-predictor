"""
Pydantic request/response schemas.
Phase 1 provides base read schemas for seed validation.
Full API schemas implemented in Phase 8 — REST API.
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Phase 8 — API response schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    last_scrape: Optional[str] = None
    last_successful_scrape: Optional[str] = None


class FighterSearchResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    weight_class: Optional[str] = None
    elo_rating: float


class CareerAverages(BaseModel):
    fights: int
    wins: int
    losses: int
    draws: int
    avg_sig_strikes_landed: Optional[float] = None
    avg_takedowns_landed: Optional[float] = None
    avg_control_time_seconds: Optional[float] = None
    ko_rate: float
    sub_rate: float
    dec_rate: float


class StyleScores(BaseModel):
    striker: float
    wrestler: float
    grappler: float
    brawler: float


class FighterProfile(BaseModel):
    id: int
    name: str
    height: Optional[float] = None
    reach: Optional[float] = None
    stance: Optional[str] = None
    dob: Optional[date] = None
    weight_class: Optional[str] = None
    elo_rating: float
    career: CareerAverages
    style: StyleScores


class FightHistoryEntry(BaseModel):
    fight_id: int
    date: date
    event: Optional[str] = None
    opponent: str
    result: str  # "Win", "Loss", "Draw", "NC"
    method: Optional[str] = None
    round: Optional[int] = None
    time: Optional[str] = None


class FightHistoryResponse(BaseModel):
    fighter: str
    page: int
    page_size: int
    total: int
    fights: list[FightHistoryEntry]


class PredictionRequest(BaseModel):
    fighter_a: str
    fighter_b: str
    scheduled_rounds: Optional[int] = None  # 3 or 5; None = unknown (model defaults to 3-round distribution)


class MethodProbs(BaseModel):
    ko_tko: float
    submission: float
    decision: float


class FinishRates(BaseModel):
    ko_tko: float
    submission: float
    decision: float


class FighterOdds(BaseModel):
    american: float
    decimal: float


class KeyDifferentials(BaseModel):
    elo_delta: float
    reach_delta: Optional[float] = None
    height_delta: Optional[float] = None
    age_delta: Optional[float] = None


class FighterPrediction(BaseModel):
    name: str
    win_prob: float
    odds: FighterOdds


class PredictionResponse(BaseModel):
    fighter_a: FighterPrediction
    fighter_b: FighterPrediction
    method_probs: MethodProbs
    key_differentials: KeyDifferentials
    fighter_a_finish_rates: FinishRates
    fighter_b_finish_rates: FinishRates
    fighter_a_win_method_rates: FinishRates
    fighter_b_win_method_rates: FinishRates


class ScheduledFight(BaseModel):
    fight_id: int
    fighter_a: str
    fighter_b: str
    prediction: Optional[PredictionResponse] = None


class UpcomingEvent(BaseModel):
    event: str
    date: date
    fights: list[ScheduledFight]


class UpcomingEventsResponse(BaseModel):
    events: list[UpcomingEvent]


class ScrapeResponse(BaseModel):
    status: str
    message: str


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
    ufcstats_id: Optional[str] = None


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
    scheduled_rounds: Optional[int] = None
    ufcstats_id: Optional[str] = None


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
