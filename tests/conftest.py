"""
Shared fixtures for API integration tests.

Uses SQLite in-memory so tests run without a live PostgreSQL instance.
The get_db dependency is overridden via app.dependency_overrides.
"""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

from api.database import get_db
from api.main import app
from models.schema import Base, Fight, FightStats, Fighter


class _MockPredictor:
    """Stub predictor that returns fixed probabilities without loading model files."""

    def predict(self, session, fighter_a_id: int, fighter_b_id: int, as_of_date=None) -> dict:
        return {
            "win_prob_a": 0.65,
            "method_probs": {
                "ko_tko": 0.30,
                "submission": 0.15,
                "decision": 0.55,
            },
        }


@pytest.fixture(scope="module")
def db_session():
    """In-memory SQLite session seeded with two fighters and two fights."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Two fighters
        fa = Fighter(
            id=1,
            name="Jon Jones",
            height=76.0,
            reach=84.5,
            stance="Orthodox",
            dob=date(1987, 7, 19),
            weight_class="Light Heavyweight",
            elo_rating=1600.0,
        )
        fb = Fighter(
            id=2,
            name="Stipe Miocic",
            height=77.0,
            reach=80.0,
            stance="Orthodox",
            dob=date(1982, 8, 19),
            weight_class="Heavyweight",
            elo_rating=1520.0,
        )
        session.add_all([fa, fb])
        session.flush()

        # One completed fight (fa won)
        completed = Fight(
            id=1,
            date=date(2023, 3, 4),
            event="UFC 285",
            fighter_a_id=fa.id,
            fighter_b_id=fb.id,
            winner_id=fa.id,
            method="KO/TKO",
            round=1,
            time="2:04",
        )
        # One upcoming fight (no winner)
        upcoming = Fight(
            id=2,
            date=date(2099, 12, 31),
            event="UFC 999",
            fighter_a_id=fa.id,
            fighter_b_id=fb.id,
            winner_id=None,
        )
        session.add_all([completed, upcoming])

        # Fight stats for the completed fight
        stats_a = FightStats(
            fight_id=1,
            fighter_id=fa.id,
            knockdowns=1,
            significant_strikes_landed=45,
            significant_strikes_attempted=80,
            takedowns_landed=2,
            takedowns_attempted=4,
            submission_attempts=0,
            control_time_seconds=120,
        )
        stats_b = FightStats(
            fight_id=1,
            fighter_id=fb.id,
            knockdowns=0,
            significant_strikes_landed=20,
            significant_strikes_attempted=50,
            takedowns_landed=0,
            takedowns_attempted=1,
            submission_attempts=0,
            control_time_seconds=0,
        )
        session.add_all([stats_a, stats_b])
        session.commit()

        yield session

    Base.metadata.drop_all(engine)


@pytest.fixture(scope="module")
def client(db_session):
    """TestClient with get_db overridden to use the in-memory session."""
    app.dependency_overrides[get_db] = lambda: db_session

    # lifespan runs on TestClient __enter__ and may set predictor to None
    # (models not found). Override it after the client has started.
    with TestClient(app, raise_server_exceptions=True) as c:
        app.state.predictor = _MockPredictor()
        yield c

    app.dependency_overrides.clear()
    app.state.predictor = None


@pytest.fixture(scope="module")
def client_no_model(db_session):
    """TestClient with predictor explicitly set to None (model not loaded)."""
    app.dependency_overrides[get_db] = lambda: db_session

    with TestClient(app, raise_server_exceptions=True) as c:
        app.state.predictor = None
        yield c

    app.dependency_overrides.clear()
