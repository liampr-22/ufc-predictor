"""
Unit tests for the background scraper job (scraper/jobs.py).

Uses SQLite in-memory — no real scraping, no real ML training.
All external side-effects (scrape_events, run_replay, ml.train.run) are patched.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

from api.database import get_db
from api.main import app
from models.schema import Base, Fight, Fighter, ScrapeJob
from scraper.jobs import run_incremental_scrape


# ---------------------------------------------------------------------------
# Shared in-memory DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def db_url(engine):
    """Return an SQLAlchemy URL that points at the shared in-memory engine.

    We monkey-patch create_engine inside jobs.py to return our fixture engine
    so both the job and the assertions share the same database.
    """
    return "sqlite:///:memory:"


# ---------------------------------------------------------------------------
# Helper: patch all external I/O so the job runs in-memory
# ---------------------------------------------------------------------------

def _patch_job(monkeypatch, fights_added: int = 0, raise_on_scrape: Exception | None = None):
    """Return a context-manager stack that patches all network / ML calls."""
    def fake_scrape_events(session, scraper, since=None):
        if raise_on_scrape:
            raise raise_on_scrape
        return (1, fights_added)

    patches = [
        patch("scraper.jobs.scrape_events", side_effect=fake_scrape_events),
        patch("scraper.jobs.UFCStatsScraper"),
        patch("scraper.jobs._latest_fight_date", return_value=date(2024, 1, 1)),
    ]
    return patches


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScrapeJobLogging:
    def test_job_created_and_marked_success_no_fights(self, engine, monkeypatch):
        """A run that inserts 0 fights should create a success ScrapeJob row."""
        with (
            patch("scraper.jobs.scrape_events", return_value=(1, 0)),
            patch("scraper.jobs.UFCStatsScraper"),
            patch("scraper.jobs._latest_fight_date", return_value=date(2024, 1, 1)),
            patch("scraper.jobs.create_engine", return_value=engine),
        ):
            run_incremental_scrape(database_url="sqlite:///:memory:")

        with Session(engine) as s:
            job = s.execute(select(ScrapeJob)).scalar_one()
        assert job.status == "success"
        assert job.fights_added == 0
        assert job.started_at is not None
        assert job.finished_at is not None
        assert job.error is None

    def test_job_records_fights_added(self, engine, monkeypatch):
        """fights_added count from scrape_events is persisted on the ScrapeJob row."""
        with (
            patch("scraper.jobs.scrape_events", return_value=(2, 3)),
            patch("scraper.jobs.UFCStatsScraper"),
            patch("scraper.jobs._latest_fight_date", return_value=date(2024, 1, 1)),
            patch("scraper.jobs.create_engine", return_value=engine),
            patch("ml.elo.run_replay"),
        ):
            run_incremental_scrape(database_url="sqlite:///:memory:")

        with Session(engine) as s:
            job = s.execute(select(ScrapeJob)).scalar_one()
        assert job.status == "success"
        assert job.fights_added == 3

    def test_job_marked_failed_on_exception(self, engine):
        """An exception during scraping must flip status to 'failed' and store error text."""
        with (
            patch("scraper.jobs.scrape_events", side_effect=RuntimeError("network timeout")),
            patch("scraper.jobs.UFCStatsScraper"),
            patch("scraper.jobs._latest_fight_date", return_value=date(2024, 1, 1)),
            patch("scraper.jobs.create_engine", return_value=engine),
        ):
            with pytest.raises(RuntimeError):
                run_incremental_scrape(database_url="sqlite:///:memory:")

        with Session(engine) as s:
            job = s.execute(select(ScrapeJob)).scalar_one()
        assert job.status == "failed"
        assert "network timeout" in job.error
        assert job.finished_at is not None


class TestEloAndRetrain:
    def test_elo_refresh_called_when_fights_added(self, engine):
        """run_replay must be called when fights_added > 0."""
        with (
            patch("scraper.jobs.scrape_events", return_value=(1, 2)),
            patch("scraper.jobs.UFCStatsScraper"),
            patch("scraper.jobs._latest_fight_date", return_value=date(2024, 1, 1)),
            patch("scraper.jobs.create_engine", return_value=engine),
            patch("ml.elo.run_replay") as mock_replay,
        ):
            run_incremental_scrape(database_url="sqlite:///:memory:")

        mock_replay.assert_called_once()

    def test_elo_refresh_skipped_when_no_fights(self, engine):
        """run_replay must NOT be called when fights_added == 0."""
        with (
            patch("scraper.jobs.scrape_events", return_value=(1, 0)),
            patch("scraper.jobs.UFCStatsScraper"),
            patch("scraper.jobs._latest_fight_date", return_value=date(2024, 1, 1)),
            patch("scraper.jobs.create_engine", return_value=engine),
            patch("ml.elo.run_replay") as mock_replay,
        ):
            run_incremental_scrape(database_url="sqlite:///:memory:")

        mock_replay.assert_not_called()

    def test_retrain_triggered_at_threshold(self, engine):
        """ml.train.run must be called when fights_added >= retrain_threshold."""
        mock_train = MagicMock()
        with (
            patch("scraper.jobs.scrape_events", return_value=(1, 5)),
            patch("scraper.jobs.UFCStatsScraper"),
            patch("scraper.jobs._latest_fight_date", return_value=date(2024, 1, 1)),
            patch("scraper.jobs.create_engine", return_value=engine),
            patch("ml.elo.run_replay"),
            patch("ml.train.run", mock_train),
        ):
            run_incremental_scrape(database_url="sqlite:///:memory:", retrain_threshold=5)

        mock_train.assert_called_once()

    def test_retrain_not_triggered_below_threshold(self, engine):
        """ml.train.run must NOT be called when fights_added < retrain_threshold."""
        mock_train = MagicMock()
        with (
            patch("scraper.jobs.scrape_events", return_value=(1, 4)),
            patch("scraper.jobs.UFCStatsScraper"),
            patch("scraper.jobs._latest_fight_date", return_value=date(2024, 1, 1)),
            patch("scraper.jobs.create_engine", return_value=engine),
            patch("ml.elo.run_replay"),
            patch("ml.train.run", mock_train),
        ):
            run_incremental_scrape(database_url="sqlite:///:memory:", retrain_threshold=5)

        mock_train.assert_not_called()

    def test_predictor_reloaded_after_retrain(self, engine):
        """app.state.predictor must be replaced after a successful retrain."""
        mock_app = MagicMock()
        mock_predictor = MagicMock()
        with (
            patch("scraper.jobs.scrape_events", return_value=(1, 5)),
            patch("scraper.jobs.UFCStatsScraper"),
            patch("scraper.jobs._latest_fight_date", return_value=date(2024, 1, 1)),
            patch("scraper.jobs.create_engine", return_value=engine),
            patch("ml.elo.run_replay"),
            patch("ml.train.run"),
            patch("ml.predict.Predictor", return_value=mock_predictor),
        ):
            run_incremental_scrape(
                database_url="sqlite:///:memory:",
                app=mock_app,
                retrain_threshold=5,
            )

        assert mock_app.state.predictor is mock_predictor


# ---------------------------------------------------------------------------
# /health endpoint shows last_successful_scrape
# ---------------------------------------------------------------------------

class _MockPredictor:
    def predict(self, session, fighter_a_id, fighter_b_id, as_of_date=None):
        return {"win_prob_a": 0.5, "method_probs": {"ko_tko": 0.3, "submission": 0.2, "decision": 0.5}}


@pytest.fixture()
def health_client(engine):
    """TestClient wired to the in-memory engine, with a seed ScrapeJob row."""
    with Session(engine) as s:
        fa = Fighter(id=10, name="A", elo_rating=1500.0)
        fb = Fighter(id=11, name="B", elo_rating=1500.0)
        s.add_all([fa, fb])
        s.add(Fight(
            id=10,
            date=date(2024, 6, 1),
            fighter_a_id=10,
            fighter_b_id=11,
        ))
        s.add(ScrapeJob(
            started_at=datetime(2024, 6, 1, 6, 0, tzinfo=timezone.utc),
            finished_at=datetime(2024, 6, 1, 6, 5, tzinfo=timezone.utc),
            fights_added=3,
            status="success",
        ))
        s.commit()

    def override_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    with TestClient(app, raise_server_exceptions=True) as c:
        app.state.predictor = _MockPredictor()
        yield c
    app.dependency_overrides.clear()
    app.state.predictor = None


def test_health_shows_last_successful_scrape(health_client):
    resp = health_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["last_successful_scrape"] is not None
    assert "2024-06-01" in data["last_successful_scrape"]


def test_health_last_successful_scrape_none_when_no_jobs():
    """When no success rows exist, last_successful_scrape should be null."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)

    def override_db():
        with Session(eng) as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app, raise_server_exceptions=True) as c:
            app.state.predictor = _MockPredictor()
            resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["last_successful_scrape"] is None
    finally:
        app.dependency_overrides.clear()
        app.state.predictor = None
        Base.metadata.drop_all(eng)
