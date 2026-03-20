"""
Migration tests — verify schema applies cleanly on SQLite in-memory.
Runs as part of the standard pytest suite with no Docker or PostgreSQL required.
"""

from datetime import date

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from models.schema import Base, Fight, FightStats, Fighter, HistoricalOdds, RoundStats


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    """In-memory SQLite engine with the full schema applied via SQLAlchemy metadata."""
    eng = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture(scope="module")
def two_fighters(engine):
    """Reusable pair of fighters inserted once for the module."""
    with Session(engine) as session:
        fa = Fighter(name="Fighter A", elo_rating=1500.0)
        fb = Fighter(name="Fighter B", elo_rating=1500.0)
        session.add_all([fa, fb])
        session.commit()
        # Return plain IDs to avoid detached-instance errors across sessions
        return fa.id, fb.id


# ── Table existence ───────────────────────────────────────────────────────────

def test_all_tables_created(engine):
    tables = set(inspect(engine).get_table_names())
    expected = {"fighters", "fights", "fight_stats", "round_stats", "historical_odds"}
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"


# ── Column presence ───────────────────────────────────────────────────────────

def test_fighters_columns(engine):
    cols = {c["name"] for c in inspect(engine).get_columns("fighters")}
    for col in ["id", "name", "height", "reach", "stance", "dob", "weight_class", "elo_rating"]:
        assert col in cols, f"Missing column: {col}"


def test_fights_columns(engine):
    cols = {c["name"] for c in inspect(engine).get_columns("fights")}
    for col in ["id", "date", "event", "fighter_a_id", "fighter_b_id", "winner_id", "method", "round", "time"]:
        assert col in cols, f"Missing column: {col}"


def test_fight_stats_columns(engine):
    cols = {c["name"] for c in inspect(engine).get_columns("fight_stats")}
    for col in [
        "id", "fight_id", "fighter_id",
        "knockdowns",
        "significant_strikes_landed", "significant_strikes_attempted",
        "total_strikes_landed", "total_strikes_attempted",
        "head_strikes_landed", "body_strikes_landed", "leg_strikes_landed",
        "distance_strikes_landed", "clinch_strikes_landed", "ground_strikes_landed",
        "takedowns_landed", "takedowns_attempted",
        "submission_attempts", "reversals", "control_time_seconds",
    ]:
        assert col in cols, f"Missing column: {col}"


def test_round_stats_columns(engine):
    cols = {c["name"] for c in inspect(engine).get_columns("round_stats")}
    for col in ["id", "fight_id", "fighter_id", "round_number", "control_time_seconds"]:
        assert col in cols, f"Missing column: {col}"


def test_historical_odds_columns(engine):
    cols = {c["name"] for c in inspect(engine).get_columns("historical_odds")}
    for col in ["id", "fight_id", "fighter_a_odds", "fighter_b_odds", "source", "recorded_at"]:
        assert col in cols, f"Missing column: {col}"


# ── Relationship round-trip ───────────────────────────────────────────────────

def test_fighter_fight_roundtrip(engine, two_fighters):
    fa_id, fb_id = two_fighters
    with Session(engine) as session:
        fight = Fight(
            date=date(2024, 1, 1), event="Test Event",
            fighter_a_id=fa_id, fighter_b_id=fb_id,
            winner_id=fa_id, method="KO/TKO", round=1, time="1:23",
        )
        session.add(fight)
        session.commit()
        fight_id = fight.id

    with Session(engine) as session:
        loaded = session.get(Fight, fight_id)
        assert loaded is not None
        assert loaded.fighter_a.name == "Fighter A"
        assert loaded.fighter_b.name == "Fighter B"
        assert loaded.winner.name == "Fighter A"


# ── Nullable field checks ─────────────────────────────────────────────────────

def test_nullable_fighter_fields(engine):
    with Session(engine) as session:
        f = Fighter(name="Minimal Fighter", elo_rating=1500.0)
        session.add(f)
        session.commit()
        fid = f.id

    with Session(engine) as session:
        loaded = session.get(Fighter, fid)
        assert loaded.height is None
        assert loaded.reach is None
        assert loaded.dob is None
        assert loaded.weight_class is None


def test_winner_id_nullable(engine, two_fighters):
    fa_id, fb_id = two_fighters
    with Session(engine) as session:
        fight = Fight(
            date=date(2024, 6, 1), event="Draw Event",
            fighter_a_id=fa_id, fighter_b_id=fb_id,
            winner_id=None, method="NC",
        )
        session.add(fight)
        session.commit()
        fid = fight.id

    with Session(engine) as session:
        loaded = session.get(Fight, fid)
        assert loaded.winner_id is None
        assert loaded.winner is None


# ── Child table cascade ───────────────────────────────────────────────────────

def test_fight_stats_cascade_delete(engine, two_fighters):
    fa_id, fb_id = two_fighters
    with Session(engine) as session:
        fight = Fight(
            date=date(2024, 3, 1), event="Cascade Test",
            fighter_a_id=fa_id, fighter_b_id=fb_id,
            winner_id=fa_id, method="SUB",
        )
        session.add(fight)
        session.flush()

        stat = FightStats(fight_id=fight.id, fighter_id=fa_id, knockdowns=1)
        session.add(stat)
        session.commit()
        fight_id = fight.id
        stat_id = stat.id

    with Session(engine) as session:
        fight = session.get(Fight, fight_id)
        session.delete(fight)
        session.commit()

    with Session(engine) as session:
        assert session.get(FightStats, stat_id) is None, "FightStats should be cascade-deleted"


# ── Alembic upgrade/downgrade via SQLite file DB ──────────────────────────────

def test_alembic_upgrade_downgrade(tmp_path, monkeypatch):
    """Run alembic upgrade head then downgrade base against a file-based SQLite DB."""
    import os
    from alembic import command
    from alembic.config import Config

    db_path = tmp_path / "test_migration.db"
    sqlite_url = f"sqlite:///{db_path}"

    # Override DATABASE_URL so env.py uses SQLite instead of the real PostgreSQL URL
    monkeypatch.setenv("DATABASE_URL", sqlite_url)

    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", sqlite_url)

    command.upgrade(alembic_cfg, "head")

    # Verify tables exist after upgrade
    from sqlalchemy import create_engine as ce, inspect as ins
    eng = ce(f"sqlite:///{db_path}")
    tables = set(ins(eng).get_table_names())
    assert {"fighters", "fights", "fight_stats", "round_stats", "historical_odds"}.issubset(tables)
    eng.dispose()

    command.downgrade(alembic_cfg, "base")

    # Verify tables are gone after downgrade
    eng = ce(f"sqlite:///{db_path}")
    tables_after = set(ins(eng).get_table_names())
    assert "fighters" not in tables_after
    eng.dispose()
