"""
Elo rating system tests.

Groups:
  1. Pure math     — no DB required
  2. Replay logic  — plain dicts, no DB
  3. Backtest      — plain dicts, no DB
  4. DB layer      — module-scoped SQLite in-memory (same pattern as test_migrations.py)
"""

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ml.elo import (
    BASE_K,
    INITIAL_ELO,
    backtest,
    effective_k,
    expected_score,
    load_fights_from_db,
    persist_ratings,
    replay_fights,
    update_rating,
)
from models.schema import Base, Fight, Fighter


# ── DB fixture ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fight(fa_id: int, fb_id: int, winner_id: int | None, fight_date: date, fid: int = 1) -> dict:
    return {
        "id": fid,
        "date": fight_date,
        "fighter_a_id": fa_id,
        "fighter_b_id": fb_id,
        "winner_id": winner_id,
    }


# ── Group 1: Pure math ────────────────────────────────────────────────────────

def test_expected_score_equal_ratings():
    assert expected_score(1500.0, 1500.0) == 0.5


def test_expected_score_higher_rated_favourite():
    assert expected_score(1600.0, 1500.0) > 0.5


def test_expected_score_lower_rated_underdog():
    assert expected_score(1400.0, 1500.0) < 0.5


def test_expected_score_known_value():
    # E_a = 1 / (1 + 10 ** ((1400 - 1600) / 400)) = 1 / (1 + 10 ** -0.5)
    expected_manual = 1.0 / (1.0 + 10.0 ** -0.5)
    assert abs(expected_score(1600.0, 1400.0) - expected_manual) < 1e-9


def test_update_rating_win():
    result = update_rating(1500.0, 1.0, 0.5, 32.0)
    assert result == 1516.0


def test_update_rating_loss():
    result = update_rating(1500.0, 0.0, 0.5, 32.0)
    assert result == 1484.0


def test_update_rating_draw():
    result = update_rating(1500.0, 0.5, 0.5, 32.0)
    assert result == 1500.0


def test_effective_k_recent_fight():
    # 1 year old — under threshold
    result = effective_k(date(2024, 1, 1), date(2025, 1, 1), base_k=32.0)
    assert result == 32.0


def test_effective_k_old_fight():
    # 10 years old — decayed
    result = effective_k(date(2015, 1, 1), date(2025, 1, 1), base_k=32.0)
    assert result == 16.0


def test_effective_k_exactly_at_threshold():
    # Exactly 5 years old — should decay (>= threshold)
    result = effective_k(date(2020, 1, 1), date(2025, 1, 1), base_k=32.0)
    assert result == 16.0


# ── Group 2: Replay logic ─────────────────────────────────────────────────────

REF_DATE = date(2025, 1, 1)  # fixed reference so recency decay is deterministic


def test_replay_single_win():
    fights = [_fight(1, 2, winner_id=1, fight_date=date(2024, 6, 1))]
    ratings = replay_fights(fights, reference_date=REF_DATE)
    assert ratings[1] > INITIAL_ELO
    assert ratings[2] < INITIAL_ELO
    # Zero-sum check
    assert abs(ratings[1] + ratings[2] - 3000.0) < 1e-9


def test_replay_draw_no_change_at_equal_ratings():
    fights = [_fight(1, 2, winner_id=None, fight_date=date(2024, 6, 1))]
    ratings = replay_fights(fights, reference_date=REF_DATE)
    assert ratings[1] == INITIAL_ELO
    assert ratings[2] == INITIAL_ELO


def test_replay_all_fighters_present_in_output():
    fights = [
        _fight(1, 2, winner_id=1, fight_date=date(2023, 1, 1), fid=1),
        _fight(2, 3, winner_id=2, fight_date=date(2023, 6, 1), fid=2),
    ]
    ratings = replay_fights(fights, reference_date=REF_DATE)
    assert 1 in ratings
    assert 2 in ratings
    assert 3 in ratings


def test_replay_resets_to_initial_rating():
    fights = [_fight(1, 2, winner_id=1, fight_date=date(2024, 1, 1))]
    result_1 = replay_fights(fights, reference_date=REF_DATE)
    result_2 = replay_fights(fights, reference_date=REF_DATE)
    assert result_1 == result_2


def test_replay_ordering_matters():
    # A wins twice, B wins once — A should end up rated higher
    fights = [
        _fight(1, 2, winner_id=1, fight_date=date(2022, 1, 1), fid=1),
        _fight(1, 2, winner_id=1, fight_date=date(2023, 1, 1), fid=2),
        _fight(1, 2, winner_id=2, fight_date=date(2024, 1, 1), fid=3),
    ]
    ratings = replay_fights(fights, reference_date=REF_DATE)
    assert ratings[1] > ratings[2]


# ── Group 3: Backtest ─────────────────────────────────────────────────────────

def _make_fights(n: int, winner_alternates: bool = False) -> list[dict]:
    """Generate n fights between fighter 1 and 2 across consecutive years."""
    fights = []
    for i in range(n):
        winner = 1 if (not winner_alternates or i % 2 == 0) else 2
        fights.append({
            "id": i + 1,
            "date": date(2010 + i, 6, 1),
            "fighter_a_id": 1,
            "fighter_b_id": 2,
            "winner_id": winner,
        })
    return fights


def test_backtest_returns_expected_keys():
    result = backtest(_make_fights(10))
    assert {"accuracy", "correct", "total", "train_size", "test_size", "skipped_draws"} == set(result.keys())


def test_backtest_split_sizes():
    result = backtest(_make_fights(10), holdout_fraction=0.20)
    assert result["train_size"] == 8
    assert result["test_size"] == 2


def test_backtest_skips_draws():
    fights = _make_fights(5)
    # Add one draw as the 6th fight
    fights.append({
        "id": 6,
        "date": date(2016, 6, 1),
        "fighter_a_id": 1,
        "fighter_b_id": 2,
        "winner_id": None,
    })
    # With holdout_fraction=0.20 and n=6 → split=5, test=[fight 6 (draw)]
    result = backtest(fights, holdout_fraction=0.20)
    assert result["skipped_draws"] == 1
    assert result["total"] == 0
    # Should not raise ZeroDivisionError; accuracy defaults to 0.0
    assert result["accuracy"] == 0.0


def test_backtest_perfect_predictor():
    # Train: fighter 1 wins 8 straight → rated much higher
    # Test: fighter 1 wins the 9th fight → perfect prediction
    fights = _make_fights(9, winner_alternates=False)
    result = backtest(fights, holdout_fraction=0.20)
    # With 9 fights, split = 9 - max(1, int(9*0.2)) = 9 - 1 = 8 train, 1 test
    assert result["train_size"] == 8
    assert result["total"] == 1
    assert result["accuracy"] == 1.0


# ── Group 4: DB layer ─────────────────────────────────────────────────────────

def test_load_fights_from_db_empty(engine):
    with Session(engine) as session:
        result = load_fights_from_db(session)
    assert result == []


def test_load_fights_from_db_ordered_by_date(engine):
    with Session(engine) as session:
        fa = Fighter(name="DB Fighter A", elo_rating=1500.0)
        fb = Fighter(name="DB Fighter B", elo_rating=1500.0)
        session.add_all([fa, fb])
        session.flush()

        # Insert out-of-order
        f2 = Fight(date=date(2020, 1, 1), fighter_a_id=fa.id, fighter_b_id=fb.id, winner_id=fa.id)
        f1 = Fight(date=date(2019, 1, 1), fighter_a_id=fa.id, fighter_b_id=fb.id, winner_id=fb.id)
        f3 = Fight(date=date(2021, 1, 1), fighter_a_id=fa.id, fighter_b_id=fb.id, winner_id=fa.id)
        session.add_all([f2, f1, f3])
        session.commit()
        fa_id, fb_id = fa.id, fb.id

    with Session(engine) as session:
        fights = load_fights_from_db(session)

    # Filter to just the fights we inserted (DB may have others from earlier tests)
    relevant = [f for f in fights if f["fighter_a_id"] == fa_id and f["fighter_b_id"] == fb_id]
    dates = [f["date"] for f in relevant]
    assert dates == sorted(dates), "Fights not returned in ascending date order"


def test_load_fights_from_db_draw_winner_none(engine):
    with Session(engine) as session:
        fa = Fighter(name="Draw Fighter A", elo_rating=1500.0)
        fb = Fighter(name="Draw Fighter B", elo_rating=1500.0)
        session.add_all([fa, fb])
        session.flush()
        fight = Fight(date=date(2022, 3, 15), fighter_a_id=fa.id, fighter_b_id=fb.id, winner_id=None)
        session.add(fight)
        session.commit()
        fa_id, fb_id = fa.id, fb.id

    with Session(engine) as session:
        fights = load_fights_from_db(session)

    draw_fights = [f for f in fights if f["fighter_a_id"] == fa_id and f["fighter_b_id"] == fb_id]
    assert len(draw_fights) == 1
    assert draw_fights[0]["winner_id"] is None


def test_persist_ratings_updates_all(engine):
    with Session(engine) as session:
        fa = Fighter(name="Persist Fighter A", elo_rating=1500.0)
        fb = Fighter(name="Persist Fighter B", elo_rating=1500.0)
        session.add_all([fa, fb])
        session.commit()
        fa_id, fb_id = fa.id, fb.id

    with Session(engine) as session:
        updated = persist_ratings(session, {fa_id: 1620.0, fb_id: 1380.0})

    with Session(engine) as session:
        fa_row = session.get(Fighter, fa_id)
        fb_row = session.get(Fighter, fb_id)
        assert fa_row.elo_rating == 1620.0
        assert fb_row.elo_rating == 1380.0

    assert updated == 2


def test_persist_ratings_returns_count(engine):
    with Session(engine) as session:
        fa = Fighter(name="Count Fighter A", elo_rating=1500.0)
        fb = Fighter(name="Count Fighter B", elo_rating=1500.0)
        fc = Fighter(name="Count Fighter C", elo_rating=1500.0)
        session.add_all([fa, fb, fc])
        session.commit()
        ids = [fa.id, fb.id, fc.id]

    with Session(engine) as session:
        count = persist_ratings(session, {fid: 1600.0 for fid in ids})

    assert count == 3
