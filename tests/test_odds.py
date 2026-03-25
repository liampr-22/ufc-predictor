"""
Odds generation and backtesting tests — Phase 7.

Covers:
    - american_odds_to_implied_prob: exact favourite / underdog formula
    - remove_vig: fair probability normalisation
    - apply_vig: symmetric vig application
    - prob_to_american_odds: favourite / underdog conversion + round-trip
    - prob_to_decimal_odds: basic correctness
    - prob_to_fractional_odds: simplified fraction output
    - run_backtest: end-to-end backtest with seeded historical odds
    - format_report: markdown report generation
"""
from __future__ import annotations

import math
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ml.backtest import FightBacktestResult, format_report, run_backtest
from ml.calibration import (
    american_odds_to_implied_prob,
    apply_vig,
    prob_to_american_odds,
    prob_to_decimal_odds,
    prob_to_fractional_odds,
    remove_vig,
)
from models.schema import Base, Fight, FightStats, Fighter, HistoricalOdds


# ---------------------------------------------------------------------------
# Odds conversion unit tests
# ---------------------------------------------------------------------------

class TestAmericanOddsToImpliedProb:
    def test_favourite_minus_200(self):
        # |-200| / (|-200| + 100) = 200/300 = 2/3
        assert math.isclose(american_odds_to_implied_prob(-200), 2 / 3, rel_tol=1e-9)

    def test_underdog_plus_200(self):
        # 100 / (200 + 100) = 100/300 = 1/3
        assert math.isclose(american_odds_to_implied_prob(200), 1 / 3, rel_tol=1e-9)

    def test_even_money_plus_100(self):
        # 100 / (100 + 100) = 0.5
        assert math.isclose(american_odds_to_implied_prob(100), 0.5, rel_tol=1e-9)

    def test_even_money_minus_100(self):
        # 100 / (100 + 100) = 0.5
        assert math.isclose(american_odds_to_implied_prob(-100), 0.5, rel_tol=1e-9)

    def test_heavy_favourite_minus_400(self):
        # 400 / 500 = 0.80
        assert math.isclose(american_odds_to_implied_prob(-400), 0.80, rel_tol=1e-9)

    def test_big_underdog_plus_400(self):
        # 100 / 500 = 0.20
        assert math.isclose(american_odds_to_implied_prob(400), 0.20, rel_tol=1e-9)

    def test_result_in_open_interval(self):
        for odds in [-500, -150, -100, 100, 150, 500]:
            p = american_odds_to_implied_prob(odds)
            assert 0 < p < 1, f"Implied prob out of (0,1) for odds={odds}: {p}"

    def test_favourite_plus_underdog_sum_gt_one(self):
        # With typical vig, -175 / +145 should sum > 1.0
        p_a = american_odds_to_implied_prob(-175)
        p_b = american_odds_to_implied_prob(145)
        assert p_a + p_b > 1.0


class TestRemoveVig:
    def test_sums_to_one(self):
        raw_a = american_odds_to_implied_prob(-200)
        raw_b = american_odds_to_implied_prob(175)
        fair_a, fair_b = remove_vig(raw_a, raw_b)
        assert math.isclose(fair_a + fair_b, 1.0, rel_tol=1e-9)

    def test_preserves_relative_order(self):
        raw_a = american_odds_to_implied_prob(-175)
        raw_b = american_odds_to_implied_prob(145)
        fair_a, fair_b = remove_vig(raw_a, raw_b)
        assert fair_a > fair_b, "Favourite should still have higher fair prob"

    def test_symmetric_market_gives_fifty_fifty(self):
        # -100 / +100 → both raw = 0.5, fair = 0.5 each
        fair_a, fair_b = remove_vig(0.5, 0.5)
        assert math.isclose(fair_a, 0.5, rel_tol=1e-9)
        assert math.isclose(fair_b, 0.5, rel_tol=1e-9)

    def test_raises_on_zero_total(self):
        with pytest.raises(ValueError, match="positive"):
            remove_vig(0.0, 0.0)

    def test_raises_on_negative_total(self):
        with pytest.raises(ValueError, match="positive"):
            remove_vig(-0.3, -0.3)


class TestApplyVig:
    def test_sum_of_both_sides_equals_one_plus_vig(self):
        p = 0.60
        vig = 0.04
        vigged_a = apply_vig(p, vig)
        vigged_b = apply_vig(1 - p, vig)
        assert math.isclose(vigged_a + vigged_b, 1.0 + vig, rel_tol=1e-9)

    def test_zero_vig_returns_unchanged(self):
        for p in (0.3, 0.5, 0.7):
            assert math.isclose(apply_vig(p, vig=0.0), p, rel_tol=1e-9)

    def test_default_vig_is_four_percent(self):
        p = 0.60
        # With 4% vig: vigged = 0.60 + 0.02 = 0.62
        assert math.isclose(apply_vig(p), 0.62, rel_tol=1e-9)

    def test_raises_on_invalid_probability(self):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            apply_vig(0.0)
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            apply_vig(1.0)

    def test_raises_on_invalid_vig(self):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            apply_vig(0.5, vig=-0.01)
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            apply_vig(0.5, vig=1.0)


class TestProbToAmericanOdds:
    def test_favourite_gives_negative_odds(self):
        assert prob_to_american_odds(0.75) < 0

    def test_underdog_gives_positive_odds(self):
        assert prob_to_american_odds(0.30) > 0

    def test_fifty_percent_gives_minus_100(self):
        # p=0.5 → -(0.5/0.5)*100 = -100
        assert math.isclose(prob_to_american_odds(0.5), -100.0, rel_tol=1e-9)

    def test_two_thirds_gives_minus_200(self):
        # p=2/3 → -(2/3 / 1/3)*100 = -200
        assert math.isclose(prob_to_american_odds(2 / 3), -200.0, rel_tol=1e-9)

    def test_one_third_gives_plus_200(self):
        # p=1/3 → ((2/3)/(1/3))*100 = +200
        assert math.isclose(prob_to_american_odds(1 / 3), 200.0, rel_tol=1e-9)

    def test_round_trip_favourite(self):
        for p in (0.55, 0.65, 0.80):
            odds = prob_to_american_odds(p)
            recovered = american_odds_to_implied_prob(int(round(odds)))
            assert math.isclose(p, recovered, rel_tol=1e-3), (
                f"Round-trip failed for p={p}: odds={odds}, recovered={recovered}"
            )

    def test_round_trip_underdog(self):
        for p in (0.30, 0.40, 0.45):
            odds = prob_to_american_odds(p)
            recovered = american_odds_to_implied_prob(int(round(odds)))
            # Underdogs have larger integer-rounding errors than favourites;
            # ±0.5 on positive odds with p≈0.30 shifts the implied prob by ~0.15%.
            assert math.isclose(p, recovered, rel_tol=2e-3)

    def test_raises_on_boundary(self):
        with pytest.raises(ValueError):
            prob_to_american_odds(0.0)
        with pytest.raises(ValueError):
            prob_to_american_odds(1.0)


class TestProbToDecimalOdds:
    def test_fifty_percent_gives_two(self):
        assert math.isclose(prob_to_decimal_odds(0.5), 2.0, rel_tol=1e-9)

    def test_always_at_least_one(self):
        for p in (0.01, 0.5, 0.99):
            assert prob_to_decimal_odds(p) >= 1.0

    def test_formula_is_reciprocal(self):
        for p in (0.25, 0.50, 0.75):
            assert math.isclose(prob_to_decimal_odds(p), 1.0 / p, rel_tol=1e-12)

    def test_raises_on_boundary(self):
        with pytest.raises(ValueError):
            prob_to_decimal_odds(0.0)
        with pytest.raises(ValueError):
            prob_to_decimal_odds(1.0)


class TestProbToFractionalOdds:
    def test_fifty_percent_gives_evens(self):
        num, den = prob_to_fractional_odds(0.5)
        assert num == 1 and den == 1, f"Expected 1/1 for p=0.5, got {num}/{den}"

    def test_two_thirds_gives_half(self):
        # p=2/3 → (1/3)/(2/3) = 1/2
        num, den = prob_to_fractional_odds(2 / 3)
        assert num == 1 and den == 2, f"Expected 1/2 for p=2/3, got {num}/{den}"

    def test_one_third_gives_two_to_one(self):
        # p=1/3 → (2/3)/(1/3) = 2/1
        num, den = prob_to_fractional_odds(1 / 3)
        assert num == 2 and den == 1, f"Expected 2/1 for p=1/3, got {num}/{den}"

    def test_output_is_positive_integers(self):
        for p in (0.2, 0.4, 0.6, 0.8):
            num, den = prob_to_fractional_odds(p)
            assert isinstance(num, int) and num > 0
            assert isinstance(den, int) and den > 0

    def test_raises_on_boundary(self):
        with pytest.raises(ValueError):
            prob_to_fractional_odds(0.0)
        with pytest.raises(ValueError):
            prob_to_fractional_odds(1.0)


# ---------------------------------------------------------------------------
# Backtest integration tests
# ---------------------------------------------------------------------------

# Per-fighter stat profiles reused from test_model.py
_STATS = {
    1: (60, 110, 3, 5, 1, 2),
    2: (45,  90, 5, 7, 3, 0),
    3: (35,  80, 6, 8, 5, 0),
    4: (70, 120, 1, 3, 0, 4),
}
_STUB = (30, 70, 1, 3, 0, 0)


def _add_stats(sess: Session, fight_id: int, a_id: int, b_id: int) -> None:
    for fid, profile in ((a_id, _STATS.get(a_id, _STUB)), (b_id, _STATS.get(b_id, _STUB))):
        sl, sa, tl, ta, sub, kd = profile
        sess.add(FightStats(
            fight_id=fight_id, fighter_id=fid,
            significant_strikes_landed=sl, significant_strikes_attempted=sa,
            takedowns_landed=tl, takedowns_attempted=ta,
            submission_attempts=sub, knockdowns=kd,
        ))


def _seed_backtest_db(sess: Session) -> None:
    """
    Minimal dataset: 4 fighters, 12 fights, 6 with historical odds.
    The first 8 fights are "training history"; the last 4 include odds.
    """
    for fid, name, stance, dob in [
        (1, "Fighter A", "Orthodox", date(1990, 1, 1)),
        (2, "Fighter B", "Orthodox", date(1991, 1, 1)),
        (3, "Fighter C", "Southpaw", date(1992, 1, 1)),
        (4, "Fighter D", "Orthodox", date(1989, 1, 1)),
    ]:
        sess.add(Fighter(
            id=fid, name=name, height=70.0, reach=71.0, stance=stance,
            dob=dob, weight_class="Lightweight", elo_rating=1500.0,
        ))

    # Stub opponents for training history
    for i in range(10, 16):
        sess.add(Fighter(id=i, name=f"Stub {i}", weight_class="Lightweight", elo_rating=1500.0))
    sess.flush()

    fight_defs = [
        # Training history (no odds)
        ( 1, date(2018,  1,  1),  1, 10,  1, "KO"),
        ( 2, date(2018,  4,  1),  2, 11,  2, "DEC"),
        ( 3, date(2018,  7,  1),  3, 12,  3, "SUB"),
        ( 4, date(2018, 10,  1),  4, 13,  4, "KO"),
        ( 5, date(2019,  1,  1),  1, 14,  1, "DEC"),
        ( 6, date(2019,  4,  1),  2, 15, 15, "KO"),
        ( 7, date(2020,  1,  1),  1,  2,  1, "DEC"),
        ( 8, date(2020,  6,  1),  3,  4,  4, "KO"),
        # Fights with historical odds (test period)
        ( 9, date(2021,  1,  1),  1,  3,  1, "KO"),
        (10, date(2021,  6,  1),  2,  4,  2, "SUB"),
        (11, date(2022,  1,  1),  1,  4,  4, "DEC"),
        (12, date(2022,  6,  1),  3,  2,  3, "KO"),
    ]

    for fid, fdate, a_id, b_id, winner_id, method in fight_defs:
        sess.add(Fight(id=fid, date=fdate, fighter_a_id=a_id,
                       fighter_b_id=b_id, winner_id=winner_id, method=method))
        sess.flush()
        _add_stats(sess, fid, a_id, b_id)

    # Historical closing odds for fights 9–12
    # Fighter A typically favoured: -175 / +145
    odds_defs = [
        (9,  -175,  145, "BestFightOdds"),   # A favoured, A wins  → model likely picks A
        (10, -150,  120, "BestFightOdds"),   # A (fighter 2) favoured, A (fighter 2) wins
        (11,  130, -160, "BestFightOdds"),   # B (fighter 4) favoured, B (fighter 4) wins
        (12,  155, -185, "BestFightOdds"),   # B (fighter 2) favoured, A (fighter 3) wins
    ]
    for fight_id, odds_a, odds_b, source in odds_defs:
        sess.add(HistoricalOdds(
            fight_id=fight_id,
            fighter_a_odds=odds_a,
            fighter_b_odds=odds_b,
            source=source,
        ))

    sess.commit()


@pytest.fixture(scope="module")
def backtest_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        _seed_backtest_db(sess)
        yield sess


class _StubPredictor:
    """
    Minimal predictor stub that returns deterministic probabilities without
    loading model files, so the backtest tests are fast and self-contained.

    Returns a fixed model_prob_a based on (fighter_a_id, fighter_b_id):
      - Fighter 1 vs anyone: 0.65
      - Fighter 2 vs anyone: 0.60
      - Fighter 3 vs anyone: 0.55
      - Fighter 4 vs anyone: 0.45
    """

    def predict_proba(
        self,
        session: Session,
        fighter_a_id: int,
        fighter_b_id: int,
        as_of_date=None,
    ) -> float:
        _probs = {1: 0.65, 2: 0.60, 3: 0.55, 4: 0.45}
        return _probs.get(fighter_a_id, 0.50)


class TestRunBacktest:
    def test_returns_required_keys(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        required = {"total_fights", "skipped", "pct_beat_closing_line", "avg_clv",
                    "total_pnl", "per_fight"}
        assert required.issubset(set(result.keys()))

    def test_total_fights_matches_seeded_odds(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        assert result["total_fights"] == 4

    def test_pct_beat_in_range(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        assert 0.0 <= result["pct_beat_closing_line"] <= 1.0

    def test_avg_clv_is_float(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        assert isinstance(result["avg_clv"], float)

    def test_per_fight_has_correct_length(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        assert len(result["per_fight"]) == result["total_fights"]

    def test_per_fight_keys(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        expected = {
            "fight_id", "fight_date", "fighter_a_id", "fighter_b_id",
            "winner_id", "model_prob_a", "fair_market_prob_a", "fair_market_prob_b",
            "model_pick", "clv", "pnl", "beat_closing_line",
        }
        for row in result["per_fight"]:
            assert expected.issubset(set(row.keys())), f"Missing keys in row: {row}"

    def test_fair_probs_sum_to_one(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        for row in result["per_fight"]:
            total = row["fair_market_prob_a"] + row["fair_market_prob_b"]
            assert math.isclose(total, 1.0, rel_tol=1e-6), (
                f"Fair probs do not sum to 1.0 for fight {row['fight_id']}: {total}"
            )

    def test_model_pick_is_a_or_b(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        for row in result["per_fight"]:
            assert row["model_pick"] in {"A", "B"}

    def test_clv_is_model_prob_minus_market(self, backtest_session):
        """CLV == model_prob_for_pick - fair_market_prob_for_pick."""
        result = run_backtest(backtest_session, _StubPredictor())
        for row in result["per_fight"]:
            if row["model_pick"] == "A":
                expected_clv = row["model_prob_a"] - row["fair_market_prob_a"]
            else:
                expected_clv = (1.0 - row["model_prob_a"]) - row["fair_market_prob_b"]
            assert math.isclose(row["clv"], expected_clv, rel_tol=1e-4), (
                f"CLV mismatch for fight {row['fight_id']}: "
                f"got {row['clv']}, expected {expected_clv}"
            )

    def test_pnl_correct_on_winning_bet(self, backtest_session):
        """When the model picks correctly, P&L should be (1/fair_prob - 1) > 0."""
        result = run_backtest(backtest_session, _StubPredictor())
        for row in result["per_fight"]:
            if row["model_pick"] == "A":
                model_correct = row["winner_id"] == row["fighter_a_id"]
                fair_pick_prob = row["fair_market_prob_a"]
            else:
                model_correct = row["winner_id"] == row["fighter_b_id"]
                fair_pick_prob = row["fair_market_prob_b"]

            if model_correct:
                expected_pnl = (1.0 / fair_pick_prob) - 1.0
                assert math.isclose(row["pnl"], expected_pnl, rel_tol=1e-4), (
                    f"Winning P&L mismatch for fight {row['fight_id']}: "
                    f"got {row['pnl']}, expected {expected_pnl}"
                )
            else:
                assert math.isclose(row["pnl"], -1.0, rel_tol=1e-9), (
                    f"Losing P&L should be -1.0 for fight {row['fight_id']}: {row['pnl']}"
                )

    def test_beat_closing_line_consistent_with_clv(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        for row in result["per_fight"]:
            assert row["beat_closing_line"] == (row["clv"] > 0)

    def test_total_pnl_equals_sum_of_per_fight(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        expected = sum(r["pnl"] for r in result["per_fight"])
        assert math.isclose(result["total_pnl"], expected, rel_tol=1e-4)

    def test_empty_backtest_on_no_odds(self):
        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        with Session(engine) as sess:
            result = run_backtest(sess, _StubPredictor())
        assert result["total_fights"] == 0
        assert result["per_fight"] == []
        assert result["avg_clv"] == 0.0
        assert result["total_pnl"] == 0.0


class TestFormatReport:
    def test_returns_string(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        report = format_report(result)
        assert isinstance(report, str)

    def test_contains_summary_heading(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        report = format_report(result)
        assert "## Summary" in report

    def test_contains_per_fight_table(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        report = format_report(result)
        assert "## Per-Fight Results" in report

    def test_empty_result_shows_no_data_message(self):
        empty = {
            "total_fights": 0, "skipped": 0, "pct_beat_closing_line": 0.0,
            "avg_clv": 0.0, "total_pnl": 0.0, "per_fight": [],
        }
        report = format_report(empty)
        assert "No historical odds" in report

    def test_report_includes_fight_ids(self, backtest_session):
        result = run_backtest(backtest_session, _StubPredictor())
        report = format_report(result)
        for row in result["per_fight"]:
            assert str(row["fight_id"]) in report
