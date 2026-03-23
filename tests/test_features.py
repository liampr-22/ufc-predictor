"""Feature engineering tests — Phase 4.

Primary fixture: Islam Makhachev (A) vs Dustin Poirier (B) at 2023-01-01.
All expected values are computed by hand from the seeded data below and
serve as ground-truth anchors for the feature pipeline.

Seeded fight histories
----------------------
Makhachev (id=1): 5 fights, all wins — 3 SUB, 2 DEC
Poirier   (id=2): 5 fights, 4 wins — 3 KO, 1 DEC; fight 8 is a DEC loss
"""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ml.features import FeatureBuilder, _GLOBAL_PRIORS
from models.schema import Base, Fight, FightStats, Fighter

# ---------------------------------------------------------------------------
# Tolerance for float assertions
# ---------------------------------------------------------------------------
ABS = 1e-4


# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        _seed(sess)
        yield sess


def _seed(sess: Session) -> None:
    """
    Populate the in-memory test DB with Makhachev and Poirier fight histories.
    Opponent fighters (id 3-12) are stub records — only IDs matter.
    All stats have been chosen to produce clean, manually-verifiable expected values.
    """
    # Stub opponents
    sess.add_all(
        Fighter(id=i, name=f"Opponent {i}", weight_class="Lightweight", elo_rating=1500.0)
        for i in range(3, 13)
    )

    sess.add(Fighter(
        id=1, name="Islam Makhachev",
        height=70.5, reach=70.0, stance="Orthodox",
        dob=date(1991, 10, 18),
        weight_class="Lightweight",
        elo_rating=1700.0,
    ))
    sess.add(Fighter(
        id=2, name="Dustin Poirier",
        height=69.0, reach=72.0, stance="Orthodox",
        dob=date(1989, 1, 19),
        weight_class="Lightweight",
        elo_rating=1650.0,
    ))

    # ── Makhachev fights (ids 1-5, fighter_a=1) ─────────────────────────────
    sess.add_all([
        Fight(id=1,  date=date(2019, 1, 1),  fighter_a_id=1, fighter_b_id=3,  winner_id=1,  method="SUB"),
        Fight(id=2,  date=date(2019, 9, 1),  fighter_a_id=1, fighter_b_id=4,  winner_id=1,  method="SUB"),
        Fight(id=3,  date=date(2020, 3, 1),  fighter_a_id=1, fighter_b_id=5,  winner_id=1,  method="DEC"),
        Fight(id=4,  date=date(2021, 2, 1),  fighter_a_id=1, fighter_b_id=6,  winner_id=1,  method="SUB"),
        Fight(id=5,  date=date(2022, 1, 1),  fighter_a_id=1, fighter_b_id=7,  winner_id=1,  method="DEC"),
    ])

    # Makhachev FightStats rows: (fight_id, fighter_id, sig_l, sig_a, td_l, td_a, sub_att, kd)
    for fid, pid, sl, sa, tl, ta, sub, kd in [
        (1, 1, 50, 100, 4, 5, 3, 0),   # Makhachev
        (1, 3, 30,  70, 0, 2, 0, 0),   # Opp
        (2, 1, 45,  90, 3, 4, 4, 0),
        (2, 4, 25,  60, 1, 3, 0, 0),
        (3, 1, 60, 110, 2, 3, 1, 0),
        (3, 5, 40,  90, 0, 2, 0, 0),
        (4, 1, 40,  80, 5, 6, 5, 0),
        (4, 6, 20,  55, 0, 3, 0, 0),
        (5, 1, 70, 130, 3, 4, 2, 0),
        (5, 7, 35,  80, 1, 4, 0, 0),
    ]:
        sess.add(FightStats(
            fight_id=fid, fighter_id=pid,
            significant_strikes_landed=sl, significant_strikes_attempted=sa,
            takedowns_landed=tl, takedowns_attempted=ta,
            submission_attempts=sub, knockdowns=kd,
        ))

    # ── Poirier fights (ids 6-10, fighter_a=2); fight 8 is a loss ───────────
    sess.add_all([
        Fight(id=6,  date=date(2019, 3,  1), fighter_a_id=2, fighter_b_id=8,  winner_id=2,  method="KO"),
        Fight(id=7,  date=date(2019, 11, 1), fighter_a_id=2, fighter_b_id=9,  winner_id=2,  method="KO"),
        Fight(id=8,  date=date(2020, 6,  1), fighter_a_id=2, fighter_b_id=10, winner_id=10, method="DEC"),
        Fight(id=9,  date=date(2021, 1,  1), fighter_a_id=2, fighter_b_id=11, winner_id=2,  method="KO"),
        Fight(id=10, date=date(2022, 6,  1), fighter_a_id=2, fighter_b_id=12, winner_id=2,  method="DEC"),
    ])

    # Poirier FightStats rows
    for fid, pid, sl, sa, tl, ta, sub, kd in [
        (6,  2, 55, 100, 0, 1, 0, 2),   # Poirier
        (6,  8, 30,  65, 0, 1, 0, 0),
        (7,  2, 60, 110, 0, 0, 0, 1),
        (7,  9, 25,  55, 0, 0, 0, 0),
        (8,  2, 50,  95, 0, 2, 0, 0),
        (8, 10, 55, 100, 2, 3, 0, 0),
        (9,  2, 65, 115, 0, 1, 0, 3),
        (9, 11, 20,  50, 0, 2, 0, 0),
        (10, 2, 70, 130, 1, 3, 0, 0),
        (10,12, 60, 120, 1, 2, 0, 0),
    ]:
        sess.add(FightStats(
            fight_id=fid, fighter_id=pid,
            significant_strikes_landed=sl, significant_strikes_attempted=sa,
            takedowns_landed=tl, takedowns_attempted=ta,
            submission_attempts=sub, knockdowns=kd,
        ))

    sess.commit()


# ---------------------------------------------------------------------------
# Primary fixture: Makhachev A vs Poirier B at 2023-01-01
# ---------------------------------------------------------------------------

AS_OF = date(2023, 1, 1)


@pytest.fixture(scope="module")
def fv(session):
    return FeatureBuilder(session).build(1, 2, AS_OF)


# ---------------------------------------------------------------------------
# Physical features
# ---------------------------------------------------------------------------

class TestPhysical:
    def test_height_delta(self, fv):
        assert fv.height_delta == pytest.approx(1.5, abs=ABS)

    def test_reach_delta(self, fv):
        assert fv.reach_delta == pytest.approx(-2.0, abs=ABS)

    def test_age_delta(self, fv):
        # Makhachev dob 1991-10-18, Poirier dob 1989-01-19
        # (date(1991,10,18) - date(1989,1,19)).days = 1002
        assert fv.age_delta == pytest.approx(1002.0, abs=ABS)

    def test_southpaw_matchup_both_orthodox(self, fv):
        assert fv.is_southpaw_matchup == 0


# ---------------------------------------------------------------------------
# Striking features
# ---------------------------------------------------------------------------

class TestStriking:
    def test_sig_strike_accuracy_delta(self, fv):
        # Makhachev: 265/510; Poirier: 300/550
        expected = 265 / 510 - 300 / 550
        assert fv.sig_strike_accuracy_delta == pytest.approx(expected, abs=ABS)

    def test_strike_defense_delta(self, fv):
        # Makhachev: 1 - 150/355; Poirier: 1 - 190/390
        expected = (1 - 150 / 355) - (1 - 190 / 390)
        assert fv.strike_defense_delta == pytest.approx(expected, abs=ABS)

    def test_knockdown_rate_delta(self, fv):
        # Makhachev: 0/5 = 0.0; Poirier: 6/5 = 1.2
        assert fv.knockdown_rate_delta == pytest.approx(-1.2, abs=ABS)


# ---------------------------------------------------------------------------
# Grappling features
# ---------------------------------------------------------------------------

class TestGrappling:
    def test_takedown_accuracy_delta(self, fv):
        # Makhachev: 17/22; Poirier: 1/7
        expected = 17 / 22 - 1 / 7
        assert fv.takedown_accuracy_delta == pytest.approx(expected, abs=ABS)

    def test_takedown_defense_delta(self, fv):
        # Makhachev: 1 - 2/14; Poirier: 1 - 3/8
        expected = (1 - 2 / 14) - (1 - 3 / 8)
        assert fv.takedown_defense_delta == pytest.approx(expected, abs=ABS)

    def test_submission_attempts_per_fight_delta(self, fv):
        # Makhachev: 15/5 = 3.0; Poirier: 0/5 = 0.0
        assert fv.submission_attempts_per_fight_delta == pytest.approx(3.0, abs=ABS)


# ---------------------------------------------------------------------------
# Finishing rates
# ---------------------------------------------------------------------------

class TestFinishingRates:
    def test_ko_rates(self, fv):
        assert fv.ko_rate_a == pytest.approx(0.0, abs=ABS)   # 0 KO wins
        assert fv.ko_rate_b == pytest.approx(0.6, abs=ABS)   # 3/5 KO wins

    def test_sub_rates(self, fv):
        assert fv.sub_rate_a == pytest.approx(0.6, abs=ABS)  # 3/5 SUB wins
        assert fv.sub_rate_b == pytest.approx(0.0, abs=ABS)

    def test_dec_rates(self, fv):
        assert fv.dec_rate_a == pytest.approx(0.4, abs=ABS)  # 2/5 DEC wins
        assert fv.dec_rate_b == pytest.approx(0.2, abs=ABS)  # 1/5 DEC wins


# ---------------------------------------------------------------------------
# Recency features
# ---------------------------------------------------------------------------

class TestRecency:
    def test_days_since_last_fight(self, fv):
        # Makhachev: 2023-01-01 − 2022-01-01 = 365 days
        assert fv.days_since_last_fight_a == pytest.approx(365.0, abs=ABS)
        # Poirier: 2023-01-01 − 2022-06-01 = 214 days
        assert fv.days_since_last_fight_b == pytest.approx(214.0, abs=ABS)

    def test_win_streak_delta(self, fv):
        # Makhachev: 5 consecutive wins; Poirier: 2 (lost fight 8, won 9 & 10)
        assert fv.win_streak_delta == 3

    def test_recent_sig_strike_accuracy_delta(self, fv):
        # Makhachev last-3 (most recent first): f5, f4, f3
        acc_a = 0.5 * (70 / 130) + 0.3 * (40 / 80) + 0.2 * (60 / 110)
        # Poirier last-3: f10, f9, f8
        acc_b = 0.5 * (70 / 130) + 0.3 * (65 / 115) + 0.2 * (50 / 95)
        assert fv.recent_sig_strike_accuracy_delta == pytest.approx(acc_a - acc_b, abs=ABS)

    def test_recent_td_accuracy_delta(self, fv):
        # Makhachev: f5=3/4, f4=5/6, f3=2/3
        td_a = 0.5 * (3 / 4) + 0.3 * (5 / 6) + 0.2 * (2 / 3)
        # Poirier: f10=1/3, f9=0/1, f8=0/2
        td_b = 0.5 * (1 / 3) + 0.3 * 0.0 + 0.2 * 0.0
        assert fv.recent_td_accuracy_delta == pytest.approx(td_a - td_b, abs=ABS)


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------

class TestElo:
    def test_elo_delta(self, fv):
        assert fv.elo_delta == pytest.approx(50.0, abs=ABS)


# ---------------------------------------------------------------------------
# Style archetype scores
# ---------------------------------------------------------------------------

class TestStyleArchetypes:
    def test_striker_scores(self, fv):
        assert fv.striker_score_a == pytest.approx(265 / 510, abs=ABS)
        assert fv.striker_score_b == pytest.approx(300 / 550, abs=ABS)

    def test_wrestler_scores(self, fv):
        assert fv.wrestler_score_a == pytest.approx(17 / 22, abs=ABS)
        assert fv.wrestler_score_b == pytest.approx(1 / 7, abs=ABS)

    def test_grappler_scores(self, fv):
        # Makhachev: min(1.0, 3.0 / 5.0) = 0.6; Poirier: min(1.0, 0.0 / 5.0) = 0.0
        assert fv.grappler_score_a == pytest.approx(0.6, abs=ABS)
        assert fv.grappler_score_b == pytest.approx(0.0, abs=ABS)

    def test_brawler_scores(self, fv):
        # Makhachev: 0.0 KDs; Poirier: min(1.0, 1.2 / 3.0) = 0.4
        assert fv.brawler_score_a == pytest.approx(0.0, abs=ABS)
        assert fv.brawler_score_b == pytest.approx(0.4, abs=ABS)


# ---------------------------------------------------------------------------
# Confidence flags
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_not_low_confidence_with_five_fights(self, fv):
        assert fv.low_confidence_a is False
        assert fv.low_confidence_b is False


# ---------------------------------------------------------------------------
# to_dict completeness
# ---------------------------------------------------------------------------

class TestToDict:
    def test_returns_dict_with_all_keys(self, fv):
        d = fv.to_dict()
        assert isinstance(d, dict)
        for key in [
            "height_delta", "reach_delta", "age_delta", "is_southpaw_matchup",
            "elo_delta", "ko_rate_a", "sub_rate_b",
            "striker_score_a", "brawler_score_b",
            "low_confidence_a", "low_confidence_b",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_dict_has_no_none_values(self, fv):
        d = fv.to_dict()
        none_keys = [k for k, v in d.items() if v is None]
        assert none_keys == [], f"Unexpected None values for: {none_keys}"


# ---------------------------------------------------------------------------
# Data leakage prevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    def test_fight_on_exact_cutoff_date_excluded(self, session):
        """Fight 5 (2022-01-01) must not count when as_of_date = 2022-01-01."""
        fv = FeatureBuilder(session).build(1, 2, date(2022, 1, 1))
        # Makhachev: fights 1-4 only (4 fights) → low_confidence
        assert fv.low_confidence_a is True

    def test_future_fights_excluded(self, session):
        """Only fights 1 and 2 exist before 2019-10-01 for Makhachev."""
        fv = FeatureBuilder(session).build(1, 2, date(2019, 10, 1))
        assert fv.low_confidence_a is True

    def test_win_streak_excludes_future_fights(self, session):
        """
        Fight 5 (2022-01-01) is excluded. Makhachev: 4 wins → streak 4.
        Poirier: fights 6-9, last win was fight 9 after losing fight 8 → streak 1.
        delta = 4 - 1 = 3. If fight 5 were included the delta would be 4 (5-1).
        """
        fv = FeatureBuilder(session).build(1, 2, date(2022, 1, 1))
        assert fv.win_streak_delta == 3


# ---------------------------------------------------------------------------
# Southpaw matchup
# ---------------------------------------------------------------------------

class TestSouthpawMatchup:
    def test_southpaw_vs_orthodox(self, session):
        fighter = session.get(Fighter, 1)
        original = fighter.stance
        fighter.stance = "Southpaw"
        session.flush()
        try:
            fv = FeatureBuilder(session).build(1, 2, AS_OF)
            assert fv.is_southpaw_matchup == 1
        finally:
            fighter.stance = original
            session.flush()

    def test_both_southpaw(self, session):
        fa = session.get(Fighter, 1)
        fb = session.get(Fighter, 2)
        orig_a, orig_b = fa.stance, fb.stance
        fa.stance = "Southpaw"
        fb.stance = "Southpaw"
        session.flush()
        try:
            fv = FeatureBuilder(session).build(1, 2, AS_OF)
            assert fv.is_southpaw_matchup == 0
        finally:
            fa.stance = orig_a
            fb.stance = orig_b
            session.flush()


# ---------------------------------------------------------------------------
# Bayesian shrinkage
# ---------------------------------------------------------------------------

class TestBayesianShrinkage:
    def test_low_confidence_flag_with_one_fight(self, session):
        fv = FeatureBuilder(session).build(1, 2, date(2019, 2, 1))
        assert fv.low_confidence_a is True

    def test_shrinkage_toward_global_prior(self, session):
        """
        At 2019-02-01: Makhachev has 1 fight (sig_acc raw = 50/100 = 0.50).
        Weight-class has only 2 FightStats rows → falls back to global prior = 0.45.
        Shrunk = (1*0.50 + 5*0.45) / 6 = 2.75/6 ≈ 0.45833.
        Poirier has 0 fights → uses prior = 0.45.
        delta = 0.45833 - 0.45 ≈ 0.00833.
        """
        prior = _GLOBAL_PRIORS["sig_strike_accuracy"]  # 0.45
        shrunk_a = (1 * 0.50 + 5 * prior) / (1 + 5)
        expected_delta = shrunk_a - prior
        fv = FeatureBuilder(session).build(1, 2, date(2019, 2, 1))
        assert fv.sig_strike_accuracy_delta == pytest.approx(expected_delta, abs=ABS)

    def test_exact_threshold_not_shrunk(self, session):
        """Fighter with exactly 5 fights must NOT be shrunk (low_confidence = False)."""
        fv = FeatureBuilder(session).build(1, 2, AS_OF)
        assert fv.low_confidence_a is False
        # Accuracy should equal raw rate, not shrunk
        assert fv.striker_score_a == pytest.approx(265 / 510, abs=ABS)


# ---------------------------------------------------------------------------
# No fights edge case
# ---------------------------------------------------------------------------

class TestNoFights:
    def test_both_no_fights_uses_prior(self, session):
        fv = FeatureBuilder(session).build(1, 2, date(2018, 1, 1))
        assert fv.low_confidence_a is True
        assert fv.low_confidence_b is True

    def test_both_no_fights_accuracy_delta_is_zero(self, session):
        fv = FeatureBuilder(session).build(1, 2, date(2018, 1, 1))
        # Both use global prior → delta = 0
        assert fv.sig_strike_accuracy_delta == pytest.approx(0.0, abs=ABS)

    def test_no_fights_days_since_is_large(self, session):
        fv = FeatureBuilder(session).build(1, 2, date(2018, 1, 1))
        assert fv.days_since_last_fight_a == pytest.approx(9999.0, abs=ABS)
        assert fv.days_since_last_fight_b == pytest.approx(9999.0, abs=ABS)


# ---------------------------------------------------------------------------
# Invalid fighter
# ---------------------------------------------------------------------------

class TestInvalidFighter:
    def test_unknown_fighter_raises_value_error(self, session):
        with pytest.raises(ValueError):
            FeatureBuilder(session).build(999, 2, AS_OF)

    def test_both_unknown_raises_value_error(self, session):
        with pytest.raises(ValueError):
            FeatureBuilder(session).build(888, 999, AS_OF)
