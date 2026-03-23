"""Feature engineering pipeline for UFC fight outcome prediction.

All features are computed as differentials (fighter_a - fighter_b) unless
asymmetric matchup information is preserved (e.g. finishing rates per fighter).

Leakage prevention: every stat query filters to Fight.date < as_of_date,
ensuring no post-fight data is ever used for training or inference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from models.schema import Fight, FightStats, Fighter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FIGHTS_FOR_CONFIDENCE = 5
PRIOR_STRENGTH = 5          # equivalent fights of prior weight
RECENCY_WEIGHTS = [0.5, 0.3, 0.2]  # most-recent-first, last-3 fights

GRAPPLER_SCORE_CAP = 5.0   # sub attempts per fight — caps grappler score at 1.0
BRAWLER_SCORE_CAP = 3.0    # knockdowns per fight — caps brawler score at 1.0

# Method string sets for finishing-rate computation
_KO_METHODS = frozenset({"KO", "KO/TKO", "TKO"})
_SUB_METHODS = frozenset({"SUB", "SUBMISSION"})
_DEC_METHODS = frozenset({"DEC", "U-DEC", "S-DEC", "M-DEC", "UD", "SD", "MD"})

# Global UFC averages — fallback prior when weight-class data is sparse
_GLOBAL_PRIORS: dict = {
    "sig_strike_accuracy": 0.45,
    "td_accuracy": 0.40,
    "td_defense": 0.65,
    "strike_defense": 0.55,
    "sub_att_per_fight": 0.5,
    "kd_per_fight": 0.3,
}


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _FightRecord:
    """Per-fight stats for one fighter (plus opponent stats for defense calc)."""

    fight_id: int
    fight_date: date
    won: bool
    method: Optional[str]    # uppercase, e.g. "KO", "SUB", "DEC"

    # Own stats
    sig_landed: int
    sig_attempted: int
    td_landed: int
    td_attempted: int
    sub_att: int
    knockdowns: int

    # Opponent stats — used for defense metrics
    opp_sig_landed: int
    opp_sig_attempted: int
    opp_td_landed: int
    opp_td_attempted: int


@dataclass
class _FighterAggStats:
    """Aggregated career statistics for one fighter up to a cut-off date."""

    fighter_id: int
    num_fights: int
    low_confidence: bool       # True when num_fights < MIN_FIGHTS_FOR_CONFIDENCE

    # Career rates (Bayesian-shrunk toward weight-class mean when low_confidence)
    sig_strike_accuracy: float
    strike_defense: float
    td_accuracy: float
    td_defense: float
    sub_att_per_fight: float
    kd_per_fight: float

    # Finishing rates (raw — no shrinkage, treated as categorical signal)
    ko_rate: float
    sub_rate: float
    dec_rate: float

    # Recency
    days_since_last_fight: Optional[float]  # None when no fights
    win_streak: int

    # Last-3-fight weighted accuracy
    recent_sig_strike_accuracy: float
    recent_td_accuracy: float

    # Style archetype scores in [0, 1]
    striker_score: float
    wrestler_score: float
    grappler_score: float
    brawler_score: float


# ---------------------------------------------------------------------------
# Public output
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """
    Leakage-free feature vector for a fighter pair at a specific date.

    Differential features are A − B (positive = A has the edge).
    Finishing rates and style scores are kept per-fighter to preserve
    asymmetric matchup information.
    """

    # Physical differentials
    height_delta: float            # inches; A − B
    reach_delta: float             # inches; A − B
    age_delta: float               # days; positive = A is younger (born later)
    is_southpaw_matchup: int       # 1 if exactly one fighter is southpaw

    # Striking differentials
    sig_strike_accuracy_delta: float
    strike_defense_delta: float
    knockdown_rate_delta: float

    # Grappling differentials
    takedown_accuracy_delta: float
    takedown_defense_delta: float
    submission_attempts_per_fight_delta: float

    # Finishing rates per fighter (not differential — asymmetric matchup info)
    ko_rate_a: float
    ko_rate_b: float
    sub_rate_a: float
    sub_rate_b: float
    dec_rate_a: float
    dec_rate_b: float

    # Recency
    days_since_last_fight_a: float
    days_since_last_fight_b: float
    win_streak_delta: int          # A streak − B streak

    # Last-3 weighted differentials
    recent_sig_strike_accuracy_delta: float
    recent_td_accuracy_delta: float

    # Elo
    elo_delta: float

    # Style archetype scores [0, 1] per fighter
    striker_score_a: float
    wrestler_score_a: float
    grappler_score_a: float
    brawler_score_a: float
    striker_score_b: float
    wrestler_score_b: float
    grappler_score_b: float
    brawler_score_b: float

    # Confidence flags
    low_confidence_a: bool
    low_confidence_b: bool

    def to_dict(self) -> dict:
        """Return a flat dict suitable for use as an ML feature matrix row."""
        return asdict(self)


# ---------------------------------------------------------------------------
# FeatureBuilder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """
    Builds leakage-free feature vectors for fight outcome prediction.

    Usage::

        builder = FeatureBuilder(session)
        fv = builder.build(fighter_a_id=1, fighter_b_id=2, as_of_date=date(2024, 1, 1))

    All features reflect only data available strictly before ``as_of_date``,
    making the builder safe for historical training without data leakage.
    """

    def __init__(self, session: Session) -> None:
        self._s = session

    def build(
        self,
        fighter_a_id: int,
        fighter_b_id: int,
        as_of_date: date,
    ) -> FeatureVector:
        """
        Build a validated feature vector for the given fighter pair.

        Parameters
        ----------
        fighter_a_id:
            Database ID of fighter A (red-corner / home side).
        fighter_b_id:
            Database ID of fighter B.
        as_of_date:
            Cut-off date. Only fights with date **strictly before** this value
            are used — the fight being predicted is excluded automatically.

        Raises
        ------
        ValueError
            If either fighter ID is not found in the database.
        """
        fa = self._s.get(Fighter, fighter_a_id)
        fb = self._s.get(Fighter, fighter_b_id)

        if fa is None or fb is None:
            missing = [i for i, f in [(fighter_a_id, fa), (fighter_b_id, fb)] if f is None]
            raise ValueError(f"Fighter(s) not found: ids {missing}")

        records_a = self._load_fight_records(fighter_a_id, as_of_date)
        records_b = self._load_fight_records(fighter_b_id, as_of_date)

        prior_a = self._get_weight_class_prior(fa.weight_class, as_of_date)
        prior_b = self._get_weight_class_prior(fb.weight_class, as_of_date)

        stats_a = self._aggregate(fighter_a_id, records_a, as_of_date, prior_a)
        stats_b = self._aggregate(fighter_b_id, records_b, as_of_date, prior_b)

        return self._build_vector(fa, fb, stats_a, stats_b)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_fight_records(
        self, fighter_id: int, as_of_date: date
    ) -> list:
        """
        Return all fight records for ``fighter_id`` strictly before ``as_of_date``,
        ordered chronologically (oldest first).
        """
        fights = (
            self._s.query(Fight)
            .filter(
                or_(
                    Fight.fighter_a_id == fighter_id,
                    Fight.fighter_b_id == fighter_id,
                ),
                Fight.date < as_of_date,
            )
            .order_by(Fight.date.asc(), Fight.id.asc())
            .all()
        )

        records = []
        for fight in fights:
            opp_id = (
                fight.fighter_b_id
                if fight.fighter_a_id == fighter_id
                else fight.fighter_a_id
            )
            my = self._get_fight_stats(fight.id, fighter_id)
            opp = self._get_fight_stats(fight.id, opp_id)
            won = fight.winner_id == fighter_id
            method = fight.method.upper().strip() if fight.method else None

            records.append(
                _FightRecord(
                    fight_id=fight.id,
                    fight_date=fight.date,
                    won=won,
                    method=method,
                    sig_landed=my.significant_strikes_landed or 0,
                    sig_attempted=my.significant_strikes_attempted or 0,
                    td_landed=my.takedowns_landed or 0,
                    td_attempted=my.takedowns_attempted or 0,
                    sub_att=my.submission_attempts or 0,
                    knockdowns=my.knockdowns or 0,
                    opp_sig_landed=opp.significant_strikes_landed or 0,
                    opp_sig_attempted=opp.significant_strikes_attempted or 0,
                    opp_td_landed=opp.takedowns_landed or 0,
                    opp_td_attempted=opp.takedowns_attempted or 0,
                )
            )

        return records

    def _get_fight_stats(self, fight_id: int, fighter_id: int) -> FightStats:
        """
        Return the FightStats row for (fight_id, fighter_id).
        Returns a zero-filled stub when no record exists so callers never
        need to handle None.
        """
        row = (
            self._s.query(FightStats)
            .filter(
                FightStats.fight_id == fight_id,
                FightStats.fighter_id == fighter_id,
            )
            .first()
        )
        if row is not None:
            return row

        stub = FightStats()
        stub.significant_strikes_landed = 0
        stub.significant_strikes_attempted = 0
        stub.takedowns_landed = 0
        stub.takedowns_attempted = 0
        stub.submission_attempts = 0
        stub.knockdowns = 0
        return stub

    def _get_weight_class_prior(
        self, weight_class, as_of_date: date
    ) -> dict:
        """
        Compute mean stats for all fighters in ``weight_class`` before
        ``as_of_date``.  Falls back to global UFC averages when fewer than
        10 FightStats rows are available (sparse or unknown weight class).
        """
        if not weight_class:
            return dict(_GLOBAL_PRIORS)

        rows = (
            self._s.query(FightStats)
            .join(Fight, FightStats.fight_id == Fight.id)
            .join(Fighter, FightStats.fighter_id == Fighter.id)
            .filter(
                Fighter.weight_class == weight_class,
                Fight.date < as_of_date,
            )
            .all()
        )

        if len(rows) < 10:
            return dict(_GLOBAL_PRIORS)

        sig_l = sum(r.significant_strikes_landed or 0 for r in rows)
        sig_a = sum(r.significant_strikes_attempted or 0 for r in rows)
        td_l = sum(r.takedowns_landed or 0 for r in rows)
        td_a = sum(r.takedowns_attempted or 0 for r in rows)
        sub_t = sum(r.submission_attempts or 0 for r in rows)
        n = len(rows)

        return {
            "sig_strike_accuracy": _safe_div(sig_l, sig_a, _GLOBAL_PRIORS["sig_strike_accuracy"]),
            "td_accuracy": _safe_div(td_l, td_a, _GLOBAL_PRIORS["td_accuracy"]),
            "td_defense": _GLOBAL_PRIORS["td_defense"],
            "strike_defense": _GLOBAL_PRIORS["strike_defense"],
            "sub_att_per_fight": _safe_div(sub_t, n, _GLOBAL_PRIORS["sub_att_per_fight"]),
            "kd_per_fight": _GLOBAL_PRIORS["kd_per_fight"],
        }

    def _aggregate(
        self,
        fighter_id: int,
        records: list,
        as_of_date: date,
        prior: dict,
    ) -> _FighterAggStats:
        """
        Aggregate fight records into career stats, applying Bayesian shrinkage
        toward the weight-class prior when ``num_fights < MIN_FIGHTS_FOR_CONFIDENCE``.
        """
        n = len(records)
        low_confidence = n < MIN_FIGHTS_FOR_CONFIDENCE

        if n == 0:
            return _FighterAggStats(
                fighter_id=fighter_id,
                num_fights=0,
                low_confidence=True,
                sig_strike_accuracy=prior["sig_strike_accuracy"],
                strike_defense=prior["strike_defense"],
                td_accuracy=prior["td_accuracy"],
                td_defense=prior["td_defense"],
                sub_att_per_fight=prior["sub_att_per_fight"],
                kd_per_fight=prior["kd_per_fight"],
                ko_rate=0.0,
                sub_rate=0.0,
                dec_rate=0.0,
                days_since_last_fight=None,
                win_streak=0,
                recent_sig_strike_accuracy=prior["sig_strike_accuracy"],
                recent_td_accuracy=prior["td_accuracy"],
                striker_score=prior["sig_strike_accuracy"],
                wrestler_score=prior["td_accuracy"],
                grappler_score=min(1.0, prior["sub_att_per_fight"] / GRAPPLER_SCORE_CAP),
                brawler_score=min(1.0, prior["kd_per_fight"] / BRAWLER_SCORE_CAP),
            )

        # --- Raw aggregates ---
        total_sig_l = sum(r.sig_landed for r in records)
        total_sig_a = sum(r.sig_attempted for r in records)
        total_td_l = sum(r.td_landed for r in records)
        total_td_a = sum(r.td_attempted for r in records)
        total_opp_sig_l = sum(r.opp_sig_landed for r in records)
        total_opp_sig_a = sum(r.opp_sig_attempted for r in records)
        total_opp_td_l = sum(r.opp_td_landed for r in records)
        total_opp_td_a = sum(r.opp_td_attempted for r in records)
        total_sub_att = sum(r.sub_att for r in records)
        total_kd = sum(r.knockdowns for r in records)

        raw_sig_acc = _safe_div(total_sig_l, total_sig_a)
        raw_td_acc = _safe_div(total_td_l, total_td_a)
        raw_strike_def = 1.0 - _safe_div(total_opp_sig_l, total_opp_sig_a)
        raw_td_def = 1.0 - _safe_div(total_opp_td_l, total_opp_td_a)
        raw_sub_per_fight = _safe_div(total_sub_att, n)
        raw_kd_per_fight = _safe_div(total_kd, n)

        # --- Bayesian shrinkage when low confidence ---
        k = PRIOR_STRENGTH
        if low_confidence:
            sig_acc = (n * raw_sig_acc + k * prior["sig_strike_accuracy"]) / (n + k)
            td_acc = (n * raw_td_acc + k * prior["td_accuracy"]) / (n + k)
            strike_def = (n * raw_strike_def + k * prior["strike_defense"]) / (n + k)
            td_def = (n * raw_td_def + k * prior["td_defense"]) / (n + k)
            sub_per_fight = (n * raw_sub_per_fight + k * prior["sub_att_per_fight"]) / (n + k)
            kd_per_fight = (n * raw_kd_per_fight + k * prior["kd_per_fight"]) / (n + k)
        else:
            sig_acc = raw_sig_acc
            td_acc = raw_td_acc
            strike_def = raw_strike_def
            td_def = raw_td_def
            sub_per_fight = raw_sub_per_fight
            kd_per_fight = raw_kd_per_fight

        # --- Finishing rates (no shrinkage — categorical) ---
        ko_rate = _safe_div(
            sum(1 for r in records if r.won and r.method in _KO_METHODS), n
        )
        sub_rate = _safe_div(
            sum(1 for r in records if r.won and r.method in _SUB_METHODS), n
        )
        dec_rate = _safe_div(
            sum(1 for r in records if r.won and r.method in _DEC_METHODS), n
        )

        # --- Recency ---
        last = records[-1]   # records are oldest-first; last = most recent
        days_since = float((as_of_date - last.fight_date).days)

        streak = 0
        for r in reversed(records):
            if r.won:
                streak += 1
            else:
                break

        # --- Last-3 weighted stats ---
        recent = list(reversed(records))[: len(RECENCY_WEIGHTS)]
        weights = RECENCY_WEIGHTS[: len(recent)]
        weight_sum = sum(weights)

        recent_sig_acc = sum(
            w * _safe_div(r.sig_landed, r.sig_attempted)
            for w, r in zip(weights, recent)
        ) / weight_sum

        recent_td_acc = sum(
            w * _safe_div(r.td_landed, r.td_attempted)
            for w, r in zip(weights, recent)
        ) / weight_sum

        # --- Style archetype scores ---
        striker_score = sig_acc
        wrestler_score = td_acc
        grappler_score = min(1.0, sub_per_fight / GRAPPLER_SCORE_CAP)
        brawler_score = min(1.0, kd_per_fight / BRAWLER_SCORE_CAP)

        return _FighterAggStats(
            fighter_id=fighter_id,
            num_fights=n,
            low_confidence=low_confidence,
            sig_strike_accuracy=sig_acc,
            strike_defense=strike_def,
            td_accuracy=td_acc,
            td_defense=td_def,
            sub_att_per_fight=sub_per_fight,
            kd_per_fight=kd_per_fight,
            ko_rate=ko_rate,
            sub_rate=sub_rate,
            dec_rate=dec_rate,
            days_since_last_fight=days_since,
            win_streak=streak,
            recent_sig_strike_accuracy=recent_sig_acc,
            recent_td_accuracy=recent_td_acc,
            striker_score=striker_score,
            wrestler_score=wrestler_score,
            grappler_score=grappler_score,
            brawler_score=brawler_score,
        )

    def _build_vector(
        self,
        fa: Fighter,
        fb: Fighter,
        stats_a: _FighterAggStats,
        stats_b: _FighterAggStats,
    ) -> FeatureVector:
        """Combine per-fighter aggregated stats into the final feature vector."""

        # Physical
        height_delta = (fa.height or 0.0) - (fb.height or 0.0)
        reach_delta = (fa.reach or 0.0) - (fb.reach or 0.0)
        age_delta = (
            float((fa.dob - fb.dob).days) if (fa.dob and fb.dob) else 0.0
        )
        southpaw_a = (fa.stance or "").lower() == "southpaw"
        southpaw_b = (fb.stance or "").lower() == "southpaw"
        is_southpaw_matchup = 1 if southpaw_a != southpaw_b else 0

        # Recency defaults for fighters with no fights
        days_a = stats_a.days_since_last_fight if stats_a.days_since_last_fight is not None else 9999.0
        days_b = stats_b.days_since_last_fight if stats_b.days_since_last_fight is not None else 9999.0

        return FeatureVector(
            # Physical
            height_delta=height_delta,
            reach_delta=reach_delta,
            age_delta=age_delta,
            is_southpaw_matchup=is_southpaw_matchup,
            # Striking
            sig_strike_accuracy_delta=stats_a.sig_strike_accuracy - stats_b.sig_strike_accuracy,
            strike_defense_delta=stats_a.strike_defense - stats_b.strike_defense,
            knockdown_rate_delta=stats_a.kd_per_fight - stats_b.kd_per_fight,
            # Grappling
            takedown_accuracy_delta=stats_a.td_accuracy - stats_b.td_accuracy,
            takedown_defense_delta=stats_a.td_defense - stats_b.td_defense,
            submission_attempts_per_fight_delta=stats_a.sub_att_per_fight - stats_b.sub_att_per_fight,
            # Finishing rates
            ko_rate_a=stats_a.ko_rate,
            ko_rate_b=stats_b.ko_rate,
            sub_rate_a=stats_a.sub_rate,
            sub_rate_b=stats_b.sub_rate,
            dec_rate_a=stats_a.dec_rate,
            dec_rate_b=stats_b.dec_rate,
            # Recency
            days_since_last_fight_a=days_a,
            days_since_last_fight_b=days_b,
            win_streak_delta=stats_a.win_streak - stats_b.win_streak,
            # Last-3 weighted
            recent_sig_strike_accuracy_delta=(
                stats_a.recent_sig_strike_accuracy - stats_b.recent_sig_strike_accuracy
            ),
            recent_td_accuracy_delta=(
                stats_a.recent_td_accuracy - stats_b.recent_td_accuracy
            ),
            # Elo
            elo_delta=fa.elo_rating - fb.elo_rating,
            # Style archetypes
            striker_score_a=stats_a.striker_score,
            wrestler_score_a=stats_a.wrestler_score,
            grappler_score_a=stats_a.grappler_score,
            brawler_score_a=stats_a.brawler_score,
            striker_score_b=stats_b.striker_score,
            wrestler_score_b=stats_b.wrestler_score,
            grappler_score_b=stats_b.grappler_score,
            brawler_score_b=stats_b.brawler_score,
            # Confidence
            low_confidence_a=stats_a.low_confidence,
            low_confidence_b=stats_b.low_confidence,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` when denominator is zero."""
    return numerator / denominator if denominator else default
