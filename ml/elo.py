"""
Elo rating system — initialise, replay, and update fighter ratings.
Implemented in Phase 3 — Elo Rating System.
"""

import argparse
import logging
import os
import sys
from datetime import date

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models.schema import Fight, Fighter

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_ELO: float = 1500.0
BASE_K: float = 48.0          # raised from 32 → wider Elo spread (~50% more differentiation)
DECAY_THRESHOLD_YEARS: int = 5
DECAY_MULTIPLIER: float = 0.5



# ── Pure math ─────────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Return the expected score for fighter A given both ratings.

    E_a = 1 / (1 + 10 ** ((Rb - Ra) / 400))
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_rating(rating: float, score: float, expected: float, k_factor: float) -> float:
    """
    Return an updated Elo rating after one fight.

    R_new = R + K * (S - E)
    """
    return rating + k_factor * (score - expected)


def effective_k(
    fight_date: date,
    reference_date: date,
    base_k: float = BASE_K,
    decay_threshold_years: int = DECAY_THRESHOLD_YEARS,
    decay_multiplier: float = DECAY_MULTIPLIER,
    opponent_elo: float | None = None,
) -> float:
    """
    Return the effective K-factor for one fighter in one fight.

    Two adjustments are applied:
    - Time decay: fights 5+ years old use base_k * decay_multiplier.
    - Opponent quality: K scales with opponent Elo relative to INITIAL_ELO.
      Beating/losing to a 1200-Elo opponent = 0.8× K (less movement).
      Beating/losing to a 1800-Elo opponent = 1.2× K (more movement), capped at 1.5×.
      This prevents fighters from farming Elo against weak opposition.
    """
    years_old = (reference_date - fight_date).days / 365.25
    k = base_k * (decay_multiplier if years_old >= decay_threshold_years else 1.0)
    if opponent_elo is not None:
        k *= min(1.5, opponent_elo / INITIAL_ELO)
    return k


# ── In-memory replay ──────────────────────────────────────────────────────────

def replay_fights(
    fights: list[dict],
    base_k: float = BASE_K,
    initial_rating: float = INITIAL_ELO,
    reference_date: date | None = None,
) -> dict[int, float]:
    """
    Replay all fights in chronological order and return final ratings.

    Resets every encountered fighter to initial_rating before replaying.
    Does not touch the database — all state is in-memory.

    Each dict in `fights` must contain:
        id, date, fighter_a_id, fighter_b_id, winner_id (int | None)

    Args:
        fights:         List of fight dicts ordered by date ascending.
        base_k:         Base K-factor.
        initial_rating: Starting rating for every fighter.
        reference_date: Date used to evaluate recency decay. Defaults to today.

    Returns:
        Dict mapping fighter_id → final Elo rating.
    """
    if reference_date is None:
        reference_date = date.today()

    ratings: dict[int, float] = {}

    for fight in fights:
        fa_id = fight["fighter_a_id"]
        fb_id = fight["fighter_b_id"]
        winner_id = fight["winner_id"]
        fight_date = fight["date"]

        # Initialise on first encounter
        if fa_id not in ratings:
            ratings[fa_id] = initial_rating
        if fb_id not in ratings:
            ratings[fb_id] = initial_rating

        ra = ratings[fa_id]
        rb = ratings[fb_id]

        ea = expected_score(ra, rb)
        eb = 1.0 - ea

        if winner_id == fa_id:
            sa, sb = 1.0, 0.0
        elif winner_id == fb_id:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5  # draw / no contest

        # Per-fighter K: each fighter's movement scales with their opponent's rating.
        ka = effective_k(fight_date, reference_date, base_k, opponent_elo=rb)
        kb = effective_k(fight_date, reference_date, base_k, opponent_elo=ra)

        ratings[fa_id] = update_rating(ra, sa, ea, ka)
        ratings[fb_id] = update_rating(rb, sb, eb, kb)

    return ratings


def backtest(
    fights: list[dict],
    holdout_fraction: float = 0.20,
    base_k: float = BASE_K,
    initial_rating: float = INITIAL_ELO,
) -> dict:
    """
    Hold out the last `holdout_fraction` of fights and measure prediction accuracy.

    Trains on the first (1 - holdout_fraction) of fights chronologically, then
    predicts each test fight by comparing pre-fight Elo ratings. Draws/NCs are
    skipped from the accuracy calculation.

    Returns:
        Dict with: accuracy, correct, total, train_size, test_size, skipped_draws
    """
    n = len(fights)
    split = n - max(1, int(n * holdout_fraction))
    train_fights = fights[:split]
    test_fights = fights[split:]

    # Compute ratings from training set only
    train_ratings = replay_fights(train_fights, base_k=base_k, initial_rating=initial_rating)

    correct = 0
    total = 0
    skipped_draws = 0

    for fight in test_fights:
        winner_id = fight["winner_id"]
        if winner_id is None:
            skipped_draws += 1
            continue

        fa_id = fight["fighter_a_id"]
        fb_id = fight["fighter_b_id"]
        ra = train_ratings.get(fa_id, initial_rating)
        rb = train_ratings.get(fb_id, initial_rating)

        predicted_winner = fa_id if ra >= rb else fb_id
        if predicted_winner == winner_id:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "train_size": len(train_fights),
        "test_size": len(test_fights),
        "skipped_draws": skipped_draws,
    }


# ── Pre-fight snapshot builder ────────────────────────────────────────────────

def build_elo_snapshots(
    fights: list[dict],
    base_k: float = BASE_K,
    initial_rating: float = INITIAL_ELO,
    reference_date: date | None = None,
) -> dict[int, dict[int, float]]:
    """
    Replay all fights chronologically and capture each fighter's Elo rating
    *before* each fight.

    This is critical for leakage-free training: when building features for a
    historical fight, we need the Elo ratings as they stood just before the
    fight occurred, not the final post-history values stored in the DB.

    Args:
        fights:         List of fight dicts ordered by date ascending.
                        Each dict must have: id, date, fighter_a_id, fighter_b_id, winner_id.
        base_k:         Base K-factor.
        initial_rating: Starting rating for every fighter.
        reference_date: Date used for time-decay. Defaults to today.

    Returns:
        Dict mapping fight_id → {fighter_id: pre_fight_elo}.
        Only the two participants are included per fight.
    """
    if reference_date is None:
        reference_date = date.today()

    ratings: dict[int, float] = {}
    snapshots: dict[int, dict[int, float]] = {}

    for fight in fights:
        fa_id = fight["fighter_a_id"]
        fb_id = fight["fighter_b_id"]
        winner_id = fight["winner_id"]
        fight_date = fight["date"]
        fight_id = fight["id"]

        if fa_id not in ratings:
            ratings[fa_id] = initial_rating
        if fb_id not in ratings:
            ratings[fb_id] = initial_rating

        # Capture pre-fight snapshot BEFORE updating ratings
        snapshots[fight_id] = {fa_id: ratings[fa_id], fb_id: ratings[fb_id]}

        # Update ratings (same logic as replay_fights)
        ra = ratings[fa_id]
        rb = ratings[fb_id]
        ea = expected_score(ra, rb)
        eb = 1.0 - ea

        if winner_id == fa_id:
            sa, sb = 1.0, 0.0
        elif winner_id == fb_id:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        ka = effective_k(fight_date, reference_date, base_k, opponent_elo=rb)
        kb = effective_k(fight_date, reference_date, base_k, opponent_elo=ra)
        ratings[fa_id] = update_rating(ra, sa, ea, ka)
        ratings[fb_id] = update_rating(rb, sb, eb, kb)

    return snapshots


# ── DB layer ──────────────────────────────────────────────────────────────────

def load_fights_from_db(session: Session) -> list[dict]:
    """
    Load all fights from the database ordered by date ASC, id ASC.

    Returns plain dicts to decouple replay logic from ORM objects.
    """
    stmt = select(Fight).order_by(Fight.date.asc(), Fight.id.asc())
    fights = session.scalars(stmt).all()
    return [
        {
            "id": f.id,
            "date": f.date,
            "fighter_a_id": f.fighter_a_id,
            "fighter_b_id": f.fighter_b_id,
            "winner_id": f.winner_id,
            "method": f.method,
        }
        for f in fights
    ]


def persist_ratings(session: Session, ratings: dict[int, float]) -> int:
    """
    Write Elo ratings back to the fighters table and commit.

    Returns the number of fighter rows updated.
    """
    updated = 0
    for fighter_id, rating in ratings.items():
        fighter = session.get(Fighter, fighter_id)
        if fighter is not None:
            fighter.elo_rating = rating
            updated += 1
    session.commit()
    return updated


def run_replay(database_url: str, base_k: float = BASE_K) -> dict[int, float]:
    """
    Full pipeline: connect → load fights → replay → persist ratings.

    Intended as the top-level callable used by the CLI and by
    scraper/scheduler.py after each scrape.

    Returns:
        Final ratings dict {fighter_id: float}.
    """
    engine = create_engine(database_url)
    with Session(engine) as session:
        fights = load_fights_from_db(session)
        ratings = replay_fights(fights, base_k=base_k)
        count = persist_ratings(session, ratings)
    logger.info("Elo replay complete. Updated %d fighters.", count)
    return ratings


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point for the Elo module.

    Usage:
        python -m ml.elo                # run replay and update DB
        python -m ml.elo --backtest     # run backtest, print accuracy, do NOT update DB
        python -m ml.elo --k 24         # override K-factor
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Elo rating system — replay fight history and update fighter ratings."
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest on held-out fights and print accuracy. Does not update the DB.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=BASE_K,
        metavar="K_FACTOR",
        help=f"Base K-factor (default: {BASE_K}).",
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set.")
        sys.exit(1)

    engine = create_engine(database_url)

    with Session(engine) as session:
        fights = load_fights_from_db(session)

    if args.backtest:
        result = backtest(fights, base_k=args.k)
        print(f"Backtest accuracy : {result['accuracy']:.1%}")
        print(f"Correct / Total   : {result['correct']} / {result['total']}")
        print(f"Draws skipped     : {result['skipped_draws']}")
        print(f"Train / Test split: {result['train_size']} / {result['test_size']}")
        if result["total"] > 0 and result["accuracy"] < 0.60:
            logger.warning(
                "Backtest accuracy %.1f%% is below the 60%% target.", result["accuracy"] * 100
            )
    else:
        ratings = run_replay(database_url, base_k=args.k)
        logger.info("Updated %d fighters.", len(ratings))


if __name__ == "__main__":
    main()
