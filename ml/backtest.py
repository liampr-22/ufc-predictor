"""
Backtesting module — Phase 7: Odds Generation & Backtesting.

Compares model-implied probabilities against historical closing lines stored
in the historical_odds table to measure:

    - Closing Line Value (CLV): positive CLV means the model's implied
      probability was sharper (higher) than the market's closing line for
      the fighter the model backed.
    - Flat-stake P&L: simulates betting $1 on every fight the model has a
      view on, collecting at fair closing odds.

Only closing lines are used (not opening lines). HistoricalOdds rows must
have both fighter_a_odds and fighter_b_odds populated and a non-null
Fight.winner_id (draws are skipped).

Usage:
    python -m ml.backtest                        # run and print report
    python -m ml.backtest --output report.md     # write markdown report
    python -m ml.backtest --db-url postgresql://...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ml.calibration import american_odds_to_implied_prob, remove_vig
from models.schema import Fight, HistoricalOdds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-fight result container
# ---------------------------------------------------------------------------

@dataclass
class FightBacktestResult:
    fight_id: int
    fight_date: str            # ISO format
    fighter_a_id: int
    fighter_b_id: int
    winner_id: Optional[int]
    model_prob_a: float        # model P(fighter_a wins)
    fair_market_prob_a: float  # closing line implied prob for A after vig removal
    fair_market_prob_b: float  # closing line implied prob for B after vig removal
    model_pick: str            # "A" or "B"
    clv: float                 # closing line value on the model's pick (higher is better)
    pnl: float                 # flat-stake P&L on this fight (units, stake = 1)
    beat_closing_line: bool    # True if clv > 0


# ---------------------------------------------------------------------------
# Core backtesting function
# ---------------------------------------------------------------------------

def run_backtest(
    session: Session,
    predictor,
    vig: float = 0.04,
) -> dict:
    """
    Run backtesting against all fights that have historical closing odds.

    For each fight the model predicts fighter_a's win probability. The
    model "bets" on whichever fighter it assigns the higher probability.
    CLV is measured against the fair closing line after vig removal.

    Parameters
    ----------
    session:
        Active SQLAlchemy session connected to a populated database.
    predictor:
        A fitted ml.predict.Predictor instance.
    vig:
        Sportsbook vig used when the market total implied probability
        deviates unexpectedly. Not used directly here — vig is removed
        from market odds via remove_vig() regardless of this value.
        Kept as a parameter for future use (e.g. logging expected overround).

    Returns
    -------
    dict
        {
            "total_fights": int,
            "skipped": int,
            "pct_beat_closing_line": float,   # 0.0–1.0
            "avg_clv": float,                 # mean CLV across all fights (probability units)
            "total_pnl": float,               # sum of per-fight P&L in stake units
            "per_fight": list[dict],
        }
    """
    stmt = (
        select(Fight, HistoricalOdds)
        .join(HistoricalOdds, Fight.id == HistoricalOdds.fight_id)
        .where(
            HistoricalOdds.fighter_a_odds.isnot(None),
            HistoricalOdds.fighter_b_odds.isnot(None),
            Fight.winner_id.isnot(None),
        )
        .order_by(Fight.date.asc(), Fight.id.asc())
    )

    rows = session.execute(stmt).all()
    results: list[FightBacktestResult] = []
    skipped = 0

    for fight, odds_row in rows:
        try:
            model_prob_a = predictor.predict_proba(
                session,
                fight.fighter_a_id,
                fight.fighter_b_id,
                as_of_date=fight.date,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping fight %d: %s", fight.id, exc)
            skipped += 1
            continue

        # Convert American closing odds → raw implied probabilities → fair probs
        raw_a = american_odds_to_implied_prob(odds_row.fighter_a_odds)
        raw_b = american_odds_to_implied_prob(odds_row.fighter_b_odds)
        fair_a, fair_b = remove_vig(raw_a, raw_b)

        model_prob_b = 1.0 - model_prob_a

        if model_prob_a >= model_prob_b:
            # Model backs fighter A
            clv = model_prob_a - fair_a
            if fight.winner_id == fight.fighter_a_id:
                pnl = (1.0 / fair_a) - 1.0  # decimal odds - 1 = profit per unit stake
            else:
                pnl = -1.0
            model_pick = "A"
        else:
            # Model backs fighter B
            clv = model_prob_b - fair_b
            if fight.winner_id == fight.fighter_b_id:
                pnl = (1.0 / fair_b) - 1.0
            else:
                pnl = -1.0
            model_pick = "B"

        results.append(FightBacktestResult(
            fight_id=fight.id,
            fight_date=str(fight.date),
            fighter_a_id=fight.fighter_a_id,
            fighter_b_id=fight.fighter_b_id,
            winner_id=fight.winner_id,
            model_prob_a=round(model_prob_a, 6),
            fair_market_prob_a=round(fair_a, 6),
            fair_market_prob_b=round(fair_b, 6),
            model_pick=model_pick,
            clv=round(clv, 6),
            pnl=round(pnl, 6),
            beat_closing_line=clv > 0,
        ))

    if not results:
        logger.warning("No fights with historical odds found — backtest is empty.")
        return {
            "total_fights": 0,
            "skipped": skipped,
            "pct_beat_closing_line": 0.0,
            "avg_clv": 0.0,
            "total_pnl": 0.0,
            "per_fight": [],
        }

    n = len(results)
    pct_beat = sum(r.beat_closing_line for r in results) / n
    avg_clv = sum(r.clv for r in results) / n
    total_pnl = sum(r.pnl for r in results)

    return {
        "total_fights": n,
        "skipped": skipped,
        "pct_beat_closing_line": round(float(pct_beat), 4),
        "avg_clv": round(float(avg_clv), 4),
        "total_pnl": round(float(total_pnl), 4),
        "per_fight": [asdict(r) for r in results],
    }


# ---------------------------------------------------------------------------
# Markdown report formatter
# ---------------------------------------------------------------------------

def format_report(result: dict) -> str:
    """
    Render a backtest result dict as a Markdown report string.

    Parameters
    ----------
    result:
        Output of run_backtest().

    Returns
    -------
    str
        Markdown-formatted report.
    """
    n = result["total_fights"]
    skipped = result["skipped"]
    pct_beat = result["pct_beat_closing_line"]
    avg_clv = result["avg_clv"]
    total_pnl = result["total_pnl"]
    per_fight = result["per_fight"]

    lines = [
        "# UFC Predictor — Backtest Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Fights backtested | {n} |",
        f"| Fights skipped (no features) | {skipped} |",
        f"| % beats closing line | {pct_beat * 100:.1f}% |",
        f"| Average CLV | {avg_clv * 100:+.2f}% |",
        f"| Flat-stake P&L (units, stake = 1) | {total_pnl:+.2f} |",
        f"| ROI | {(total_pnl / n * 100) if n else 0:+.2f}% |",
        "",
    ]

    if n == 0:
        lines += [
            "> **No historical odds found in the database.**",
            "> Populate the `historical_odds` table and re-run:",
            "> `docker-compose exec api python -m ml.backtest`",
            "",
        ]
        return "\n".join(lines)

    lines += [
        "## Interpretation",
        "",
        "- **CLV > 0%** means the model's implied line was sharper than the market close for that fight.",
        "- **Average CLV > 0%** across all fights indicates the model consistently finds value.",
        "- **P&L** is computed by betting 1 unit on every fight and collecting at fair closing odds.",
        "",
        "## Per-Fight Results",
        "",
        "| Fight ID | Date | Model Pick | Model Prob | Mkt Close | CLV | P&L | Beat CLV? |",
        "|----------|------|------------|------------|-----------|-----|-----|-----------|",
    ]

    for f in per_fight:
        if f["model_pick"] == "A":
            model_prob = f["model_prob_a"]
            mkt_prob = f["fair_market_prob_a"]
        else:
            model_prob = 1.0 - f["model_prob_a"]
            mkt_prob = f["fair_market_prob_b"]

        clv_str = f"{f['clv'] * 100:+.1f}%"
        pnl_str = f"{f['pnl']:+.2f}"
        beat_str = "✓" if f["beat_closing_line"] else "✗"
        lines.append(
            f"| {f['fight_id']} | {f['fight_date']} | {f['model_pick']} "
            f"| {model_prob:.1%} | {mkt_prob:.1%} | {clv_str} | {pnl_str} | {beat_str} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _run_with_session(
    session: Session,
    output_path: Optional[str] = None,
) -> dict:
    """Run backtest given an open session. Exposed for tests."""
    from ml.predict import Predictor  # lazy — avoids circular import at module level

    model_dir = os.environ.get("MODEL_DIR", "models/")
    try:
        predictor = Predictor(model_dir=model_dir)
    except FileNotFoundError as exc:
        logger.error("Model not found: %s — train first with `python -m ml.train`.", exc)
        raise

    result = run_backtest(session, predictor)

    if output_path:
        report_md = format_report(result)
        Path(output_path).write_text(report_md, encoding="utf-8")
        logger.info("Backtest report written to %s", output_path)
    else:
        print(json.dumps(result, indent=2))

    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Backtest model against historical closing odds.")
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy database URL (defaults to DATABASE_URL env var).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write Markdown report to this path instead of printing JSON.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "models/"),
        help="Directory containing trained model files (default: models/).",
    )
    args = parser.parse_args()

    if not args.db_url:
        logger.error("DATABASE_URL not set and --db-url not provided.")
        sys.exit(1)

    os.environ["MODEL_DIR"] = args.model_dir

    engine = create_engine(args.db_url)
    with Session(engine) as session:
        _run_with_session(session, output_path=args.output)


if __name__ == "__main__":
    main()
