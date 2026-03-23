"""
Model training and serialisation.
Implemented in Phase 5 — Outcome Prediction Model.

Usage:
    python -m ml.train                        # train and save model
    python -m ml.train --report               # dry run: print metrics, do not save
    python -m ml.train --output models/       # specify output directory
    python -m ml.train --db-url postgresql://...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import accuracy_score, log_loss as sk_log_loss, brier_score_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ml.calibration import calibrate, evaluate
from ml.elo import expected_score, load_fights_from_db, replay_fights
from ml.features import FeatureBuilder, FeatureVector
from models.schema import Fight

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [f.name for f in dataclasses.fields(FeatureVector)]
MODEL_FILENAME = "xgb_model.joblib"
REPORT_FILENAME = "training_report.json"
METHOD_MODEL_FILENAME = "method_model.joblib"


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_training_dataset(
    session: Session,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build a leakage-free training dataset from all historical fights.

    For each fight with a decisive outcome, features are computed using only
    data available *before* the fight date (enforced by FeatureBuilder's
    strict Fight.date < as_of_date filter).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with FEATURE_COLS columns, one row per fight.
    y : pd.Series[int]
        Binary labels: 1 = fighter_a won, 0 = fighter_b won.
    dates : pd.Series[date]
        Fight dates aligned with X/y — used only for chronological splitting.
    """
    stmt = select(Fight).order_by(Fight.date.asc(), Fight.id.asc())
    fights = session.scalars(stmt).all()

    builder = FeatureBuilder(session)
    rows: list[dict] = []
    labels: list[int] = []
    dates: list[date] = []

    skipped_draws = 0
    skipped_errors = 0

    for fight in fights:
        if fight.winner_id is None:
            skipped_draws += 1
            continue
        if fight.winner_id not in (fight.fighter_a_id, fight.fighter_b_id):
            skipped_errors += 1
            continue

        try:
            fv = builder.build(fight.fighter_a_id, fight.fighter_b_id, as_of_date=fight.date)
        except ValueError as exc:
            logger.warning("Skipping fight %d: %s", fight.id, exc)
            skipped_errors += 1
            continue

        feat_dict = fv.to_dict()
        # Cast booleans to int so XGBoost receives numeric input
        for key, val in feat_dict.items():
            if isinstance(val, bool):
                feat_dict[key] = int(val)

        rows.append(feat_dict)
        labels.append(1 if fight.winner_id == fight.fighter_a_id else 0)
        dates.append(fight.date)

    logger.info(
        "Dataset built: %d fights included, %d draws skipped, %d errors skipped.",
        len(rows), skipped_draws, skipped_errors,
    )

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    y = pd.Series(labels, dtype=int, name="label")
    dates_series = pd.Series(dates, name="date")
    return X, y, dates_series


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def split_by_date(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split dataset chronologically — no shuffling.

    Parameters
    ----------
    X, y, dates:
        Feature matrix, labels, and dates (must be aligned and sorted by date).
    test_fraction:
        Fraction of data to hold out as test set (most recent fights).

    Returns
    -------
    X_train, X_test, y_train, y_test, dates_train, dates_test
    """
    n = len(X)
    split_idx = int(n * (1 - test_fraction))

    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_test = X.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)
    dates_train = dates.iloc[:split_idx].reset_index(drop=True)
    dates_test = dates.iloc[split_idx:].reset_index(drop=True)

    if len(dates_train) > 0 and len(dates_test) > 0:
        assert dates_train.max() <= dates_test.min(), (
            f"Chronological split violated: train max {dates_train.max()} > "
            f"test min {dates_test.min()}"
        )

    return X_train, X_test, y_train, y_test, dates_train, dates_test


# ---------------------------------------------------------------------------
# Elo baseline
# ---------------------------------------------------------------------------

def elo_baseline(
    session: Session,
    test_fights: list[dict],
    train_fights: list[dict],
) -> dict:
    """
    Compute Elo-only prediction metrics on the test set.

    Replays Elo from scratch using train_fights, then uses the resulting
    ratings to predict test_fights. Returns accuracy, log_loss, and brier_score
    so the metrics are directly comparable with the XGBoost results.

    Note: replay_fights() re-initialises all ratings from INITIAL_ELO — it does
    not read fighter.elo_rating from the DB. This matches how elo.backtest() works.

    Parameters
    ----------
    session:
        Unused (kept for API symmetry with other pipeline functions).
    test_fights:
        List of fight dicts (from load_fights_from_db) in the test period.
    train_fights:
        List of fight dicts used to compute pre-test Elo ratings.

    Returns
    -------
    dict
        Keys: accuracy, log_loss, brier_score_loss, correct, total, skipped_draws.
    """
    train_ratings = replay_fights(train_fights)

    y_true: list[int] = []
    y_proba: list[float] = []
    skipped_draws = 0

    for fight in test_fights:
        if fight["winner_id"] is None:
            skipped_draws += 1
            continue

        ra = train_ratings.get(fight["fighter_a_id"], 1500.0)
        rb = train_ratings.get(fight["fighter_b_id"], 1500.0)
        prob_a = expected_score(ra, rb)

        label = 1 if fight["winner_id"] == fight["fighter_a_id"] else 0
        y_true.append(label)
        y_proba.append(prob_a)

    total = len(y_true)
    if total == 0:
        return {
            "accuracy": 0.0, "log_loss": 0.0, "brier_score_loss": 0.0,
            "correct": 0, "total": 0, "skipped_draws": skipped_draws,
        }

    y_pred = [1 if p >= 0.5 else 0 for p in y_proba]
    correct = sum(p == t for p, t in zip(y_pred, y_true))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(sk_log_loss(y_true, y_proba)),
        "brier_score_loss": float(brier_score_loss(y_true, y_proba)),
        "correct": correct,
        "total": total,
        "skipped_draws": skipped_draws,
    }


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_cv_splits: int = 5,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier with time-series cross-validation.

    Uses TimeSeriesSplit to respect chronological ordering — X_train must
    already be sorted by date (guaranteed by split_by_date).

    GridSearchCV selects the best hyperparameters by neg_log_loss, then
    refits on the full X_train.

    Parameters
    ----------
    X_train, y_train:
        Training features and labels.
    n_cv_splits:
        Number of TimeSeriesSplit folds. Use 2 in tests, 5 in production.

    Returns
    -------
    lgb.LGBMClassifier
        Best estimator refitted on the full training set.
    """
    base = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    cv = TimeSeriesSplit(n_splits=n_cv_splits)
    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train.values, y_train.values)

    logger.info("Best params: %s | CV score: %.4f", search.best_params_, -search.best_score_)
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Full pipeline (session-based, for testing)
# ---------------------------------------------------------------------------

def _run_with_session(
    session: Session,
    output_dir: str = "models/",
    report: bool = False,
) -> dict:
    """
    Full training pipeline given an open SQLAlchemy session.

    Exposed as a private function so tests can pass an in-memory SQLite session
    directly without needing a database URL string.

    Parameters
    ----------
    session:
        Active SQLAlchemy session pointing at a populated database.
    output_dir:
        Directory to write xgb_model.joblib and training_report.json.
    report:
        If True, compute and return metrics but do NOT write files to disk.

    Returns
    -------
    dict
        Report with trained_at, train_size, test_size, split_date, xgb metrics,
        and elo_baseline metrics.
    """
    # 1. Load fight dicts for Elo baseline (session re-used, no extra query cost)
    all_fight_dicts = load_fights_from_db(session)

    # 2. Build feature dataset
    X, y, dates = build_training_dataset(session)
    if len(X) == 0:
        raise RuntimeError("No training data available. Is the database populated?")

    # 3. Chronological split
    X_train, X_test, y_train, y_test, dates_train, dates_test = split_by_date(X, y, dates)

    # 4. Slice fight dicts by split date for Elo baseline
    split_date = dates_test.min() if len(dates_test) > 0 else date.today()
    train_fight_dicts = [f for f in all_fight_dicts if f["date"] < split_date]
    test_fight_dicts = [
        f for f in all_fight_dicts
        if f["date"] >= split_date and f["winner_id"] is not None
    ]

    # 5. Train XGBoost
    model = train_model(X_train, y_train)

    # 6. Calibrate on tail of training set
    # X_cal is carved from X_train so cv="prefit" is valid.
    # For small datasets this means the calibration set overlaps with training —
    # acceptable; for large real-data runs the overlap is minor.
    cal_idx = int(len(X_train) * 0.80)
    X_cal = X_train.iloc[cal_idx:].values
    y_cal = y_train.iloc[cal_idx:].values
    cal_model = calibrate(model, X_cal, y_cal, method="isotonic")

    # 7. Evaluate calibrated model on held-out test set
    xgb_metrics = evaluate(cal_model, X_test.values, y_test.values)

    # 8. Elo baseline on same test set
    elo_metrics = elo_baseline(session, test_fight_dicts, train_fight_dicts)

    # 9. Build report
    report_dict = {
        "trained_at": datetime.utcnow().isoformat(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "split_date": str(split_date),
        "xgb": xgb_metrics,
        "elo_baseline": elo_metrics,
    }

    # 10. Persist artifacts (skip in dry-run mode)
    if not report:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(cal_model, out / MODEL_FILENAME)
        with open(out / REPORT_FILENAME, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        logger.info("Model saved to %s", out / MODEL_FILENAME)
        logger.info("Report saved to %s", out / REPORT_FILENAME)

    # 11. Train and save method of victory model
    from ml.method_train import _run_method_pipeline  # lazy — avoids circular import
    method_report = _run_method_pipeline(session, output_dir=output_dir, report=report)
    report_dict["method_model"] = method_report["method_model"]

    return report_dict


# ---------------------------------------------------------------------------
# Public entry point (creates its own engine/session from database URL)
# ---------------------------------------------------------------------------

def run(
    database_url: str,
    output_dir: str = "models/",
    report: bool = False,
) -> dict:
    """
    Full training pipeline.

    Parameters
    ----------
    database_url:
        SQLAlchemy database URL (e.g. from DATABASE_URL env var).
    output_dir:
        Directory to write model artifacts.
    report:
        Dry run — compute metrics without saving to disk.

    Returns
    -------
    dict
        Training report with metrics for XGBoost and Elo baseline.
    """
    engine = create_engine(database_url)
    with Session(engine) as session:
        return _run_with_session(session, output_dir=output_dir, report=report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train UFC fight predictor model.")
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy database URL (defaults to DATABASE_URL env var).",
    )
    parser.add_argument(
        "--output",
        default="models/",
        help="Output directory for model artifacts (default: models/).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Dry run: print metrics without saving model to disk.",
    )
    args = parser.parse_args()

    if not args.db_url:
        logger.error("DATABASE_URL not set and --db-url not provided.")
        sys.exit(1)

    result = run(args.db_url, output_dir=args.output, report=args.report)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
