"""
Method of Victory model — Phase 6.

Trains a multiclass LightGBM classifier predicting how a fight ends:
    0 = KO/TKO
    1 = Submission
    2 = Decision

Reuses FeatureBuilder from ml/features.py. The existing feature set already
contains finishing-rate and grappling-ratio features that are strongly
predictive of method, so no additional feature engineering is required.

Calibration: one-vs-rest (OvR) via CalibratedClassifierCV with cv="prefit".
sklearn applies OvR calibration automatically for multiclass estimators.

Usage:
    python -m ml.method_train                  # train and save
    python -m ml.method_train --report         # dry run: print metrics only
    python -m ml.method_train --output models/
    python -m ml.method_train --db-url postgresql://...
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    log_loss as sk_log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ml.elo import build_elo_snapshots, load_fights_from_db
from ml.features import FeatureBuilder, FeatureVector
from models.schema import Fight

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_MODEL_FILENAME = "method_model.joblib"
METHOD_REPORT_FILENAME = "method_training_report.json"

METHOD_LABEL_KO = 0
METHOD_LABEL_SUB = 1
METHOD_LABEL_DEC = 2

METHOD_NAMES: dict[int, str] = {
    METHOD_LABEL_KO: "ko_tko",
    METHOD_LABEL_SUB: "submission",
    METHOD_LABEL_DEC: "decision",
}

# Computed independently from FeatureVector — avoids importing from train.py
FEATURE_COLS: list[str] = [f.name for f in dataclasses.fields(FeatureVector)]

_KO_METHODS = frozenset({"KO", "KO/TKO", "TKO", "TKO - DOCTOR'S STOPPAGE"})
_SUB_METHODS = frozenset({"SUB", "SUBMISSION"})
_DEC_METHODS = frozenset({
    "DEC", "U-DEC", "S-DEC", "M-DEC", "UD", "SD", "MD",
    "DECISION - UNANIMOUS", "DECISION - SPLIT", "DECISION - MAJORITY",
})


def _method_label(method_str: str | None) -> int | None:
    """Return 0/1/2 for KO/SUB/DEC, or None if unclassifiable."""
    if method_str is None:
        return None
    m = method_str.upper().strip()
    if m in _KO_METHODS:
        return METHOD_LABEL_KO
    if m in _SUB_METHODS:
        return METHOD_LABEL_SUB
    if m in _DEC_METHODS:
        return METHOD_LABEL_DEC
    return None


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_method_dataset(
    session: Session,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build a leakage-free method-of-victory dataset from all historical fights.

    Includes all fights with a classifiable method (KO/TKO, Submission,
    Decision). Fights with None or unknown methods are excluded.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (same FEATURE_COLS as outcome model).
    y : pd.Series[int]
        Method labels: 0=KO/TKO, 1=Submission, 2=Decision.
    dates : pd.Series[date]
        Fight dates aligned with X/y — used for chronological splitting.
    """
    stmt = select(Fight).order_by(Fight.date.asc(), Fight.id.asc())
    fights = session.scalars(stmt).all()

    # Build pre-fight Elo snapshots so the method model uses leakage-free Elo,
    # matching the approach in train.py's build_training_dataset().
    all_fight_dicts = load_fights_from_db(session)
    elo_snapshots = build_elo_snapshots(all_fight_dicts)

    builder = FeatureBuilder(session)
    rows: list[dict] = []
    labels: list[int] = []
    dates: list[date] = []

    skipped_no_method = 0
    skipped_errors = 0

    for fight in fights:
        label = _method_label(fight.method)
        if label is None:
            skipped_no_method += 1
            continue

        snapshot = elo_snapshots.get(fight.id, {})
        elo_a = snapshot.get(fight.fighter_a_id)
        elo_b = snapshot.get(fight.fighter_b_id)

        try:
            fv = builder.build(
                fight.fighter_a_id, fight.fighter_b_id,
                as_of_date=fight.date,
                elo_a=elo_a, elo_b=elo_b,
                elo_snapshots=elo_snapshots,
                scheduled_rounds=fight.scheduled_rounds,
            )
        except ValueError as exc:
            logger.warning("Skipping fight %d: %s", fight.id, exc)
            skipped_errors += 1
            continue

        feat_dict = fv.to_dict()
        for key, val in feat_dict.items():
            if isinstance(val, bool):
                feat_dict[key] = int(val)

        rows.append(feat_dict)
        labels.append(label)
        dates.append(fight.date)

    logger.info(
        "Method dataset built: %d fights included, %d skipped (no/unknown method), %d errors.",
        len(rows), skipped_no_method, skipped_errors,
    )

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    y = pd.Series(labels, dtype=int, name="method_label")
    dates_series = pd.Series(dates, name="date")
    return X, y, dates_series


# ---------------------------------------------------------------------------
# Chronological split (inlined to avoid circular import with train.py)
# ---------------------------------------------------------------------------

def _split_by_date(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Chronological 80/20 split — no shuffling."""
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
# Model training
# ---------------------------------------------------------------------------

def train_method_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_cv_splits: int = 5,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM multiclass classifier for method of victory.

    Uses TimeSeriesSplit to respect chronological ordering. GridSearchCV
    selects the best hyperparameters by neg_log_loss (multiclass), then
    refits on the full X_train.

    Parameters
    ----------
    X_train, y_train:
        Training features and method labels (0=KO/TKO, 1=SUB, 2=DEC).
    n_cv_splits:
        Number of TimeSeriesSplit folds. Use 2 in tests, 5 in production.

    Returns
    -------
    lgb.LGBMClassifier
        Best estimator refitted on the full training set.
    """
    base = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }

    cv = TimeSeriesSplit(n_splits=3)
    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train.values, y_train.values)

    logger.info(
        "Method model best params: %s | CV score: %.4f",
        search.best_params_, -search.best_score_,
    )
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_method_model(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """
    Wrap a pretrained multiclass model with OvR probability calibration.

    Uses cv="prefit" — the base model is not retrained. sklearn's
    CalibratedClassifierCV applies one-vs-rest calibration for multiclass
    estimators automatically.

    Parameters
    ----------
    model:
        Fitted LGBMClassifier with objective="multiclass".
    X_cal, y_cal:
        Calibration set (must not overlap with the model's training data).
    method:
        "isotonic" (default) or "sigmoid".

    Returns
    -------
    CalibratedClassifierCV
        Fitted calibrated wrapper.
    """
    # Only calibrate when all 3 classes are present in the calibration set.
    # If a class is missing the CalibratedClassifierCV output shape will be
    # wrong (< 3 columns), causing downstream errors.
    if len(np.unique(y_cal)) < 3:
        logger.warning(
            "Calibration set missing one or more method classes — skipping calibration."
        )
        return model
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_method_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Compute multiclass metrics for the method model.

    Parameters
    ----------
    model:
        Fitted sklearn-compatible multiclass classifier.
    X:
        Feature matrix (numpy array).
    y:
        True method labels (0/1/2).

    Returns
    -------
    dict
        Keys: log_loss, accuracy, per_class.
        per_class has ko_tko, submission, decision — each with precision,
        recall, f1. Returns zeros when X is empty.
    """
    _zero_per_class = {
        name: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for name in METHOD_NAMES.values()
    }

    if len(X) == 0:
        warnings.warn(
            "evaluate_method_model() called with empty dataset — returning zeros.",
            stacklevel=2,
        )
        return {"log_loss": 0.0, "accuracy": 0.0, "per_class": _zero_per_class}

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)  # shape [n, 3]

    y_proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
    ll = float(sk_log_loss(y, y_proba_clipped, labels=[0, 1, 2]))
    acc = float(accuracy_score(y, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1, 2], zero_division=0
    )

    per_class = {
        METHOD_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i in range(3)
    }

    return {"log_loss": ll, "accuracy": acc, "per_class": per_class}


# ---------------------------------------------------------------------------
# Full pipeline (session-based, for testing and internal use)
# ---------------------------------------------------------------------------

def _run_method_pipeline(
    session: Session,
    output_dir: str = "models/",
    report: bool = False,
    n_cv_splits: int = 5,
) -> dict:
    """
    Full method model training pipeline given an open SQLAlchemy session.

    Parameters
    ----------
    session:
        Active SQLAlchemy session pointing at a populated database.
    output_dir:
        Directory to write method_model.joblib and method_training_report.json.
    report:
        If True, compute metrics but do NOT write files to disk.
    n_cv_splits:
        CV folds for GridSearchCV. Pass 2 in tests for speed.

    Returns
    -------
    dict
        Report with trained_at, train_size, test_size, method_model metrics.
    """
    X, y, dates = build_method_dataset(session)
    if len(X) == 0:
        raise RuntimeError("No method training data available. Is the database populated?")

    X_train, X_test, y_train, y_test, _, _ = _split_by_date(X, y, dates)

    model = train_method_model(X_train, y_train, n_cv_splits=n_cv_splits)

    # 30% calibration carve-out (was 20%) gives more data for the 3-class calibration.
    # sigmoid (Platt) instead of isotonic — fewer parameters, less overfit on small cal sets.
    cal_idx = int(len(X_train) * 0.70)
    X_cal = X_train.iloc[cal_idx:].values
    y_cal = y_train.iloc[cal_idx:].values
    cal_model = calibrate_method_model(model, X_cal, y_cal, method="sigmoid")

    metrics = evaluate_method_model(cal_model, X_test.values, y_test.values)

    report_dict = {
        "trained_at": datetime.utcnow().isoformat(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "method_model": metrics,
    }

    if not report:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(cal_model, out / METHOD_MODEL_FILENAME)
        with open(out / METHOD_REPORT_FILENAME, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        logger.info("Method model saved to %s", out / METHOD_MODEL_FILENAME)
        logger.info("Method report saved to %s", out / METHOD_REPORT_FILENAME)

    return report_dict


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_method(
    database_url: str,
    output_dir: str = "models/",
    report: bool = False,
) -> dict:
    """
    Full method model training pipeline.

    Parameters
    ----------
    database_url:
        SQLAlchemy database URL.
    output_dir:
        Directory to write model artifacts.
    report:
        Dry run — compute metrics without saving to disk.
    """
    engine = create_engine(database_url)
    with Session(engine) as session:
        return _run_method_pipeline(session, output_dir=output_dir, report=report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train UFC method of victory model.")
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

    result = run_method(args.db_url, output_dir=args.output, report=args.report)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
