"""
Probability calibration — wraps a pretrained model with sklearn CalibratedClassifierCV
and provides a shared evaluate() helper used by both train.py and predict.py.

Implemented in Phase 5 — Outcome Prediction Model.
Odds generation will be added in Phase 7.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logger = logging.getLogger(__name__)

_VALID_METHODS = {"isotonic", "sigmoid"}


def calibrate(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """
    Wrap a pretrained model with probability calibration.

    Uses cv="prefit" so the base model is NOT retrained — only the calibration
    layer is fitted on X_cal / y_cal. The caller is responsible for ensuring
    X_cal was not used to train the base model.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible classifier (e.g. XGBClassifier).
    X_cal:
        Calibration feature matrix (numpy array, shape [n, features]).
    y_cal:
        Binary labels (0/1) for the calibration set.
    method:
        "isotonic" (default) or "sigmoid" (Platt scaling).

    Returns
    -------
    CalibratedClassifierCV
        Fitted calibrated wrapper.

    Raises
    ------
    ValueError
        If method is not "isotonic" or "sigmoid".
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Invalid calibration method {method!r}. Must be one of {_VALID_METHODS}."
        )
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Compute accuracy, log-loss, and Brier score for a fitted model.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible classifier with predict() and predict_proba().
    X:
        Feature matrix (numpy array).
    y:
        True binary labels (0/1).

    Returns
    -------
    dict
        Keys: accuracy, log_loss, brier_score_loss — all floats.
        Returns zeros with a warning if X is empty.
    """
    if len(X) == 0:
        warnings.warn("evaluate() called with empty dataset — returning zeros.", stacklevel=2)
        return {"accuracy": 0.0, "log_loss": 0.0, "brier_score_loss": 0.0}

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "log_loss": float(log_loss(y, y_proba)),
        "brier_score_loss": float(brier_score_loss(y, y_proba)),
    }
