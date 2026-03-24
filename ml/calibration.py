"""
Probability calibration — wraps a pretrained model with sklearn CalibratedClassifierCV
and provides a shared evaluate() helper used by both train.py and predict.py.

Also contains odds conversion utilities added in Phase 7:
    american_odds_to_implied_prob, remove_vig, apply_vig,
    prob_to_american_odds, prob_to_decimal_odds, prob_to_fractional_odds.

Implemented in Phase 5 — Outcome Prediction Model.
Odds generation added in Phase 7.
"""
from __future__ import annotations

import logging
import warnings
from fractions import Fraction

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

logger = logging.getLogger(__name__)

_VALID_METHODS = {"isotonic", "sigmoid"}


# ---------------------------------------------------------------------------
# Odds conversion (Phase 7)
# ---------------------------------------------------------------------------

def american_odds_to_implied_prob(odds: int) -> float:
    """
    Convert American odds to raw implied probability.

    Parameters
    ----------
    odds:
        American odds (negative = favourite, positive = underdog).
        e.g. -200, +150.

    Returns
    -------
    float
        Implied probability in (0, 1). Does NOT remove vig.
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Remove bookmaker vig from a pair of raw implied probabilities.

    Divides each raw implied probability by their sum so the result
    sums to 1.0 — the fair (no-vig) market price.

    Parameters
    ----------
    prob_a, prob_b:
        Raw implied probabilities for each side (sum > 1 due to vig).

    Returns
    -------
    tuple[float, float]
        (fair_prob_a, fair_prob_b) summing to 1.0.

    Raises
    ------
    ValueError
        If the sum of probabilities is not positive.
    """
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError(
            f"Sum of implied probabilities must be positive, got {total}."
        )
    return prob_a / total, prob_b / total


def apply_vig(p: float, vig: float = 0.04) -> float:
    """
    Apply bookmaker vig to a fair probability for a two-way market.

    Adds vig/2 to each side so that the sum of both vigged prices = 1 + vig.
    This is the standard symmetric vig model.

    Parameters
    ----------
    p:
        Fair win probability for one side, in (0, 1).
    vig:
        Total overround, e.g. 0.04 for a 4 % vig. Default: 0.04.

    Returns
    -------
    float
        Vigged implied probability for this side.

    Raises
    ------
    ValueError
        If p is not in (0, 1) or vig is not in [0, 1).
    """
    if not 0 < p < 1:
        raise ValueError(f"Probability must be in (0, 1), got {p}.")
    if not 0 <= vig < 1:
        raise ValueError(f"Vig must be in [0, 1), got {vig}.")
    return p + vig / 2


def prob_to_american_odds(p: float) -> float:
    """
    Convert a win probability to American odds.

    Uses exact asymmetric American odds formula:
        favourite (p >= 0.5): -(p / (1-p)) * 100
        underdog  (p <  0.5):  ((1-p) / p) * 100

    Parameters
    ----------
    p:
        Win probability in (0, 1).

    Returns
    -------
    float
        American odds (negative for favourite, positive for underdog).

    Raises
    ------
    ValueError
        If p is not in (0, 1).
    """
    if not 0 < p < 1:
        raise ValueError(f"Probability must be in (0, 1), got {p}.")
    if p >= 0.5:
        return -(p / (1 - p)) * 100   # favourite: negative
    else:
        return ((1 - p) / p) * 100    # underdog: positive


def prob_to_decimal_odds(p: float) -> float:
    """
    Convert a win probability to decimal odds.

    Decimal odds = 1 / p. A £1 bet returns £(decimal_odds) total.

    Parameters
    ----------
    p:
        Win probability in (0, 1).

    Returns
    -------
    float
        Decimal odds >= 1.0.

    Raises
    ------
    ValueError
        If p is not in (0, 1).
    """
    if not 0 < p < 1:
        raise ValueError(f"Probability must be in (0, 1), got {p}.")
    return 1.0 / p


def prob_to_fractional_odds(p: float) -> tuple[int, int]:
    """
    Convert a win probability to fractional odds (numerator, denominator).

    Fractional odds represent profit relative to stake: a win at 3/1 returns
    £3 profit on a £1 stake. Uses Python's Fraction with limit_denominator(100)
    for a clean, human-readable result.

    Parameters
    ----------
    p:
        Win probability in (0, 1).

    Returns
    -------
    tuple[int, int]
        (numerator, denominator) in simplest form, e.g. (3, 1) or (1, 2).

    Raises
    ------
    ValueError
        If p is not in (0, 1).
    """
    if not 0 < p < 1:
        raise ValueError(f"Probability must be in (0, 1), got {p}.")
    # (1-p)/p = profit / stake
    frac = Fraction((1 - p) / p).limit_denominator(100)
    return frac.numerator, frac.denominator


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

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
