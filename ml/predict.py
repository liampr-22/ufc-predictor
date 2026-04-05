"""
Inference wrapper — loads trained models and returns predictions.
Implemented in Phase 5 (outcome model) and Phase 6 (method model).
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ml.features import FeatureBuilder
from ml.method_train import METHOD_LABEL_DEC, METHOD_LABEL_KO, METHOD_LABEL_SUB, METHOD_MODEL_FILENAME
from ml.train import FEATURE_COLS, MODEL_FILENAME

logger = logging.getLogger(__name__)


class Predictor:
    """
    Loads both serialised calibrated models once at init and exposes
    predict_proba() (outcome only) and predict() (outcome + method).

    Designed to be instantiated once at API startup and reused for all
    requests — joblib.load is not cheap.

    Parameters
    ----------
    model_dir:
        Directory containing xgb_model.joblib and method_model.joblib
        (default: "models/").

    Raises
    ------
    FileNotFoundError
        If either model file does not exist.
    """

    def __init__(self, model_dir: str = "models/") -> None:
        model_path = Path(model_dir) / MODEL_FILENAME
        method_model_path = Path(model_dir) / METHOD_MODEL_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(
                f"Outcome model not found at {model_path}. "
                f"Run `python -m ml.train` to train the model first."
            )
        if not method_model_path.exists():
            raise FileNotFoundError(
                f"Method model not found at {method_model_path}. "
                f"Run `python -m ml.train` to train the model first."
            )

        self._model = joblib.load(model_path)
        self._method_model = joblib.load(method_model_path)
        logger.info("Loaded outcome model from %s", model_path)
        logger.info("Loaded method model from %s", method_model_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_row(
        self,
        session: Session,
        fighter_a_id: int,
        fighter_b_id: int,
        as_of_date: date,
        scheduled_rounds: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        """Build a single-row feature matrix and return (row, feature_dict)."""
        builder = FeatureBuilder(session)
        fv = builder.build(fighter_a_id, fighter_b_id, as_of_date, scheduled_rounds=scheduled_rounds)

        feat_dict = fv.to_dict()
        for key, val in feat_dict.items():
            if isinstance(val, bool):
                feat_dict[key] = int(val)

        return pd.DataFrame([feat_dict])[FEATURE_COLS].values, feat_dict

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        session: Session,
        fighter_a_id: int,
        fighter_b_id: int,
        as_of_date: Optional[date] = None,
        scheduled_rounds: Optional[int] = None,
    ) -> float:
        """
        Return P(fighter_a wins) in [0.0, 1.0].

        Parameters
        ----------
        session:
            Active SQLAlchemy session (caller manages lifecycle).
        fighter_a_id, fighter_b_id:
            Fighter database IDs.
        as_of_date:
            Feature cut-off date. Defaults to today (live prediction mode).

        Returns
        -------
        float
            Calibrated win probability for fighter A.

        Raises
        ------
        ValueError
            If either fighter ID is not found (propagated from FeatureBuilder).
        """
        if as_of_date is None:
            as_of_date = date.today()

        row, _ = self._build_row(session, fighter_a_id, fighter_b_id, as_of_date, scheduled_rounds=scheduled_rounds)
        proba = float(self._model.predict_proba(row)[0, 1])
        return proba

    def predict(
        self,
        session: Session,
        fighter_a_id: int,
        fighter_b_id: int,
        as_of_date: Optional[date] = None,
        scheduled_rounds: Optional[int] = None,
    ) -> dict:
        """
        Return combined outcome and method-of-victory probabilities.

        Both models share the same feature vector, built once and passed
        to each model.

        Parameters
        ----------
        session:
            Active SQLAlchemy session.
        fighter_a_id, fighter_b_id:
            Fighter database IDs.
        as_of_date:
            Feature cut-off date. Defaults to today.

        Returns
        -------
        dict
            {
                "win_prob_a": float,           # P(fighter_a wins)
                "method_probs": {
                    "ko_tko":     float,       # P(fight ends by KO/TKO)
                    "submission": float,       # P(fight ends by submission)
                    "decision":   float,       # P(fight ends by decision)
                }
            }

        Raises
        ------
        ValueError
            If either fighter ID is not found.
        """
        if as_of_date is None:
            as_of_date = date.today()

        row, feat_dict = self._build_row(session, fighter_a_id, fighter_b_id, as_of_date, scheduled_rounds=scheduled_rounds)

        win_prob = float(self._model.predict_proba(row)[0, 1])
        method_proba = self._method_model.predict_proba(row)[0]  # shape [3]

        def _normalize(rates: dict) -> dict:
            total = sum(rates.values())
            if total <= 0:
                return {k: 1 / 3 for k in rates}
            return {k: v / total for k, v in rates.items()}

        def _conditional_win_methods(ko: float, sub: float, dec: float, win_rate: float) -> dict:
            denom = max(win_rate, 0.01)
            return _normalize({"ko_tko": ko / denom, "submission": sub / denom, "decision": dec / denom})

        win_rate_a = float(feat_dict.get("win_rate_a", 0.5))
        win_rate_b = float(feat_dict.get("win_rate_b", 0.5))

        return {
            "win_prob_a": win_prob,
            "method_probs": {
                "ko_tko":     float(method_proba[METHOD_LABEL_KO]),
                "submission": float(method_proba[METHOD_LABEL_SUB]),
                "decision":   float(method_proba[METHOD_LABEL_DEC]),
            },
            "fighter_a_finish_rates": {
                "ko_tko":     float(feat_dict.get("ko_rate_a", 0.0)),
                "submission": float(feat_dict.get("sub_rate_a", 0.0)),
                "decision":   float(feat_dict.get("dec_rate_a", 0.0)),
            },
            "fighter_b_finish_rates": {
                "ko_tko":     float(feat_dict.get("ko_rate_b", 0.0)),
                "submission": float(feat_dict.get("sub_rate_b", 0.0)),
                "decision":   float(feat_dict.get("dec_rate_b", 0.0)),
            },
            "fighter_a_win_method_rates": _conditional_win_methods(
                float(feat_dict.get("ko_rate_a", 0.0)),
                float(feat_dict.get("sub_rate_a", 0.0)),
                float(feat_dict.get("dec_rate_a", 0.0)),
                win_rate_a,
            ),
            "fighter_b_win_method_rates": _conditional_win_methods(
                float(feat_dict.get("ko_rate_b", 0.0)),
                float(feat_dict.get("sub_rate_b", 0.0)),
                float(feat_dict.get("dec_rate_b", 0.0)),
                win_rate_b,
            ),
        }
