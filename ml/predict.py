"""
Inference wrapper — loads trained model and returns predictions.
Implemented in Phase 5 — Outcome Prediction Model.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from ml.features import FeatureBuilder
from ml.train import FEATURE_COLS, MODEL_FILENAME

logger = logging.getLogger(__name__)


class Predictor:
    """
    Loads the serialised calibrated model once at init and exposes a single
    predict_proba() method.

    Designed to be instantiated once at API startup and reused for all
    requests — joblib.load is not cheap.

    Parameters
    ----------
    model_dir:
        Directory containing xgb_model.joblib (default: "models/").

    Raises
    ------
    FileNotFoundError
        If the model file does not exist. Fail-fast at startup so the issue
        is caught immediately rather than on the first prediction request.
    """

    def __init__(self, model_dir: str = "models/") -> None:
        model_path = Path(model_dir) / MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                f"Run `python -m ml.train` to train the model first."
            )
        self._model = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)

    def predict_proba(
        self,
        session: Session,
        fighter_a_id: int,
        fighter_b_id: int,
        as_of_date: Optional[date] = None,
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

        builder = FeatureBuilder(session)
        fv = builder.build(fighter_a_id, fighter_b_id, as_of_date)

        feat_dict = fv.to_dict()
        # Cast bool columns to int (low_confidence_a, low_confidence_b)
        for key, val in feat_dict.items():
            if isinstance(val, bool):
                feat_dict[key] = int(val)

        row = pd.DataFrame([feat_dict])[FEATURE_COLS]
        proba = self._model.predict_proba(row.values)[0, 1]
        return float(proba)
