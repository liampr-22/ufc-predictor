"""
Method of Victory model tests — Phase 6.

Synthetic dataset: same 4 primary fighters and 25-fight history as test_model.py.
Fight 25 is a draw with method=None → excluded from both datasets.

Fixture dependency chain:
    session → method_dataset → method_split → method_trained → method_calibrated
    saved_both_dir → TestPredictCombined (requires both outcome + method models)

Note: n_cv_splits=2 throughout to keep runtime short.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ml.method_train import (
    FEATURE_COLS,
    METHOD_MODEL_FILENAME,
    METHOD_NAMES,
    _method_label,
    _run_method_pipeline,
    build_method_dataset,
    calibrate_method_model,
    evaluate_method_model,
    train_method_model,
)
from ml.predict import Predictor
from ml.train import _run_with_session
from models.schema import Base, Fight, FightStats, Fighter


# ---------------------------------------------------------------------------
# Seed helpers (self-contained — same data as test_model.py)
# ---------------------------------------------------------------------------

_STATS = {
    1: (60, 110, 3, 5, 1, 2),
    2: (45,  90, 5, 7, 3, 0),
    3: (35,  80, 6, 8, 5, 0),
    4: (70, 120, 1, 3, 0, 4),
}
_STUB = (30, 70, 1, 3, 0, 0)


def _add_stats(sess: Session, fight_id: int, a_id: int, b_id: int) -> None:
    for fid, profile in ((a_id, _STATS.get(a_id, _STUB)), (b_id, _STATS.get(b_id, _STUB))):
        sl, sa, tl, ta, sub, kd = profile
        sess.add(FightStats(
            fight_id=fight_id, fighter_id=fid,
            significant_strikes_landed=sl,
            significant_strikes_attempted=sa,
            takedowns_landed=tl,
            takedowns_attempted=ta,
            submission_attempts=sub,
            knockdowns=kd,
        ))


def _seed(sess: Session) -> None:
    sess.add(Fighter(id=1, name="Fighter A", height=71.0, reach=72.0, stance="Orthodox",
                     dob=date(1990, 1, 1), weight_class="Lightweight", elo_rating=1500.0))
    sess.add(Fighter(id=2, name="Fighter B", height=70.0, reach=70.0, stance="Orthodox",
                     dob=date(1991, 1, 1), weight_class="Lightweight", elo_rating=1500.0))
    sess.add(Fighter(id=3, name="Fighter C", height=69.0, reach=71.0, stance="Southpaw",
                     dob=date(1992, 1, 1), weight_class="Lightweight", elo_rating=1500.0))
    sess.add(Fighter(id=4, name="Fighter D", height=72.0, reach=73.0, stance="Orthodox",
                     dob=date(1989, 1, 1), weight_class="Lightweight", elo_rating=1500.0))

    sess.add_all(
        Fighter(id=i, name=f"Stub {i}", weight_class="Lightweight", elo_rating=1500.0)
        for i in range(10, 22)
    )
    sess.flush()

    fight_defs = [
        ( 1, date(2015,  1,  1), 1, 10, 1,  "KO"),
        ( 2, date(2015,  4,  1), 2, 11, 2,  "DEC"),
        ( 3, date(2015,  7,  1), 3, 12, 3,  "SUB"),
        ( 4, date(2015, 10,  1), 4, 13, 4,  "KO"),
        ( 5, date(2016,  1,  1), 1, 14, 1,  "DEC"),
        ( 6, date(2016,  4,  1), 2, 15, 15, "KO"),
        ( 7, date(2017,  1,  1), 3, 16, 3,  "KO"),
        ( 8, date(2017,  6,  1), 4, 17, 4,  "SUB"),
        ( 9, date(2018,  1,  1), 1,  2, 1,  "DEC"),
        (10, date(2018,  6,  1), 3,  4, 4,  "KO"),
        (11, date(2019,  1,  1), 1, 18, 18, "SUB"),
        (12, date(2019,  6,  1), 2, 19, 2,  "DEC"),
        (13, date(2020,  1,  1), 3,  1, 3,  "KO"),
        (14, date(2020,  6,  1), 4,  2, 4,  "DEC"),
        (15, date(2021,  1,  1), 1, 20, 1,  "KO"),
        (16, date(2021,  6,  1), 2,  3, 2,  "SUB"),
        (17, date(2021, 10,  1), 4, 21, 4,  "KO"),
        (18, date(2022,  1,  1), 1,  3, 1,  "DEC"),
        (19, date(2022,  6,  1), 2,  4, 2,  "KO"),
        (20, date(2023,  1,  1), 1,  4, 4,  "SUB"),
        (21, date(2023,  6,  1), 2,  3, 3,  "KO"),
        (22, date(2023, 10,  1), 1,  2, 2,  "DEC"),
        (23, date(2024,  1,  1), 3,  4, 3,  "SUB"),
        (24, date(2024,  6,  1), 1,  3, 1,  "KO"),
        (25, date(2024, 10,  1), 2,  4, None, None),  # DRAW — no method
    ]

    for fid, fdate, a_id, b_id, winner_id, method in fight_defs:
        sess.add(Fight(id=fid, date=fdate, fighter_a_id=a_id, fighter_b_id=b_id,
                       winner_id=winner_id, method=method))
        sess.flush()
        _add_stats(sess, fid, a_id, b_id)

    sess.commit()


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        _seed(sess)
        yield sess


@pytest.fixture(scope="module")
def method_dataset(session):
    return build_method_dataset(session)


@pytest.fixture(scope="module")
def method_split(method_dataset):
    from ml.method_train import _split_by_date
    X, y, dates = method_dataset
    return _split_by_date(X, y, dates, test_fraction=0.20)


@pytest.fixture(scope="module")
def method_trained(method_split):
    X_train, _, y_train, _, _, _ = method_split
    return train_method_model(X_train, y_train, n_cv_splits=2)


@pytest.fixture(scope="module")
def method_calibrated(method_trained, method_split):
    X_train, _, y_train, _, _, _ = method_split
    cal_idx = int(len(X_train) * 0.80)
    return calibrate_method_model(
        method_trained,
        X_train.iloc[cal_idx:].values,
        y_train.iloc[cal_idx:].values,
        method="isotonic",
    )


@pytest.fixture(scope="module")
def saved_both_dir(session, tmp_path_factory):
    """Run the full pipeline (outcome + method) and save to a temp dir."""
    model_dir = str(tmp_path_factory.mktemp("models"))
    _run_with_session(session, output_dir=model_dir, report=False)
    return model_dir


# ---------------------------------------------------------------------------
# Tests: method label helper
# ---------------------------------------------------------------------------

class TestMethodLabel:
    def test_ko_variants(self):
        assert _method_label("KO") == 0
        assert _method_label("TKO") == 0
        assert _method_label("KO/TKO") == 0
        assert _method_label("ko") == 0        # case-insensitive

    def test_submission_variants(self):
        assert _method_label("SUB") == 1
        assert _method_label("SUBMISSION") == 1
        assert _method_label("sub") == 1

    def test_decision_variants(self):
        assert _method_label("DEC") == 2
        assert _method_label("U-DEC") == 2
        assert _method_label("S-DEC") == 2
        assert _method_label("M-DEC") == 2

    def test_none_returns_none(self):
        assert _method_label(None) is None

    def test_unknown_returns_none(self):
        assert _method_label("OVERTURNED") is None
        assert _method_label("") is None


# ---------------------------------------------------------------------------
# Tests: dataset building
# ---------------------------------------------------------------------------

class TestMethodDataset:
    def test_returns_three_element_tuple(self, method_dataset):
        assert len(method_dataset) == 3

    def test_x_is_dataframe(self, method_dataset):
        X, _, _ = method_dataset
        assert isinstance(X, pd.DataFrame)

    def test_y_is_series(self, method_dataset):
        _, y, _ = method_dataset
        assert isinstance(y, pd.Series)

    def test_correct_column_count(self, method_dataset):
        X, _, _ = method_dataset
        assert len(X.columns) == len(FEATURE_COLS)

    def test_all_expected_columns_present(self, method_dataset):
        X, _, _ = method_dataset
        for col in FEATURE_COLS:
            assert col in X.columns, f"Missing column: {col}"

    def test_labels_in_valid_range(self, method_dataset):
        _, y, _ = method_dataset
        assert set(y.unique()).issubset({0, 1, 2})

    def test_all_three_classes_represented(self, method_dataset):
        # Seed data has KO, SUB, and DEC fights
        _, y, _ = method_dataset
        assert {0, 1, 2}.issubset(set(y.unique()))

    def test_no_method_fight_excluded(self, method_dataset):
        # Fight 25 has method=None — should be excluded
        X, _, _ = method_dataset
        assert len(X) <= 24

    def test_at_least_one_row(self, method_dataset):
        X, _, _ = method_dataset
        assert len(X) > 0

    def test_x_y_dates_same_length(self, method_dataset):
        X, y, dates = method_dataset
        assert len(X) == len(y) == len(dates)

    def test_dates_non_decreasing(self, method_dataset):
        _, _, dates = method_dataset
        assert all(dates.iloc[i] <= dates.iloc[i + 1] for i in range(len(dates) - 1))

    def test_bool_columns_cast_to_int(self, method_dataset):
        X, _, _ = method_dataset
        for col in ("low_confidence_a", "low_confidence_b"):
            assert X[col].dtype != bool, f"{col} should be int, not bool"

    def test_no_all_nan_columns(self, method_dataset):
        X, _, _ = method_dataset
        assert not X.isnull().all().any()


# ---------------------------------------------------------------------------
# Tests: model training
# ---------------------------------------------------------------------------

class TestMethodTraining:
    def test_returns_lgbm_classifier(self, method_trained):
        assert isinstance(method_trained, lgb.LGBMClassifier)

    def test_predict_proba_shape(self, method_trained, method_split):
        _, X_test, _, _, _, _ = method_split
        proba = method_trained.predict_proba(X_test.values)
        assert proba.shape == (len(X_test), 3)

    def test_predict_proba_rows_sum_to_one(self, method_trained, method_split):
        _, X_test, _, _, _, _ = method_split
        proba = method_trained.predict_proba(X_test.values)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_predictions_in_valid_class_range(self, method_trained, method_split):
        _, X_test, _, _, _, _ = method_split
        preds = method_trained.predict(X_test.values)
        assert set(preds).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# Tests: calibration
# ---------------------------------------------------------------------------

class TestMethodCalibration:
    def test_returns_calibrated_classifier(self, method_calibrated):
        assert isinstance(method_calibrated, CalibratedClassifierCV)

    def test_probas_in_range(self, method_calibrated, method_split):
        _, X_test, _, _, _, _ = method_split
        proba = method_calibrated.predict_proba(X_test.values)
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()

    def test_probas_sum_to_one(self, method_calibrated, method_split):
        _, X_test, _, _, _, _ = method_split
        proba = method_calibrated.predict_proba(X_test.values)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_sigmoid_calibration_works(self, method_trained, method_split):
        X_train, _, y_train, _, _, _ = method_split
        cal_idx = int(len(X_train) * 0.80)
        cal = calibrate_method_model(
            method_trained,
            X_train.iloc[cal_idx:].values,
            y_train.iloc[cal_idx:].values,
            method="sigmoid",
        )
        assert cal is not None


# ---------------------------------------------------------------------------
# Tests: evaluation
# ---------------------------------------------------------------------------

class TestMethodEvaluate:
    def test_returns_required_keys(self, method_calibrated, method_split):
        _, X_test, _, y_test, _, _ = method_split
        result = evaluate_method_model(method_calibrated, X_test.values, y_test.values)
        assert {"log_loss", "accuracy", "per_class"}.issubset(set(result.keys()))

    def test_per_class_has_all_methods(self, method_calibrated, method_split):
        _, X_test, _, y_test, _, _ = method_split
        result = evaluate_method_model(method_calibrated, X_test.values, y_test.values)
        assert set(result["per_class"].keys()) == set(METHOD_NAMES.values())

    def test_per_class_metrics_structure(self, method_calibrated, method_split):
        _, X_test, _, y_test, _, _ = method_split
        result = evaluate_method_model(method_calibrated, X_test.values, y_test.values)
        for class_name in METHOD_NAMES.values():
            assert {"precision", "recall", "f1"}.issubset(set(result["per_class"][class_name]))

    def test_accuracy_in_valid_range(self, method_calibrated, method_split):
        _, X_test, _, y_test, _, _ = method_split
        result = evaluate_method_model(method_calibrated, X_test.values, y_test.values)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_log_loss_positive(self, method_calibrated, method_split):
        _, X_test, _, y_test, _, _ = method_split
        result = evaluate_method_model(method_calibrated, X_test.values, y_test.values)
        assert result["log_loss"] >= 0.0

    def test_precision_recall_in_range(self, method_calibrated, method_split):
        _, X_test, _, y_test, _, _ = method_split
        result = evaluate_method_model(method_calibrated, X_test.values, y_test.values)
        for cls in result["per_class"].values():
            assert 0.0 <= cls["precision"] <= 1.0
            assert 0.0 <= cls["recall"] <= 1.0
            assert 0.0 <= cls["f1"] <= 1.0

    def test_empty_dataset_returns_zeros(self, method_calibrated):
        result = evaluate_method_model(
            method_calibrated,
            np.empty((0, len(FEATURE_COLS))),
            np.empty(0),
        )
        assert result["log_loss"] == 0.0
        assert result["accuracy"] == 0.0
        for cls in result["per_class"].values():
            assert cls == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ---------------------------------------------------------------------------
# Tests: full method pipeline
# ---------------------------------------------------------------------------

class TestRunMethodPipeline:
    def test_returns_required_keys(self, session):
        result = _run_method_pipeline(session, output_dir="/tmp/_test_ufc_method", report=True)
        assert {"trained_at", "train_size", "test_size", "method_model"}.issubset(
            set(result.keys())
        )

    def test_method_model_sub_keys(self, session):
        result = _run_method_pipeline(session, output_dir="/tmp/_test_ufc_method", report=True)
        assert {"log_loss", "accuracy", "per_class"}.issubset(set(result["method_model"].keys()))

    def test_sizes_positive(self, session):
        result = _run_method_pipeline(session, output_dir="/tmp/_test_ufc_method", report=True)
        assert result["train_size"] > 0
        assert result["test_size"] > 0

    def test_dry_run_does_not_write_file(self, session, tmp_path):
        _run_method_pipeline(session, output_dir=str(tmp_path), report=True)
        assert not (tmp_path / METHOD_MODEL_FILENAME).exists()

    def test_saves_model_file_when_not_dry_run(self, saved_both_dir):
        assert (Path(saved_both_dir) / METHOD_MODEL_FILENAME).exists()

    def test_train_also_produces_method_model(self, session):
        """_run_with_session report dict should include method_model key."""
        result = _run_with_session(session, output_dir="/tmp/_test_ufc_both", report=True)
        assert "method_model" in result
        assert {"log_loss", "accuracy", "per_class"}.issubset(set(result["method_model"].keys()))


# ---------------------------------------------------------------------------
# Tests: Predictor.predict() combined inference
# ---------------------------------------------------------------------------

class TestPredictCombined:
    def test_predict_returns_dict(self, session, saved_both_dir):
        predictor = Predictor(model_dir=saved_both_dir)
        result = predictor.predict(session, 1, 2)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, session, saved_both_dir):
        predictor = Predictor(model_dir=saved_both_dir)
        result = predictor.predict(session, 1, 2)
        assert "win_prob_a" in result
        assert "method_probs" in result

    def test_win_prob_in_range(self, session, saved_both_dir):
        predictor = Predictor(model_dir=saved_both_dir)
        result = predictor.predict(session, 1, 2)
        assert 0.0 <= result["win_prob_a"] <= 1.0

    def test_method_probs_keys(self, session, saved_both_dir):
        predictor = Predictor(model_dir=saved_both_dir)
        result = predictor.predict(session, 1, 2)
        assert set(result["method_probs"].keys()) == {"ko_tko", "submission", "decision"}

    def test_method_probs_sum_to_one(self, session, saved_both_dir):
        predictor = Predictor(model_dir=saved_both_dir)
        result = predictor.predict(session, 1, 2)
        total = sum(result["method_probs"].values())
        assert abs(total - 1.0) < 1e-5

    def test_method_probs_in_range(self, session, saved_both_dir):
        predictor = Predictor(model_dir=saved_both_dir)
        result = predictor.predict(session, 1, 2)
        for prob in result["method_probs"].values():
            assert 0.0 <= prob <= 1.0

    def test_predict_proba_still_works(self, session, saved_both_dir):
        """Backwards-compatible predict_proba() should still return a float."""
        predictor = Predictor(model_dir=saved_both_dir)
        prob = predictor.predict_proba(session, 1, 2)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_unknown_fighter_raises_value_error(self, session, saved_both_dir):
        predictor = Predictor(model_dir=saved_both_dir)
        with pytest.raises(ValueError):
            predictor.predict(session, 9999, 2)

    def test_missing_method_model_raises_file_not_found(self, tmp_path, saved_both_dir):
        """If only the outcome model is present, init should fail fast."""
        import shutil
        # Copy outcome model only
        shutil.copy(
            Path(saved_both_dir) / "xgb_model.joblib",
            tmp_path / "xgb_model.joblib",
        )
        with pytest.raises(FileNotFoundError):
            Predictor(model_dir=str(tmp_path))
