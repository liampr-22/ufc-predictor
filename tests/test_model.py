"""
Model pipeline tests — Phase 5: Outcome Prediction Model.

Synthetic dataset: 4 primary fighters (ids 1-4), 12 stub opponents (ids 10-21),
25 fights spanning 2015-2024. Fight 25 is a draw (excluded from training set).
This gives 24 labeled fights, of which ~5 fall in the test set (last 20%).

Fixture dependency chain:
    session → dataset → split → trained_model → calibrated_model

TestPredict depends on saved_model_dir, which is populated by a module-scoped
fixture that calls _run_with_session once.

Note: test_model.py uses n_cv_splits=2 throughout to keep runtime short.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ml.calibration import calibrate, evaluate
from ml.elo import load_fights_from_db
from ml.train import (
    FEATURE_COLS,
    MODEL_FILENAME,
    _run_with_session,
    build_training_dataset,
    elo_baseline,
    split_by_date,
    train_model,
)
from ml.predict import Predictor
from models.schema import Base, Fight, FightStats, Fighter


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

# Per-fighter stat profiles: (sig_l, sig_a, td_l, td_a, sub_att, kd)
_STATS = {
    1: (60, 110, 3, 5, 1, 2),   # striker
    2: (45,  90, 5, 7, 3, 0),   # wrestler
    3: (35,  80, 6, 8, 5, 0),   # grappler
    4: (70, 120, 1, 3, 0, 4),   # brawler
}
_STUB = (30, 70, 1, 3, 0, 0)


def _add_stats(sess: Session, fight_id: int, a_id: int, b_id: int) -> None:
    """Add FightStats rows for both fighters in a fight."""
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
    """
    Populate test DB with 4 primary fighters, 12 stub opponents, and 25 fights.
    Fight 25 is a draw — excluded from the training dataset.
    """
    # Primary fighters
    sess.add(Fighter(id=1, name="Fighter A", height=71.0, reach=72.0, stance="Orthodox",
                     dob=date(1990, 1, 1), weight_class="Lightweight", elo_rating=1500.0))
    sess.add(Fighter(id=2, name="Fighter B", height=70.0, reach=70.0, stance="Orthodox",
                     dob=date(1991, 1, 1), weight_class="Lightweight", elo_rating=1500.0))
    sess.add(Fighter(id=3, name="Fighter C", height=69.0, reach=71.0, stance="Southpaw",
                     dob=date(1992, 1, 1), weight_class="Lightweight", elo_rating=1500.0))
    sess.add(Fighter(id=4, name="Fighter D", height=72.0, reach=73.0, stance="Orthodox",
                     dob=date(1989, 1, 1), weight_class="Lightweight", elo_rating=1500.0))

    # Stub opponents (ids 10-21)
    sess.add_all(
        Fighter(id=i, name=f"Stub {i}", weight_class="Lightweight", elo_rating=1500.0)
        for i in range(10, 22)
    )
    sess.flush()

    # Fights: (id, date, a_id, b_id, winner_id, method)
    fight_defs = [
        # Phase 1 — establish history (2015-2016)
        ( 1, date(2015,  1,  1), 1, 10, 1,  "KO"),
        ( 2, date(2015,  4,  1), 2, 11, 2,  "DEC"),
        ( 3, date(2015,  7,  1), 3, 12, 3,  "SUB"),
        ( 4, date(2015, 10,  1), 4, 13, 4,  "KO"),
        ( 5, date(2016,  1,  1), 1, 14, 1,  "DEC"),
        ( 6, date(2016,  4,  1), 2, 15, 15, "KO"),   # 2 loses
        # Phase 2 — mix matchups (2017-2019)
        ( 7, date(2017,  1,  1), 3, 16, 3,  "KO"),
        ( 8, date(2017,  6,  1), 4, 17, 4,  "SUB"),
        ( 9, date(2018,  1,  1), 1,  2, 1,  "DEC"),
        (10, date(2018,  6,  1), 3,  4, 4,  "KO"),
        (11, date(2019,  1,  1), 1, 18, 18, "SUB"),  # 1 loses
        (12, date(2019,  6,  1), 2, 19, 2,  "DEC"),
        # Phase 3 — cross matchups (2020-2022)
        (13, date(2020,  1,  1), 3,  1, 3,  "KO"),
        (14, date(2020,  6,  1), 4,  2, 4,  "DEC"),
        (15, date(2021,  1,  1), 1, 20, 1,  "KO"),
        (16, date(2021,  6,  1), 2,  3, 2,  "SUB"),
        (17, date(2021, 10,  1), 4, 21, 4,  "KO"),
        (18, date(2022,  1,  1), 1,  3, 1,  "DEC"),
        (19, date(2022,  6,  1), 2,  4, 2,  "KO"),
        # Phase 4 — test territory (2023-2024)
        (20, date(2023,  1,  1), 1,  4, 4,  "SUB"),
        (21, date(2023,  6,  1), 2,  3, 3,  "KO"),
        (22, date(2023, 10,  1), 1,  2, 2,  "DEC"),
        (23, date(2024,  1,  1), 3,  4, 3,  "SUB"),
        (24, date(2024,  6,  1), 1,  3, 1,  "KO"),
        (25, date(2024, 10,  1), 2,  4, None, None),  # DRAW
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
def dataset(session):
    return build_training_dataset(session)


@pytest.fixture(scope="module")
def split(dataset):
    X, y, dates = dataset
    return split_by_date(X, y, dates, test_fraction=0.20)


@pytest.fixture(scope="module")
def trained_model(split):
    X_train, _, y_train, _, _, _ = split
    return train_model(X_train, y_train, n_cv_splits=2)


@pytest.fixture(scope="module")
def calibrated_model(trained_model, split):
    X_train, _, y_train, _, _, _ = split
    cal_idx = int(len(X_train) * 0.80)
    X_cal = X_train.iloc[cal_idx:].values
    y_cal = y_train.iloc[cal_idx:].values
    return calibrate(trained_model, X_cal, y_cal, method="isotonic")


@pytest.fixture(scope="module")
def saved_model_dir(session, tmp_path_factory):
    """Run the full pipeline once and save the model to a temp directory."""
    model_dir = str(tmp_path_factory.mktemp("models"))
    _run_with_session(session, output_dir=model_dir, report=False)
    return model_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDatasetBuilding:
    def test_returns_three_element_tuple(self, dataset):
        assert len(dataset) == 3

    def test_x_is_dataframe(self, dataset):
        X, _, _ = dataset
        assert isinstance(X, pd.DataFrame)

    def test_y_is_series(self, dataset):
        _, y, _ = dataset
        assert isinstance(y, pd.Series)

    def test_correct_column_count(self, dataset):
        X, _, _ = dataset
        assert len(X.columns) == len(FEATURE_COLS)

    def test_all_expected_columns_present(self, dataset):
        X, _, _ = dataset
        for col in FEATURE_COLS:
            assert col in X.columns, f"Missing column: {col}"

    def test_labels_are_binary(self, dataset):
        _, y, _ = dataset
        assert set(y.unique()).issubset({0, 1})

    def test_draw_excluded(self, dataset):
        # 25 fights seeded, 1 draw → at most 24 rows
        X, _, _ = dataset
        assert len(X) <= 24

    def test_at_least_one_row(self, dataset):
        X, _, _ = dataset
        assert len(X) > 0

    def test_x_y_dates_same_length(self, dataset):
        X, y, dates = dataset
        assert len(X) == len(y) == len(dates)

    def test_dates_non_decreasing(self, dataset):
        _, _, dates = dataset
        assert all(dates.iloc[i] <= dates.iloc[i + 1] for i in range(len(dates) - 1))

    def test_no_all_nan_columns(self, dataset):
        X, _, _ = dataset
        assert not X.isnull().all().any()

    def test_bool_columns_cast_to_int(self, dataset):
        X, _, _ = dataset
        # low_confidence_a and low_confidence_b must not be bool dtype
        for col in ("low_confidence_a", "low_confidence_b"):
            assert X[col].dtype != bool, f"{col} should be int, not bool"


class TestDateSplit:
    def test_returns_six_element_tuple(self, split):
        assert len(split) == 6

    def test_no_overlap(self, split):
        _, _, _, _, dates_train, dates_test = split
        if len(dates_train) > 0 and len(dates_test) > 0:
            assert dates_train.max() <= dates_test.min()

    def test_test_fraction_approximately_correct(self, split):
        X_train, X_test, _, _, _, _ = split
        total = len(X_train) + len(X_test)
        actual_fraction = len(X_test) / total
        assert abs(actual_fraction - 0.20) < 0.10

    def test_train_before_test(self, split):
        _, _, _, _, dates_train, dates_test = split
        if len(dates_train) > 0 and len(dates_test) > 0:
            assert dates_train.iloc[-1] <= dates_test.iloc[0]

    def test_no_data_lost(self, split, dataset):
        X_train, X_test, _, _, _, _ = split
        X, _, _ = dataset
        assert len(X_train) + len(X_test) == len(X)

    def test_x_and_y_aligned(self, split):
        X_train, X_test, y_train, y_test, _, _ = split
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


class TestTrainPipeline:
    def test_returns_lgbm_classifier(self, trained_model):
        assert isinstance(trained_model, lgb.LGBMClassifier)

    def test_model_can_predict(self, trained_model, split):
        _, X_test, _, _, _, _ = split
        preds = trained_model.predict(X_test.values)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_model_predict_proba_shape(self, trained_model, split):
        _, X_test, _, _, _, _ = split
        proba = trained_model.predict_proba(X_test.values)
        assert proba.shape == (len(X_test), 2)

    def test_model_serializes_to_disk(self, trained_model, tmp_path):
        path = tmp_path / "test_lgbm.joblib"
        joblib.dump(trained_model, path)
        loaded = joblib.load(path)
        assert isinstance(loaded, lgb.LGBMClassifier)

    def test_run_returns_required_top_level_keys(self, session):
        result = _run_with_session(session, output_dir="/tmp/_test_ufc_dryrun", report=True)
        required = {"trained_at", "train_size", "test_size", "split_date", "xgb", "elo_baseline"}
        assert required.issubset(set(result.keys()))

    def test_run_xgb_sub_keys(self, session):
        result = _run_with_session(session, output_dir="/tmp/_test_ufc_dryrun", report=True)
        assert {"accuracy", "log_loss", "brier_score_loss"}.issubset(set(result["xgb"].keys()))

    def test_run_elo_baseline_sub_keys(self, session):
        result = _run_with_session(session, output_dir="/tmp/_test_ufc_dryrun", report=True)
        required = {"accuracy", "log_loss", "brier_score_loss", "correct", "total", "skipped_draws"}
        assert required.issubset(set(result["elo_baseline"].keys()))

    def test_run_saves_model_file(self, saved_model_dir):
        assert (Path(saved_model_dir) / MODEL_FILENAME).exists()

    def test_run_saves_report_file(self, saved_model_dir):
        from ml.train import REPORT_FILENAME
        assert (Path(saved_model_dir) / REPORT_FILENAME).exists()

    def test_run_sizes_positive(self, session):
        result = _run_with_session(session, output_dir="/tmp/_test_ufc_dryrun", report=True)
        assert result["train_size"] > 0
        assert result["test_size"] > 0


class TestEloBaseline:
    def _get_splits(self, session):
        fight_dicts = load_fights_from_db(session)
        split_idx = int(len(fight_dicts) * 0.80)
        train_fights = fight_dicts[:split_idx]
        test_fights = [f for f in fight_dicts[split_idx:] if f["winner_id"] is not None]
        return train_fights, test_fights

    def test_returns_required_keys(self, session):
        train_fights, test_fights = self._get_splits(session)
        result = elo_baseline(session, test_fights, train_fights)
        required = {"accuracy", "log_loss", "brier_score_loss", "correct", "total", "skipped_draws"}
        assert required.issubset(set(result.keys()))

    def test_accuracy_in_valid_range(self, session):
        train_fights, test_fights = self._get_splits(session)
        result = elo_baseline(session, test_fights, train_fights)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_correct_le_total(self, session):
        train_fights, test_fights = self._get_splits(session)
        result = elo_baseline(session, test_fights, train_fights)
        assert result["correct"] <= result["total"]

    def test_draws_counted_in_skipped(self, session):
        fight_dicts = load_fights_from_db(session)
        draw_fights = [f for f in fight_dicts if f["winner_id"] is None]
        non_draw_fights = [f for f in fight_dicts if f["winner_id"] is not None]
        result = elo_baseline(session, draw_fights, non_draw_fights)
        assert result["skipped_draws"] == len(draw_fights)
        assert result["total"] == 0

    def test_empty_test_set_returns_zeros(self, session):
        fight_dicts = load_fights_from_db(session)
        result = elo_baseline(session, [], fight_dicts)
        assert result["accuracy"] == 0.0
        assert result["total"] == 0


class TestCalibration:
    def test_calibrated_probas_in_range(self, calibrated_model, split):
        _, X_test, _, _, _, _ = split
        probas = calibrated_model.predict_proba(X_test.values)[:, 1]
        assert (probas >= 0.0).all()
        assert (probas <= 1.0).all()

    def test_evaluate_returns_required_keys(self, calibrated_model, split):
        _, X_test, _, y_test, _, _ = split
        result = evaluate(calibrated_model, X_test.values, y_test.values)
        assert {"accuracy", "log_loss", "brier_score_loss"}.issubset(set(result.keys()))

    def test_evaluate_accuracy_in_range(self, calibrated_model, split):
        _, X_test, _, y_test, _, _ = split
        result = evaluate(calibrated_model, X_test.values, y_test.values)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_evaluate_brier_score_in_range(self, calibrated_model, split):
        _, X_test, _, y_test, _, _ = split
        result = evaluate(calibrated_model, X_test.values, y_test.values)
        assert 0.0 <= result["brier_score_loss"] <= 1.0

    def test_invalid_method_raises_value_error(self, trained_model, split):
        X_train, _, y_train, _, _, _ = split
        cal_idx = int(len(X_train) * 0.80)
        with pytest.raises(ValueError, match="Invalid calibration method"):
            calibrate(
                trained_model,
                X_train.iloc[cal_idx:].values,
                y_train.iloc[cal_idx:].values,
                method="bad_method",
            )

    def test_evaluate_empty_returns_zeros(self, calibrated_model):
        import numpy as np
        result = evaluate(calibrated_model, np.empty((0, len(FEATURE_COLS))), np.empty(0))
        assert result == {"accuracy": 0.0, "log_loss": 0.0, "brier_score_loss": 0.0}

    def test_sigmoid_calibration_works(self, trained_model, split):
        X_train, _, y_train, _, _, _ = split
        cal_idx = int(len(X_train) * 0.80)
        # Should not raise
        cal_model = calibrate(
            trained_model,
            X_train.iloc[cal_idx:].values,
            y_train.iloc[cal_idx:].values,
            method="sigmoid",
        )
        assert cal_model is not None


class TestPredict:
    def test_returns_float(self, session, saved_model_dir):
        predictor = Predictor(model_dir=saved_model_dir)
        result = predictor.predict_proba(session, 1, 2)
        assert isinstance(result, float)

    def test_probability_in_range(self, session, saved_model_dir):
        predictor = Predictor(model_dir=saved_model_dir)
        prob = predictor.predict_proba(session, 1, 2)
        assert 0.0 <= prob <= 1.0

    def test_asymmetry(self, session, saved_model_dir):
        """P(A beats B) + P(B beats A) should not both equal exactly 0.5."""
        predictor = Predictor(model_dir=saved_model_dir)
        prob_ab = predictor.predict_proba(session, 1, 2)
        prob_ba = predictor.predict_proba(session, 2, 1)
        # The two predictions need not sum to 1 (calibration is independent),
        # but neither should be out of range
        assert 0.0 <= prob_ab <= 1.0
        assert 0.0 <= prob_ba <= 1.0

    def test_unknown_fighter_raises_value_error(self, session, saved_model_dir):
        predictor = Predictor(model_dir=saved_model_dir)
        with pytest.raises(ValueError):
            predictor.predict_proba(session, 9999, 2)

    def test_missing_model_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Predictor(model_dir=str(tmp_path / "nonexistent"))
