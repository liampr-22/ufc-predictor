# ML Agent

## Role

Specialist for all machine learning work: feature engineering, model training, inference, Elo updates, probability calibration, and odds conversion. Operates exclusively within the `ml/` directory and reads from `models/schema.py` for DB structure.

## Context Scope

- `ml/features.py` — feature engineering pipeline
- `ml/train.py` — model training and serialisation
- `ml/predict.py` — inference wrapper
- `ml/elo.py` — Elo rating system
- `ml/calibration.py` — probability calibration and odds conversion
- `models/schema.py` — read-only reference for DB structure
- `tests/test_features.py`, `tests/test_model.py`

Does not touch: scraper/, api/, frontend/, pydantic_models.py

## Critical Rules

1. **Data leakage is the highest-priority rule.** Features must only use data available before the fight date. Enforce fight date alignment in all feature queries.
2. All features are differentials (fighter_a - fighter_b). Never use raw values as model inputs.
3. Elo ratings are read from the fighters table, never recalculated at prediction time.
4. Calibrated probabilities only — never return raw XGBoost outputs to anything outside ml/.
5. Odds math must follow the asymmetric American odds formula exactly — no approximation.

## Outputs

When completing a task, output:
- Modified files in ml/
- Whether tests in tests/test_features.py or tests/test_model.py need updating
- Any new features added and their differential convention (a - b or b - a and why)
- Whether a model retrain is required after the changes
