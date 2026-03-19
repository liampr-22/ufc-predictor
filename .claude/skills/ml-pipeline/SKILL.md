---
name: ml-pipeline
description: Feature engineering, model training, inference, Elo rating updates, probability calibration, or anything touching the ml/ directory. Activate for tasks involving XGBoost/LightGBM training, adding new features, updating the Elo system, calibration math, or backtesting model sharpness against historical odds.
---

# ML Pipeline

## Feature Engineering

All features are **differentials**: `fighter_a_value - fighter_b_value`. Never use raw values as model inputs.

Feature categories:
- **Physical:** height delta, reach delta, age delta, stance matchup (orthodox vs southpaw encoded)
- **Striking:** sig strike accuracy delta, strikes absorbed delta, knockdown rate delta
- **Grappling:** takedown accuracy delta, takedown defense delta, submission attempts per fight delta
- **Finishing rate:** KO rate, submission rate, decision rate per fighter
- **Recency:** time since last fight, win streak, last-3-fight performance (weighted, not averaged)
- **Elo delta:** fighter_a_elo - fighter_b_elo from the fighters table
- **Style archetype:** wrestler/striker/grappler/brawler encoded from stat ratios at compute time

## Data Leakage Rule — Never Violate

Features must only use data available **before the fight date**. This is enforced via fight date alignment in `features.py`. When adding new features, always filter stats to `fight_date < current_fight_date`. Violating this produces inflated training accuracy that does not generalise.

## Elo System

- Ratings live in the `fighters` table and are updated incrementally after each result
- Do not recalculate Elo from scratch at prediction time — read from DB
- K-factor and expected score formula should remain consistent with `ml/elo.py` — do not change the formula without updating all historical ratings

## Model Training

- Target: binary win/loss for outcome model, multiclass for method of victory (KO_TKO / Submission / Decision)
- Calibration: probability outputs must be calibrated — use `ml/calibration.py`, do not return raw XGBoost probabilities to the API
- Serialise trained models to disk — `ml/train.py` handles this

## Odds Conversion

American odds have asymmetric logic:
- Favourite (negative): implied prob = |odds| / (|odds| + 100)
- Underdog (positive): implied prob = 100 / (odds + 100)
- Overround (vig) must be removed before converting model probabilities to odds
- Do not approximate or simplify this math — it lives in `ml/calibration.py`

## Backtesting

Historical odds are stored in `historical_odds` table. Backtesting compares model implied odds against closing lines. A sharp model should beat or match closing lines on average. Do not use opening lines for backtesting.

## Style Archetypes

Computed from stat ratios at feature engineering time, not stored in DB:
- **Wrestler:** high takedown accuracy + high takedown attempts
- **Grappler:** high submission attempts + high clinch control
- **Striker:** high sig strike accuracy + low takedown attempts
- **Brawler:** high strikes absorbed + high knockdown rate
Encode as categorical integers for model input.
