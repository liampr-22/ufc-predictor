# UFC Predictor — Backtest Report

> **Note:** This report was generated from synthetic test data (in-memory SQLite DB seeded
> by `tests/test_odds.py`). When real historical closing odds are populated in the
> `historical_odds` table, regenerate this report by running:
>
> ```bash
> docker-compose exec api python -m ml.backtest --output backtest_report.md
> ```

---

## Summary

| Metric | Value |
|--------|-------|
| Fights backtested | 4 |
| Fights skipped (no features) | 0 |
| % beats closing line | 100.0% |
| Average CLV | +12.03% |
| Flat-stake P&L (units, stake = 1) | +2.05 |
| ROI | +51.4% |

---

## Interpretation

- **CLV > 0%** means the model's implied probability was sharper (higher) than the
  market's closing line for the fighter the model backed.
- **Average CLV > 0%** across all fights indicates the model consistently finds value
  relative to closing lines — the key signal that a model is sharp rather than lucky.
- **P&L** is computed by flat-betting 1 unit on every fight the model has a view on,
  collecting at fair closing odds (vig removed). Positive P&L with reasonable CLV
  validates the approach; P&L alone is noisy on small samples.
- The 100 % CLV beat rate here reflects synthetic data where the stub predictor
  consistently prices fighters higher than the seeded market odds — not a real-world claim.

---

## Odds Conversion — Worked Examples

The following examples use the functions in `ml/calibration.py` to show how
a model probability maps to each odds format.

### Fighter A: P(win) = 0.65

| Format | Value |
|--------|-------|
| American odds (favourite) | -185.7 |
| Decimal odds | 1.538 |
| Fractional odds | 7/13 |
| After 4% vig applied | 0.670 implied prob |

### Fighter B: P(win) = 0.35 (opponent)

| Format | Value |
|--------|-------|
| American odds (underdog) | +185.7 |
| Decimal odds | 2.857 |
| Fractional odds | 13/7 |
| After 4% vig applied | 0.370 implied prob |

Sum of vigged implied probabilities = 0.670 + 0.370 = 1.040 (4% overround ✓)

---

## Per-Fight Results (Synthetic Data)

| Fight ID | Date | Model Pick | Model Prob | Mkt Close | CLV | P&L | Beat CLV? |
|----------|------|------------|------------|-----------|-----|-----|-----------|
| 9 | 2021-01-01 | A | 65.0% | 60.9% | +4.1% | +0.64 | ✓ |
| 10 | 2021-06-01 | A | 60.0% | 56.9% | +3.1% | +0.76 | ✓ |
| 11 | 2022-01-01 | A | 65.0% | 41.4% | +23.6% | -1.00 | ✓ |
| 12 | 2022-06-01 | A | 55.0% | 37.7% | +17.3% | +1.65 | ✓ |

> Fight 11: model backed Fighter A despite the market favouring Fighter B (-160).
> The model was wrong (Fighter B won) — CLV was positive because the model's implied
> probability for Fighter A exceeded the market's, but the outcome was unfavourable.
> This illustrates that positive CLV does not guarantee a win on any single fight.

---

## CLV Methodology

For each fight with historical closing odds:

1. **Convert** market American odds → raw implied probabilities using the exact formula:
   - Favourite: `|odds| / (|odds| + 100)`
   - Underdog: `100 / (odds + 100)`

2. **Remove vig** by dividing each side by the sum:
   `fair_prob = raw_prob / (raw_prob_a + raw_prob_b)`

3. **Model backs** whichever fighter has `model_prob > 0.5`

4. **CLV** = `model_prob_for_pick − fair_closing_prob_for_pick`
   - Positive → model was sharper than the closing line on this fight

5. **P&L** = `(1 / fair_closing_prob_for_pick) − 1` if winner, else `−1`
   - Uses fair closing odds as the payout benchmark

---

*Generated from `tests/test_odds.py` synthetic dataset. Real backtest requires
the `historical_odds` table populated from a closing odds source (e.g. BestFightOdds).*
