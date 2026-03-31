# UFC Fight Predictor & Odds Engine

End-to-end ML system for UFC fight outcome prediction, style matchup analysis, and implied odds generation.

## Stack

| Layer | Technology |
|---|---|
| Scraping | BeautifulSoup, httpx |
| Database | PostgreSQL + SQLAlchemy |
| API | FastAPI + Pydantic |
| ML | XGBoost / LightGBM, scikit-learn |
| Odds | Custom calibration layer |
| Frontend | React + Recharts |
| Infra | Docker Compose |
| Testing | pytest + Starlette TestClient |

## Commands

```bash
# Start all services
docker-compose up --build

# Full initial scrape (4–5 hrs — fetches all fighter profiles then all fight pages)
docker-compose exec api python -m scraper.scheduler --full

# Incremental scrape
docker-compose exec api python -m scraper.scheduler

# Train model
docker-compose exec api python -m ml.train

# Run tests
docker-compose exec api pytest tests/ -v
```

Services: API at http://localhost:8000 · Frontend at http://localhost:3000 · Docs at http://localhost:8000/docs

## Project Structure

```
ufc-predictor/
├── scraper/
│   ├── ufcstats.py        # UFCStats scraping logic
│   ├── scheduler.py       # Incremental update job
│   └── parser.py          # HTML → structured data
├── models/
│   ├── schema.py          # SQLAlchemy ORM models
│   └── pydantic_models.py # Request/response schemas
├── ml/
│   ├── features.py        # Feature engineering pipeline
│   ├── train.py           # Model training and serialisation
│   ├── predict.py         # Inference wrapper
│   ├── elo.py             # Elo rating system
│   └── calibration.py     # Probability calibration and odds
├── api/
│   ├── main.py            # FastAPI app
│   └── routers/           # fighters, predict, events, admin
├── frontend/
│   └── src/               # React app
└── tests/
```

## Architecture

- Scraper hits UFCStats.com → parses HTML → persists to PostgreSQL
- Feature engineering computes differentials at prediction time from persisted stats
- XGBoost model trained on differential features, calibrated for probability output
- FastAPI exposes prediction and fighter data endpoints
- React frontend consumes the API for matchup search and odds display

## Critical Rules

**Data leakage — most important rule in this codebase:**
Features must only use data available *before* the fight date. Never use post-fight stats in training. Fight date alignment is enforced in `ml/features.py` — do not break this.

**Feature engineering convention:**
All features are computed as differentials: `fighter_a_value - fighter_b_value`. Maintain this convention consistently when adding new features.

**Odds math:**
American odds have asymmetric sign logic (negative = favourite, positive = underdog). Implied probability = 100 / (odds + 100) for positives, |odds| / (|odds| + 100) for negatives. Do not simplify or approximate.

**Docker is the assumed runtime:**
All commands run inside Docker Compose. Never assume a local Python environment.

**Schema consistency:**
SQLAlchemy models in `models/schema.py` and Pydantic schemas in `models/pydantic_models.py` must stay in sync. When modifying one, always update the other.

**Never trigger a full scrape casually:**
`--full` flag scrapes the entire UFCStats history and takes 15–30 minutes. Only use for initial setup or explicit data resets.

## Data Model

- **fighters** — id, name, height, reach, stance, dob, weight_class, elo_rating
- **fights** — id, date, event, fighter_a_id, fighter_b_id, winner_id, method, round, time
- **fight_stats** — per-fighter-per-fight striking, grappling, clinch stats
- **round_stats** — round-by-round breakdown
- **historical_odds** — closing lines for backtesting

## Gotchas

- Elo ratings are updated after each fight result — do not recalculate from scratch on every prediction, read from the fighters table
- Style archetypes (wrestler/striker/grappler/brawler) are derived from stat ratios, not stored — computed at feature engineering time
- The `/admin/scrape` endpoint is authenticated — do not remove auth middleware
- Recency features use last-3-fight weighting, not a simple average
