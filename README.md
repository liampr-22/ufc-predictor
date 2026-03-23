# UFC Fight Predictor & Odds Engine

A machine learning system for UFC fight outcome prediction, style matchup analysis, and implied odds generation.

**Stack:** Python · FastAPI · PostgreSQL · LightGBM · BeautifulSoup · Docker · React

---

## Overview

UFC Fight Predictor is an end-to-end ML backend that scrapes and persists historical UFC fight data, engineers fighter-level features across 50+ attributes, trains a calibrated gradient boosted model to predict fight outcomes, and exposes a REST API for arbitrary matchup queries. A minimal React frontend allows fighter search and matchup prediction with a sportsbook-style odds display.

Given any two active UFC fighters, the system returns win probabilities, predicted method of victory, key stat differentials, and implied American, decimal, and fractional betting odds.

---

## Features

- Full historical UFC fight data scraped from UFCStats.com and persisted in PostgreSQL
- Fighter profiles with physical attributes, stance, reach, age, and career statistics
- Round-by-round fight breakdowns across striking, grappling, and clinch dimensions
- Style matchup feature engineering with differential stats between any two fighters
- Elo-style dynamic rating system updated after each fight result
- XGBoost/LightGBM model with calibrated win probability output
- Method of victory prediction — KO/TKO, Decision, Submission with per-class probabilities
- Implied odds generation in American, decimal, and fractional formats
- Historical odds backtesting against closing lines to validate model sharpness
- REST API with fighter search, matchup prediction, and fight history endpoints
- Background scraper job with incremental update support
- Minimal React frontend — fighter search inputs, prediction card, stat differentials
- Fully containerised via Docker Compose

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data scraping | BeautifulSoup, httpx | Scrape UFCStats.com fight & fighter data |
| Database | PostgreSQL + SQLAlchemy | Persist fighters, fights, rounds, odds |
| API | FastAPI + Pydantic | REST endpoints for prediction & fighter data |
| ML | XGBoost / LightGBM, scikit-learn | Fight outcome & method of victory models |
| Odds | Custom calibration layer | Convert probabilities to American/decimal/fractional |
| Frontend | React + Recharts | Fighter search, prediction card, stat viz |
| Infra | Docker Compose | Containerised local deployment |
| Testing | pytest + Starlette TestClient | API and model pipeline tests |

---

## Project Structure

```
ufc-predictor/
├── scraper/
│   ├── ufcstats.py        # UFCStats scraping logic
│   ├── scheduler.py       # Incremental update job
│   └── parser.py          # HTML to structured data
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
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   └── test_model.py
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Liveness check with last scrape timestamp |
| GET | /fighters/{name} | Full fighter profile, stats, and Elo rating |
| GET | /fighters/{name}/history | Fight log with outcomes and methods |
| GET | /fighters/search?q={query} | Fuzzy search across all fighters |
| POST | /predict | Matchup prediction — win probability, method, odds |
| GET | /events/upcoming | Next UFC card with predicted outcomes per fight |
| POST | /admin/scrape | Trigger incremental data scrape (authenticated) |

### Example `/predict` Request & Response

```json
POST /predict
{ "fighter_a": "Islam Makhachev", "fighter_b": "Dustin Poirier" }

{
  "fighter_a_win_prob": 0.71,
  "fighter_b_win_prob": 0.29,
  "method_probs": { "KO_TKO": 0.18, "Submission": 0.39, "Decision": 0.43 },
  "odds": {
    "fighter_a": { "american": -245, "decimal": 1.41, "fractional": "20/49" },
    "fighter_b": { "american": 210, "decimal": 3.10, "fractional": "21/10" }
  },
  "key_differentials": {
    "reach": "+2in (Makhachev)",
    "takedown_accuracy": "+31% (Makhachev)",
    "sig_strike_accuracy": "+8% (Poirier)"
  }
}
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Node.js 18+ (frontend only)
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ufc-predictor.git
   cd ufc-predictor
   ```

2. **Copy environment variables**
   ```bash
   cp .env.example .env
   ```

3. **Start services**
   ```bash
   docker-compose up --build
   ```

4. **Run initial data scrape** (15–30 min first time)
   ```bash
   docker-compose exec api python -m scraper.scheduler --full
   ```

5. **Train the model**
   ```bash
   docker-compose exec api python -m ml.train
   ```

6. API available at `http://localhost:8000`, frontend at `http://localhost:3000`
7. Interactive API docs at `http://localhost:8000/docs`

### Running Tests

```bash
docker-compose exec api pytest tests/ -v
```

---

## Data Model

Core tables:

- **fighters** — id, name, height, reach, stance, dob, weight_class, elo_rating
- **fights** — id, date, event, fighter_a_id, fighter_b_id, winner_id, method, round, time
- **fight_stats** — per-fighter-per-fight striking, grappling, and clinch statistics
- **round_stats** — round-by-round breakdown for each fight
- **historical_odds** — closing lines from external sources for backtesting

Fight date alignment ensures features are computed only from data available before each fight — no post-fight stats leak into training.

---

## Feature Engineering

All features are computed as differentials between the two fighters at prediction time:

- **Physical:** height delta, reach delta, age, orthodox vs southpaw matchup
- **Striking:** significant strike accuracy delta, strikes absorbed delta, knockdown rate delta
- **Grappling:** takedown accuracy delta, takedown defense delta, submission attempts per fight
- **Finishing rate:** KO rate, submission rate, decision rate per fighter
- **Recency:** time since last fight, win streak, last-3-fight performance weighting
- **Elo delta:** difference in current Elo rating between fighters
- **Style archetype:** wrestler, striker, grappler, brawler — encoded categorically from stat ratios

---

## Deployment

- Backend: Railway or Fly.io (Docker, free tier)
- Frontend: Vercel or Netlify
- CI: GitHub Actions — lint, test, build on every push to `main`
