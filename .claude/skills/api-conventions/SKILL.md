---
name: api-conventions
description: FastAPI endpoints, routers, Pydantic schemas, or anything touching the api/ directory. Activate for tasks involving adding routes, modifying request/response schemas, middleware, authentication, or the predict endpoint logic.
---

# API Conventions

## Framework

FastAPI with Pydantic v2. Use `pydantic_models.py` for all request/response schemas — never return raw SQLAlchemy objects from endpoints.

## Router Structure

Routers live in `api/routers/`. Current routers: fighters, predict, events, admin. Add new domain areas as new router files, not as routes on `main.py`.

## Endpoints Reference

| Method | Endpoint | Notes |
|---|---|---|
| GET | /health | Returns liveness + last scrape timestamp |
| GET | /fighters/{name} | Full profile, stats, Elo |
| GET | /fighters/{name}/history | Fight log |
| GET | /fighters/search?q={query} | Fuzzy search |
| POST | /predict | Core prediction endpoint |
| GET | /events/upcoming | Next card with predictions |
| POST | /admin/scrape | Authenticated — do not remove auth |

## Predict Endpoint

Input: `{ "fighter_a": str, "fighter_b": str }`

Output must include:
- `fighter_a_win_prob` and `fighter_b_win_prob` (sum to 1.0)
- `method_probs`: `{ KO_TKO, Submission, Decision }` (sum to 1.0)
- `odds`: American, decimal, and fractional for both fighters
- `key_differentials`: human-readable stat deltas with directional attribution

Do not change this response shape without updating the React frontend.

## Authentication

The `/admin/scrape` endpoint requires authentication. Do not remove or bypass auth middleware on admin routes.

## Error Handling

Use FastAPI's `HTTPException` with appropriate status codes. Do not let SQLAlchemy or ML errors surface as 500s without a meaningful message.

## Schema Consistency

Pydantic models in `models/pydantic_models.py` must mirror the SQLAlchemy schema in `models/schema.py`. When modifying one, always check the other. Mismatches cause silent serialisation bugs.

## Testing

API tests use Starlette TestClient in `tests/test_api.py`. New endpoints need test coverage. Run with `docker-compose exec api pytest tests/test_api.py -v`.
