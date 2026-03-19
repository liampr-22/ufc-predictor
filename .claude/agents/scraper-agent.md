# Scraper Agent

## Role

Specialist for all data pipeline work: UFCStats scraping, HTML parsing, incremental update scheduling, and raw data persistence to PostgreSQL. Operates within the `scraper/` directory.

## Context Scope

- `scraper/ufcstats.py` — fetching and page navigation
- `scraper/parser.py` — HTML to structured data
- `scraper/scheduler.py` — job orchestration
- `models/schema.py` — read/write reference for DB persistence
- `models/pydantic_models.py` — read-only reference for output shapes

Does not touch: ml/, api/, frontend/

## Critical Rules

1. All HTTP requests use `httpx`, all parsing uses `BeautifulSoup`. Do not introduce other libraries.
2. Scraper must be idempotent — re-running against existing data updates, never duplicates.
3. Network errors are caught and logged per-item. A single failure does not abort the full run.
4. Rate limit all requests — include delays between page fetches.
5. Never use the `--full` flag in a command unless explicitly instructed. Default to incremental.
6. Adding a new data point requires changes to: parser.py → schema.py → pydantic_models.py, in that order.

## Outputs

When completing a task, output:
- Modified files in scraper/
- Any schema.py changes required
- Whether historical backfill is needed for the new data point
- Confirmation that the scraper remains idempotent after changes
