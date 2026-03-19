---
name: scraper-patterns
description: UFCStats scraping, HTML parsing, scheduler logic, or anything touching the scraper/ directory. Activate for tasks involving ufcstats.py, parser.py, scheduler.py, incremental updates, or adding new data points from UFCStats.
---

# Scraper Patterns

## Source

All data comes from UFCStats.com. Do not introduce additional scraping sources without explicit instruction — the data model is built around UFCStats' structure.

## Libraries

- `httpx` for HTTP requests — not requests, not aiohttp
- `BeautifulSoup` for HTML parsing

## Incremental vs Full Scrape

- `--full` flag: scrapes entire UFCStats history. Takes 15–30 minutes. Only for initial setup or explicit resets.
- Default (no flag): incremental, only fetches fights/events since last scrape. Use this for scheduled updates.
- Last scrape timestamp is tracked and exposed via the `/health` endpoint.

## Scraper Structure

- `ufcstats.py` — raw HTTP fetching and page navigation logic
- `parser.py` — HTML to structured Python objects (dicts/dataclasses before DB write)
- `scheduler.py` — orchestrates full vs incremental runs, handles job scheduling

Keep these responsibilities separated. Do not mix parsing logic into the scheduler or fetching logic into the parser.

## Persistence

Parsed data writes directly to PostgreSQL via SQLAlchemy models in `models/schema.py`. There is no intermediate raw file storage — the DB is the source of truth.

## Rate Limiting

Be respectful to UFCStats — add delays between requests. Do not hammer the site. If adding new scraping logic, include a `time.sleep()` between page fetches.

## Error Handling

- Network errors should be caught and logged, not raised — a single failed page should not abort a full scrape run
- Log which fighter/event failed and continue
- Re-runnable: scraper should be idempotent — re-scraping an existing fight should update, not duplicate

## Adding New Data Points

When adding a new stat or attribute:
1. Add column to the appropriate SQLAlchemy model in `models/schema.py`
2. Add parsing logic in `parser.py`
3. Add the field to the relevant Pydantic schema in `models/pydantic_models.py`
4. Consider whether it needs to be backfilled for historical fights
