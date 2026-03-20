"""
UFCStats.com scraping logic.
Handles HTTP fetching, rate limiting, and page enumeration.
Parsing is delegated to scraper/parser.py.
"""

import logging
import random
import string
import time
from typing import Iterator

import httpx

logger = logging.getLogger(__name__)

_BASE = "http://www.ufcstats.com"
_FIGHTER_INDEX = _BASE + "/statistics/fighters"
_EVENT_INDEX = _BASE + "/statistics/events/completed?page=all"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class UFCStatsScraper:
    """
    Thin HTTP wrapper around UFCStats.com with polite rate limiting.

    All methods return raw HTML strings. Callers (scheduler) are responsible
    for parsing via scraper.parser.
    """

    def __init__(self, delay_min: float = 0.8, delay_max: float = 2.0):
        self._client = httpx.Client(
            timeout=30.0,
            headers=_HEADERS,
            follow_redirects=True,
        )
        self._delay_min = delay_min
        self._delay_max = delay_max
        self._last_request_at: float = 0.0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        delay = random.uniform(self._delay_min, self._delay_max)
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def fetch(self, url: str) -> str:
        """
        Fetch a URL with rate limiting. Raises httpx.HTTPStatusError on 4xx/5xx.
        A failed fetch is logged and re-raised so the caller can catch it.
        """
        self._wait()
        self._last_request_at = time.monotonic()
        logger.debug("GET %s", url)
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.text

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "UFCStatsScraper":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Fighter index ─────────────────────────────────────────────────────────

    def fighter_index_pages(self) -> Iterator[tuple[str, str]]:
        """
        Yield (letter, html) for each letter a–z of the fighter index.
        Skips letters that return an HTTP error.
        """
        for letter in string.ascii_lowercase:
            url = f"{_FIGHTER_INDEX}?char={letter}&page=all"
            try:
                html = self.fetch(url)
                yield letter, html
            except httpx.HTTPError as exc:
                logger.error("Failed to fetch fighter index for %r: %s", letter, exc)

    def fetch_fighter_profile(self, url: str) -> str:
        """Fetch a single fighter profile page."""
        return self.fetch(url)

    # ── Event index ───────────────────────────────────────────────────────────

    def fetch_event_index(self) -> str:
        """Fetch the full completed-events listing page."""
        return self.fetch(_EVENT_INDEX)

    def fetch_event_page(self, url: str) -> str:
        """Fetch a single event detail page."""
        return self.fetch(url)

    # ── Fight detail ──────────────────────────────────────────────────────────

    def fetch_fight_page(self, url: str) -> str:
        """Fetch a single fight detail page."""
        return self.fetch(url)
