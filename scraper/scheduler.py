"""
Incremental scrape job scheduler.

Usage:
    python -m scraper.scheduler           # incremental (events since last DB fight)
    python -m scraper.scheduler --full    # full historical scrape
    python -m scraper.scheduler --since 2024-01-01
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from typing import Optional

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from models.schema import Fight, FightStats, Fighter, RoundStats
from scraper.parser import (
    parse_event_index,
    parse_event_page,
    parse_fight_page,
    parse_fighter_index,
    parse_fighter_profile,
)
from scraper.ufcstats import UFCStatsScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_or_scrape_fighter(
    session: Session,
    scraper: UFCStatsScraper,
    ufcstats_id: Optional[str],
    name: Optional[str],
    fighter_url: Optional[str],
) -> Optional[Fighter]:
    """
    Return the Fighter row for the given ufcstats_id.
    If not in DB: scrape the profile page and insert.
    Falls back to name lookup if ufcstats_id is absent.
    """
    if ufcstats_id:
        fighter = session.scalar(select(Fighter).where(Fighter.ufcstats_id == ufcstats_id))
        if fighter:
            return fighter

    if name:
        fighter = session.scalar(select(Fighter).where(Fighter.name == name))
        if fighter:
            if ufcstats_id and not fighter.ufcstats_id:
                fighter.ufcstats_id = ufcstats_id
            return fighter

    # Not in DB — scrape profile
    if not fighter_url:
        logger.warning("No URL to scrape profile for %r (ufcstats_id=%r)", name, ufcstats_id)
        return None

    try:
        profile_html = scraper.fetch_fighter_profile(fighter_url)
        profile = parse_fighter_profile(profile_html, fighter_url)
    except httpx.HTTPError as exc:
        logger.error("Failed to fetch fighter profile %s: %s", fighter_url, exc)
        return None

    fighter = Fighter(
        name=profile.get("name") or name or "Unknown",
        height=profile.get("height"),
        reach=profile.get("reach"),
        stance=profile.get("stance"),
        dob=profile.get("dob"),
        ufcstats_id=profile.get("ufcstats_id") or ufcstats_id,
        elo_rating=1500.0,
    )
    session.add(fighter)
    session.flush()
    logger.info("  Inserted fighter: %s", fighter.name)
    return fighter


def _persist_fight(
    session: Session,
    scraper: UFCStatsScraper,
    fight_stub: dict,
) -> bool:
    """
    Fetch and persist a single fight + its stats.
    Returns True if inserted, False if skipped (already in DB or parse failure).
    """
    ufcstats_id = fight_stub.get("ufcstats_id")

    # Idempotency check
    if ufcstats_id:
        existing = session.scalar(select(Fight).where(Fight.ufcstats_id == ufcstats_id))
        if existing:
            return False

    # Ensure both fighters exist
    fa = _get_or_scrape_fighter(
        session, scraper,
        fight_stub.get("fighter_a_ufcstats_id"),
        fight_stub.get("fighter_a_name"),
        fight_stub.get("fighter_a_url"),
    )
    fb = _get_or_scrape_fighter(
        session, scraper,
        fight_stub.get("fighter_b_ufcstats_id"),
        fight_stub.get("fighter_b_name"),
        fight_stub.get("fighter_b_url"),
    )
    if not fa or not fb:
        logger.warning("Skipping fight %s — could not resolve fighters", ufcstats_id)
        return False

    # Fetch and parse fight detail page
    fight_url = fight_stub["url"]
    try:
        fight_html = scraper.fetch_fight_page(fight_url)
    except httpx.HTTPError as exc:
        logger.error("Failed to fetch fight %s: %s", fight_url, exc)
        return False

    fight_data = parse_fight_page(fight_html, fight_url)
    if not fight_data:
        logger.warning("Could not parse fight page: %s", fight_url)
        return False

    # Resolve winner
    winner_id: Optional[int] = None
    wid = fight_data.get("winner_ufcstats_id")
    if wid == fa.ufcstats_id:
        winner_id = fa.id
    elif wid == fb.ufcstats_id:
        winner_id = fb.id

    fight = Fight(
        date=fight_stub["event_date"],
        event=fight_stub["event_name"],
        fighter_a_id=fa.id,
        fighter_b_id=fb.id,
        winner_id=winner_id,
        method=fight_data.get("method"),
        round=fight_data.get("round"),
        time=fight_data.get("time"),
        ufcstats_id=ufcstats_id,
    )
    session.add(fight)
    session.flush()

    # Update fighters' weight class to their most recently seen one
    wc = fight_stub.get("weight_class")
    if wc:
        fa.weight_class = wc
        fb.weight_class = wc

    # Fight stats (one row per fighter per fight)
    def _make_fight_stats(fighter: Fighter, stats: dict) -> FightStats:
        return FightStats(
            fight_id=fight.id,
            fighter_id=fighter.id,
            knockdowns=stats.get("knockdowns"),
            significant_strikes_landed=stats.get("significant_strikes_landed"),
            significant_strikes_attempted=stats.get("significant_strikes_attempted"),
            total_strikes_landed=stats.get("total_strikes_landed"),
            total_strikes_attempted=stats.get("total_strikes_attempted"),
            head_strikes_landed=stats.get("head_strikes_landed"),
            head_strikes_attempted=stats.get("head_strikes_attempted"),
            body_strikes_landed=stats.get("body_strikes_landed"),
            body_strikes_attempted=stats.get("body_strikes_attempted"),
            leg_strikes_landed=stats.get("leg_strikes_landed"),
            leg_strikes_attempted=stats.get("leg_strikes_attempted"),
            distance_strikes_landed=stats.get("distance_strikes_landed"),
            distance_strikes_attempted=stats.get("distance_strikes_attempted"),
            clinch_strikes_landed=stats.get("clinch_strikes_landed"),
            clinch_strikes_attempted=stats.get("clinch_strikes_attempted"),
            ground_strikes_landed=stats.get("ground_strikes_landed"),
            ground_strikes_attempted=stats.get("ground_strikes_attempted"),
            takedowns_landed=stats.get("takedowns_landed"),
            takedowns_attempted=stats.get("takedowns_attempted"),
            submission_attempts=stats.get("submission_attempts"),
            reversals=stats.get("reversals"),
            control_time_seconds=stats.get("control_time_seconds"),
        )

    session.add(_make_fight_stats(fa, fight_data.get("stats_a", {})))
    session.add(_make_fight_stats(fb, fight_data.get("stats_b", {})))

    # Round stats
    for rs in fight_data.get("round_stats", []):
        rs_fid = rs.get("fighter_ufcstats_id")
        fighter = fa if rs_fid == fa.ufcstats_id else fb
        session.add(RoundStats(
            fight_id=fight.id,
            fighter_id=fighter.id,
            round_number=rs["round_number"],
            knockdowns=rs.get("knockdowns"),
            significant_strikes_landed=rs.get("significant_strikes_landed"),
            significant_strikes_attempted=rs.get("significant_strikes_attempted"),
            total_strikes_landed=rs.get("total_strikes_landed"),
            total_strikes_attempted=rs.get("total_strikes_attempted"),
            head_strikes_landed=rs.get("head_strikes_landed"),
            head_strikes_attempted=rs.get("head_strikes_attempted"),
            body_strikes_landed=rs.get("body_strikes_landed"),
            body_strikes_attempted=rs.get("body_strikes_attempted"),
            leg_strikes_landed=rs.get("leg_strikes_landed"),
            leg_strikes_attempted=rs.get("leg_strikes_attempted"),
            distance_strikes_landed=rs.get("distance_strikes_landed"),
            distance_strikes_attempted=rs.get("distance_strikes_attempted"),
            clinch_strikes_landed=rs.get("clinch_strikes_landed"),
            clinch_strikes_attempted=rs.get("clinch_strikes_attempted"),
            ground_strikes_landed=rs.get("ground_strikes_landed"),
            ground_strikes_attempted=rs.get("ground_strikes_attempted"),
            takedowns_landed=rs.get("takedowns_landed"),
            takedowns_attempted=rs.get("takedowns_attempted"),
            submission_attempts=rs.get("submission_attempts"),
            reversals=rs.get("reversals"),
            control_time_seconds=rs.get("control_time_seconds"),
        ))

    return True


# ── Scrape passes ─────────────────────────────────────────────────────────────

def scrape_all_fighters(session: Session, scraper: UFCStatsScraper) -> int:
    """
    Enumerate fighter index (A–Z) and upsert all fighter profiles into the DB.
    Returns count of new fighters inserted.
    """
    inserted = 0
    for letter, html in scraper.fighter_index_pages():
        stubs = parse_fighter_index(html)
        logger.info("Fighter index [%s]: %d entries", letter.upper(), len(stubs))
        for stub in stubs:
            uid = stub.get("ufcstats_id")
            if uid:
                existing = session.scalar(select(Fighter).where(Fighter.ufcstats_id == uid))
                if existing:
                    continue
            # On-demand profile fetch
            try:
                profile_html = scraper.fetch_fighter_profile(stub["url"])
                profile = parse_fighter_profile(profile_html, stub["url"])
            except httpx.HTTPError as exc:
                logger.error("Profile fetch failed for %r: %s", stub["name"], exc)
                continue

            fighter = Fighter(
                name=profile.get("name") or stub["name"],
                height=profile.get("height"),
                reach=profile.get("reach"),
                stance=profile.get("stance"),
                dob=profile.get("dob"),
                ufcstats_id=profile.get("ufcstats_id") or uid,
                elo_rating=1500.0,
            )
            session.add(fighter)
            try:
                session.flush()
                inserted += 1
            except Exception as exc:
                session.rollback()
                logger.error("Could not insert fighter %r: %s", stub["name"], exc)

    session.commit()
    logger.info("Fighter index scrape complete. Inserted %d new fighters.", inserted)
    return inserted


def scrape_events(
    session: Session,
    scraper: UFCStatsScraper,
    since: Optional[date] = None,
) -> tuple[int, int]:
    """
    Scrape all events (or events on/after `since`).
    For each event: scrape fight pages and persist results.

    Returns (events_processed, fights_inserted).
    """
    logger.info("Fetching event index…")
    event_index_html = scraper.fetch_event_index()
    events = parse_event_index(event_index_html)
    logger.info("Found %d completed events total.", len(events))

    if since:
        events = [e for e in events if e["date"] and e["date"] >= since]
        logger.info("Filtered to %d events on/after %s.", len(events), since)

    events_processed = 0
    fights_inserted = 0

    for event in events:
        event_name = event["name"]
        event_date = event["date"]
        event_url = event["url"]

        logger.info("Event: %s (%s)", event_name, event_date)

        try:
            event_html = scraper.fetch_event_page(event_url)
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch event %s: %s", event_url, exc)
            continue

        fight_stubs = parse_event_page(event_html, event_name, event_date)
        logger.info("  %d fights on card.", len(fight_stubs))

        for stub in fight_stubs:
            try:
                inserted = _persist_fight(session, scraper, stub)
                if inserted:
                    fights_inserted += 1
                    logger.info(
                        "  + %s vs %s",
                        stub.get("fighter_a_name", "?"),
                        stub.get("fighter_b_name", "?"),
                    )
            except Exception as exc:
                session.rollback()
                logger.error(
                    "  Error persisting fight %s vs %s: %s",
                    stub.get("fighter_a_name", "?"),
                    stub.get("fighter_b_name", "?"),
                    exc,
                )
                continue

        session.commit()
        events_processed += 1

    return events_processed, fights_inserted


# ── Entry point ───────────────────────────────────────────────────────────────

def _latest_fight_date(session: Session) -> Optional[date]:
    """Return the date of the most recent fight in the DB, or None."""
    result = session.scalar(select(func.max(Fight.date)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="UFC fight data scraper")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--full",
        action="store_true",
        help="Full historical scrape (15–30 min). Only use for initial setup.",
    )
    mode.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="Scrape events on or after this date.",
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set.")
        sys.exit(1)

    engine = create_engine(database_url)

    with UFCStatsScraper() as scraper:
        with Session(engine) as session:
            if args.full:
                logger.info("=== FULL SCRAPE ===")
                scrape_all_fighters(session, scraper)
                events_done, fights_done = scrape_events(session, scraper, since=None)

            elif args.since:
                try:
                    since_date = date.fromisoformat(args.since)
                except ValueError:
                    logger.error("Invalid date format for --since: %r (expected YYYY-MM-DD)", args.since)
                    sys.exit(1)
                logger.info("=== INCREMENTAL SCRAPE since %s ===", since_date)
                events_done, fights_done = scrape_events(session, scraper, since=since_date)

            else:
                # Default incremental: start from the day after the latest fight in DB
                latest = _latest_fight_date(session)
                since_date = (latest + timedelta(days=1)) if latest else date(1993, 11, 12)
                logger.info("=== INCREMENTAL SCRAPE since %s ===", since_date)
                events_done, fights_done = scrape_events(session, scraper, since=since_date)

    logger.info(
        "Done. Processed %d events, inserted %d fights.",
        events_done,
        fights_done,
    )


if __name__ == "__main__":
    main()
