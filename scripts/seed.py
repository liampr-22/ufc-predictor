"""
Seed script — populates the database with hand-crafted rows to validate the
schema before scraping begins. Idempotent: safe to run multiple times.

Usage (inside Docker):
    docker-compose exec api python -m scripts.seed
"""

import os
from datetime import date, datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models.schema import Base, Fight, FightStats, Fighter, HistoricalOdds, RoundStats

DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL)


def seed_fighters(session: Session) -> dict[str, Fighter]:
    """Insert fighters if they don't already exist. Returns name → Fighter map."""
    rows = [
        # (name, height_in, reach_in, stance, dob, weight_class, elo_rating)
        ("Jon Jones",              76.0, 84.5, "Orthodox",  date(1987,  7, 19), "Light Heavyweight", 1800.0),
        ("Daniel Cormier",         69.0, 73.0, "Orthodox",  date(1979,  3, 20), "Light Heavyweight", 1720.0),
        ("Alexander Gustafsson",   76.0, 79.0, "Orthodox",  date(1987,  1, 15), "Light Heavyweight", 1680.0),
        ("Conor McGregor",         69.0, 74.0, "Southpaw",  date(1988,  7, 14), "Featherweight",     1750.0),
        ("Khabib Nurmagomedov",    70.0, 70.0, "Orthodox",  date(1988,  9, 20), "Lightweight",       1790.0),
        ("Stipe Miocic",           76.0, 79.0, "Orthodox",  date(1982,  8, 19), "Heavyweight",       1720.0),
        ("Francis Ngannou",        76.0, 83.0, "Orthodox",  date(1986,  9,  5), "Heavyweight",       1700.0),
        ("Amanda Nunes",           68.0, 69.0, "Orthodox",  date(1988,  5, 30), "Bantamweight",      1780.0),
        ("Israel Adesanya",        76.0, 80.0, "Orthodox",  date(1989,  7, 22), "Middleweight",      1760.0),
        ("Robert Whittaker",       72.0, 73.5, "Orthodox",  date(1990, 10, 20), "Middleweight",      1700.0),
    ]

    fighters: dict[str, Fighter] = {}
    for name, height, reach, stance, dob, wc, elo in rows:
        existing = session.scalar(select(Fighter).where(Fighter.name == name))
        if existing:
            fighters[name] = existing
            continue
        f = Fighter(
            name=name, height=height, reach=reach, stance=stance,
            dob=dob, weight_class=wc, elo_rating=elo,
        )
        session.add(f)
        session.flush()
        fighters[name] = f

    return fighters


def seed_fights(session: Session, fighters: dict[str, Fighter]) -> list[Fight]:
    """Insert fights if they don't already exist. Returns list of Fight objects."""
    rows = [
        # (event, date, fighter_a, fighter_b, winner, method, round, time)
        ("UFC 182",  date(2015,  1,  3), "Jon Jones",           "Daniel Cormier",        "Jon Jones",           "DEC",    5, "5:00"),
        ("UFC 165",  date(2013,  9, 21), "Jon Jones",           "Alexander Gustafsson",  "Jon Jones",           "DEC",    5, "5:00"),
        ("UFC 229",  date(2018, 10,  6), "Khabib Nurmagomedov", "Conor McGregor",         "Khabib Nurmagomedov", "SUB",    4, "3:03"),
        ("UFC 220",  date(2018,  1, 20), "Stipe Miocic",        "Francis Ngannou",        "Stipe Miocic",        "DEC",    5, "5:00"),
        ("UFC 243",  date(2019, 10,  5), "Israel Adesanya",     "Robert Whittaker",       "Israel Adesanya",     "KO/TKO", 2, "3:33"),
    ]

    fights: list[Fight] = []
    for event, fight_date, fa_name, fb_name, winner_name, method, rnd, t in rows:
        fa = fighters[fa_name]
        fb = fighters[fb_name]
        winner = fighters[winner_name]
        existing = session.scalar(
            select(Fight).where(Fight.event == event, Fight.fighter_a_id == fa.id)
        )
        if existing:
            fights.append(existing)
            continue
        fight = Fight(
            date=fight_date, event=event,
            fighter_a_id=fa.id, fighter_b_id=fb.id,
            winner_id=winner.id, method=method, round=rnd, time=t,
        )
        session.add(fight)
        session.flush()
        fights.append(fight)

    return fights


def seed_fight_stats(session: Session, fights: list[Fight], fighters: dict[str, Fighter]) -> None:
    """Seed per-fight stats for UFC 182 (Jones vs DC)."""
    fight = fights[0]  # UFC 182
    jones = fighters["Jon Jones"]
    dc = fighters["Daniel Cormier"]

    existing = session.scalar(
        select(FightStats).where(
            FightStats.fight_id == fight.id, FightStats.fighter_id == jones.id
        )
    )
    if existing:
        return

    jones_stats = FightStats(
        fight_id=fight.id, fighter_id=jones.id,
        knockdowns=1,
        significant_strikes_landed=128, significant_strikes_attempted=212,
        total_strikes_landed=209,       total_strikes_attempted=339,
        head_strikes_landed=73,         head_strikes_attempted=140,
        body_strikes_landed=30,         body_strikes_attempted=40,
        leg_strikes_landed=25,          leg_strikes_attempted=32,
        distance_strikes_landed=101,    distance_strikes_attempted=178,
        clinch_strikes_landed=12,       clinch_strikes_attempted=18,
        ground_strikes_landed=15,       ground_strikes_attempted=16,
        takedowns_landed=2,             takedowns_attempted=5,
        submission_attempts=1, reversals=0, control_time_seconds=302,
    )
    dc_stats = FightStats(
        fight_id=fight.id, fighter_id=dc.id,
        knockdowns=0,
        significant_strikes_landed=83,  significant_strikes_attempted=155,
        total_strikes_landed=132,       total_strikes_attempted=235,
        head_strikes_landed=50,         head_strikes_attempted=100,
        body_strikes_landed=20,         body_strikes_attempted=30,
        leg_strikes_landed=13,          leg_strikes_attempted=25,
        distance_strikes_landed=60,     distance_strikes_attempted=118,
        clinch_strikes_landed=10,       clinch_strikes_attempted=14,
        ground_strikes_landed=13,       ground_strikes_attempted=23,
        takedowns_landed=3,             takedowns_attempted=9,
        submission_attempts=0, reversals=0, control_time_seconds=450,
    )
    session.add_all([jones_stats, dc_stats])


def seed_round_stats(session: Session, fights: list[Fight], fighters: dict[str, Fighter]) -> None:
    """Seed round 1 stats for UFC 182 (Jones vs DC)."""
    fight = fights[0]  # UFC 182
    jones = fighters["Jon Jones"]

    existing = session.scalar(
        select(RoundStats).where(
            RoundStats.fight_id == fight.id,
            RoundStats.fighter_id == jones.id,
            RoundStats.round_number == 1,
        )
    )
    if existing:
        return

    from models.schema import RoundStats as RS  # noqa: re-import just for clarity
    jones_r1 = RoundStats(
        fight_id=fight.id, fighter_id=jones.id, round_number=1,
        knockdowns=0,
        significant_strikes_landed=22, significant_strikes_attempted=38,
        total_strikes_landed=35,       total_strikes_attempted=58,
        head_strikes_landed=14,        head_strikes_attempted=26,
        body_strikes_landed=5,         body_strikes_attempted=7,
        leg_strikes_landed=3,          leg_strikes_attempted=5,
        distance_strikes_landed=18,    distance_strikes_attempted=32,
        clinch_strikes_landed=2,       clinch_strikes_attempted=3,
        ground_strikes_landed=2,       ground_strikes_attempted=3,
        takedowns_landed=0, takedowns_attempted=1,
        submission_attempts=0, reversals=0, control_time_seconds=42,
    )
    session.add(jones_r1)


def seed_historical_odds(session: Session, fights: list[Fight]) -> None:
    """Seed closing line odds for UFC 182."""
    fight = fights[0]  # UFC 182

    existing = session.scalar(
        select(HistoricalOdds).where(HistoricalOdds.fight_id == fight.id)
    )
    if existing:
        return

    odds = HistoricalOdds(
        fight_id=fight.id,
        fighter_a_odds=-550,   # Jones was a heavy favourite
        fighter_b_odds=400,
        source="BestFightOdds",
        recorded_at=datetime(2015, 1, 3, 12, 0, 0, tzinfo=timezone.utc),
    )
    session.add(odds)


def main() -> None:
    with Session(engine) as session:
        fighters = seed_fighters(session)
        fights = seed_fights(session, fighters)
        seed_fight_stats(session, fights, fighters)
        seed_round_stats(session, fights, fighters)
        seed_historical_odds(session, fights)
        session.commit()

    print(f"Seed complete: {len(fighters)} fighters, {len(fights)} fights.")


if __name__ == "__main__":
    main()
