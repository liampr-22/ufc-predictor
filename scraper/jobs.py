"""
Reusable incremental scrape job.

Called by:
  - APScheduler (weekly cron in api/main.py lifespan)
  - POST /admin/scrape (via FastAPI BackgroundTasks)

Logs every run to the scrape_jobs table.
"""

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from models.schema import ScrapeJob
from scraper.scheduler import _latest_fight_date, scrape_events
from scraper.ufcstats import UFCStatsScraper

logger = logging.getLogger(__name__)


def run_incremental_scrape(
    database_url: str,
    app=None,
    retrain_threshold: int = 5,
) -> None:
    """
    Run an incremental scrape, update Elo ratings, and optionally retrain the model.

    Parameters
    ----------
    database_url:
        SQLAlchemy-compatible database URL.
    app:
        FastAPI application instance. When provided, app.state.predictor is
        reloaded after a successful retrain so new predictions use the fresh model.
    retrain_threshold:
        Minimum number of new fights required to trigger model retraining.
    """
    engine = create_engine(database_url)
    job_id: Optional[int] = None

    with Session(engine) as session:
        job = ScrapeJob(
            started_at=datetime.now(timezone.utc),
            status="running",
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    logger.info("ScrapeJob #%d started.", job_id)

    fights_added = 0
    try:
        with Session(engine) as session:
            latest = _latest_fight_date(session)
            since_date: date = (
                (latest + timedelta(days=1)) if latest else date(1993, 11, 12)
            )

        logger.info("Incremental scrape from %s.", since_date)
        with UFCStatsScraper() as scraper:
            with Session(engine) as session:
                _events_done, fights_added = scrape_events(
                    session, scraper, since=since_date
                )

        if fights_added > 0:
            logger.info("Refreshing Elo ratings (%d new fights)…", fights_added)
            from ml.elo import run_replay
            run_replay(database_url)

        if fights_added >= retrain_threshold:
            logger.info(
                "fights_added=%d >= threshold=%d — retraining model.",
                fights_added,
                retrain_threshold,
            )
            from ml.train import run as train_run
            train_run(database_url)
            if app is not None:
                try:
                    from ml.predict import Predictor
                    app.state.predictor = Predictor()
                    logger.info("Predictor reloaded after retrain.")
                except Exception as exc:
                    logger.warning("Could not reload predictor after retrain: %s", exc)

        with Session(engine) as session:
            job = session.get(ScrapeJob, job_id)
            job.finished_at = datetime.now(timezone.utc)
            job.fights_added = fights_added
            job.status = "success"
            session.commit()

        logger.info("ScrapeJob #%d finished. fights_added=%d.", job_id, fights_added)

    except Exception as exc:
        logger.exception("ScrapeJob #%d failed: %s", job_id, exc)
        try:
            with Session(engine) as session:
                job = session.get(ScrapeJob, job_id)
                job.finished_at = datetime.now(timezone.utc)
                job.fights_added = fights_added
                job.status = "failed"
                job.error = str(exc)[:2000]
                session.commit()
        except Exception as inner:
            logger.error("Could not update ScrapeJob #%d status to failed: %s", job_id, inner)
        raise
