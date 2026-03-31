import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.database import get_db
from api.routers.admin import router as admin_router
from api.routers.events import router as events_router
from api.routers.fighters import router as fighters_router
from api.routers.predict import router as predict_router
from models.pydantic_models import HealthResponse
from models.schema import Fight, ScrapeJob

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure all tables exist (idempotent — won't alter existing columns)
    from api.database import engine
    from models.schema import Base
    Base.metadata.create_all(bind=engine)

    try:
        from ml.predict import Predictor  # deferred to avoid importing ML deps at collection time
        app.state.predictor = Predictor()
        logger.info("Predictor loaded successfully")
    except Exception as exc:
        app.state.predictor = None
        logger.warning("Models not found — prediction endpoints will return 503. (%s)", exc)

    database_url = os.environ.get("DATABASE_URL", "")
    retrain_threshold = int(os.environ.get("RETRAIN_THRESHOLD", "5"))

    from apscheduler.schedulers.background import BackgroundScheduler
    from scraper.jobs import run_incremental_scrape

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_incremental_scrape,
        trigger="cron",
        day_of_week="tue",
        hour=6,
        minute=0,
        kwargs={
            "database_url": database_url,
            "app": app,
            "retrain_threshold": retrain_threshold,
        },
        id="weekly_scrape",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.start()
    app.state.scheduler = scheduler
    logger.info("APScheduler started — weekly scrape scheduled for Tuesday 06:00 UTC.")

    yield

    scheduler.shutdown(wait=False)
    logger.info("APScheduler shut down.")


app = FastAPI(
    title="UFC Fight Predictor",
    description="ML-powered UFC fight outcome prediction, style matchup analysis, and implied odds generation.",
    version="0.1.0",
    lifespan=lifespan,
)

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(fighters_router)
app.include_router(predict_router)
app.include_router(events_router)
app.include_router(admin_router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(db: Session = Depends(get_db)):
    last_fight_date = db.execute(select(func.max(Fight.date))).scalar_one_or_none()
    last_job = db.execute(
        select(ScrapeJob)
        .where(ScrapeJob.status == "success")
        .order_by(ScrapeJob.finished_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        last_scrape=last_fight_date.isoformat() if last_fight_date else None,
        last_successful_scrape=last_job.finished_at.isoformat() if last_job else None,
    )
