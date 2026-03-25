import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.database import get_db
from api.routers.admin import router as admin_router
from api.routers.events import router as events_router
from api.routers.fighters import router as fighters_router
from api.routers.predict import router as predict_router
from models.pydantic_models import HealthResponse
from models.schema import Fight

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from ml.predict import Predictor  # deferred to avoid importing ML deps at collection time
        app.state.predictor = Predictor()
        logger.info("Predictor loaded successfully")
    except (FileNotFoundError, ImportError) as exc:
        app.state.predictor = None
        logger.warning("Models not found — prediction endpoints will return 503. (%s)", exc)
    yield


app = FastAPI(
    title="UFC Fight Predictor",
    description="ML-powered UFC fight outcome prediction, style matchup analysis, and implied odds generation.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(fighters_router)
app.include_router(predict_router)
app.include_router(events_router)
app.include_router(admin_router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(db: Session = Depends(get_db)):
    last_scrape = db.execute(select(func.max(Fight.date))).scalar_one_or_none()
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        last_scrape=last_scrape.isoformat() if last_scrape else None,
    )
