import os

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request

from models.pydantic_models import ScrapeResponse

router = APIRouter(prefix="/admin", tags=["admin"])


def _verify_api_key(x_api_key: str = Header(...)):
    """Dependency that validates the X-Api-Key header against ADMIN_API_KEY env var."""
    expected = os.environ.get("ADMIN_API_KEY", "")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.post("/scrape", response_model=ScrapeResponse, dependencies=[Depends(_verify_api_key)])
def trigger_scrape(request: Request, background_tasks: BackgroundTasks):
    """
    Trigger an incremental scrape of UFCStats.

    Authenticated via X-Api-Key header. The scrape runs as a background
    task — this endpoint returns immediately.
    """
    from scraper.jobs import run_incremental_scrape

    database_url = os.environ.get("DATABASE_URL", "")
    retrain_threshold = int(os.environ.get("RETRAIN_THRESHOLD", "5"))
    background_tasks.add_task(
        run_incremental_scrape,
        database_url=database_url,
        app=request.app,
        retrain_threshold=retrain_threshold,
    )
    return ScrapeResponse(status="accepted", message="Incremental scrape started")
