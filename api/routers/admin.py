import os
import subprocess
import sys

from fastapi import APIRouter, Depends, Header, HTTPException

from models.pydantic_models import ScrapeResponse

router = APIRouter(prefix="/admin", tags=["admin"])


def _verify_api_key(x_api_key: str = Header(...)):
    """Dependency that validates the X-Api-Key header against ADMIN_API_KEY env var."""
    expected = os.environ.get("ADMIN_API_KEY", "")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.post("/scrape", response_model=ScrapeResponse, dependencies=[Depends(_verify_api_key)])
def trigger_scrape():
    """
    Trigger an incremental scrape of UFCStats.

    Authenticated via X-Api-Key header. The scrape runs as a background
    subprocess — this endpoint returns immediately.
    """
    subprocess.Popen(
        [sys.executable, "-m", "scraper.scheduler"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return ScrapeResponse(status="accepted", message="Incremental scrape started")
