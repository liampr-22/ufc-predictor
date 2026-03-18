from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="UFC Fight Predictor",
    description="ML-powered UFC fight outcome prediction, style matchup analysis, and implied odds generation.",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    last_scrape: str | None


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        last_scrape=None,
    )
