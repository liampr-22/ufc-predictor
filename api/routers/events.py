import logging
from collections import defaultdict
from datetime import date

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.database import get_db
from api.routers.predict import _build_prediction_response
from models.pydantic_models import (
    PredictionResponse,
    ScheduledFight,
    UpcomingEvent,
    UpcomingEventsResponse,
)
from models.schema import Fight, Fighter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


@router.get("/upcoming", response_model=UpcomingEventsResponse)
def upcoming_events(request: Request, db: Session = Depends(get_db)):
    """
    Return the next UFC card(s) with model predictions per scheduled fight.

    Scheduled fights are those with date >= today and no winner recorded.
    """
    today = date.today()

    fights = db.execute(
        select(Fight)
        .where(Fight.date >= today, Fight.winner_id.is_(None))
        .order_by(Fight.date, Fight.event)
    ).scalars().all()

    predictor = request.app.state.predictor

    # Group by (event, date)
    grouped: dict[tuple[str, date], list[ScheduledFight]] = defaultdict(list)

    for fight in fights:
        fa: Fighter = fight.fighter_a
        fb: Fighter = fight.fighter_b

        prediction: PredictionResponse | None = None
        if predictor is not None:
            try:
                raw = predictor.predict(db, fa.id, fb.id)
                prediction = _build_prediction_response(fa, fb, raw)
            except Exception as exc:
                logger.debug("Skipping prediction for fight %d: %s", fight.id, exc)

        event_key = (fight.event or "Unannounced", fight.date)
        grouped[event_key].append(
            ScheduledFight(
                fight_id=fight.id,
                fighter_a=fa.name,
                fighter_b=fb.name,
                prediction=prediction,
            )
        )

    events = [
        UpcomingEvent(event=event_name, date=event_date, fights=fight_list)
        for (event_name, event_date), fight_list in sorted(grouped.items(), key=lambda x: x[0][1])
    ]

    return UpcomingEventsResponse(events=events)
