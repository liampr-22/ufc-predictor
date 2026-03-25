from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.database import get_db
from ml.calibration import prob_to_american_odds, prob_to_decimal_odds
from models.pydantic_models import (
    FighterOdds,
    FighterPrediction,
    KeyDifferentials,
    MethodProbs,
    PredictionRequest,
    PredictionResponse,
)
from models.schema import Fighter

router = APIRouter(prefix="/predict", tags=["predict"])


def _resolve_fighter(name: str, db: Session) -> Fighter:
    fighter = db.execute(
        select(Fighter).where(Fighter.name == name)
    ).scalar_one_or_none()
    if fighter is None:
        fighter = db.execute(
            select(Fighter).where(Fighter.name.ilike(f"%{name}%")).limit(1)
        ).scalar_one_or_none()
    if fighter is None:
        raise HTTPException(status_code=404, detail=f"Fighter '{name}' not found")
    return fighter


def _build_prediction_response(
    fa: Fighter,
    fb: Fighter,
    raw: dict,
) -> PredictionResponse:
    win_prob_a = raw["win_prob_a"]
    win_prob_b = 1.0 - win_prob_a

    # Clamp to avoid odds math errors at exactly 0.0 or 1.0
    win_prob_a = max(0.001, min(0.999, win_prob_a))
    win_prob_b = max(0.001, min(0.999, win_prob_b))

    method_probs = raw["method_probs"]

    # Key differentials from DB values
    reach_delta = (
        (fa.reach - fb.reach) if fa.reach is not None and fb.reach is not None else None
    )
    height_delta = (
        (fa.height - fb.height) if fa.height is not None and fb.height is not None else None
    )
    today = date.today()
    age_delta: float | None = None
    if fa.dob is not None and fb.dob is not None:
        age_a = (today - fa.dob).days / 365.25
        age_b = (today - fb.dob).days / 365.25
        age_delta = round(age_a - age_b, 2)

    return PredictionResponse(
        fighter_a=FighterPrediction(
            name=fa.name,
            win_prob=round(win_prob_a, 4),
            odds=FighterOdds(
                american=round(prob_to_american_odds(win_prob_a), 1),
                decimal=round(prob_to_decimal_odds(win_prob_a), 3),
            ),
        ),
        fighter_b=FighterPrediction(
            name=fb.name,
            win_prob=round(win_prob_b, 4),
            odds=FighterOdds(
                american=round(prob_to_american_odds(win_prob_b), 1),
                decimal=round(prob_to_decimal_odds(win_prob_b), 3),
            ),
        ),
        method_probs=MethodProbs(
            ko_tko=round(method_probs["ko_tko"], 4),
            submission=round(method_probs["submission"], 4),
            decision=round(method_probs["decision"], 4),
        ),
        key_differentials=KeyDifferentials(
            elo_delta=round(fa.elo_rating - fb.elo_rating, 2),
            reach_delta=round(reach_delta, 2) if reach_delta is not None else None,
            height_delta=round(height_delta, 2) if height_delta is not None else None,
            age_delta=age_delta,
        ),
    )


@router.post("", response_model=PredictionResponse)
def predict(body: PredictionRequest, request: Request, db: Session = Depends(get_db)):
    """
    Predict the outcome of a fight between two fighters.

    Returns win probabilities, method probabilities, implied American and decimal
    odds, and key physical/rating differentials.
    """
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction models are not loaded. Run `python -m ml.train` first.",
        )

    fa = _resolve_fighter(body.fighter_a, db)
    fb = _resolve_fighter(body.fighter_b, db)

    try:
        raw = predictor.predict(db, fa.id, fb.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _build_prediction_response(fa, fb, raw)
