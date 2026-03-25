from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from api.database import get_db
from models.pydantic_models import (
    CareerAverages,
    FightHistoryEntry,
    FightHistoryResponse,
    FighterProfile,
    FighterSearchResult,
    StyleScores,
)
from models.schema import Fight, FightStats, Fighter

router = APIRouter(prefix="/fighters", tags=["fighters"])

# Style archetype caps — must match ml/features.py constants
_GRAPPLER_SCORE_CAP = 5.0
_BRAWLER_SCORE_CAP = 3.0

_KO_METHODS = {"KO", "KO/TKO", "TKO"}
_SUB_METHODS = {"SUB", "Submission"}
_DEC_METHODS = {"DEC", "Decision", "U-DEC", "S-DEC", "M-DEC"}


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _resolve_fighter(name: str, db: Session) -> Fighter:
    """Exact match first, ILIKE fallback. Raises 404 if not found."""
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


def _compute_career(fighter_id: int, db: Session) -> tuple[CareerAverages, StyleScores]:
    """Aggregate fight_stats + fight records to compute career averages and style scores."""
    # All fights for this fighter
    all_fights = db.execute(
        select(Fight).where(
            or_(Fight.fighter_a_id == fighter_id, Fight.fighter_b_id == fighter_id)
        ).order_by(Fight.date)
    ).scalars().all()

    n = len(all_fights)
    wins = sum(1 for f in all_fights if f.winner_id == fighter_id)
    losses = sum(
        1 for f in all_fights
        if f.winner_id is not None and f.winner_id != fighter_id
    )
    draws = n - wins - losses

    # Count method-based finishing rates
    ko_wins = sum(1 for f in all_fights if f.winner_id == fighter_id and f.method in _KO_METHODS)
    sub_wins = sum(1 for f in all_fights if f.winner_id == fighter_id and f.method in _SUB_METHODS)
    dec_wins = sum(1 for f in all_fights if f.winner_id == fighter_id and f.method in _DEC_METHODS)

    ko_rate = _safe_div(ko_wins, n)
    sub_rate = _safe_div(sub_wins, n)
    dec_rate = _safe_div(dec_wins, n)

    # Aggregate fight_stats for this fighter
    stats_rows = db.execute(
        select(FightStats).where(FightStats.fighter_id == fighter_id)
    ).scalars().all()

    if stats_rows:
        total_sig_l = sum(r.significant_strikes_landed or 0 for r in stats_rows)
        total_sig_a = sum(r.significant_strikes_attempted or 0 for r in stats_rows)
        total_td_l = sum(r.takedowns_landed or 0 for r in stats_rows)
        total_td_a = sum(r.takedowns_attempted or 0 for r in stats_rows)
        total_sub = sum(r.submission_attempts or 0 for r in stats_rows)
        total_kd = sum(r.knockdowns or 0 for r in stats_rows)
        total_ctrl = sum(r.control_time_seconds or 0 for r in stats_rows)
        ns = len(stats_rows)

        sig_acc = _safe_div(total_sig_l, total_sig_a)
        td_acc = _safe_div(total_td_l, total_td_a)
        sub_per_fight = _safe_div(total_sub, ns)
        kd_per_fight = _safe_div(total_kd, ns)

        avg_sig = total_sig_l / ns
        avg_td = total_td_l / ns
        avg_ctrl = total_ctrl / ns
    else:
        sig_acc = td_acc = sub_per_fight = kd_per_fight = 0.0
        avg_sig = avg_td = avg_ctrl = None
        ns = 0

    career = CareerAverages(
        fights=n,
        wins=wins,
        losses=losses,
        draws=draws,
        avg_sig_strikes_landed=round(avg_sig, 2) if avg_sig is not None else None,
        avg_takedowns_landed=round(avg_td, 2) if avg_td is not None else None,
        avg_control_time_seconds=round(avg_ctrl, 1) if avg_ctrl is not None else None,
        ko_rate=round(ko_rate, 4),
        sub_rate=round(sub_rate, 4),
        dec_rate=round(dec_rate, 4),
    )

    style = StyleScores(
        striker=round(sig_acc, 4),
        wrestler=round(td_acc, 4),
        grappler=round(min(1.0, sub_per_fight / _GRAPPLER_SCORE_CAP), 4),
        brawler=round(min(1.0, kd_per_fight / _BRAWLER_SCORE_CAP), 4),
    )

    return career, style


@router.get("/search", response_model=list[FighterSearchResult])
def search_fighters(q: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    """Fuzzy fighter name search via ILIKE."""
    rows = db.execute(
        select(Fighter).where(Fighter.name.ilike(f"%{q}%")).limit(20)
    ).scalars().all()
    return rows


@router.get("/{name}/history", response_model=FightHistoryResponse)
def fighter_history(
    name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Paginated fight history for a fighter."""
    fighter = _resolve_fighter(name, db)

    total = db.execute(
        select(func.count()).where(
            or_(Fight.fighter_a_id == fighter.id, Fight.fighter_b_id == fighter.id)
        )
    ).scalar_one()

    fights = db.execute(
        select(Fight)
        .where(or_(Fight.fighter_a_id == fighter.id, Fight.fighter_b_id == fighter.id))
        .order_by(Fight.date.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).scalars().all()

    entries: list[FightHistoryEntry] = []
    for fight in fights:
        if fight.fighter_a_id == fighter.id:
            opp = fight.fighter_b
        else:
            opp = fight.fighter_a

        if fight.winner_id is None:
            result = "NC"
        elif fight.winner_id == fighter.id:
            result = "Win"
        else:
            result = "Loss"

        # Distinguish draw: no winner but also not a no-contest
        # (method "DEC" with no winner_id is likely a draw)
        if fight.winner_id is None and fight.method and fight.method.upper() in {"DEC", "DRAW", "M-DEC", "S-DEC"}:
            result = "Draw"

        entries.append(FightHistoryEntry(
            fight_id=fight.id,
            date=fight.date,
            event=fight.event,
            opponent=opp.name if opp else "Unknown",
            result=result,
            method=fight.method,
            round=fight.round,
            time=fight.time,
        ))

    return FightHistoryResponse(
        fighter=fighter.name,
        page=page,
        page_size=page_size,
        total=total,
        fights=entries,
    )


@router.get("/{name}", response_model=FighterProfile)
def fighter_profile(name: str, db: Session = Depends(get_db)):
    """Full fighter profile: physical stats, career averages, Elo, style scores."""
    fighter = _resolve_fighter(name, db)
    career, style = _compute_career(fighter.id, db)

    return FighterProfile(
        id=fighter.id,
        name=fighter.name,
        height=fighter.height,
        reach=fighter.reach,
        stance=fighter.stance,
        dob=fighter.dob,
        weight_class=fighter.weight_class,
        elo_rating=fighter.elo_rating,
        career=career,
        style=style,
    )
