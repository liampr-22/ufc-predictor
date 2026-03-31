"""
HTML to structured data parsing helpers.
All parsing targets UFCStats.com HTML — do not add other sources.
"""

import logging
import re
from datetime import date, datetime
from typing import Optional

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# ── Known weight classes ──────────────────────────────────────────────────────

_WEIGHT_CLASSES = [
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
    "Strawweight", "Flyweight", "Bantamweight", "Featherweight",
    "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight",
    "Super Heavyweight", "Heavyweight", "Open Weight", "Catch Weight",
]


# ── String helpers ────────────────────────────────────────────────────────────

def _text(el: Optional[Tag]) -> str:
    """Collapse all whitespace from a tag's text."""
    if el is None:
        return ""
    return " ".join(el.get_text().split())


def _parse_of(value: str) -> tuple[Optional[int], Optional[int]]:
    """Parse 'X of Y' → (X, Y). Returns (None, None) on failure."""
    m = re.match(r"(\d+)\s+of\s+(\d+)", value.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _parse_ctrl(value: str) -> Optional[int]:
    """Parse 'MM:SS' control time → total seconds. '--' or empty → None."""
    value = value.strip()
    if not value or value == "--":
        return None
    m = re.match(r"(\d+):(\d+)", value)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def _parse_height(value: str) -> Optional[float]:
    """Parse '5\' 11"' → 71.0 inches. '--' or empty → None."""
    value = value.strip()
    if not value or value == "--":
        return None
    m = re.match(r"""(\d+)'\s*(\d+)""", value)
    if m:
        return float(int(m.group(1)) * 12 + int(m.group(2)))
    return None


def _parse_reach(value: str) -> Optional[float]:
    """Parse '84"' or '84.5"' → float inches. '--' or empty → None."""
    value = value.strip()
    if not value or value == "--":
        return None
    m = re.match(r"([\d.]+)", value)
    if m:
        return float(m.group(1))
    return None


def _parse_date(value: str) -> Optional[date]:
    """Parse date strings in multiple formats → date. '--' or empty → None."""
    value = value.strip()
    if not value or value == "--":
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    logger.debug("Could not parse date: %r", value)
    return None


def _parse_int(value: str) -> Optional[int]:
    """Parse integer string. '--' or empty or non-numeric → None."""
    value = value.strip()
    if not value or value == "--":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _extract_id(url: str) -> Optional[str]:
    """Extract the trailing hash ID from a UFCStats URL."""
    if not url:
        return None
    return url.rstrip("/").split("/")[-1] or None


def _normalize_weight_class(text: str) -> Optional[str]:
    """Match text against known UFC weight class names (case-insensitive)."""
    text = text.strip()
    for wc in _WEIGHT_CLASSES:
        if wc.lower() in text.lower():
            return wc
    return None


# ── Fighter index ─────────────────────────────────────────────────────────────

def parse_fighter_index(html: str) -> list[dict]:
    """
    Parse one letter page of the fighter index.
    URL: /statistics/fighters?char={letter}&page=all

    Returns list of dicts: {name, ufcstats_id, url}
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="b-statistics__table")
    if not table:
        return []

    results = []
    for row in table.select("tbody tr.b-statistics__table-row"):
        cols = row.find_all("td")
        if len(cols) < 2:
            continue
        first_link = cols[0].find("a")
        if not first_link:
            continue
        url = (first_link.get("href") or "").strip()
        if not url:
            continue
        first_name = first_link.get_text(strip=True)
        last_link = cols[1].find("a")
        last_name = last_link.get_text(strip=True) if last_link else cols[1].get_text(strip=True)
        name = f"{first_name} {last_name}".strip()
        if not name:
            continue
        results.append({
            "name": name,
            "ufcstats_id": _extract_id(url),
            "url": url,
        })
    return results


# ── Fighter profile ───────────────────────────────────────────────────────────

def parse_fighter_profile(html: str, url: str) -> dict:
    """
    Parse a fighter's profile page.
    URL: /fighter-details/{id}

    Returns dict: {name, height, reach, stance, dob, ufcstats_id}
    Weight class is not reliably available on the profile — set externally.
    """
    soup = BeautifulSoup(html, "lxml")
    data: dict = {"ufcstats_id": _extract_id(url)}

    name_el = soup.find("span", class_="b-content__title-highlight")
    data["name"] = _text(name_el) if name_el else None

    stats: dict[str, str] = {}
    for li in soup.select("ul.b-list__box-list li.b-list__box-list-item"):
        label_el = li.find("i", class_="b-list__box-item-title")
        if not label_el:
            continue
        label = label_el.get_text(strip=True).rstrip(":").upper()
        label_el.extract()
        value = _text(li)
        stats[label] = value

    data["height"] = _parse_height(stats.get("HEIGHT", ""))
    data["reach"] = _parse_reach(stats.get("REACH", ""))
    stance = stats.get("STANCE", "").strip()
    data["stance"] = stance if stance and stance != "--" else None
    data["dob"] = _parse_date(stats.get("DOB", ""))
    return data


# ── Event index ───────────────────────────────────────────────────────────────

def parse_event_index(html: str) -> list[dict]:
    """
    Parse the completed events listing page.
    URL: /statistics/events/completed?page=all

    Returns list of dicts: {name, date, ufcstats_id, url}
    Sorted oldest-first so the caller can process in chronological order.
    """
    soup = BeautifulSoup(html, "lxml")
    results = []

    for row in soup.select("tr.b-statistics__table-row"):
        link = row.find("a", class_="b-link_style_black")
        if not link:
            continue
        url = (link.get("href") or "").strip()
        if not url or "event-details" not in url:
            continue
        name = link.get_text(strip=True)
        date_el = row.find("span", class_="b-statistics__date")
        event_date = _parse_date(date_el.get_text(strip=True)) if date_el else None
        results.append({
            "name": name,
            "date": event_date,
            "ufcstats_id": _extract_id(url),
            "url": url,
        })

    # UFCStats lists newest first — reverse so we process oldest first
    results.reverse()
    return results


# ── Event page ────────────────────────────────────────────────────────────────

def parse_event_page(html: str, event_name: str, event_date: Optional[date]) -> list[dict]:
    """
    Parse an event detail page to enumerate its fights.
    URL: /event-details/{id}

    Returns list of dicts per fight:
    {
      ufcstats_id, url, event_name, event_date, weight_class,
      fighter_a_url, fighter_a_name, fighter_a_ufcstats_id,
      fighter_b_url, fighter_b_name, fighter_b_ufcstats_id,
    }
    """
    soup = BeautifulSoup(html, "lxml")
    results = []

    for row in soup.select("tr.b-fight-details__table-row__hover"):
        fight_url = (row.get("data-link") or "").strip()
        if not fight_url or "fight-details" not in fight_url:
            continue

        cols = row.find_all("td")
        if not cols:
            continue

        # Fighter links are in the second column (col[0] is the result status link)
        fighter_links = cols[1].find_all("a")
        if len(fighter_links) < 2:
            continue

        fa_url = (fighter_links[0].get("href") or "").strip()
        fa_name = fighter_links[0].get_text(strip=True)
        fb_url = (fighter_links[1].get("href") or "").strip()
        fb_name = fighter_links[1].get_text(strip=True)

        # Weight class is in the last column
        weight_class: Optional[str] = None
        for col in reversed(cols):
            txt = col.get_text(strip=True)
            wc = _normalize_weight_class(txt)
            if wc:
                weight_class = wc
                break

        results.append({
            "ufcstats_id": _extract_id(fight_url),
            "url": fight_url,
            "event_name": event_name,
            "event_date": event_date,
            "weight_class": weight_class,
            "fighter_a_url": fa_url,
            "fighter_a_name": fa_name,
            "fighter_a_ufcstats_id": _extract_id(fa_url),
            "fighter_b_url": fb_url,
            "fighter_b_name": fb_name,
            "fighter_b_ufcstats_id": _extract_id(fb_url),
        })

    return results


# ── Fight detail ──────────────────────────────────────────────────────────────

def _parse_totals_row(cols: list, fighter_index: int = 0) -> dict:
    """
    Parse one fighter's stats from a totals table row (10 columns).
    Cols: Fighter | KD | Sig.str | Sig.str% | Total str | TD | TD% | Sub.att | Rev | Ctrl

    Each cell now contains two <p class="b-fight-details__table-text"> elements,
    one per fighter. fighter_index selects which fighter (0=A, 1=B).
    """
    links = cols[0].find_all("a")
    fighter_url = (links[fighter_index].get("href") or "").strip() if len(links) > fighter_index else None

    ssl, ssa = _parse_of(_get_cell_text(cols[2], fighter_index))
    tsl, tsa = _parse_of(_get_cell_text(cols[4], fighter_index))
    tdl, tda = _parse_of(_get_cell_text(cols[5], fighter_index))

    return {
        "fighter_ufcstats_id": _extract_id(fighter_url),
        "knockdowns": _parse_int(_get_cell_text(cols[1], fighter_index)),
        "significant_strikes_landed": ssl,
        "significant_strikes_attempted": ssa,
        "total_strikes_landed": tsl,
        "total_strikes_attempted": tsa,
        "takedowns_landed": tdl,
        "takedowns_attempted": tda,
        "submission_attempts": _parse_int(_get_cell_text(cols[7], fighter_index)),
        "reversals": _parse_int(_get_cell_text(cols[8], fighter_index)),
        "control_time_seconds": _parse_ctrl(_get_cell_text(cols[9], fighter_index)),
    }


def _parse_sig_row(cols: list, fighter_index: int = 0) -> dict:
    """
    Parse one fighter's stats from a significant strikes table row (9 columns).
    Cols: Fighter | Sig.str | Sig.str% | Head | Body | Leg | Distance | Clinch | Ground

    Each cell now contains two <p class="b-fight-details__table-text"> elements,
    one per fighter. fighter_index selects which fighter (0=A, 1=B).
    """
    links = cols[0].find_all("a")
    fighter_url = (links[fighter_index].get("href") or "").strip() if len(links) > fighter_index else None

    hl, ha = _parse_of(_get_cell_text(cols[3], fighter_index))
    bl, ba = _parse_of(_get_cell_text(cols[4], fighter_index))
    ll, la = _parse_of(_get_cell_text(cols[5], fighter_index))
    dl, da = _parse_of(_get_cell_text(cols[6], fighter_index))
    cl, ca = _parse_of(_get_cell_text(cols[7], fighter_index))
    gl, ga = _parse_of(_get_cell_text(cols[8], fighter_index))

    return {
        "fighter_ufcstats_id": _extract_id(fighter_url),
        "head_strikes_landed": hl,
        "head_strikes_attempted": ha,
        "body_strikes_landed": bl,
        "body_strikes_attempted": ba,
        "leg_strikes_landed": ll,
        "leg_strikes_attempted": la,
        "distance_strikes_landed": dl,
        "distance_strikes_attempted": da,
        "clinch_strikes_landed": cl,
        "clinch_strikes_attempted": ca,
        "ground_strikes_landed": gl,
        "ground_strikes_attempted": ga,
    }


def _merge_stats(totals: dict, sig: dict) -> dict:
    """Merge sig-strikes fields into totals dict (fighter_ufcstats_id from totals wins)."""
    merged = dict(totals)
    for k, v in sig.items():
        if k != "fighter_ufcstats_id":
            merged[k] = v
    return merged


def _get_cell_text(col: Tag, idx: int) -> str:
    """Return text of the idx-th <p class="b-fight-details__table-text"> in a cell.
    Falls back to full cell text if the multi-p structure is absent."""
    ps = col.find_all("p", class_="b-fight-details__table-text")
    if len(ps) > idx:
        return ps[idx].get_text(strip=True)
    return col.get_text(strip=True)


def parse_fight_page(html: str, url: str) -> Optional[dict]:
    """
    Parse a fight detail page.
    URL: /fight-details/{id}

    Returns dict:
    {
      ufcstats_id,
      fighter_a_ufcstats_id, fighter_a_name,
      fighter_b_ufcstats_id, fighter_b_name,
      winner_ufcstats_id,       # None for draw/NC
      method, round, time,
      stats_a, stats_b,         # full per-fight striking/grappling stats
      round_stats,              # list of per-round stats dicts
    }
    Returns None if the page cannot be parsed.
    """
    soup = BeautifulSoup(html, "lxml")
    result: dict = {"ufcstats_id": _extract_id(url)}

    # ── Fighter identities ────────────────────────────────────────────────────
    persons = soup.select("div.b-fight-details__person")
    if len(persons) < 2:
        logger.warning("Could not find two fighters on fight page: %s", url)
        return None

    def _parse_person(div: Tag) -> dict:
        status_el = div.find("i", class_="b-fight-details__person-status")
        status = status_el.get_text(strip=True) if status_el else ""
        link = div.find("a", class_="b-fight-details__person-link")
        fighter_url = (link.get("href") or "").strip() if link else None
        name_el = div.find("h3", class_="b-fight-details__person-name")
        return {
            "status": status,
            "ufcstats_id": _extract_id(fighter_url),
            "name": _text(name_el),
        }

    person_a = _parse_person(persons[0])
    person_b = _parse_person(persons[1])

    result["fighter_a_ufcstats_id"] = person_a["ufcstats_id"]
    result["fighter_a_name"] = person_a["name"]
    result["fighter_b_ufcstats_id"] = person_b["ufcstats_id"]
    result["fighter_b_name"] = person_b["name"]

    if person_a["status"] == "W":
        result["winner_ufcstats_id"] = person_a["ufcstats_id"]
    elif person_b["status"] == "W":
        result["winner_ufcstats_id"] = person_b["ufcstats_id"]
    else:
        result["winner_ufcstats_id"] = None  # Draw / NC

    # ── Fight result ──────────────────────────────────────────────────────────
    fight_info: dict[str, str] = {}
    content_div = soup.find("div", class_="b-fight-details__content")
    if content_div:
        text_para = content_div.find("p", class_="b-fight-details__text")
        if text_para:
            items = [text_para.find("i", class_="b-fight-details__text-item_first")]
            items += text_para.find_all("i", class_="b-fight-details__text-item")
            for item in items:
                if not item:
                    continue
                label_el = item.find("i", class_="b-fight-details__label")
                if not label_el:
                    continue
                label = label_el.get_text(strip=True).rstrip(":").upper()
                label_el.extract()
                fight_info[label] = _text(item)

    result["method"] = fight_info.get("METHOD") or None
    result["round"] = _parse_int(fight_info.get("ROUND", ""))
    result["time"] = fight_info.get("TIME") or None

    # ── Stats tables ──────────────────────────────────────────────────────────
    # UFCStats fight pages now have 2 tables (totals + sig strikes).
    # Each table has one row where every cell contains two <p> elements —
    # one per fighter. fighter_index 0=A, 1=B.
    tables = soup.select("table.b-fight-details__table.js-fight-table")

    stats_a: dict = {}
    stats_b: dict = {}

    if len(tables) >= 1:
        rows = tables[0].select("tbody tr")
        if rows:
            cols = rows[0].find_all("td")
            if len(cols) >= 10:
                stats_a = _parse_totals_row(cols, 0)
                stats_b = _parse_totals_row(cols, 1)

    if len(tables) >= 2:
        rows = tables[1].select("tbody tr")
        if rows:
            cols = rows[0].find_all("td")
            if len(cols) >= 9:
                stats_a = _merge_stats(stats_a, _parse_sig_row(cols, 0))
                stats_b = _merge_stats(stats_b, _parse_sig_row(cols, 1))

    result["stats_a"] = stats_a
    result["stats_b"] = stats_b

    # ── Round-by-round stats ──────────────────────────────────────────────────
    round_stats: list[dict] = []
    round_tables = tables[2:]

    for i in range(0, len(round_tables) - 1, 2):
        round_number = (i // 2) + 1
        totals_tbl = round_tables[i]
        sig_tbl = round_tables[i + 1]

        t_rows = totals_tbl.select("tbody tr")
        if len(t_rows) < 2:
            continue

        for fighter_idx, person in enumerate([person_a, person_b]):
            t_cols = t_rows[fighter_idx].find_all("td")
            if len(t_cols) < 10:
                continue
            r = _parse_totals_row(t_cols)
            r["fighter_ufcstats_id"] = person["ufcstats_id"]
            r["round_number"] = round_number

            s_rows = sig_tbl.select("tbody tr")
            if len(s_rows) > fighter_idx:
                s_cols = s_rows[fighter_idx].find_all("td")
                if len(s_cols) >= 9:
                    r = _merge_stats(r, _parse_sig_row(s_cols))
                    r["fighter_ufcstats_id"] = person["ufcstats_id"]
                    r["round_number"] = round_number

            round_stats.append(r)

    result["round_stats"] = round_stats
    return result
