"""
Parser unit tests — no HTTP, no database, no Docker required.
All tests use hand-crafted HTML that mirrors the UFCStats page structure.
"""

from datetime import date

import pytest

from scraper.parser import (
    parse_event_index,
    parse_event_page,
    parse_fight_page,
    parse_fighter_index,
    parse_fighter_profile,
)


# ── Fixtures: minimal HTML snippets ──────────────────────────────────────────

FIGHTER_INDEX_HTML = """
<html><body>
<table class="b-statistics__table">
  <thead>
    <tr class="b-statistics__table-row">
      <th>First</th><th>Last</th><th>Nickname</th>
    </tr>
  </thead>
  <tbody>
    <tr class="b-statistics__table-row">
      <td><a href="http://www.ufcstats.com/fighter-details/abc123">Jon</a></td>
      <td><a href="http://www.ufcstats.com/fighter-details/abc123">Jones</a></td>
      <td>Bones</td>
    </tr>
    <tr class="b-statistics__table-row">
      <td><a href="http://www.ufcstats.com/fighter-details/def456">Conor</a></td>
      <td><a href="http://www.ufcstats.com/fighter-details/def456">McGregor</a></td>
      <td>The Notorious</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

FIGHTER_PROFILE_HTML = """
<html><body>
<span class="b-content__title-highlight">Jon Jones</span>
<ul class="b-list__box-list">
  <li class="b-list__box-list-item">
    <i class="b-list__box-item-title">Height:</i> 6' 4"
  </li>
  <li class="b-list__box-list-item">
    <i class="b-list__box-item-title">Weight:</i> 205 lbs.
  </li>
  <li class="b-list__box-list-item">
    <i class="b-list__box-item-title">Reach:</i> 84.5"
  </li>
  <li class="b-list__box-list-item">
    <i class="b-list__box-item-title">STANCE:</i> Orthodox
  </li>
  <li class="b-list__box-list-item">
    <i class="b-list__box-item-title">DOB:</i> Jul 19, 1987
  </li>
</ul>
</body></html>
"""

EVENT_INDEX_HTML = """
<html><body>
<table class="b-statistics__table-events">
  <tbody>
    <tr class="b-statistics__table-row b-statistics__table-row_type_first">
      <td>
        <i class="b-statistics__table-content">
          <a href="http://www.ufcstats.com/event-details/ev001" class="b-link b-link_style_black">
            UFC 300: Pereira vs. Hill
          </a>
        </i>
        <span class="b-statistics__date">April 13, 2024</span>
      </td>
    </tr>
    <tr class="b-statistics__table-row">
      <td>
        <i class="b-statistics__table-content">
          <a href="http://www.ufcstats.com/event-details/ev002" class="b-link b-link_style_black">
            UFC 299: O'Malley vs. Vera 2
          </a>
        </i>
        <span class="b-statistics__date">March 09, 2024</span>
      </td>
    </tr>
  </tbody>
</table>
</body></html>
"""

EVENT_PAGE_HTML = """
<html><body>
<table class="b-fight-details__table">
  <tbody>
    <tr class="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"
        data-link="http://www.ufcstats.com/fight-details/fight001">
      <td>
        <a href="http://www.ufcstats.com/fighter-details/abc123">Jon Jones</a>
        <a href="http://www.ufcstats.com/fighter-details/ghi789">Daniel Cormier</a>
      </td>
      <td>W</td>
      <td>2</td>
      <td>128</td>
      <td>83</td>
      <td>2</td>
      <td>0</td>
      <td>Light Heavyweight</td>
    </tr>
    <tr class="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click"
        data-link="http://www.ufcstats.com/fight-details/fight002">
      <td>
        <a href="http://www.ufcstats.com/fighter-details/def456">Conor McGregor</a>
        <a href="http://www.ufcstats.com/fighter-details/jkl012">Dustin Poirier</a>
      </td>
      <td>W</td>
      <td>1</td>
      <td>45</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>Lightweight</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

# Minimal fight detail page — overall totals + sig strikes (no round tables)
FIGHT_PAGE_HTML = """
<html><body>
<div class="b-fight-details__persons clearfix">
  <div class="b-fight-details__person">
    <i class="b-fight-details__person-status">W</i>
    <a href="http://www.ufcstats.com/fighter-details/abc123" class="b-fight-details__person-link">
      <h3 class="b-fight-details__person-name">Jon Jones</h3>
    </a>
  </div>
  <div class="b-fight-details__person">
    <i class="b-fight-details__person-status">L</i>
    <a href="http://www.ufcstats.com/fighter-details/ghi789" class="b-fight-details__person-link">
      <h3 class="b-fight-details__person-name">Daniel Cormier</h3>
    </a>
  </div>
</div>

<div class="b-fight-details__fight-info">
  <ul>
    <li class="b-fight-details__list-item">
      <i class="b-fight-details__label">Method:</i> DEC
    </li>
    <li class="b-fight-details__list-item">
      <i class="b-fight-details__label">Round:</i> 5
    </li>
    <li class="b-fight-details__list-item">
      <i class="b-fight-details__label">Time:</i> 5:00
    </li>
  </ul>
</div>

<!-- Overall totals table -->
<table class="b-fight-details__table js-fight-table">
  <thead><tr>
    <th>Fighter</th><th>KD</th><th>Sig. str.</th><th>Sig. str. %</th>
    <th>Total str.</th><th>Td</th><th>Td %</th><th>Sub. att</th><th>Rev.</th><th>Ctrl</th>
  </tr></thead>
  <tbody>
    <tr class="b-fight-details__table-row">
      <td><a href="http://www.ufcstats.com/fighter-details/abc123">Jon Jones</a></td>
      <td>1</td>
      <td>128 of 212</td>
      <td>60%</td>
      <td>209 of 339</td>
      <td>2 of 5</td>
      <td>40%</td>
      <td>1</td>
      <td>0</td>
      <td>5:02</td>
    </tr>
    <tr class="b-fight-details__table-row">
      <td><a href="http://www.ufcstats.com/fighter-details/ghi789">Daniel Cormier</a></td>
      <td>0</td>
      <td>83 of 155</td>
      <td>53%</td>
      <td>132 of 235</td>
      <td>3 of 9</td>
      <td>33%</td>
      <td>0</td>
      <td>0</td>
      <td>7:30</td>
    </tr>
  </tbody>
</table>

<!-- Overall sig strikes table -->
<table class="b-fight-details__table js-fight-table">
  <thead><tr>
    <th>Fighter</th><th>Sig. str.</th><th>Sig. str. %</th>
    <th>Head</th><th>Body</th><th>Leg</th>
    <th>Distance</th><th>Clinch</th><th>Ground</th>
  </tr></thead>
  <tbody>
    <tr class="b-fight-details__table-row">
      <td><a href="http://www.ufcstats.com/fighter-details/abc123">Jon Jones</a></td>
      <td>128 of 212</td>
      <td>60%</td>
      <td>73 of 140</td>
      <td>30 of 40</td>
      <td>25 of 32</td>
      <td>101 of 178</td>
      <td>12 of 18</td>
      <td>15 of 16</td>
    </tr>
    <tr class="b-fight-details__table-row">
      <td><a href="http://www.ufcstats.com/fighter-details/ghi789">Daniel Cormier</a></td>
      <td>83 of 155</td>
      <td>53%</td>
      <td>50 of 100</td>
      <td>20 of 30</td>
      <td>13 of 25</td>
      <td>60 of 118</td>
      <td>10 of 14</td>
      <td>13 of 23</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

# Fight page with one round of per-round stats appended
FIGHT_PAGE_WITH_ROUNDS_HTML = FIGHT_PAGE_HTML.replace(
    "</body>",
    """
<!-- Round 1 totals -->
<table class="b-fight-details__table js-fight-table">
  <thead><tr>
    <th>Fighter</th><th>KD</th><th>Sig. str.</th><th>Sig. str. %</th>
    <th>Total str.</th><th>Td</th><th>Td %</th><th>Sub. att</th><th>Rev.</th><th>Ctrl</th>
  </tr></thead>
  <tbody>
    <tr><td><a href="http://www.ufcstats.com/fighter-details/abc123">Jon Jones</a></td>
      <td>0</td><td>22 of 38</td><td>57%</td><td>35 of 58</td>
      <td>0 of 1</td><td>0%</td><td>0</td><td>0</td><td>0:42</td></tr>
    <tr><td><a href="http://www.ufcstats.com/fighter-details/ghi789">Daniel Cormier</a></td>
      <td>0</td><td>15 of 30</td><td>50%</td><td>25 of 45</td>
      <td>1 of 3</td><td>33%</td><td>0</td><td>0</td><td>1:10</td></tr>
  </tbody>
</table>
<!-- Round 1 sig strikes -->
<table class="b-fight-details__table js-fight-table">
  <thead><tr>
    <th>Fighter</th><th>Sig. str.</th><th>Sig. str. %</th>
    <th>Head</th><th>Body</th><th>Leg</th>
    <th>Distance</th><th>Clinch</th><th>Ground</th>
  </tr></thead>
  <tbody>
    <tr><td><a href="http://www.ufcstats.com/fighter-details/abc123">Jon Jones</a></td>
      <td>22 of 38</td><td>57%</td>
      <td>14 of 26</td><td>5 of 7</td><td>3 of 5</td>
      <td>18 of 32</td><td>2 of 3</td><td>2 of 3</td></tr>
    <tr><td><a href="http://www.ufcstats.com/fighter-details/ghi789">Daniel Cormier</a></td>
      <td>15 of 30</td><td>50%</td>
      <td>10 of 20</td><td>3 of 6</td><td>2 of 4</td>
      <td>12 of 24</td><td>2 of 4</td><td>1 of 2</td></tr>
  </tbody>
</table>
</body>""",
)


# ── Tests: parse_fighter_index ────────────────────────────────────────────────

def test_fighter_index_count():
    result = parse_fighter_index(FIGHTER_INDEX_HTML)
    assert len(result) == 2


def test_fighter_index_names():
    result = parse_fighter_index(FIGHTER_INDEX_HTML)
    names = [r["name"] for r in result]
    assert "Jon Jones" in names
    assert "Conor McGregor" in names


def test_fighter_index_ufcstats_id():
    result = parse_fighter_index(FIGHTER_INDEX_HTML)
    ids = {r["ufcstats_id"] for r in result}
    assert "abc123" in ids
    assert "def456" in ids


def test_fighter_index_empty_table():
    html = "<html><body><table class='b-statistics__table'><tbody></tbody></table></body></html>"
    assert parse_fighter_index(html) == []


def test_fighter_index_no_table():
    assert parse_fighter_index("<html><body></body></html>") == []


# ── Tests: parse_fighter_profile ─────────────────────────────────────────────

def test_fighter_profile_name():
    result = parse_fighter_profile(FIGHTER_PROFILE_HTML, "http://www.ufcstats.com/fighter-details/abc123")
    assert result["name"] == "Jon Jones"


def test_fighter_profile_height():
    result = parse_fighter_profile(FIGHTER_PROFILE_HTML, "http://www.ufcstats.com/fighter-details/abc123")
    assert result["height"] == 76.0  # 6*12 + 4


def test_fighter_profile_reach():
    result = parse_fighter_profile(FIGHTER_PROFILE_HTML, "http://www.ufcstats.com/fighter-details/abc123")
    assert result["reach"] == 84.5


def test_fighter_profile_stance():
    result = parse_fighter_profile(FIGHTER_PROFILE_HTML, "http://www.ufcstats.com/fighter-details/abc123")
    assert result["stance"] == "Orthodox"


def test_fighter_profile_dob():
    result = parse_fighter_profile(FIGHTER_PROFILE_HTML, "http://www.ufcstats.com/fighter-details/abc123")
    assert result["dob"] == date(1987, 7, 19)


def test_fighter_profile_ufcstats_id():
    result = parse_fighter_profile(FIGHTER_PROFILE_HTML, "http://www.ufcstats.com/fighter-details/abc123")
    assert result["ufcstats_id"] == "abc123"


def test_fighter_profile_missing_stats():
    html = """<html><body>
    <span class="b-content__title-highlight">Test Fighter</span>
    <ul class="b-list__box-list">
      <li class="b-list__box-list-item">
        <i class="b-list__box-item-title">Height:</i> --
      </li>
    </ul>
    </body></html>"""
    result = parse_fighter_profile(html, "http://www.ufcstats.com/fighter-details/xyz")
    assert result["height"] is None
    assert result["reach"] is None
    assert result["stance"] is None
    assert result["dob"] is None


# ── Tests: parse_event_index ──────────────────────────────────────────────────

def test_event_index_count():
    result = parse_event_index(EVENT_INDEX_HTML)
    assert len(result) == 2


def test_event_index_sorted_oldest_first():
    result = parse_event_index(EVENT_INDEX_HTML)
    assert result[0]["date"] < result[1]["date"]


def test_event_index_names():
    result = parse_event_index(EVENT_INDEX_HTML)
    names = [r["name"] for r in result]
    assert any("UFC 299" in n for n in names)
    assert any("UFC 300" in n for n in names)


def test_event_index_ids():
    result = parse_event_index(EVENT_INDEX_HTML)
    ids = {r["ufcstats_id"] for r in result}
    assert "ev001" in ids
    assert "ev002" in ids


def test_event_index_dates():
    result = parse_event_index(EVENT_INDEX_HTML)
    dates = {r["date"] for r in result}
    assert date(2024, 4, 13) in dates
    assert date(2024, 3, 9) in dates


# ── Tests: parse_event_page ───────────────────────────────────────────────────

def test_event_page_count():
    result = parse_event_page(EVENT_PAGE_HTML, "UFC 182", date(2015, 1, 3))
    assert len(result) == 2


def test_event_page_fight_ids():
    result = parse_event_page(EVENT_PAGE_HTML, "UFC 182", date(2015, 1, 3))
    ids = {r["ufcstats_id"] for r in result}
    assert "fight001" in ids
    assert "fight002" in ids


def test_event_page_fighters():
    result = parse_event_page(EVENT_PAGE_HTML, "UFC 182", date(2015, 1, 3))
    first = next(r for r in result if r["ufcstats_id"] == "fight001")
    assert first["fighter_a_name"] == "Jon Jones"
    assert first["fighter_b_name"] == "Daniel Cormier"
    assert first["fighter_a_ufcstats_id"] == "abc123"
    assert first["fighter_b_ufcstats_id"] == "ghi789"


def test_event_page_weight_class():
    result = parse_event_page(EVENT_PAGE_HTML, "UFC 182", date(2015, 1, 3))
    first = next(r for r in result if r["ufcstats_id"] == "fight001")
    assert first["weight_class"] == "Light Heavyweight"


def test_event_page_event_meta_propagated():
    result = parse_event_page(EVENT_PAGE_HTML, "UFC 182", date(2015, 1, 3))
    for fight in result:
        assert fight["event_name"] == "UFC 182"
        assert fight["event_date"] == date(2015, 1, 3)


# ── Tests: parse_fight_page ───────────────────────────────────────────────────

def test_fight_page_winner():
    result = parse_fight_page(FIGHT_PAGE_HTML, "http://www.ufcstats.com/fight-details/fight001")
    assert result is not None
    assert result["winner_ufcstats_id"] == "abc123"  # Jones won


def test_fight_page_fighters():
    result = parse_fight_page(FIGHT_PAGE_HTML, "http://www.ufcstats.com/fight-details/fight001")
    assert result["fighter_a_name"] == "Jon Jones"
    assert result["fighter_b_name"] == "Daniel Cormier"
    assert result["fighter_a_ufcstats_id"] == "abc123"
    assert result["fighter_b_ufcstats_id"] == "ghi789"


def test_fight_page_method_round_time():
    result = parse_fight_page(FIGHT_PAGE_HTML, "http://www.ufcstats.com/fight-details/fight001")
    assert result["method"] == "DEC"
    assert result["round"] == 5
    assert result["time"] == "5:00"


def test_fight_page_totals_stats_a():
    result = parse_fight_page(FIGHT_PAGE_HTML, "http://www.ufcstats.com/fight-details/fight001")
    sa = result["stats_a"]
    assert sa["knockdowns"] == 1
    assert sa["significant_strikes_landed"] == 128
    assert sa["significant_strikes_attempted"] == 212
    assert sa["total_strikes_landed"] == 209
    assert sa["takedowns_landed"] == 2
    assert sa["takedowns_attempted"] == 5
    assert sa["submission_attempts"] == 1
    assert sa["control_time_seconds"] == 302  # 5*60+2


def test_fight_page_sig_strikes_a():
    result = parse_fight_page(FIGHT_PAGE_HTML, "http://www.ufcstats.com/fight-details/fight001")
    sa = result["stats_a"]
    assert sa["head_strikes_landed"] == 73
    assert sa["body_strikes_landed"] == 30
    assert sa["leg_strikes_landed"] == 25
    assert sa["distance_strikes_landed"] == 101
    assert sa["clinch_strikes_landed"] == 12
    assert sa["ground_strikes_landed"] == 15


def test_fight_page_totals_stats_b():
    result = parse_fight_page(FIGHT_PAGE_HTML, "http://www.ufcstats.com/fight-details/fight001")
    sb = result["stats_b"]
    assert sb["knockdowns"] == 0
    assert sb["significant_strikes_landed"] == 83
    assert sb["takedowns_landed"] == 3
    assert sb["control_time_seconds"] == 450  # 7*60+30


def test_fight_page_no_round_stats():
    result = parse_fight_page(FIGHT_PAGE_HTML, "http://www.ufcstats.com/fight-details/fight001")
    assert result["round_stats"] == []


def test_fight_page_with_round_stats():
    result = parse_fight_page(
        FIGHT_PAGE_WITH_ROUNDS_HTML,
        "http://www.ufcstats.com/fight-details/fight001",
    )
    assert result is not None
    rs = result["round_stats"]
    # 2 fighters × 1 round = 2 entries
    assert len(rs) == 2
    jones_r1 = next(r for r in rs if r["fighter_ufcstats_id"] == "abc123")
    assert jones_r1["round_number"] == 1
    assert jones_r1["significant_strikes_landed"] == 22
    assert jones_r1["head_strikes_landed"] == 14
    assert jones_r1["control_time_seconds"] == 42  # 0*60+42


def test_fight_page_draw_has_no_winner():
    html = FIGHT_PAGE_HTML.replace(
        '<i class="b-fight-details__person-status">W</i>',
        '<i class="b-fight-details__person-status">D</i>',
    ).replace(
        '<i class="b-fight-details__person-status">L</i>',
        '<i class="b-fight-details__person-status">D</i>',
    )
    result = parse_fight_page(html, "http://www.ufcstats.com/fight-details/draw001")
    assert result is not None
    assert result["winner_ufcstats_id"] is None


def test_fight_page_missing_persons_returns_none():
    result = parse_fight_page("<html><body></body></html>", "http://www.ufcstats.com/fight-details/bad")
    assert result is None


def test_fight_page_ufcstats_id():
    result = parse_fight_page(FIGHT_PAGE_HTML, "http://www.ufcstats.com/fight-details/fight001")
    assert result["ufcstats_id"] == "fight001"
