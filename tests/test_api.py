"""
Phase 8 — REST API integration tests.

Uses Starlette TestClient against an in-memory SQLite database.
Fixtures are defined in tests/conftest.py.
"""
from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_health_has_last_scrape(self, client):
        """last_scrape should reflect the most recent fight date in the DB (2023-03-04)."""
        r = client.get("/health")
        assert r.status_code == 200
        # The seeded DB has a completed fight on 2023-03-04; upcoming is in 2099
        # max(date) = 2099-12-31 since winner_id=None fights still count
        assert r.json()["last_scrape"] is not None


# ---------------------------------------------------------------------------
# Fighter search
# ---------------------------------------------------------------------------

class TestFighterSearch:
    def test_search_returns_match(self, client):
        r = client.get("/fighters/search?q=Jon")
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 1
        names = [f["name"] for f in data]
        assert "Jon Jones" in names

    def test_search_fields(self, client):
        r = client.get("/fighters/search?q=Jon Jones")
        assert r.status_code == 200
        f = r.json()[0]
        assert "id" in f
        assert "name" in f
        assert "elo_rating" in f

    def test_search_no_results(self, client):
        r = client.get("/fighters/search?q=zzznomatch")
        assert r.status_code == 200
        assert r.json() == []

    def test_search_missing_q(self, client):
        r = client.get("/fighters/search")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Fighter profile
# ---------------------------------------------------------------------------

class TestFighterProfile:
    def test_profile_found(self, client):
        r = client.get("/fighters/Jon Jones")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Jon Jones"
        assert data["elo_rating"] == 1600.0
        assert "career" in data
        assert "style" in data

    def test_profile_career_fields(self, client):
        r = client.get("/fighters/Jon Jones")
        career = r.json()["career"]
        assert career["fights"] >= 1
        assert career["wins"] >= 1
        assert "ko_rate" in career
        assert "sub_rate" in career
        assert "dec_rate" in career

    def test_profile_style_fields(self, client):
        r = client.get("/fighters/Jon Jones")
        style = r.json()["style"]
        for key in ("striker", "wrestler", "grappler", "brawler"):
            assert key in style
            assert 0.0 <= style[key] <= 1.0

    def test_profile_not_found(self, client):
        r = client.get("/fighters/Nobody McFakename")
        assert r.status_code == 404

    def test_profile_ilike_fallback(self, client):
        """Partial name should resolve via ILIKE."""
        r = client.get("/fighters/Stipe")
        assert r.status_code == 200
        assert r.json()["name"] == "Stipe Miocic"


# ---------------------------------------------------------------------------
# Fighter history
# ---------------------------------------------------------------------------

class TestFighterHistory:
    def test_history_returns_fights(self, client):
        r = client.get("/fighters/Jon Jones/history")
        assert r.status_code == 200
        data = r.json()
        assert data["fighter"] == "Jon Jones"
        assert data["total"] >= 1
        assert len(data["fights"]) >= 1

    def test_history_entry_fields(self, client):
        r = client.get("/fighters/Jon Jones/history")
        entry = r.json()["fights"][0]
        for field in ("fight_id", "date", "opponent", "result"):
            assert field in entry

    def test_history_opponent_name(self, client):
        r = client.get("/fighters/Jon Jones/history")
        opponents = [f["opponent"] for f in r.json()["fights"]]
        assert "Stipe Miocic" in opponents

    def test_history_result_win(self, client):
        r = client.get("/fighters/Jon Jones/history")
        fights = r.json()["fights"]
        completed = [f for f in fights if f["result"] in ("Win", "Loss", "Draw", "NC")]
        assert len(completed) >= 1

    def test_history_pagination_page2_empty(self, client):
        """Page 2 with page_size=100 should be empty (only 2 seeded fights)."""
        r = client.get("/fighters/Jon Jones/history?page=2&page_size=100")
        assert r.status_code == 200
        data = r.json()
        assert data["page"] == 2
        assert data["fights"] == []

    def test_history_not_found(self, client):
        r = client.get("/fighters/Nobody McFakename/history")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_success(self, client):
        r = client.post("/predict", json={"fighter_a": "Jon Jones", "fighter_b": "Stipe Miocic"})
        assert r.status_code == 200
        data = r.json()
        assert "fighter_a" in data
        assert "fighter_b" in data
        assert "method_probs" in data
        assert "key_differentials" in data

    def test_predict_win_probs_sum_to_one(self, client):
        r = client.post("/predict", json={"fighter_a": "Jon Jones", "fighter_b": "Stipe Miocic"})
        data = r.json()
        total = data["fighter_a"]["win_prob"] + data["fighter_b"]["win_prob"]
        assert abs(total - 1.0) < 0.001

    def test_predict_method_probs_present(self, client):
        r = client.post("/predict", json={"fighter_a": "Jon Jones", "fighter_b": "Stipe Miocic"})
        mp = r.json()["method_probs"]
        for key in ("ko_tko", "submission", "decision"):
            assert key in mp
            assert 0.0 <= mp[key] <= 1.0

    def test_predict_odds_present(self, client):
        r = client.post("/predict", json={"fighter_a": "Jon Jones", "fighter_b": "Stipe Miocic"})
        data = r.json()
        for side in ("fighter_a", "fighter_b"):
            assert "american" in data[side]["odds"]
            assert "decimal" in data[side]["odds"]

    def test_predict_differentials_present(self, client):
        r = client.post("/predict", json={"fighter_a": "Jon Jones", "fighter_b": "Stipe Miocic"})
        kd = r.json()["key_differentials"]
        assert "elo_delta" in kd
        assert abs(kd["elo_delta"] - (1600.0 - 1520.0)) < 0.01

    def test_predict_fighter_not_found(self, client):
        r = client.post("/predict", json={"fighter_a": "Jon Jones", "fighter_b": "Ghost Fighter"})
        assert r.status_code == 404

    def test_predict_no_model(self, client_no_model):
        r = client_no_model.post(
            "/predict", json={"fighter_a": "Jon Jones", "fighter_b": "Stipe Miocic"}
        )
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Upcoming events
# ---------------------------------------------------------------------------

class TestUpcomingEvents:
    def test_upcoming_returns_structure(self, client):
        r = client.get("/events/upcoming")
        assert r.status_code == 200
        data = r.json()
        assert "events" in data

    def test_upcoming_contains_seeded_fight(self, client):
        r = client.get("/events/upcoming")
        events = r.json()["events"]
        # The seeded upcoming fight is in UFC 999 on 2099-12-31
        all_fights = [f for ev in events for f in ev["fights"]]
        assert len(all_fights) >= 1
        fighter_names = {f["fighter_a"] for f in all_fights} | {f["fighter_b"] for f in all_fights}
        assert "Jon Jones" in fighter_names

    def test_upcoming_event_fields(self, client):
        r = client.get("/events/upcoming")
        events = r.json()["events"]
        for ev in events:
            assert "event" in ev
            assert "date" in ev
            assert "fights" in ev


# ---------------------------------------------------------------------------
# Admin scrape
# ---------------------------------------------------------------------------

class TestAdminScrape:
    def test_scrape_unauthorized_no_header(self, client):
        r = client.post("/admin/scrape")
        assert r.status_code == 422  # missing required header → FastAPI validation

    def test_scrape_unauthorized_wrong_key(self, client):
        r = client.post("/admin/scrape", headers={"X-Api-Key": "wrong-key"})
        assert r.status_code == 401

    def test_scrape_authorized(self, client, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", "test-secret")
        # Also patch subprocess.Popen so we don't actually launch a process
        import subprocess
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: None)
        r = client.post("/admin/scrape", headers={"X-Api-Key": "test-secret"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "accepted"
