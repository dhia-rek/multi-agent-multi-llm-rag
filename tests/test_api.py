import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from tests.conftest import make_settings

CASE = "Alexandra runs a small bike shop with two stores and wants to digitize repairs."


@pytest.fixture
def client(tmp_path, monkeypatch, orchestrator):
    index_dir = tmp_path / "api_index"
    index_dir.mkdir()
    (index_dir / "faiss.index").touch()
    monkeypatch.setattr(api_main, "settings", make_settings(demo_mode="on", index_dir=index_dir))
    monkeypatch.setattr(api_main, "_orchestrator", orchestrator)
    return TestClient(api_main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["demo_mode"] is True
    assert body["vector_index_ready"] is True


def test_pipeline_info(client):
    assert client.get("/pipeline-info").status_code == 200


def test_generate_roadmap(client):
    r = client.post("/generate-roadmap", json={"business_case": CASE})
    assert r.status_code == 200
    body = r.json()
    assert body["demo_mode"] is True
    assert body["roadmap"]["output"]["phases"]
    assert body["evaluation"] is not None


def test_generate_roadmap_without_evaluator(client):
    r = client.post("/generate-roadmap", json={"business_case": CASE, "run_evaluator": False})
    assert r.status_code == 200
    assert r.json()["evaluation"] is None


def test_analyze_stops_after_canvas(client):
    r = client.post("/analyze", json={"business_case": CASE})
    assert r.status_code == 200
    body = r.json()
    assert body["canvas"]["fields"]
    assert "roadmap" not in body
    assert "strategy" not in body
    assert "strategist" not in body["models_used"]
    assert body["demo_mode"] is True


def test_short_payload_is_422(client):
    assert client.post("/analyze", json={"business_case": "too short"}).status_code == 422
    assert client.post("/generate-roadmap", json={"business_case": "x" * 19}).status_code == 422


def test_missing_index_is_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "settings", make_settings(index_dir=tmp_path / "nope"))
    assert client.post("/analyze", json={"business_case": CASE}).status_code == 503
