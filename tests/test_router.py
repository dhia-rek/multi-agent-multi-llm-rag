import pytest

from src.llm_clients.demo import DemoLLMClient
from src.llm_clients.mock import MockLLMClient
from src.routing.router import Router
from tests.conftest import make_settings


def _router(monkeypatch, **overrides):
    monkeypatch.setattr("src.routing.router.settings", make_settings(**overrides))
    return Router()


@pytest.mark.parametrize("task", ["reasoning", "generation", "evaluation"])
def test_heavy_task_types_go_powerful(monkeypatch, task):
    r = _router(monkeypatch, demo_mode="on")
    assert r.decide(task, "low", "low").tier == "powerful"


@pytest.mark.parametrize("kwargs", [{"complexity": "high"}, {"criticality": "high"}])
def test_high_complexity_or_criticality_goes_powerful(monkeypatch, kwargs):
    r = _router(monkeypatch, demo_mode="on")
    assert r.decide("extraction", **kwargs).tier == "powerful"


def test_low_criticality_with_ollama_goes_local(monkeypatch):
    r = _router(monkeypatch, demo_mode="on", ollama_model="llama3")
    assert r.decide("extraction", "medium", "low").tier == "local"


def test_low_criticality_without_ollama_goes_fast(monkeypatch):
    r = _router(monkeypatch, demo_mode="on", ollama_model=None)
    assert r.decide("extraction", "medium", "low").tier == "fast"


def test_default_goes_fast(monkeypatch):
    r = _router(monkeypatch, demo_mode="on", ollama_model="llama3")
    assert r.decide("synthesis").tier == "fast"


def test_demo_mode_uses_demo_clients(monkeypatch):
    r = _router(monkeypatch, demo_mode="on")
    assert isinstance(r.fast(), DemoLLMClient)
    assert r.fast().name == "demo-fixture-fast"
    assert r.powerful().name == "demo-fixture-powerful"


def test_demo_off_without_key_uses_mock(monkeypatch):
    r = _router(monkeypatch, demo_mode="off", google_api_key=None)
    assert not r.demo
    assert isinstance(r.fast(), MockLLMClient)
    assert r.powerful().name == "mock-powerful"


def test_auto_demo_follows_api_key(monkeypatch):
    assert _router(monkeypatch, demo_mode="auto", google_api_key=None).demo
    assert not _router(monkeypatch, demo_mode="auto", google_api_key="k").demo


def test_signature_differs_between_demo_and_mock(monkeypatch):
    demo = _router(monkeypatch, demo_mode="on").signature()
    mock = _router(monkeypatch, demo_mode="off", google_api_key=None).signature()
    assert demo != mock
    assert demo == {"mode": "demo"}
    assert mock["fast"] == mock["powerful"] == "mock"


def test_mark_exhausted_swaps_in_mock(monkeypatch):
    r = _router(monkeypatch, demo_mode="on")
    r.mark_exhausted("powerful")
    assert isinstance(r.powerful(), MockLLMClient)


def test_route_returns_client_and_decision(monkeypatch):
    r = _router(monkeypatch, demo_mode="on")
    client, decision = r.route("reasoning")
    assert decision.tier == "powerful"
    assert client.name == "demo-fixture-powerful"
