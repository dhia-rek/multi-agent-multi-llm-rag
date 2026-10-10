import pytest
from google.genai import errors

from src.llm_clients import gemini
from src.llm_clients.base import QuotaExhaustedError
from tests.conftest import make_settings


class _Models:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def generate_content(self, **kwargs):
        self.calls += 1
        raise self.exc


class _StubClient:
    def __init__(self, exc):
        self.models = _Models(exc)


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(gemini.time, "sleep", slept.append)
    return slept


@pytest.mark.parametrize(
    ("msg", "expected"),
    [
        ("... 'retryDelay': '34s' ...", 34.0),
        ("retry_delay { seconds: 12 }", 12.0),
        ("no hint here", None),
    ],
)
def test_suggested_delay(msg, expected):
    assert gemini._suggested_delay(RuntimeError(msg)) == expected


def _client(monkeypatch, exc):
    monkeypatch.setattr(gemini, "settings", make_settings(google_api_key="dummy"))
    client = gemini.GeminiClient("gemini-test", tier="fast")
    client._client = _StubClient(exc)
    return client


def test_per_day_429_raises_quota_exhausted_immediately(monkeypatch, no_sleep):
    err = errors.ClientError(
        429, {"error": {"code": 429, "message": "Quota exceeded: GenerateRequestsPerDay", "status": "X"}}
    )
    client = _client(monkeypatch, err)
    with pytest.raises(QuotaExhaustedError) as ei:
        client._call("hi", None)
    assert ei.value.model == "gemini-test"
    assert client._client.models.calls == 1  # no retry
    assert no_sleep == []


def test_per_minute_429_sleeps_then_gives_up_after_retries(monkeypatch, no_sleep):
    err = errors.ClientError(
        429, {"error": {"code": 429, "message": "slow down 'retryDelay': '7s'", "status": "X"}}
    )
    client = _client(monkeypatch, err)
    with pytest.raises(gemini.RateLimitedError):
        client._call("hi", None)
    assert client._client.models.calls == 3
    # exactly the server-suggested delay between attempts, nothing after the last one
    assert no_sleep == [7.0, 7.0]


def test_per_minute_429_recovers_and_caps_delay(monkeypatch, no_sleep):
    err = errors.ClientError(
        429, {"error": {"code": 429, "message": "slow down 'retryDelay': '90s'", "status": "X"}}
    )
    client = _client(monkeypatch, err)
    models = client._client.models

    def flaky(**kwargs):
        models.calls += 1
        if models.calls == 1:
            raise err
        return "ok"

    models.generate_content = flaky
    assert client._call("hi", None) == "ok"
    assert models.calls == 2
    assert no_sleep == [gemini.MAX_RETRY_DELAY]


def test_per_minute_429_without_hint_uses_default_delay(monkeypatch, no_sleep):
    err = errors.ClientError(429, {"error": {"code": 429, "message": "slow down", "status": "X"}})
    client = _client(monkeypatch, err)
    with pytest.raises(gemini.RateLimitedError):
        client._call("hi", None)
    assert no_sleep == [gemini.DEFAULT_RETRY_DELAY] * 2


def test_other_client_errors_propagate(monkeypatch, no_sleep):
    err = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "X"}})
    client = _client(monkeypatch, err)
    with pytest.raises(errors.ClientError):
        client._call("hi", None)
    assert client._client.models.calls == 1


def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr(gemini, "settings", make_settings(google_api_key=None))
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        gemini.GeminiClient("m", tier="fast")
