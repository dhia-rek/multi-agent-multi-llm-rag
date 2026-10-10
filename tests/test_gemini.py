import pytest
from google.genai import errors
from langchain_core.messages import AIMessage

from src.llm_clients import gemini
from src.llm_clients.base import QuotaExhaustedError
from tests.conftest import make_settings


class _Model:
    """Stands in for a ChatGoogleGenerativeAI: raises `exc` (or returns `ok`) on invoke."""

    model = "gemini-test"

    def __init__(self, exc=None, fail_times=None, ok="ok"):
        self.exc, self.fail_times, self.ok, self.calls = exc, fail_times, ok, 0

    def invoke(self, messages):
        self.calls += 1
        if self.exc is not None and (self.fail_times is None or self.calls <= self.fail_times):
            raise self.exc
        return self.ok


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(gemini.time, "sleep", slept.append)
    return slept


def _429(msg):
    return errors.ClientError(429, {"error": {"code": 429, "message": msg, "status": "X"}})


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


def test_per_day_429_raises_quota_exhausted_immediately(no_sleep):
    m = _Model(_429("Quota exceeded: GenerateRequestsPerDay"))
    with pytest.raises(QuotaExhaustedError) as ei:
        gemini.invoke_with_retry(m, [])
    assert ei.value.model == "gemini-test"
    assert m.calls == 1
    assert no_sleep == []


def test_per_minute_429_sleeps_then_gives_up_after_retries(no_sleep):
    m = _Model(_429("slow down 'retryDelay': '7s'"))
    with pytest.raises(gemini.RateLimitedError):
        gemini.invoke_with_retry(m, [])
    assert m.calls == 3
    assert no_sleep == [7.0, 7.0]  # nothing after the last attempt


def test_per_minute_429_recovers_and_caps_delay(no_sleep):
    m = _Model(_429("slow down 'retryDelay': '90s'"), fail_times=1)
    assert gemini.invoke_with_retry(m, []) == "ok"
    assert m.calls == 2
    assert no_sleep == [gemini.MAX_RETRY_DELAY]


def test_very_long_suggested_wait_is_treated_as_quota(no_sleep):
    m = _Model(_429("Please retry in 'retryDelay': '32000s'"))
    with pytest.raises(QuotaExhaustedError):
        gemini.invoke_with_retry(m, [])
    assert no_sleep == []


def test_per_minute_429_without_hint_uses_default_delay(no_sleep):
    m = _Model(_429("slow down"))
    with pytest.raises(gemini.RateLimitedError):
        gemini.invoke_with_retry(m, [])
    assert no_sleep == [gemini.DEFAULT_RETRY_DELAY] * 2


def test_503_retries_with_fixed_delay(no_sleep):
    m = _Model(errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}))
    with pytest.raises(gemini.TransientError):
        gemini.invoke_with_retry(m, [])
    assert m.calls == 3
    assert no_sleep == [gemini.SERVER_ERROR_DELAY] * 2


def test_other_errors_propagate(no_sleep):
    err = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "X"}})
    m = _Model(err)
    with pytest.raises(errors.ClientError):
        gemini.invoke_with_retry(m, [])
    assert m.calls == 1


def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr(gemini, "settings", make_settings(google_api_key=None))
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        gemini.GeminiClient("m", tier="fast")


@pytest.mark.parametrize(("tier", "thinking"), [("fast", 0), ("powerful", 1024)])
def test_thinking_budget_is_added_to_output_budget(monkeypatch, tier, thinking):
    monkeypatch.setattr(gemini, "settings", make_settings(google_api_key="dummy", gemini_thinking_budget=1024))
    chat = gemini.GeminiClient("gemini-2.5-flash", tier=tier).as_chat_model()
    assert chat.thinking_budget == thinking
    assert chat.max_output_tokens == gemini.MAX_OUTPUT_TOKENS + thinking
    assert chat.response_mime_type == "application/json"


def test_agent_override_of_thinking_budget(monkeypatch):
    monkeypatch.setattr(gemini, "settings", make_settings(google_api_key="dummy", gemini_thinking_budget=1024))
    client = gemini.GeminiClient("gemini-2.5-flash", tier="powerful")
    assert client.as_chat_model(0).thinking_budget == 0
    assert client.as_chat_model(0) is client.as_chat_model(0)  # cached per budget


def test_usage_is_normalised_and_thinking_split_out():
    msg = AIMessage(
        content="{}",
        usage_metadata={
            "input_tokens": 100, "output_tokens": 300, "total_tokens": 400,
            "output_token_details": {"reasoning": 120},
        },
        response_metadata={"finish_reason": "STOP"},
    )  # fmt: skip
    assert gemini.usage_from_message(msg) == {
        "prompt_tokens": 100, "output_tokens": 180, "thinking_tokens": 120, "finish_reason": "STOP",
    }  # fmt: skip
    assert gemini.usage_from_message(AIMessage(content="x")) is None
