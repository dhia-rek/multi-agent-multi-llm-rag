"""Gemini tiers (Flash-Lite / Flash) on `langchain-google-genai`.

Behaviour
---------
- The tier is a `ChatGoogleGenerativeAI` model. The powerful tier thinks with a
  capped budget (GEMINI_THINKING_BUDGET) added on top of max_output_tokens so
  thinking can't truncate the answer; the fast tier runs with thinking off.
  Agents may override the budget. Responses are requested as JSON.
- Retries live in `invoke_with_retry`: per-minute 429s wait the server-suggested
  delay, 5xx wait a fixed 10 s, up to 3 attempts, no sleep after the last one.
- Daily-quota errors (or a suggested wait longer than a minute) raise
  `QuotaExhaustedError` instantly so the router can fail over to a mock.
"""
from __future__ import annotations

import re
import time

from google.genai import errors
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt

from src.llm_clients.base import BaseLLMClient, LLMResponse, QuotaExhaustedError
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_RETRY_DELAY = 30.0
MAX_RETRY_DELAY = 60.0
SERVER_ERROR_DELAY = 10.0
MAX_OUTPUT_TOKENS = 4096
_TRANSIENT_SERVER_CODES = {500, 502, 503, 504}


class TransientError(RuntimeError):
    """Safe to retry after `delay` seconds."""

    def __init__(self, message: str, delay: float):
        super().__init__(message)
        self.delay = delay


class RateLimitedError(TransientError):
    """Per-minute rate limit hit."""


def _suggested_delay(err: Exception) -> float | None:
    """Gemini 429s embed a RetryInfo block (`'retryDelay': '34s'`). Honour it."""
    m = re.search(r"retry_?delay['\"]?\s*[:{]\s*['\"]?(?:seconds:\s*)?(\d+)", str(err), re.IGNORECASE)
    return float(m.group(1)) if m else None


def _wait_server_delay(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    delay = getattr(exc, "delay", DEFAULT_RETRY_DELAY)
    return min(delay, MAX_RETRY_DELAY)


def _classify(e: Exception, model: str) -> Exception | None:
    """Map a Gemini/LangChain error to Transient/QuotaExhausted, or None to propagate it as is."""
    code = getattr(e, "code", None)
    if isinstance(e, GoogleRateLimitError) or code == 429:
        msg = str(e)
        delay = _suggested_delay(e)
        # Per-day quotas, or a wait too long to sit through, are permanent for this run.
        if "PerDay" in msg or (delay is not None and delay > MAX_RETRY_DELAY * 2):
            log.error("Gemini daily quota exhausted for %s.", model)
            return QuotaExhaustedError(model, msg)
        delay = delay or DEFAULT_RETRY_DELAY
        log.warning("Gemini per-minute rate-limit on %s, retry in %.1fs", model, min(delay, MAX_RETRY_DELAY))
        return RateLimitedError(msg, delay)
    if isinstance(e, errors.ServerError) and code in _TRANSIENT_SERVER_CODES:
        log.warning("Gemini %s returned %s, retry in %.0fs", model, code, SERVER_ERROR_DELAY)
        return TransientError(str(e), SERVER_ERROR_DELAY)
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=_wait_server_delay,
    retry=retry_if_exception_type(TransientError),
    sleep=lambda s: time.sleep(s),  # late-bound so tests can patch time.sleep
    reraise=True,
)
def invoke_with_retry(model: BaseChatModel, messages: list[BaseMessage]):
    """`model.invoke` with Gemini-aware retry. Offline models never raise these errors."""
    try:
        return model.invoke(messages)
    except Exception as e:
        mapped = _classify(e, getattr(model, "model", "gemini"))
        if mapped is None:
            raise
        raise mapped from e


def usage_from_message(msg) -> dict | None:
    """Normalise LangChain usage metadata to the keys the pipeline has always reported."""
    um = getattr(msg, "usage_metadata", None)
    if not um:
        return None
    thinking = (um.get("output_token_details") or {}).get("reasoning")
    out = um.get("output_tokens") or 0
    meta = getattr(msg, "response_metadata", None) or {}
    return {
        "prompt_tokens": um.get("input_tokens"),
        "output_tokens": max(out - (thinking or 0), 0),  # LangChain counts thinking inside output_tokens
        "thinking_tokens": thinking,
        "finish_reason": meta.get("finish_reason"),
    }


class GeminiClient(BaseLLMClient):
    def __init__(self, model: str, tier: str):
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Get a free key at "
                "https://aistudio.google.com/app/apikey and put it in .env"
            )
        self.name = model
        self.tier = tier
        self._models: dict[int, ChatGoogleGenerativeAI] = {}

    def as_chat_model(self, thinking_budget: int | None = None) -> ChatGoogleGenerativeAI:
        if thinking_budget is None:
            thinking_budget = settings.gemini_thinking_budget if self.tier == "powerful" else 0
        if thinking_budget not in self._models:
            self._models[thinking_budget] = ChatGoogleGenerativeAI(
                model=self.name,
                google_api_key=settings.google_api_key,
                temperature=0.3,
                max_output_tokens=MAX_OUTPUT_TOKENS + thinking_budget,
                thinking_budget=thinking_budget,
                response_mime_type="application/json",
                max_retries=1,  # retries are ours (invoke_with_retry), one wait strategy
            )
        return self._models[thinking_budget]

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
        messages: list[BaseMessage] = ([SystemMessage(system)] if system else []) + [HumanMessage(prompt)]
        msg = invoke_with_retry(self.as_chat_model(), messages)
        text = msg.content if isinstance(msg.content, str) else str(msg.text)
        return LLMResponse(text=text.strip(), model=self.name, tier=self.tier, usage=usage_from_message(msg))
