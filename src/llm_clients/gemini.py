"""Google Gemini client (Flash-Lite / Flash tiers) on the `google-genai` SDK.

Behaviour
---------
- On a per-minute rate limit (HTTP 429) waits exactly the retry delay Gemini
  suggests (30 s if it gives none, capped at 60 s), up to 3 attempts. The wait
  lives in tenacity only, so there is no double wait and no sleep after the
  last attempt.
- Transient server errors (500/502/503/504, e.g. "model overloaded") get the
  same retry path with a fixed 10 s wait. The SDK itself doesn't retry.
- Daily-quota errors raise instantly so the router can fail over to another tier
  rather than retrying for minutes.
"""
from __future__ import annotations

import re
import time

from google import genai
from google.genai import errors, types
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from src.llm_clients.base import BaseLLMClient, LLMResponse, QuotaExhaustedError
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


DEFAULT_RETRY_DELAY = 30.0
MAX_RETRY_DELAY = 60.0
SERVER_ERROR_DELAY = 10.0
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
    """tenacity wait: use the delay carried by the TransientError."""
    exc = state.outcome.exception() if state.outcome else None
    delay = getattr(exc, "delay", DEFAULT_RETRY_DELAY)
    return min(delay, MAX_RETRY_DELAY)


class GeminiClient(BaseLLMClient):
    def __init__(self, model: str, tier: str):
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Get a free key at "
                "https://aistudio.google.com/app/apikey and put it in .env"
            )
        self.name = model
        self.tier = tier
        self._client = genai.Client(api_key=settings.google_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=_wait_server_delay,
        retry=retry_if_exception_type(TransientError),
        sleep=lambda s: time.sleep(s),  # late-bound so tests can patch time.sleep
        reraise=True,
    )
    def _call(self, prompt: str, cfg: types.GenerateContentConfig):
        try:
            return self._client.models.generate_content(model=self.name, contents=prompt, config=cfg)
        except errors.ServerError as e:
            if e.code not in _TRANSIENT_SERVER_CODES:
                raise
            log.warning("Gemini %s returned %s, retry in %.0fs", self.name, e.code, SERVER_ERROR_DELAY)
            raise TransientError(str(e), SERVER_ERROR_DELAY) from e
        except errors.ClientError as e:
            if e.code != 429:
                raise
            msg = str(e)
            # Per-day: permanent for the rest of the run → tell the router.
            if "PerDay" in msg:
                log.error("Gemini daily quota exhausted for %s.", self.name)
                raise QuotaExhaustedError(self.name, msg) from e
            # Per-minute: tenacity waits the suggested delay, then retries.
            delay = _suggested_delay(e) or DEFAULT_RETRY_DELAY
            log.warning("Gemini per-minute rate-limit on %s, retry in %.1fs", self.name, min(delay, MAX_RETRY_DELAY))
            raise RateLimitedError(msg, delay) from e

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=kwargs.get("temperature", 0.3),
            max_output_tokens=kwargs.get("max_output_tokens", 2048),
        )
        resp = self._call(prompt, cfg)
        text = (resp.text or "").strip()
        finish = resp.candidates[0].finish_reason if resp.candidates else None
        if finish == types.FinishReason.MAX_TOKENS:
            log.warning("Gemini %s hit max_output_tokens; output is truncated.", self.name)
        usage = None
        if resp.usage_metadata:
            usage = {
                "prompt_tokens": resp.usage_metadata.prompt_token_count,
                "output_tokens": resp.usage_metadata.candidates_token_count,
                "thinking_tokens": resp.usage_metadata.thoughts_token_count,
                "finish_reason": finish.name if finish else None,
            }
        return LLMResponse(text=text, model=self.name, tier=self.tier, usage=usage)
