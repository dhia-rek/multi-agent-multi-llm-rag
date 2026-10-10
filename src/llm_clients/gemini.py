"""Google Gemini client (Flash-Lite / Flash tiers) on the `google-genai` SDK.

Behaviour
---------
- Honours the retry delay returned by Gemini on a per-minute rate limit (HTTP 429)
  so we don't hammer the API uselessly.
- Retries with exponential backoff on those rate limits.
- Daily-quota errors raise instantly so the router can fail over to another tier
  rather than retrying for minutes.
"""
from __future__ import annotations

import re
import time

from google import genai
from google.genai import errors, types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from src.llm_clients.base import BaseLLMClient, LLMResponse, QuotaExhaustedError
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


class RateLimitedError(RuntimeError):
    """Per-minute rate limit hit; safe to retry after a short sleep."""


def _suggested_delay(err: Exception) -> float | None:
    """Gemini 429s embed a RetryInfo block (`'retryDelay': '34s'`). Honour it."""
    m = re.search(r"retry_?delay['\"]?\s*[:{]\s*['\"]?(?:seconds:\s*)?(\d+)", str(err), re.IGNORECASE)
    return float(m.group(1)) if m else None


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
        # No tenacity wait: _call already sleeps the server-suggested delay.
        retry=retry_if_exception_type(RateLimitedError),
        reraise=True,
    )
    def _call(self, prompt: str, cfg: types.GenerateContentConfig):
        try:
            return self._client.models.generate_content(model=self.name, contents=prompt, config=cfg)
        except errors.ClientError as e:
            if e.code != 429:
                raise
            msg = str(e)
            # Per-day: permanent for the rest of the run → tell the router.
            if "PerDay" in msg:
                log.error("Gemini daily quota exhausted for %s.", self.name)
                raise QuotaExhaustedError(self.name, msg) from e
            # Per-minute: sleep the suggested delay, then let tenacity retry.
            wait = _suggested_delay(e) or 30.0
            log.warning("Gemini per-minute rate-limit on %s, sleeping %.1fs", self.name, wait)
            time.sleep(min(wait, 60))
            raise RateLimitedError(msg) from e

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=kwargs.get("temperature", 0.3),
            max_output_tokens=kwargs.get("max_output_tokens", 2048),
        )
        resp = self._call(prompt, cfg)
        text = (resp.text or "").strip()
        usage = None
        if resp.usage_metadata:
            usage = {
                "prompt_tokens": resp.usage_metadata.prompt_token_count,
                "output_tokens": resp.usage_metadata.candidates_token_count,
            }
        return LLMResponse(text=text, model=self.name, tier=self.tier, usage=usage)
