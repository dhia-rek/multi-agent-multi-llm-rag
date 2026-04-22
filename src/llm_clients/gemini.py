"""Google Gemini client (Flash variants + Pro).

Behaviour
---------
- Honours `retry_delay` returned by Gemini on rate-limit (`ResourceExhausted`)
  so we don't hammer the API uselessly.
- Retries with exponential backoff on transient errors.
- Daily-quota errors raise instantly so the orchestrator can surface them
  to the user rather than retrying for minutes.
"""
from __future__ import annotations

import re
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm_clients.base import BaseLLMClient, LLMResponse, QuotaExhaustedError
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


def _suggested_delay(err: ResourceExhausted) -> float | None:
    """Gemini 429s embed a `retry_delay { seconds: N }` block. Honour it."""
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(err))
    return float(m.group(1)) if m else None


class GeminiClient(BaseLLMClient):
    def __init__(self, model: str, tier: str):
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Get a free key at "
                "https://aistudio.google.com/app/apikey and put it in .env"
            )
        genai.configure(api_key=settings.google_api_key)
        self.name = model
        self.tier = tier
        self._model = genai.GenerativeModel(model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(ResourceExhausted),
        reraise=True,
    )
    def _call(self, full_prompt: str, cfg: dict):
        try:
            return self._model.generate_content(full_prompt, generation_config=cfg)
        except ResourceExhausted as e:
            msg = str(e)
            # Per-day: permanent for the rest of the run → tell the router.
            if "PerDay" in msg:
                log.error("Gemini daily quota exhausted for %s.", self.name)
                raise QuotaExhaustedError(self.name, msg) from e
            # Per-minute: sleep the suggested delay, then let tenacity retry.
            wait = _suggested_delay(e) or 30.0
            log.warning("Gemini per-minute rate-limit on %s, sleeping %.1fs", self.name, wait)
            time.sleep(min(wait, 60))
            raise

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> LLMResponse:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        cfg = {
            "temperature": kwargs.get("temperature", 0.3),
            "max_output_tokens": kwargs.get("max_output_tokens", 2048),
        }
        resp = self._call(full_prompt, cfg)
        text = (resp.text or "").strip() if hasattr(resp, "text") else ""
        usage = None
        if getattr(resp, "usage_metadata", None):
            usage = {
                "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", None),
                "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", None),
            }
        return LLMResponse(text=text, model=self.name, tier=self.tier, usage=usage)
