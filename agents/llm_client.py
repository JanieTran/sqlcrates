import json
import sys
import time

import requests
from pydantic import BaseModel
from typing import Any

from config import logger, settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_llm(
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel] | None = None,
    max_retries: int = 3,
) -> dict[str, Any] | str:
    """Call OpenRouter and return the parsed JSON response.

    If *response_model* is provided, the response is validated against it
    and a plain dict is returned (same keys as the model fields).
    Retries up to *max_retries* times on HTTP, JSON, or validation errors.
    """
    logger.info(f"Calling LLM model: {settings.model}")
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }

    body: dict[str, Any] = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if response_model:
        body["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=body, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            usage = data.get("usage", {})
            if usage:
                logger.info(
                    "Tokens: %d prompt + %d completion = %d total",
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0),
                )

            raw = data["choices"][0]["message"]["content"]
            if response_model:
                parsed = json.loads(raw)
                validated = response_model(**parsed)
                return validated.model_dump()
            return raw

        except Exception as exc:
            if attempt < max_retries - 1:
                # Default exponential backoff: 1 s, 2 s, 4 s, …
                wait = 2 ** attempt

                # OpenRouter returns 429 with an X-RateLimit-Reset header
                # that gives the epoch timestamp (seconds) when the rate
                # limit resets.  The exception from raise_for_status()
                # carries the original response so we can inspect headers.
                if (
                    hasattr(exc, "response")
                    and exc.response is not None
                    and exc.response.status_code == 429
                ):
                    reset_ts = exc.response.headers.get("X-RateLimit-Reset")
                    if reset_ts is not None:
                        try:
                            reset_secs = int(reset_ts) / 1000
                            wait = int(reset_secs - time.time())
                        except (ValueError, TypeError):
                            pass  # fall through to exponential backoff

                        if wait > 180:
                            from datetime import datetime
                            readable = datetime.fromtimestamp(reset_secs).strftime("%Y-%m-%d %H:%M:%S")
                            logger.error(
                                "Rate limit exceeded, reset at %s",
                                readable
                            )
                            sys.exit(1)

                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "LLM call failed after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise
