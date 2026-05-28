"""Shared OpenRouter model factory — imported by core.py and sandbox_subagent.py."""

from __future__ import annotations

from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

import config


def make_model() -> OpenRouterModel:
    """Build an OpenRouter model from config.

    MODEL_NAME may be 'openrouter/google/gemini-2.0-flash-001' (env format)
    or 'google/gemini-2.0-flash-001' (slug). Strip the prefix if present.
    """
    slug = config.MODEL_NAME.removeprefix("openrouter/")
    return OpenRouterModel(
        slug,
        provider=OpenRouterProvider(api_key=config.OPENROUTER_API_KEY or "no-key-set"),
    )
