"""Shared OpenRouter model factory — imported by core.py and sandbox_subagent.py."""

from __future__ import annotations

from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

import config


def make_model(model_name: str | None = None) -> OpenRouterModel:
    """Build an OpenRouter model from config.

    model_name overrides MODEL_NAME env when provided (used by the verifier sub-agent
    to pick its own cheaper model independently of the investigator's model).
    MODEL_NAME / model_name may be 'openrouter/provider/model' or 'provider/model';
    the 'openrouter/' prefix is stripped before passing to OpenRouterModel.
    """
    name = model_name or config.MODEL_NAME or "deepseek/deepseek-chat-v3-0324"
    slug = name.removeprefix("openrouter/")
    return OpenRouterModel(
        slug,
        provider=OpenRouterProvider(api_key=config.OPENROUTER_API_KEY or "no-key-set"),
    )
