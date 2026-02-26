from __future__ import annotations
from typing import Optional
from gw.llm.providers.base import LLMProvider


def _active_provider_name() -> str:
    """Return the provider name from the user's saved LLM config."""
    try:
        from gw.api.llm_config import get_active_provider
        return get_active_provider()
    except Exception:
        return "openai"


def get_provider(name: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    """Return an LLM provider instance.

    If *name* is ``None`` or empty the currently-configured provider is used.
    """
    name_l = (name or _active_provider_name() or "openai").lower()

    if name_l == "openai":
        from gw.llm.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model or "gpt-4o-mini")

    if name_l == "anthropic":
        from gw.llm.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514")

    raise ValueError(f"Unknown provider: {name}")
