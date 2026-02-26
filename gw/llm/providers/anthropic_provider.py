from __future__ import annotations

import json as _json
import re
from typing import Any, Dict, Optional


class AnthropicProvider:
    """LLM provider backed by Anthropic Claude API.

    Implements the same ``draft`` / ``generate_markdown`` interface as
    ``OpenAIProvider`` so it can be used interchangeably via the registry.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            ) from e
        if not api_key:
            try:
                from gw.api.llm_config import get_api_key
                api_key = get_api_key("anthropic")
            except Exception:
                pass
        if not api_key:
            raise RuntimeError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY or configure in Settings."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    # ------------------------------------------------------------------
    # draft – structured JSON output
    # ------------------------------------------------------------------

    def draft(self, prompt: str, context: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Structured drafting: returns parsed JSON conforming to *schema*."""
        system = (
            "You draft MODFLOW 6 configuration JSON for a deterministic builder. "
            "Return ONLY valid JSON conforming to the provided JSON schema. "
            "Do NOT include markdown formatting, code fences, or commentary."
        )

        user_text = (
            f"PROMPT:\n{prompt}\n\n"
            f"CONTEXT:\n{_json.dumps(context, default=str)}\n\n"
            f"JSON SCHEMA:\n{_json.dumps(schema, default=str)}"
        )

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_text}],
        )

        raw = self._extract_text(resp)

        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)

        return _json.loads(raw)

    # ------------------------------------------------------------------
    # generate_markdown – freeform text
    # ------------------------------------------------------------------

    def generate_markdown(self, *, instructions: str, user_input: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=instructions,
            messages=[{"role": "user", "content": user_input}],
        )
        return self._extract_text(resp)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(resp: Any) -> str:
        if hasattr(resp, "content") and resp.content:
            parts = []
            for block in resp.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "".join(parts).strip()
        return ""
