from typing import Any, Dict, Optional

class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "OpenAI provider requires the 'openai' package. Install it with: pip install openai"
            ) from e
        # Use explicit key if provided, otherwise fall back to config / env
        if not api_key:
            try:
                from gw.api.llm_config import get_api_key
                api_key = get_api_key("openai")
            except Exception:
                pass
        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)
        self.model = model

    def draft(self, prompt: str, context: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Structured drafting used by llm-draft-config."""
        system = (
            "You draft MODFLOW 6 configuration JSON for a deterministic builder. "
            "Return ONLY JSON conforming to the provided JSON schema."
        )
        resp = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"PROMPT:\n{prompt}\n\nCONTEXT:\n{context}"},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "draft_response",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        import json as _json
        return _json.loads(resp.output_text)

    def generate_markdown(self, *, instructions: str, user_input: str) -> str:
        """Freeform Markdown generation (no schema)."""
        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
        )
        return resp.output_text or ""
