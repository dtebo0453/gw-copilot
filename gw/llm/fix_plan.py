
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, Annotated, Tuple

from pydantic import BaseModel, Field, ValidationError, ConfigDict

from gw.llm.registry import get_provider


_UNSUPPORTED_KEYS = {
    # You already saw oneOf rejected.
    "oneOf",
    # Often rejected in strict schema mode too (varies); safest to block early.
    "anyOf",
    "allOf",
    # OpenAI strict schema frequently chokes on refs/defs unless fully inlined.
    "$ref",
    "$defs",
    "definitions",
}

FixActionType = Literal[
    "drop_rows",
    "shift_indexing",
    "set_config_value",
    "note",
]

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, List[JsonScalar]]


class DropRowsParams(BaseModel):
    """Parameters for dropping problematic stress rows (deterministic + safe)."""

    model_config = ConfigDict(extra="forbid")

    # Which CSV/package we are editing.
    package: Literal["wel", "chd", "rch"]
    # 0-based CSV row indices to drop (excluding header). Deterministic engine decides if/when.
    row_indices: List[int] = Field(default_factory=list)
    # Optional filename (e.g., wells.csv). Used for reporting only.
    filename: Optional[str] = None


class ShiftIndexingParams(BaseModel):
    """Parameters for switching 0-based vs 1-based indexing expectations."""

    model_config = ConfigDict(extra="forbid")

    from_indexing: Literal["zero", "one"]
    to_indexing: Literal["zero", "one"]


class SetConfigValueParams(BaseModel):
    """Parameters for setting a config value (restricted JSON primitives only)."""

    model_config = ConfigDict(extra="forbid")

    # Dot-path into config, e.g. "indexing" or "periods.nstp".
    path: str
    # Restricted to primitives or list-of-primitives (no free-form dicts to keep schema strict).
    value: JsonValue


class NoteParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str


class _BaseAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Literal["safe", "caution", "manual"] = "safe"
    rationale: str = ""


class DropRowsAction(_BaseAction):
    action: Literal["drop_rows"]
    params: DropRowsParams


class ShiftIndexingAction(_BaseAction):
    action: Literal["shift_indexing"]
    params: ShiftIndexingParams


class SetConfigValueAction(_BaseAction):
    action: Literal["set_config_value"]
    params: SetConfigValueParams


class NoteAction(_BaseAction):
    action: Literal["note"]
    params: NoteParams


FixAction = Annotated[
    Union[DropRowsAction, ShiftIndexingAction, SetConfigValueAction, NoteAction],
    Field(discriminator="action"),
]


class FixPlan(BaseModel):
    # Enforce JSON schema strictness (OpenAI json_schema requires additionalProperties:false).
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., description="One-paragraph summary of issues and proposed fixes.")
    actions: List[FixAction] = Field(default_factory=list)
    checks_to_rerun: List[str] = Field(default_factory=list)
    user_confirmations_needed: List[str] = Field(default_factory=list)


def _read_text_if_exists(p: Optional[str]) -> str:
    if not p:
        return ""
    path = Path(p)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _coerce_fixplan(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, FixPlan):
        return obj.model_dump()
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {"summary": obj.strip(), "actions": []}

    if not isinstance(obj, dict):
        return {"summary": "LLM returned non-dict; no actionable fixes.", "actions": []}

    if "fix_plan" in obj and isinstance(obj["fix_plan"], dict):
        obj = obj["fix_plan"]

    obj.setdefault("summary", "Fix plan (auto-generated).")
    obj.setdefault("actions", [])
    obj.setdefault("checks_to_rerun", [])
    obj.setdefault("user_confirmations_needed", [])
    return obj



def _find_unsupported(schema: Any, path: Tuple[Union[str, int], ...] = ()) -> List[Tuple[Tuple[Union[str, int], ...], str]]:
    hits: List[Tuple[Tuple[Union[str, int], ...], str]] = []
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k in _UNSUPPORTED_KEYS:
                hits.append((path, k))
            hits.extend(_find_unsupported(v, path + (k,)))
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            hits.extend(_find_unsupported(v, path + (i,)))
    return hits

def _enforce_openai_strict_json_schema(schema: Any) -> Any:
    """
    Make a JSON schema compatible with OpenAI Responses API `text.format: {type: json_schema, strict: true}`.

    Enforces:
    - For every object schema: `additionalProperties: false`
    - For every object schema: `required` must include every key in `properties`

    Also *fails fast* if schema contains unsupported constructs (oneOf/anyOf/allOf/$ref/$defs),
    since OpenAI strict mode rejects them.

    NOTE: If you need unions/discriminators, hand-write an OpenAI-friendly schema (like FixPlan),
    or implement a ref-inliner (more complex).
    """
    # Fail fast on unsupported features
    hits = _find_unsupported(schema)
    if hits:
        lines = []
        for path, key in hits[:20]:
            pretty_path = " -> ".join(str(p) for p in path) if path else "<root>"
            lines.append(f"- unsupported key '{key}' at {pretty_path}")
        extra = "" if len(hits) <= 20 else f"\n... plus {len(hits) - 20} more"
        raise ValueError(
            "Schema contains constructs not permitted by OpenAI strict json_schema:\n"
            + "\n".join(lines)
            + extra
            + "\n\nRecommendation: hand-write an OpenAI-safe schema (no oneOf/anyOf/$ref) "
              "or inline/flatten the schema before sending."
        )

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            # Recurse first
            for k, v in list(node.items()):
                node[k] = _walk(v)

            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                props: Dict[str, Any] = node["properties"]
                node["additionalProperties"] = False  # force
                # OpenAI strict requires required to list every property key
                node["required"] = list(props.keys())

            return node

        if isinstance(node, list):
            return [_walk(x) for x in node]

        return node

    return _walk(schema)

def suggest_fix_plan(
    *,
    provider: str = "openai",
    model: Optional[str] = None,
    config_path: Optional[str] = None,
    inputs_dir: Optional[str] = None,
    stress_validation_report_path: Optional[str] = None,
    max_actions: int = 8,
) -> FixPlan:
    base_cfg: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        base_cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    report_txt = _read_text_if_exists(stress_validation_report_path)

    # --- Early return if no validation report exists ---
    if not report_txt.strip():
        return FixPlan(
            summary=(
                "No stress validation report found. "
                "Run 'Validate' first to generate findings, then try 'Suggest Fix' again."
            ),
            actions=[
                NoteAction(
                    action="note",
                    severity="safe",
                    rationale="Validation must complete before fixes can be suggested.",
                    params=NoteParams(
                        message="Run the 'Validate' button first to produce a validation report, "
                        "then click 'Suggest Fix' to get AI-powered fix suggestions."
                    ),
                )
            ],
            checks_to_rerun=["validate"],
            user_confirmations_needed=[],
        )

    files = []
    if inputs_dir and Path(inputs_dir).exists():
        files = sorted([p.name for p in Path(inputs_dir).iterdir() if p.is_file()])

    context: Dict[str, Any] = {
        "base_config": base_cfg,
        "inputs_dir_files": files,
        "stress_validation_report_md": report_txt[:12000],
        "constraints": {
            "no_raw_data": True,
            "produce_deterministic_fix_plan": True,
            "max_actions": max_actions,
            "safe_actions_only_default": True,
        },
    }

    schema = _openai_fixplan_response_schema(max_actions=max_actions)
    schema = _enforce_openai_strict_json_schema(schema)

    try:
        prov = get_provider(provider, model=model)

        raw = prov.draft(
            prompt=(
                "You are a groundwater-modeling copilot. "
                "Given a deterministic stress validation report and the current MODFLOW 6 config, "
                "propose a FixPlan JSON that a deterministic engine could execute.\n\n"
                "Rules:\n"
                "1) Return JSON ONLY.\n"
                "2) Use top-level keys: summary, actions, checks_to_rerun, user_confirmations_needed.\n"
                "3) Prefer SAFE actions (drop invalid rows) over risky changes.\n"
                "4) Only propose shift_indexing if you have strong evidence (e.g., most indices appear 1-based).\n"
                "5) Never fabricate data; base suggestions on the report/config.\n"
            ),
            context=context,
            schema=schema,
        )

        raw = _coerce_fixplan(raw)
        return FixPlan.model_validate(raw)

    except ValidationError as e:
        raise RuntimeError(
            "LLM output failed FixPlan schema validation.\n"
            "Try re-running, or apply deterministic fixes without LLM.\n\n"
            f"Validation errors:\n{e}"
        ) from e
    except Exception as e:
        # Wrap any other failure (network, auth, timeout) in a helpful FixPlan
        # instead of crashing with an opaque exit code.
        return FixPlan(
            summary=f"Suggest Fix encountered an error: {type(e).__name__}: {e}",
            actions=[
                NoteAction(
                    action="note",
                    severity="safe",
                    rationale="The LLM call failed. Check your API key configuration and network.",
                    params=NoteParams(
                        message=f"Error: {e}. Verify LLM provider settings and try again."
                    ),
                )
            ],
            checks_to_rerun=[],
            user_confirmations_needed=[],
        )

def _openai_fixplan_response_schema(max_actions: int = 8) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "actions": {
                "type": "array",
                "maxItems": max_actions,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "severity": {"type": "string", "enum": ["safe", "caution", "manual"]},
                        "rationale": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["drop_rows", "shift_indexing", "set_config_value", "note"],
                        },
                        "params": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "package": {"type": "string", "enum": ["wel", "chd", "rch"]},
                                "row_indices": {"type": "array", "items": {"type": "integer"}},
                                "filename": {"type": ["string", "null"]},

                                "from_indexing": {"type": "string", "enum": ["zero", "one"]},
                                "to_indexing": {"type": "string", "enum": ["zero", "one"]},

                                "path": {"type": "string"},
                                "value": {"type": "string"},  # coerce later

                                "message": {"type": "string"},
                            },
                            "required": [],
                        },
                    },
                    "required": ["severity", "rationale", "action", "params"],
                },
            },
            "checks_to_rerun": {"type": "array", "items": {"type": "string"}},
            "user_confirmations_needed": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "actions", "checks_to_rerun", "user_confirmations_needed"],
    }



def _strip_unsupported_openai_keywords(schema):
    if isinstance(schema, dict):
        schema.pop("discriminator", None)
        for k, v in schema.items():
            schema[k] = _strip_unsupported_openai_keywords(v)
    elif isinstance(schema, list):
        schema = [_strip_unsupported_openai_keywords(x) for x in schema]
    return schema
