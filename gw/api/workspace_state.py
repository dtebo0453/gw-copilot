from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


"""Project-scoped, lightweight memory.

This is intentionally *not* an autonomous agent memory. It's a small, stable
state file that helps the chat + plot planner keep context for a given
workspace:

  - user goals / constraints
  - known issues
  - last actions / decisions

It lives under:
  <workspace>/.gw_copilot/workspace_state.json
"""


def _state_path(ws_root: Path) -> Path:
    return (ws_root / ".gw_copilot" / "workspace_state.json").resolve()


def load_workspace_state(ws_root: Path) -> Dict[str, Any]:
    p = _state_path(ws_root)
    if not p.exists():
        return {
            "version": 1,
            "updated_at": None,
            "goals": [],
            "constraints": [],
            "known_issues": [],
            "recent_notes": [],
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {
            "version": 1,
            "updated_at": None,
            "goals": [],
            "constraints": [],
            "known_issues": [],
            "recent_notes": ["(workspace_state.json was unreadable and was reset)"],
        }


def save_workspace_state(ws_root: Path, state: Dict[str, Any]) -> None:
    p = _state_path(ws_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state or {})
    state.setdefault("version", 1)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p.write_text(json.dumps(state, indent=2, sort_keys=False), encoding="utf-8")


def _dedupe_push(lst: List[str], item: str, max_n: int) -> List[str]:
    item = (item or "").strip()
    if not item:
        return lst
    out: List[str] = []
    seen = set()
    for x in ([item] + list(lst or [])):
        k = x.strip().lower()
        if not k or k in seen:
            continue
        out.append(x.strip())
        seen.add(k)
        if len(out) >= max_n:
            break
    return out


def update_workspace_state(
    ws_root: Path,
    *,
    user_message: str,
    assistant_reply: Optional[str] = None,
) -> Dict[str, Any]:
    """Heuristically update goals/constraints/issues from the conversation."""
    state = load_workspace_state(ws_root)

    um = (user_message or "").strip()
    ar = (assistant_reply or "").strip()

    # Very small heuristics — deliberately conservative.
    lowered = um.lower()
    if any(k in lowered for k in ["goal", "trying to", "i want to", "we want to", "objective"]):
        state["goals"] = _dedupe_push(state.get("goals", []), um, 8)
    if any(k in lowered for k in ["do not", "don't", "must not", "read-only", "no physical parameters"]):
        state["constraints"] = _dedupe_push(state.get("constraints", []), um, 10)
    if any(k in lowered for k in ["error", "failed", "traceback", "bug", "crash", "exception"]):
        state["known_issues"] = _dedupe_push(state.get("known_issues", []), um, 10)

    # Always keep a short rolling note trail.
    note = um
    if ar:
        note = f"User: {um}\nAssistant: {ar[:600]}".strip()
    state["recent_notes"] = _dedupe_push(state.get("recent_notes", []), note, 12)

    save_workspace_state(ws_root, state)
    return state


def workspace_state_summary(state: Dict[str, Any]) -> str:
    """Compact summary string for prompts."""
    if not state:
        return ""
    parts: List[str] = []
    if state.get("goals"):
        parts.append("Goals:\n- " + "\n- ".join(state["goals"][:5]))
    if state.get("constraints"):
        parts.append("Constraints:\n- " + "\n- ".join(state["constraints"][:5]))
    if state.get("known_issues"):
        parts.append("Known issues:\n- " + "\n- ".join(state["known_issues"][:5]))
    if state.get("recent_notes"):
        parts.append("Recent context:\n- " + "\n- ".join(state["recent_notes"][:3]))
    return "\n\n".join(parts).strip()
