from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gw.api.workspace_files import resolve_file, resolve_workspace_root, read_file_text
from gw.api.workspace_scan import ensure_workspace_scan

# Reuse the LLM JSON-call helper from plots.py to keep model/env behavior consistent.
from gw.api.plots import _call_llm_json_maybe  # type: ignore


router = APIRouter()


# -----------------------------
# Models
# -----------------------------


class PatchPlanRequest(BaseModel):
    inputs_dir: str = Field(..., description="Base inputs directory (server-side path or relative to repo root)")
    workspace: Optional[str] = Field(None, description="Optional workspace subfolder")
    goal: str = Field(..., description="What the user wants to change / improve")
    force_scan: bool = Field(False, description="Force re-scan of workspace cache")
    max_files: int = Field(18, ge=1, le=60, description="Max number of files to include as context")
    max_total_chars: int = Field(120_000, ge=10_000, le=400_000, description="Max total chars of context")


class PatchApplyRequest(BaseModel):
    inputs_dir: str
    workspace: Optional[str] = None
    patch_id: str
    selections: Optional[List[str]] = None  # list of path_rel to apply; if None apply all


class PatchValidateRequest(BaseModel):
    inputs_dir: str
    workspace: Optional[str] = None
    patch_id: str


# -----------------------------
# Utils
# -----------------------------


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _patches_root(ws: Path) -> Path:
    p = (ws / "run_artifacts" / "patches").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run_dir(ws: Path, patch_id: str) -> Path:
    p = _patches_root(ws) / patch_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_text_file_rel(path_rel: str) -> bool:
    ext = (Path(path_rel).suffix or "").lower()
    return ext in {
        ".nam",
        ".dis",
        ".disu",
        ".disv",
        ".tdis",
        ".ims",
        ".npf",
        ".sto",
        ".ic",
        ".oc",
        ".wel",
        ".chd",
        ".ghb",
        ".riv",
        ".drn",
        ".rch",
        ".evt",
        ".sfr",
        ".uzf",
        ".lak",
        ".txt",
        ".csv",
        ".tsv",
        ".dat",
        ".md",
        ".yml",
        ".yaml",
        ".json",
        ".lst",
        ".log",
    }


def _pick_candidate_files(file_index: List[Dict[str, Any]], goal: str, max_files: int) -> List[str]:
    """Pick a deterministic set of likely-relevant files for editing.

    This is intentionally conservative: we prioritize model configuration and
    stress packages, and include any explicitly mentioned filenames.
    """

    goal_l = (goal or "").lower()
    # Explicit mentions: tokens that look like filenames
    mentioned: List[str] = []
    for tok in re.findall(r"[A-Za-z0-9_\-\.]+\.[A-Za-z0-9]{2,5}", goal or ""):
        mentioned.append(tok)
    mentioned_l = {m.lower() for m in mentioned}

    # Helper to score relevance.
    def score(rel: str) -> int:
        r = rel.lower()
        s = 0
        # Strong priority: sim/model nam and core packages
        if r.endswith("mfsim.nam"):
            s += 100
        if r.endswith(".nam"):
            s += 60
        if r.endswith((".tdis", ".ims", ".dis", ".disu", ".disv")):
            s += 50
        if r.endswith((".npf", ".sto", ".ic", ".oc")):
            s += 45
        if r.endswith((".wel", ".chd", ".ghb", ".riv", ".drn", ".rch", ".evt")):
            s += 40
        if r.endswith(".lst"):
            s += 25
        # Mentioned filename or substring in goal
        if Path(r).name.lower() in mentioned_l:
            s += 80
        if any(k in goal_l for k in ["layer 2", "layer2", "hk", "k33", "npf", "sto", "sy", "ss", "ims"]):
            if r.endswith((".npf", ".sto", ".ims")):
                s += 20
        # De-prioritize large misc files
        if "/run_artifacts/" in r or r.startswith("run_artifacts/"):
            s -= 200
        return s

    # Filter to text-ish files with manageable size
    candidates: List[Tuple[int, str]] = []
    for ent in file_index:
        rel = ent.get("path_rel") or ent.get("path") or ""
        if not rel:
            continue
        if not _is_text_file_rel(rel):
            continue
        # Skip artifacts folders
        if rel.replace("\\", "/").startswith("run_artifacts/"):
            continue
        candidates.append((score(rel), rel))

    candidates.sort(key=lambda x: (-x[0], x[1].lower()))
    out: List[str] = []
    for sc, rel in candidates:
        if sc <= 0 and len(out) >= 6:
            break
        out.append(rel)
        if len(out) >= max_files:
            break

    # Ensure mentioned filenames are included if present in index
    if mentioned_l:
        existing = {p.lower() for p in out}
        idx_by_name = {Path((e.get("path_rel") or "")).name.lower(): (e.get("path_rel") or "") for e in file_index}
        for name_l in mentioned_l:
            rel = idx_by_name.get(name_l)
            if rel and rel.lower() not in existing and _is_text_file_rel(rel):
                out.insert(0, rel)
                existing.add(rel.lower())
                if len(out) > max_files:
                    out = out[:max_files]
    # de-dupe preserving order
    seen = set()
    uniq: List[str] = []
    for p in out:
        pl = p.lower()
        if pl in seen:
            continue
        seen.add(pl)
        uniq.append(p)
    return uniq[:max_files]


@dataclass
class FileContext:
    path_rel: str
    sha256: str
    size: int
    truncated: bool
    content: str


def _load_file_context(ws: Path, rels: List[str], max_total_chars: int) -> List[FileContext]:
    out: List[FileContext] = []
    budget = int(max_total_chars)
    for rel in rels:
        if budget <= 0:
            break
        try:
            p = resolve_file(ws, rel)
            text, truncated, sha, size, kind = read_file_text(p, max_bytes=min(2_000_000, max(50_000, budget)))
            if kind != "text" or text is None:
                continue
            # Hard cap per file for prompt stability
            text = text[: min(len(text), 60_000)]
            budget -= len(text)
            out.append(FileContext(path_rel=rel, sha256=sha, size=size, truncated=truncated, content=text))
        except Exception:
            continue
    return out


_HUNK_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def _apply_unified_diff(original: str, diff_text: str) -> str:
    """Apply a unified diff (single-file) to original text.

    Raises ValueError if it cannot be applied cleanly.
    """

    orig_lines = original.splitlines(keepends=True)
    diff_lines = diff_text.splitlines(keepends=True)

    # Drop file header lines if present
    i = 0
    while i < len(diff_lines) and (diff_lines[i].startswith("--- ") or diff_lines[i].startswith("+++ ")):
        i += 1

    out = orig_lines[:]
    offset = 0
    while i < len(diff_lines):
        line = diff_lines[i]
        if not line.startswith("@@"):
            i += 1
            continue
        m = _HUNK_RE.match(line.rstrip("\r\n"))
        if not m:
            raise ValueError("invalid hunk header")
        old_start = int(m.group(1))
        # old_len = int(m.group(2) or "1")
        # new_start = int(m.group(3))
        i += 1
        # Convert 1-based line to 0-based index in current output
        idx = (old_start - 1) + offset
        if idx < 0:
            idx = 0
        cursor = idx

        # Apply hunk lines until next hunk or end
        while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
            hl = diff_lines[i]
            if hl.startswith(" "):
                expected = hl[1:]
                if cursor >= len(out) or out[cursor] != expected:
                    raise ValueError("context mismatch while applying patch")
                cursor += 1
            elif hl.startswith("-"):
                expected = hl[1:]
                if cursor >= len(out) or out[cursor] != expected:
                    raise ValueError("delete mismatch while applying patch")
                del out[cursor]
                offset -= 1
            elif hl.startswith("+"):
                insert = hl[1:]
                out.insert(cursor, insert)
                cursor += 1
                offset += 1
            elif hl.startswith("\\"):
                # "\\ No newline at end of file" - ignore
                pass
            else:
                # Unexpected line
                raise ValueError("invalid diff line")
            i += 1

    return "".join(out)


def _validate_single_file_diff(ws: Path, path_rel: str, diff_text: str) -> Tuple[bool, str]:
    """Validate that diff applies cleanly and stays within allowed constraints."""
    try:
        p = resolve_file(ws, path_rel)
        text, _, _, _, kind = read_file_text(p)
        if kind != "text" or text is None:
            return False, "target is not a text file"
        _ = _apply_unified_diff(text, diff_text)
        return True, "ok"
    except Exception as e:
        return False, f"cannot apply diff: {e}"


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _backup_path(ws: Path, patch_id: str, path_rel: str) -> Path:
    bdir = (ws / ".gw_copilot" / "backups" / patch_id).resolve()
    bdir.mkdir(parents=True, exist_ok=True)
    # Preserve directory structure inside backups
    out = bdir / Path(path_rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


# -----------------------------
# Routes
# -----------------------------


@router.post("/patch/plan")
def patch_plan(req: PatchPlanRequest) -> Dict[str, Any]:
    """Plan a set of safe, reviewable edits to model text files.

    Returns unified diffs per file and stores them under run_artifacts/patches/<patch_id>/.
    """

    ws = resolve_workspace_root(req.inputs_dir, req.workspace)

    # Ensure scan exists; provides file index + snapshot + health
    scan = ensure_workspace_scan(ws, force=req.force_scan)
    file_index = scan.get("file_index", []) or []

    candidates = _pick_candidate_files(file_index, req.goal, req.max_files)
    ctx_files = _load_file_context(ws, candidates, req.max_total_chars)

    # Build prompt
    system = """You are a senior groundwater modeler and software engineer.

Your task: propose a set of READ-ONLY recommendations expressed as unified diffs that a user can review and optionally apply.

Hard requirements:
- Only modify existing text files provided in context.
- Do NOT propose new files or deletions.
- Keep changes minimal and targeted.
- Output MUST be strict JSON and MUST NOT contain invalid JSON escapes.
- Diffs must be unified diff format for a single file each (no multi-file concatenation), with headers:
  --- a/<path_rel>\n  +++ b/<path_rel>
- If you are unsure, propose NO edits and explain why.

Return JSON with keys:
{
  "status": "ok" | "no_changes" | "needs_more_info",
  "notes": "...",
  "questions": ["..."],
  "edits": [
     {"path_rel": "...", "diff": "...", "rationale": "..."}
  ]
}
"""

    # Provide scan summary + file contexts
    snap = scan.get("snapshot")
    health = scan.get("health")

    files_blob_parts: List[str] = []
    for fc in ctx_files:
        files_blob_parts.append(
            f"\n### FILE: {fc.path_rel}\n# sha256: {fc.sha256} size: {fc.size} truncated: {fc.truncated}\n" + fc.content
        )
    files_blob = "\n".join(files_blob_parts)

    user = (
        f"GOAL:\n{req.goal}\n\n"
        f"WORKSPACE_SNAPSHOT (may be partial):\n{json.dumps(snap, indent=2) if snap else '{}'}\n\n"
        f"WORKSPACE_HEALTH:\n{json.dumps(health, indent=2) if health else '{}'}\n\n"
        f"AVAILABLE_FILES_CONTEXT:\n{files_blob}\n"
    )

    try:
        plan_obj = _call_llm_json_maybe(system=system, user=user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patch planner failed: {e}")

    edits = plan_obj.get("edits") if isinstance(plan_obj, dict) else None
    if not isinstance(edits, list):
        edits = []

    # Validate diffs
    valid_edits: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for ed in edits:
        if not isinstance(ed, dict):
            continue
        path_rel = str(ed.get("path_rel") or "").strip()
        diff = ed.get("diff")
        rationale = str(ed.get("rationale") or "").strip()
        if not path_rel or not isinstance(diff, str) or diff.strip() == "":
            continue
        # Basic constraints
        if not _is_text_file_rel(path_rel):
            rejected.append({"path_rel": path_rel, "reason": "unsupported file type"})
            continue
        # Ensure headers match
        if f"--- a/{path_rel}" not in diff or f"+++ b/{path_rel}" not in diff:
            rejected.append({"path_rel": path_rel, "reason": "diff headers must match path_rel"})
            continue
        ok, msg = _validate_single_file_diff(ws, path_rel, diff)
        if not ok:
            rejected.append({"path_rel": path_rel, "reason": msg})
            continue
        valid_edits.append({"path_rel": path_rel, "diff": diff, "rationale": rationale})

    patch_id = f"{_utc_now_compact()}_{os.urandom(6).hex()}"
    rdir = _run_dir(ws, patch_id)

    # Save artifacts
    _write_json(
        rdir / "plan.json",
        {
            "patch_id": patch_id,
            "goal": req.goal,
            "notes": plan_obj.get("notes") if isinstance(plan_obj, dict) else "",
            "questions": plan_obj.get("questions") if isinstance(plan_obj, dict) else [],
            "status": plan_obj.get("status") if isinstance(plan_obj, dict) else "ok",
            "edits": valid_edits,
            "rejected": rejected,
            "context_files": [fc.path_rel for fc in ctx_files],
        },
    )
    for ed in valid_edits:
        safe_name = ed["path_rel"].replace("/", "__")
        (rdir / f"diff__{safe_name}.patch").write_text(ed["diff"], encoding="utf-8")

    return {
        "status": plan_obj.get("status") if isinstance(plan_obj, dict) else "ok",
        "patch_id": patch_id,
        "notes": plan_obj.get("notes") if isinstance(plan_obj, dict) else "",
        "questions": plan_obj.get("questions") if isinstance(plan_obj, dict) else [],
        "edits": valid_edits,
        "rejected": rejected,
    }


@router.post("/patch/validate")
def patch_validate(req: PatchValidateRequest) -> Dict[str, Any]:
    ws = resolve_workspace_root(req.inputs_dir, req.workspace)
    rdir = _run_dir(ws, req.patch_id)
    plan_path = rdir / "plan.json"
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="patch plan not found")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    edits = plan.get("edits") or []
    results = []
    ok_all = True
    for ed in edits:
        path_rel = ed.get("path_rel")
        diff = ed.get("diff")
        if not isinstance(path_rel, str) or not isinstance(diff, str):
            continue
        ok, msg = _validate_single_file_diff(ws, path_rel, diff)
        results.append({"path_rel": path_rel, "ok": ok, "message": msg})
        ok_all = ok_all and ok
    return {"patch_id": req.patch_id, "ok": ok_all, "results": results}


@router.post("/patch/apply")
def patch_apply(req: PatchApplyRequest) -> Dict[str, Any]:
    ws = resolve_workspace_root(req.inputs_dir, req.workspace)
    rdir = _run_dir(ws, req.patch_id)
    plan_path = rdir / "plan.json"
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="patch plan not found")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    edits = plan.get("edits") or []

    sel = None
    if req.selections is not None:
        sel = {s.strip().replace("\\", "/") for s in req.selections if s and s.strip()}

    applied = []
    skipped = []
    for ed in edits:
        path_rel = ed.get("path_rel")
        diff = ed.get("diff")
        if not isinstance(path_rel, str) or not isinstance(diff, str):
            continue
        if sel is not None and path_rel.replace("\\", "/") not in sel:
            skipped.append({"path_rel": path_rel, "reason": "not selected"})
            continue
        try:
            p = resolve_file(ws, path_rel)
            text, _, sha, _, kind = read_file_text(p)
            if kind != "text" or text is None:
                skipped.append({"path_rel": path_rel, "reason": "not a text file"})
                continue
            new_text = _apply_unified_diff(text, diff)
            # Backup
            bpath = _backup_path(ws, req.patch_id, path_rel)
            bpath.write_text(text, encoding="utf-8")
            # Write
            p.write_text(new_text, encoding="utf-8")
            applied.append({"path_rel": path_rel, "backup": str(bpath.relative_to(ws)).replace("\\", "/"), "sha_before": sha})
        except Exception as e:
            skipped.append({"path_rel": path_rel, "reason": str(e)})

    # Record apply
    _write_json(rdir / "apply.json", {"patch_id": req.patch_id, "applied": applied, "skipped": skipped})
    return {"patch_id": req.patch_id, "applied": applied, "skipped": skipped}
