from __future__ import annotations

import json
import logging
import os
import re
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from gw.api.workspace_files import resolve_workspace_root, read_file_text, list_workspace_files
from gw.api.workspace_scan import ensure_workspace_scan
from gw.api.workspace_state import update_workspace_state, workspace_state_summary, load_workspace_state
from gw.llm.docs_retriever import search_docs
from gw.llm.read_router import llm_route_read_plan, execute_read_plan
from gw.llm.mf6_filetype_knowledge import (
    PACKAGE_PROPERTIES,
    property_to_package,
    package_property_summary,
)
from gw.api.output_probes import probe_hds, probe_cbc, extract_hds_data, extract_cbc_data

"""GW Copilot chat agent — unified LLM-first architecture.

The chat agent routes ALL user questions through the LLM with rich workspace context:

* Text files are read directly and injected into context.
* Binary output files (.hds, .cbc) are probed via FloPy for metadata (shapes,
  times, value ranges, record names) and that metadata is injected.
* The LLM has full visibility of every file in the workspace.
* The only deterministic (non-LLM) paths are:
    - validate / revalidate / suggest-fix / apply-fixes commands
    - validate vs stress validate explanation

If no LLM is configured the agent falls back to helpful deterministic guidance.
"""



# -----------------------------
# Lightweight workspace file grounding
# -----------------------------

_FILE_MENTION_RE = re.compile(
    r"(?:(?:\./|/)?[\w\-./\\]+\.(?:"
    r"dis|disv|disu|nam|nams|tdis|ims|npf|sto|oc|ic|chd|wel|riv|drn|evt|rch|ghb|lak|sfr|uzf|maw|csub|gwt|grb|"
    r"hds|cbc|lst|"
    r"csv|json|txt|py|ts|tsx|md|yaml|yml"
    r"))", re.IGNORECASE
)

def _extract_file_mentions(message: str) -> List[str]:
    """Extract probable file paths mentioned in a user message."""
    hits = []
    for m in _FILE_MENTION_RE.finditer(message or ""):
        p = m.group(0).strip().strip('"').strip("'")
        # normalize slashes
        p = p.replace("\\", "/")
        # avoid absolute paths (we only allow relative within workspace)
        if p.startswith("/") or re.match(r"^[A-Za-z]:/", p):
            continue
        if ".." in p.split("/"):
            continue
        hits.append(p)
    # de-dupe preserving order
    out=[]
    seen=set()
    for h in hits:
        if h.lower() not in seen:
            out.append(h)
            seen.add(h.lower())
    return out[:3]



def _mf6_extract_blocks(text: str, wanted: List[str]) -> Dict[str, str]:
    """Extract selected MF6 BEGIN/END blocks from a (possibly large) text file."""
    out: Dict[str, str] = {}
    if not text:
        return out
    want = {w.upper() for w in wanted}
    cur = None
    buf: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = re.match(r"^\s*BEGIN\s+([A-Z0-9_]+)", raw, re.IGNORECASE)
        if m and cur is None:
            blk = m.group(1).upper()
            if blk in want:
                cur = blk
                buf = [raw.rstrip()]
            continue
        m2 = re.match(r"^\s*END\s+([A-Z0-9_]+)", raw, re.IGNORECASE)
        if m2 and cur is not None:
            blk2 = m2.group(1).upper()
            if blk2 == cur:
                buf.append(raw.rstrip())
                out[cur] = "\n".join(buf[:4000])
            cur = None
            buf = []
            continue
        if cur is not None:
            buf.append(raw.rstrip())
            if len(buf) > 4000:
                out[cur] = "\n".join(buf[:4000]) + "\n... (truncated)"
                cur = None
                buf = []
    return out


def _deterministic_answer_from_scan(text: str, scan: Optional[Dict[str, Any]]) -> Optional[str]:
    """Answer a few high-signal model questions deterministically from the workspace scan."""
    if not scan or not isinstance(scan, dict):
        return None
    snap = scan.get("snapshot") if isinstance(scan.get("snapshot"), dict) else {}
    grid = snap.get("grid") if isinstance(snap.get("grid"), dict) else {}
    facts = snap.get("facts") if isinstance(snap.get("facts"), list) else []

    q = (text or "").strip().lower()

    if re.search(r"\bhow\s+many\s+layer(s)?\b|\bnumber\s+of\s+layer(s)?\b|\bnlay\b", q):
        nlay = grid.get("nlay")
        if isinstance(nlay, int) and nlay > 0:
            src = grid.get("discretization_file") or "snapshot"
            return f"The model has **{nlay} layers** (from {src})."
        for f in facts:
            if str(f.get('key','')).upper()=='NLAY':
                return f"The model has **{f.get('value')} layers** (from snapshot)."

    if re.search(r"\bhow\s+many\s+(stress\s+)?periods\b|\bnper\b", q):
        tdis = snap.get("tdis") if isinstance(snap.get("tdis"), dict) else {}
        nper = tdis.get("nper")
        if isinstance(nper, int) and nper > 0:
            src = tdis.get("file") or "TDIS"
            return f"The simulation defines **{nper} stress periods** (from {src})."

    return None

def _deterministic_property_answer(text: str, ws_root: Optional[Path], scan: Optional[Dict[str, Any]]) -> Optional[str]:
    """Answer property-value questions by directly reading package files.

    Handles questions like:
    - "What is the average hydraulic conductivity for layer 1?"
    - "What are the storage properties?"
    - "What is the starting head for layer 1?"
    """
    if ws_root is None or not ws_root.exists():
        return None

    q = (text or "").strip().lower()

    # -------------------------------------------------------------------
    # Map common question patterns to (property_key, layer_number|None)
    # -------------------------------------------------------------------
    prop_key: Optional[str] = None
    layer_num: Optional[int] = None

    # Extract layer number if mentioned
    layer_m = re.search(r"\blayer\s*(\d+)\b", q)
    if layer_m:
        layer_num = int(layer_m.group(1))

    # Hydraulic conductivity
    if re.search(r"\b(hydraulic\s+conductivit|hk\b|k\s+value|\bk\b\s+for\s+layer|conductivit)", q):
        prop_key = "k"
    # Vertical conductivity / K33
    elif re.search(r"\b(vertical\s+conductivit|k33|vertical\s+k\b)", q):
        prop_key = "k33"
    # Specific storage
    elif re.search(r"\b(specific\s+storage|ss\b)", q) and "steady" not in q:
        prop_key = "ss"
    # Specific yield
    elif re.search(r"\b(specific\s+yield|sy\b)", q):
        prop_key = "sy"
    # Starting head / initial head
    elif re.search(r"\b(starting\s+head|initial\s+head|strt\b|initial\s+condition)", q):
        prop_key = "strt"
    # Cell type
    elif re.search(r"\b(cell\s*type|icelltype)\b", q):
        prop_key = "icelltype"

    if prop_key is None:
        return None

    # Resolve property → package → file
    pkg_info = property_to_package(prop_key)
    if pkg_info is None:
        return None
    pkg_type, arr_name, label = pkg_info

    # Get the package file path from snapshot
    snapshot = (scan or {}).get("snapshot", {}) if isinstance(scan, dict) else {}
    pkg_files = snapshot.get("package_files", {}) if isinstance(snapshot, dict) else {}
    grid = snapshot.get("grid", {}) if isinstance(snapshot, dict) else {}

    # Also check packages dict (which maps to filenames, not full paths)
    packages = snapshot.get("packages", {}) if isinstance(snapshot, dict) else {}

    pkg_file_rel = pkg_files.get(pkg_type.upper()) or packages.get(pkg_type.upper())
    if not pkg_file_rel:
        # Try finding by extension
        pkg_prop_info = PACKAGE_PROPERTIES.get(pkg_type.upper())
        if pkg_prop_info:
            for p in ws_root.rglob(f"*{pkg_prop_info.file_ext}"):
                try:
                    p.relative_to(ws_root.resolve())
                    pkg_file_rel = str(p.relative_to(ws_root).as_posix())
                    break
                except Exception:
                    continue
    if not pkg_file_rel:
        return None

    pkg_path = (ws_root / pkg_file_rel).resolve()
    try:
        pkg_path.relative_to(ws_root.resolve())
    except Exception:
        return None
    if not pkg_path.exists() or not pkg_path.is_file():
        return None

    # Parse the GRIDDATA block
    nlay = int(grid.get("nlay") or 1)
    nrow = int(grid.get("nrow") or 1)
    ncol = int(grid.get("ncol") or 1)

    try:
        from gw.api.viz import _parse_package_griddata
        arrays = _parse_package_griddata(ws_root, pkg_path, nlay, nrow, ncol)
    except Exception as e:
        return f"I found the file `{pkg_file_rel}` but couldn't parse the {arr_name} array: {type(e).__name__}: {e}"

    if arr_name not in arrays:
        return f"The file `{pkg_file_rel}` was parsed but the **{arr_name}** array was not found in its GRIDDATA block."

    arr = arrays[arr_name]
    expected = nlay * nrow * ncol

    # If user asked for a specific layer
    if layer_num is not None:
        if layer_num < 1 or layer_num > nlay:
            return f"Layer {layer_num} is out of range. The model has **{nlay} layers** (1 to {nlay})."

        if arr.size == expected:
            layer_arr = arr.reshape((nlay, nrow, ncol))[layer_num - 1].ravel()
        elif arr.size == nrow * ncol:
            layer_arr = arr.reshape((nrow, ncol)).ravel()
        else:
            layer_arr = arr

        import numpy as np
        finite = layer_arr[np.isfinite(layer_arr)]
        if finite.size == 0:
            return f"Layer {layer_num} **{label}** has no finite values (from `{pkg_file_rel}`)."

        mn = float(np.min(finite))
        mx = float(np.max(finite))
        mean = float(np.mean(finite))
        median = float(np.median(finite))

        if abs(mn - mx) < 1e-12:
            return (
                f"The **{label}** for layer {layer_num} is **{mean:g}** "
                f"(uniform/CONSTANT value across all {finite.size} cells, from `{pkg_file_rel}`)."
            )
        return (
            f"**{label}** statistics for layer {layer_num} (from `{pkg_file_rel}`):\n"
            f"- Mean: **{mean:g}**\n"
            f"- Median: {median:g}\n"
            f"- Min: {mn:g}\n"
            f"- Max: {mx:g}\n"
            f"- Cell count: {finite.size}"
        )

    # No specific layer - give summary across all layers
    import numpy as np
    lines = [f"**{label}** summary (from `{pkg_file_rel}`):"]
    if arr.size == expected:
        arr_3d = arr.reshape((nlay, nrow, ncol))
        for k in range(nlay):
            layer_vals = arr_3d[k].ravel()
            finite = layer_vals[np.isfinite(layer_vals)]
            if finite.size == 0:
                lines.append(f"- Layer {k+1}: no finite values")
            elif abs(float(np.min(finite)) - float(np.max(finite))) < 1e-12:
                lines.append(f"- Layer {k+1}: **{float(np.mean(finite)):g}** (uniform)")
            else:
                lines.append(f"- Layer {k+1}: mean={float(np.mean(finite)):g}, min={float(np.min(finite)):g}, max={float(np.max(finite)):g}")
    elif arr.size == nrow * ncol:
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            lines.append(f"- Single-layer array: mean={float(np.mean(finite)):g}, min={float(np.min(finite)):g}, max={float(np.max(finite)):g}")
    else:
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            lines.append(f"- Overall: mean={float(np.mean(finite)):g}, count={finite.size}")

    return "\n".join(lines)


def _maybe_attach_workspace_files(ctx: Dict[str, Any], *, inputs_dir: str, workspace: str | None, message: str) -> None:
    """Attach relevant file snippets when user explicitly references files.

    For text files: reads content directly (with MF6 block extraction for large files).
    For binary files (.hds, .cbc): probes via FloPy for metadata (times, shapes, records).
    For other binary files: reports size and kind.
    """
    try:
        ws_root = resolve_workspace_root(inputs_dir, workspace)
    except Exception:
        return
    if not ws_root.exists():
        return

    mentions = _extract_file_mentions(message)
    if not mentions:
        return

    attached=[]
    for rel in mentions:
        p = (ws_root / rel).resolve()
        # ensure still under workspace
        try:
            p.relative_to(ws_root.resolve())
        except Exception:
            continue
        if not p.exists() or not p.is_file():
            continue

        ext = p.suffix.lower()

        # Binary output files: extract actual data via FloPy
        if ext == ".hds":
            result = extract_hds_data(ws_root, rel, max_chars=40_000)
            if result.get("ok") and result.get("summary_text"):
                attached.append({"path": rel, "kind": "binary_hds_extracted", "content": result["summary_text"], "metadata": result.get("metadata", {})})
            else:
                attached.append({"path": rel, "kind": "binary_hds", "probe": probe_hds(ws_root, rel)})
            continue
        if ext == ".cbc":
            result = extract_cbc_data(ws_root, rel, max_chars=30_000)
            if result.get("ok") and result.get("summary_text"):
                attached.append({"path": rel, "kind": "binary_cbc_extracted", "content": result["summary_text"], "metadata": result.get("metadata", {})})
            else:
                attached.append({"path": rel, "kind": "binary_cbc", "probe": probe_cbc(ws_root, rel)})
            continue

        try:
            text, truncated, sha, size, kind = read_file_text(p, max_bytes=200_000)
        except Exception:
            continue
        if kind != "text" or text is None:
            attached.append({"path": rel, "kind": kind, "size": size, "sha256": sha, "note": "binary or unreadable"})
            continue
        # If the file is large/truncated and looks like MF6 text, extract the most relevant blocks.
        blocks = {}
        try:
            if bool(truncated) or (size and size > 200_000):
                wanted = []
                q = (message or "").lower()
                if ext in (".dis", ".disv", ".disu") and ("layer" in q or "nlay" in q or "layers" in q):
                    wanted = ["DIMENSIONS", "GRIDDATA"]
                elif ext == ".nam":
                    wanted = ["OPTIONS", "TIMING", "EXCHANGES", "MODELS"]
                elif ext in (".wel", ".chd", ".ghb", ".riv", ".drn", ".rcha", ".evt"):
                    wanted = ["OPTIONS", "PACKAGEDATA", "PERIOD"]
                if wanted:
                    blocks = _mf6_extract_blocks(text, wanted)
        except Exception:
            blocks = {}

        if blocks:
            # Replace full content with extracted blocks to stay within token limits.
            text = "\n\n".join(blocks.values())
            truncated = True

        attached.append({
            "path": rel,
            "kind": kind,
            "size": size,
            "sha256": sha,
            "truncated": bool(truncated),
            "content": text,
            "blocks": blocks,
        })
    if attached:
        ctx["workspace_files_referenced"] = attached


# -----------------------------
# Deterministic spine: policy
# -----------------------------


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str
    confirmations_needed: List[str]




def _maybe_attach_docs(ctx: Dict[str, Any], *, inputs_dir: str, workspace: str | None, message: str) -> None:
    """Attach retrieved documentation snippets (if docs folder exists).

    Users can add docs under:
      - <workspace>/docs
      - <workspace>/documentation
    Recommended: include MODFLOW 6 and FloPy docs (markdown/html/text) plus any GW Copilot rules.
    """
    try:
        ws_root = resolve_workspace_root(inputs_dir, workspace)
    except Exception:
        return

    # Only retrieve when query looks "knowledge-seeking" (simple heuristic),
    # and avoid doing it for very short messages.
    q = (message or "").strip()
    if len(q) < 12:
        return

    docs_roots = [ws_root / "docs", ws_root / "documentation"]
    docs_root = next((p for p in docs_roots if p.exists() and p.is_dir()), None)
    if not docs_root:
        return

    try:
        hits = search_docs(ws_root, docs_root, q, top_k=5)
    except Exception:
        hits = []

    if not hits:
        return

    ctx["retrieved_docs"] = [
        {"source": h.source, "title": h.title, "snippet": h.snippet, "score": h.score} for h in hits
    ]

def _gate_action(action: str, *, severity: str) -> GateDecision:
    """A tiny deterministic gate for chat-suggested actions.

    * safe: can be suggested without extra confirmation
    * caution: suggest but remind user to review
    * manual: suggest ONLY, and require explicit user confirmation text
    """

    severity = (severity or "").lower().strip()
    if severity not in {"safe", "caution", "manual"}:
        severity = "caution"

    if severity == "manual":
        return GateDecision(
            allowed=True,
            reason=(
                "This action is marked 'manual' and must be explicitly confirmed "
                "by the user before execution."
            ),
            confirmations_needed=[
                f"confirm:{action}",
                "confirm:reviewed_changes",
            ],
        )

    if severity == "caution":
        return GateDecision(
            allowed=True,
            reason="This action is 'caution' — review outputs before applying.",
            confirmations_needed=[],
        )

    return GateDecision(
        allowed=True,
        reason="This action is 'safe'.",
        confirmations_needed=[],
    )


# -----------------------------
# Local deterministic helpers
# -----------------------------


_CMD_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*validate\s*$", re.I), "validate"),
    (re.compile(r"^\s*revalidate\s*$", re.I), "revalidate"),
    (re.compile(r"^\s*suggest[-_ ]?fix(es)?\s*$", re.I), "suggest-fix"),
    (re.compile(r"^\s*apply[-_ ]?fix(es)?\s*$", re.I), "apply-fixes"),
]


# -----------------------------
# vNext-3: "recommendations -> patch" helper
# -----------------------------


# Heuristic: if a user asks to *improve* / *adjust* / *tune* the model (solver, sensitivity, calibration,
# stability, etc.), we automatically generate a reviewable patch plan (unified diffs) instead of only
# returning prose. This keeps the deterministic spine intact: no files are modified until the user applies.
_PATCH_INTENT_RE = re.compile(
    r"\b("
    r"recommend|recommendations|improve|improvement|optimi[sz]e|tune|tighten|loosen|"
    r"robust|robustness|stabil|converge|convergence|diverg|solver|ims|"
    r"sensitivity|calibrat|parameteriz|parameterise|adjust|refine|"
    r"reduce\s+error|fix\s+convergence|make\s+it\s+run"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_patch_request(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 18:
        return False
    # Don't treat basic "what is this file" questions as patch requests.
    if re.search(r"\b(what\s+is|explain|describe)\b", t, re.IGNORECASE) and _FILE_MENTION_RE.search(t):
        return False
    return bool(_PATCH_INTENT_RE.search(t))


def _format_patch_plan_reply(plan: dict) -> str:
    status = str(plan.get("status") or "ok")
    patch_id = str(plan.get("patch_id") or "")
    notes = str(plan.get("notes") or "").strip()
    questions = plan.get("questions") if isinstance(plan.get("questions"), list) else []
    edits = plan.get("edits") if isinstance(plan.get("edits"), list) else []

    lines: List[str] = []
    if status == "needs_more_info":
        lines.append("I can propose an exact, reviewable patch, but I need a bit more info first.")
    elif status == "no_changes":
        lines.append("I didn’t propose any file edits yet (nothing confident/targeted enough to change).")
    else:
        lines.append("I generated a reviewable patch plan (no files modified).")

    if patch_id:
        lines.append(f"\nPatch ID: {patch_id}")
        lines.append("You can validate/apply it from the UI (Patches) or via API: /patch/validate then /patch/apply.")

    if notes:
        lines.append(f"\nNotes:\n{notes}")

    if questions:
        lines.append("\nQuestions:")
        for q in questions[:6]:
            lines.append(f"- {q}")

    if edits:
        lines.append("\nProposed edits:")
        for ed in edits[:10]:
            path_rel = str(ed.get("path_rel") or "")
            rationale = str(ed.get("rationale") or "").strip()
            lines.append(f"- {path_rel}" + (f": {rationale}" if rationale else ""))
        if len(edits) > 10:
            lines.append(f"(and {len(edits) - 10} more)")
    else:
        lines.append("\nNo diffs were proposed.")

    return "\n".join(lines).strip()


def _ims_convergence_best_practices_note() -> str:
    """Compact, practical checklist for MF6 IMS convergence robustness.

    This is used as an answer-first fallback when a patch plan can't be
    confidently generated (e.g., missing context or planner returned no edits).
    """

    return (
        "Here are concrete, non-physical ways to improve convergence robustness in MODFLOW 6 (IMS-only):\n\n"
        "**1) Increase iteration headroom**\n"
        "- Increase `OUTER_MAXIMUM` and/or `INNER_MAXIMUM` (gives the solver more chances to converge).\n"
        "- If you do this, also consider tightening `OUTER_DVCLOSE`/`INNER_DVCLOSE` so you’re not just looping longer.\n\n"
        "**2) Tighten / rationalize convergence criteria**\n"
        "- Use consistent `*_DVCLOSE` and `*_RCLOSE` settings appropriate for your units and expected head changes.\n"
        "- A common failure mode is criteria that are too tight for early iterations (causing non-convergence) *or*\n"
        "  too loose (masking instability).\n\n"
        "**3) Use under-relaxation deliberately (when needed)**\n"
        "- Enable/configure under-relaxation to damp oscillations, especially with strong stresses or nonlinear packages.\n"
        "- If the model oscillates (residuals bounce), more damping can help; if it stalls, reduce damping.\n\n"
        "**4) Choose a stable linear acceleration for the problem**\n"
        "- PCG can be robust for many models; BICGSTAB can be faster but may be less forgiving in some cases.\n"
        "- If you have frequent non-convergence, try switching linear acceleration and compare.\n\n"
        "**5) Improve diagnostics (read-only changes)**\n"
        "- Add/clarify comments in the `.ims` file describing why each choice was made.\n"
        "- Ensure listing output provides enough info to see *where* it fails (outer vs inner, residual behavior).\n\n"
        "If you want, I can generate an exact, reviewable patch that edits the active IMS file used by `mfsim.nam` "
        "and only touches solver settings + comments (no physical parameters)."
    )


def _plan_patch_from_chat(*, inputs_dir: str, workspace: str | None, goal: str) -> dict:
    """Create a patch plan deterministically via the same backend planner used by /patch/plan.

    This does NOT modify any files. It writes artifacts under run_artifacts/patches/<patch_id>/.
    """
    from gw.api.patches import PatchPlanRequest, patch_plan

    # Keep context bounded for chat latency/stability.
    req = PatchPlanRequest(
        inputs_dir=inputs_dir,
        workspace=workspace,
        goal=goal,
        force_scan=False,
        max_files=18,
        max_total_chars=120_000,
    )
    return patch_plan(req)


def _detect_direct_command(text: str) -> Optional[str]:
    for pat, cmd in _CMD_PATTERNS:
        if pat.match(text or ""):
            return cmd
    return None


def _explain_validate_vs_stress_validate() -> str:
    return (
        "Here’s the practical difference in *this project*:\n\n"
        "- **validate / validate-stresses**: checks *input stress CSVs* (wells/chd/recharge/etc.) "
        "against the grid/idomain/time discretization. It answers: ‘Are my boundary/pumping/recharge inputs valid?’\n"
        "- **revalidate**: re-runs the same deterministic stress validation and writes artifacts + (optional) diffs "
        "vs a previous `stress_validation.json`. It answers: ‘After I changed inputs/config, did the issues improve?’\n\n"
        "Separately, *model execution* checks are things like **materialize-mf6** and **run-mf6** (write MF6 inputs, run MF6, parse listing, percent discrepancy, etc.)."
    )


def _read_project_context(inputs_dir: Optional[str]) -> Dict[str, Any]:
    """Lightweight context for the assistant.

    This intentionally avoids heavy IO. We only try to read config + a couple of
    recent artifacts if present.
    """

    ctx: Dict[str, Any] = {}
    if not inputs_dir:
        return ctx

    base = Path(inputs_dir)
    if not base.exists():
        return ctx

    cfg_path = base / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            ctx["config"] = {
                "model_name": cfg.get("model_name"),
                "workspace": cfg.get("workspace"),
                "grid": cfg.get("grid"),
                "periods": cfg.get("periods"),
                "inputs": cfg.get("inputs"),
            }
        except Exception:
            pass

    # try to attach summary stats from latest artifacts if available
    ws = (ctx.get("config") or {}).get("workspace")
    if ws:
        artifacts = Path(ws) / "run_artifacts"
        if artifacts.exists():
            stress_json = artifacts / "stress_validation.json"
            if stress_json.exists():
                try:
                    sj = json.loads(stress_json.read_text(encoding="utf-8"))
                    # don't include huge arrays; just counts
                    ctx["stress_validation"] = {
                        "errors": sj.get("counts", {}).get("errors"),
                        "warnings": sj.get("counts", {}).get("warnings"),
                        "info": sj.get("counts", {}).get("info"),
                    }
                except Exception:
                    pass
    return ctx


# -----------------------------
# LLM Provider Integration
# -----------------------------


def _openai_client():
    """Create OpenAI client if available and configured."""
    # Check config first, then env
    try:
        from gw.api.llm_config import get_api_key
        api_key = get_api_key("openai")
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _anthropic_client():
    """Create Anthropic client if available and configured."""
    # Check config first, then env
    try:
        from gw.api.llm_config import get_api_key
        api_key = get_api_key("anthropic")
    except Exception:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        return None
    try:
        import anthropic  # type: ignore
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def _get_active_provider() -> str:
    """Get the currently configured provider."""
    try:
        from gw.api.llm_config import get_active_provider
        return get_active_provider()
    except Exception:
        return "openai"


def _get_configured_model() -> Optional[str]:
    """Get the configured model name."""
    try:
        from gw.api.llm_config import get_model
        return get_model()
    except Exception:
        return None


def _model_looks_anthropic(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("claude") or "anthropic" in m

def _model_looks_openai(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("gpt-") or m.startswith("o") or m.startswith("chatgpt") or m.startswith("text-")

def _select_model_for_provider(provider: str, configured_model: str | None, env_default: str) -> str:
    """Choose a model name that matches the provider.

    Users sometimes paste a Claude model name while the provider is OpenAI (or vice versa).
    In that case we ignore the mismatched configured model instead of hard-failing mid-chat.
    """
    if provider == "openai":
        if configured_model and _model_looks_anthropic(configured_model):
            return env_default
        return configured_model or env_default
    if provider == "anthropic":
        if configured_model and _model_looks_openai(configured_model):
            return "claude-sonnet-4-20250514"
        return configured_model or "claude-sonnet-4-20250514"
    return configured_model or env_default

def _looks_like_model_not_found(err: Exception) -> bool:
    s = str(err).lower()
    return ("model_not_found" in s) or ("does not exist" in s and "model" in s)


def _llm_reply(
    *,
    messages: List[Dict[str, str]],
    inputs_dir: Optional[str],
    workspace: Optional[str] = None,
    message: str = "",
    model: Optional[str] = None,
    extra_ctx: Optional[Dict[str, Any]] = None,
    ws_root: Optional[Path] = None,
    enable_tools: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Generate an LLM reply, optionally with tool-use loop.

    Returns (reply_text, tool_audit_log).
    When enable_tools=True and ws_root is available, the agentic tool loop
    lets the LLM request additional file reads, binary extraction, and plot
    generation mid-conversation. Falls back to single-shot if tools fail.
    """
    provider = _get_active_provider()

    # Try tool-use loop first when enabled and workspace is available
    if enable_tools and ws_root is not None and inputs_dir:
        try:
            from gw.llm.tool_loop import tool_loop

            ctx, model_facts_text = _build_context(inputs_dir, workspace, message, extra_ctx)
            system = _build_system_prompt(ctx, model_facts_text)

            text, audit = tool_loop(
                messages=messages,
                system=system,
                ws_root=ws_root,
                inputs_dir=inputs_dir,
                workspace=workspace,
                provider=provider,
                model=model,
            )
            return text, audit
        except Exception as tool_err:
            logger.warning(
                "Tool loop failed, falling back to single-shot: %s: %s",
                type(tool_err).__name__, tool_err,
            )
            # Fall through to single-shot

    # Single-shot fallback (no tools)
    def _try_anthropic():
        return _anthropic_reply(
            messages=messages,
            inputs_dir=inputs_dir,
            workspace=workspace,
            message=message,
            model=model,
            extra_ctx=extra_ctx,
        )

    def _try_openai():
        return _openai_reply(
            messages=messages,
            inputs_dir=inputs_dir,
            workspace=workspace,
            message=message,
            model=model,
            extra_ctx=extra_ctx,
        )

    if provider == "anthropic":
        try:
            return _try_anthropic(), []
        except RuntimeError:
            # Fallback to OpenAI if Anthropic package/key unavailable
            try:
                return _try_openai(), []
            except Exception:
                pass
            raise
    else:
        try:
            return _try_openai(), []
        except RuntimeError:
            # Fallback to Anthropic if OpenAI package/key unavailable
            try:
                return _try_anthropic(), []
            except Exception:
                pass
            raise


def _try_compute_location_from_config(inputs_dir: str) -> Optional[Dict[str, Any]]:
    """Fallback: read spatial_ref directly from GW_Copilot/config.json and compute centroid.

    This is a self-contained fallback that doesn't depend on viz.py's grid loading.
    It uses the spatial_ref origin as an approximate centroid (close enough for
    geographic region identification even without the full grid extent).
    """
    import math as _math

    # Find config
    for candidate in [
        Path(inputs_dir) / "GW_Copilot" / "config.json",
        Path(inputs_dir).parent / "GW_Copilot" / "config.json",
    ]:
        if candidate.exists():
            cfg_path = candidate
            break
    else:
        logger.debug("_try_compute_location_from_config: no config.json found")
        return None

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Check for existing location_context first
    loc = cfg.get("location_context")
    if loc and loc.get("centroid_lat") is not None:
        return loc

    sr = cfg.get("spatial_ref")
    if not sr or not sr.get("epsg"):
        return None

    epsg = sr["epsg"]
    xorigin = float(sr.get("xorigin", 0))
    yorigin = float(sr.get("yorigin", 0))
    crs_name = sr.get("crs_name", f"EPSG:{epsg}")

    # Use origin directly as approximate centroid (for region identification this is fine)
    wx, wy = xorigin, yorigin

    logger.info("_try_compute_location_from_config: EPSG:%s origin=(%.1f, %.1f)", epsg, wx, wy)

    # Try pyproj first
    try:
        from pyproj import Transformer  # type: ignore
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(wx, wy)
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            logger.info("_try_compute_location_from_config: pyproj result: lat=%.6f lon=%.6f", lat, lon)
            result = {"centroid_lat": round(lat, 6), "centroid_lon": round(lon, 6),
                      "epsg": epsg, "crs_name": crs_name}
            # Persist
            try:
                cfg["location_context"] = result
                cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            except Exception:
                pass
            return result
    except Exception:
        pass

    # Fallback: get proj4 string (built-in dict or epsg.io with proper User-Agent)
    try:
        from gw.api.viz import _fetch_proj4_string
        proj_str = _fetch_proj4_string(epsg)
        if not proj_str:
            logger.warning("_try_compute_location_from_config: no proj4 string for EPSG:%s", epsg)
            return None

        # Parse proj4 params
        params: Dict[str, str] = {}
        for token in proj_str.split():
            if token.startswith("+") and "=" in token:
                k, v = token[1:].split("=", 1)
                params[k] = v

        proj_type = params.get("proj", "")
        units = params.get("units", "m")

        # Convert to meters
        x_m, y_m = wx, wy
        if units == "us-ft":
            x_m = wx * 0.3048006096012192
            y_m = wy * 0.3048006096012192
        elif units == "ft":
            x_m = wx * 0.3048
            y_m = wy * 0.3048

        lat_deg, lon_deg = None, None

        if proj_type == "lcc":
            # Lambert Conformal Conic — full inverse
            lat_0 = _math.radians(float(params.get("lat_0", "0")))
            lon_0 = _math.radians(float(params.get("lon_0", "0")))
            lat_1 = _math.radians(float(params.get("lat_1", params.get("lat_0", "0"))))
            lat_2 = _math.radians(float(params.get("lat_2", params.get("lat_1", params.get("lat_0", "0")))))
            x_0 = float(params.get("x_0", "0"))
            y_0 = float(params.get("y_0", "0"))

            x_m -= x_0
            y_m -= y_0

            a_e = 6378137.0
            f_e = 1 / 298.257222101
            if params.get("ellps", "").upper() in ("WGS84",):
                f_e = 1 / 298.257223563
            e2 = 2 * f_e - f_e ** 2
            e = _math.sqrt(e2)

            def _m(phi):
                return _math.cos(phi) / _math.sqrt(1 - e2 * _math.sin(phi) ** 2)
            def _t(phi):
                sp = e * _math.sin(phi)
                return _math.tan(_math.pi / 4 - phi / 2) / ((1 - sp) / (1 + sp)) ** (e / 2)

            m1, m2 = _m(lat_1), _m(lat_2)
            t0, t1, t2 = _t(lat_0), _t(lat_1), _t(lat_2)

            n = (_math.log(m1) - _math.log(m2)) / (_math.log(t1) - _math.log(t2)) if abs(lat_1 - lat_2) > 1e-10 else _math.sin(lat_1)
            F = m1 / (n * t1 ** n)
            rho0 = a_e * F * t0 ** n

            rs = 1 if n > 0 else -1
            xs, ys = rs * x_m, rs * (rho0 - y_m)
            rho = rs * _math.sqrt(xs ** 2 + ys ** 2)
            theta = _math.atan2(xs, ys)

            lon_deg = _math.degrees(theta / n + lon_0)
            if abs(rho) < 1e-10:
                lat_deg = _math.degrees(_math.copysign(_math.pi / 2, n))
            else:
                t = (rho / (a_e * F)) ** (1 / n)
                phi = _math.pi / 2 - 2 * _math.atan(t)
                for _ in range(10):
                    sp = e * _math.sin(phi)
                    phi_new = _math.pi / 2 - 2 * _math.atan(t * ((1 - sp) / (1 + sp)) ** (e / 2))
                    if abs(phi_new - phi) < 1e-12:
                        break
                    phi = phi_new
                lat_deg = _math.degrees(phi)

        elif proj_type == "tmerc":
            lat_0 = _math.radians(float(params.get("lat_0", "0")))
            lon_0 = _math.radians(float(params.get("lon_0", "0")))
            k_0 = float(params.get("k", params.get("k_0", "1")))
            x_0 = float(params.get("x_0", "0"))
            y_0 = float(params.get("y_0", "0"))
            x_m -= x_0
            y_m -= y_0

            a_e = 6378137.0
            f_e = 1 / 298.257222101
            e2 = 2 * f_e - f_e ** 2

            M = y_m / k_0
            mu = M / (a_e * (1 - e2 / 4 - 3 * e2 ** 2 / 64))
            e1 = (1 - _math.sqrt(1 - e2)) / (1 + _math.sqrt(1 - e2))
            phi1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * _math.sin(2 * mu) + \
                   (21 * e1 ** 2 / 16) * _math.sin(4 * mu)
            sp1, cp1, tp1 = _math.sin(phi1), _math.cos(phi1), _math.tan(phi1)
            N1 = a_e / _math.sqrt(1 - e2 * sp1 ** 2)
            R1 = a_e * (1 - e2) / ((1 - e2 * sp1 ** 2) ** 1.5)
            D = x_m / (N1 * k_0)
            lat_deg = _math.degrees(phi1 - (N1 * tp1 / R1) * D ** 2 / 2)
            lon_deg = _math.degrees(lon_0 + D / cp1)

        elif proj_type == "utm":
            zone = int(params.get("zone", "0"))
            south = "+south" in proj_str
            if 1 <= zone <= 60:
                k0 = 0.9996
                a_e = 6378137.0
                f_e = 1 / 298.257223563
                e2 = 2 * f_e - f_e ** 2
                e_p2 = e2 / (1 - e2)
                lon0 = _math.radians((zone - 1) * 6 - 180 + 3)
                x_m -= 500000.0
                if south:
                    y_m -= 10000000.0
                M = y_m / k0
                mu = M / (a_e * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
                e1 = (1 - _math.sqrt(1 - e2)) / (1 + _math.sqrt(1 - e2))
                phi1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * _math.sin(2 * mu) + \
                       (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * _math.sin(4 * mu) + \
                       (151 * e1 ** 3 / 96) * _math.sin(6 * mu)
                sp1, cp1, tp1 = _math.sin(phi1), _math.cos(phi1), _math.tan(phi1)
                N1 = a_e / _math.sqrt(1 - e2 * sp1 ** 2)
                T1 = tp1 ** 2
                C1 = e_p2 * cp1 ** 2
                R1 = a_e * (1 - e2) / ((1 - e2 * sp1 ** 2) ** 1.5)
                D = x_m / (N1 * k0)
                lat_deg = _math.degrees(
                    phi1 - (N1 * tp1 / R1) * (
                        D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * e_p2) * D ** 4 / 24 +
                        (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * e_p2 - 3 * C1 ** 2) * D ** 6 / 720
                    )
                )
                lon_deg = _math.degrees(
                    lon0 + (D - (1 + 2 * T1 + C1) * D ** 3 / 6 +
                            (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * e_p2 + 24 * T1 ** 2) * D ** 5 / 120) / cp1
                )

        elif proj_type == "longlat":
            if -180 <= wx <= 180 and -90 <= wy <= 90:
                lat_deg, lon_deg = wy, wx

        if lat_deg is not None and lon_deg is not None:
            if -180 <= lon_deg <= 180 and -90 <= lat_deg <= 90:
                logger.info("_try_compute_location_from_config: computed lat=%.6f lon=%.6f via %s",
                           lat_deg, lon_deg, proj_type)
                result = {"centroid_lat": round(lat_deg, 6), "centroid_lon": round(lon_deg, 6),
                          "epsg": epsg, "crs_name": crs_name}
                try:
                    cfg["location_context"] = result
                    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                except Exception:
                    pass
                return result

    except Exception as exc:
        logger.warning("_try_compute_location_from_config: fallback failed: %s: %s",
                      type(exc).__name__, exc)

    return None


def _build_context(
    inputs_dir: Optional[str],
    workspace: Optional[str],
    message: str,
    extra_ctx: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    """Build context and model facts for LLM calls."""
    ctx = _read_project_context(inputs_dir)
    
    # Router/executor can provide additional bounded reads
    if extra_ctx and isinstance(extra_ctx, dict):
        try:
            ctx.update(extra_ctx)
        except Exception:
            pass

    # Model facts section
    model_facts_text = ""

    try:
        if inputs_dir:
            ws_root = resolve_workspace_root(inputs_dir, workspace)

            # Load location context from GW_Copilot/config.json if available
            loc_ctx = None
            try:
                from gw.api.viz import load_location_context
                loc_ctx = load_location_context(inputs_dir)
                logger.info("_build_context: load_location_context returned: %s", loc_ctx)
            except Exception as exc:
                logger.warning("_build_context: load_location_context failed: %s: %s",
                              type(exc).__name__, exc)

            # Fallback: if load_location_context didn't return a result, try reading
            # the config.json directly and computing the centroid right here.
            if not loc_ctx or loc_ctx.get("centroid_lat") is None:
                loc_ctx = _try_compute_location_from_config(inputs_dir)

            if loc_ctx and loc_ctx.get("centroid_lat") is not None:
                lat = loc_ctx["centroid_lat"]
                lon = loc_ctx["centroid_lon"]
                crs = loc_ctx.get("crs_name", "")
                ns = "N" if lat >= 0 else "S"
                ew = "W" if lon < 0 else "E"
                loc_text = (
                    "MODEL GEOGRAPHIC LOCATION (IMPORTANT — you KNOW where this model is):\n"
                    f"- Centroid coordinates: {abs(lat):.4f}\u00b0{ns}, {abs(lon):.4f}\u00b0{ew}\n"
                    f"- Coordinate reference system: {crs}\n"
                    "- YOU HAVE THIS MODEL'S REAL-WORLD LOCATION. When the user asks about\n"
                    "  the model's location, geology, hydrogeology, or site-specific conditions,\n"
                    "  USE these coordinates to provide specific information:\n"
                    "  * Identify the geographic region, state/province, and nearby cities\n"
                    "  * Describe the local geology and aquifer systems typical for this area\n"
                    "  * Reference typical hydraulic properties (K, Ss, Sy) for local formations\n"
                    "  * Mention relevant regulatory bodies and groundwater management frameworks\n"
                    "  * Note climate conditions (arid/humid, precipitation, recharge patterns)\n"
                    "  * Suggest relevant USGS, state survey, or local data sources\n"
                    "  * Compare model parameters against expected values for the region\n"
                    "- Do NOT say you don't know the location or ask the user for it.\n"
                    "  The coordinates above ARE the model's real-world position.\n"
                )
                ctx["location_context"] = loc_ctx
                model_facts_text = loc_text + "\n"
                logger.info("_build_context: location text injected: %.4f deg %s, %.4f deg %s (%s)",
                           abs(lat), ns, abs(lon), ew, crs)
            else:
                logger.warning("_build_context: no location context available for inputs_dir=%s",
                              inputs_dir)

            try:
                scan = ensure_workspace_scan(ws_root, force=False)

                snapshot = scan.get("snapshot", {}) if isinstance(scan, dict) else {}
                facts = snapshot.get("facts", []) if isinstance(snapshot, dict) else []

                if facts:
                    fact_lines = []
                    for f in facts:
                        if isinstance(f, dict):
                            key = f.get("key", "")
                            value = f.get("value", "")
                            if key and value is not None:
                                fact_lines.append(f"- {key}: {value}")
                    if fact_lines:
                        model_facts_text += "MODEL FACTS (from workspace scan):\n" + "\n".join(fact_lines)
                
                ctx["model_snapshot"] = {
                    "ok": snapshot.get("ok"),
                    "extraction_method": snapshot.get("extraction_method"),
                    "grid": snapshot.get("grid"),
                    "tdis": snapshot.get("tdis"),
                    "packages": list(snapshot.get("packages", {}).keys()) if isinstance(snapshot.get("packages"), dict) else [],
                    "outputs_present": snapshot.get("outputs_present"),
                }
                ctx["workspace_health"] = scan.get("health") if isinstance(scan, dict) else None
            except Exception:
                pass

            try:
                state = load_workspace_state(ws_root)
                ctx["workspace_state"] = state
                summary = workspace_state_summary(state)
                if summary:
                    ctx["workspace_state_summary"] = summary
            except Exception:
                pass

            try:
                from gw.api.model_profile import get_model_profile
                profile = get_model_profile(ws_root)
                cross_pkg = profile.get("cross_package_analysis")
                if cross_pkg:
                    ctx["cross_package_analysis"] = cross_pkg
            except Exception:
                pass

            try:
                files, truncated = list_workspace_files(ws_root, max_files=300, include_hash=False)
                # Classify files by type for the LLM
                text_files = []
                binary_output_files = []
                other_files = []
                for f in files:
                    ext = (f.path_rel.rsplit(".", 1)[-1].lower() if "." in f.path_rel else "")
                    if ext in ("hds", "cbc", "grb"):
                        binary_output_files.append(f.path_rel)
                    elif ext in ("dis", "disv", "disu", "tdis", "ims", "nam", "npf",
                                 "sto", "ic", "oc", "wel", "chd", "ghb", "riv", "drn",
                                 "rch", "evt", "uzf", "maw", "lak", "sfr", "csub",
                                 "csv", "json", "txt", "py", "md", "yaml", "yml", "lst"):
                        text_files.append(f.path_rel)
                    else:
                        other_files.append(f.path_rel)

                ctx["workspace_file_index"] = {
                    "total_count": len(files),
                    "truncated": bool(truncated),
                    "text_files": text_files[:150],
                    "binary_output_files": binary_output_files[:30],
                    "other_files": other_files[:30],
                }

                # Extract actual data from binary output files (not just metadata).
                # This gives the LLM per-layer head stats, drawdown, budget breakdowns.
                binary_extractions = {}
                for bf in binary_output_files[:4]:
                    ext = bf.rsplit(".", 1)[-1].lower() if "." in bf else ""
                    try:
                        if ext == "hds":
                            result = extract_hds_data(ws_root, bf, max_chars=40_000)
                            if result.get("ok") and result.get("summary_text"):
                                binary_extractions[bf] = {
                                    "type": "hds",
                                    "extracted_data": result["summary_text"],
                                    "metadata": result.get("metadata", {}),
                                }
                            else:
                                # Fallback to metadata probe
                                binary_extractions[bf] = {"type": "hds", "probe": probe_hds(ws_root, bf)}
                        elif ext == "cbc":
                            result = extract_cbc_data(ws_root, bf, max_chars=30_000)
                            if result.get("ok") and result.get("summary_text"):
                                binary_extractions[bf] = {
                                    "type": "cbc",
                                    "extracted_data": result["summary_text"],
                                    "metadata": result.get("metadata", {}),
                                }
                            else:
                                binary_extractions[bf] = {"type": "cbc", "probe": probe_cbc(ws_root, bf)}
                    except Exception:
                        pass
                if binary_extractions:
                    ctx["binary_output_data"] = binary_extractions
            except Exception:
                pass
                
    except Exception:
        pass
    
    # Attach referenced workspace files and docs
    _maybe_attach_workspace_files(ctx, inputs_dir=inputs_dir or "", workspace=workspace, message=message)
    _maybe_attach_docs(ctx, inputs_dir=inputs_dir or "", workspace=workspace, message=message)
    
    return ctx, model_facts_text


def _build_system_prompt(ctx: Dict[str, Any], model_facts_text: str) -> str:
    """Build system prompt with facts and context."""
    facts_section = f"\n\n{model_facts_text}\n" if model_facts_text else ""

    # Compact model brief for LLM grounding
    model_brief_section = ""
    snapshot = ctx.get("model_snapshot", {}) if isinstance(ctx, dict) else {}
    if isinstance(snapshot, dict) and snapshot:
        try:
            from gw.api.model_snapshot import build_model_brief
            brief = build_model_brief(snapshot)
            if brief:
                model_brief_section = f"\n\nMODEL BRIEF:\n{brief}\n"
        except Exception:
            pass

    # Always include the static knowledge base so the LLM knows which files to check
    pkg_summary = package_property_summary()
    pkg_section = ""
    if pkg_summary:
        pkg_section = f"\n\nPACKAGE FILES AND THEIR CONTENTS:\n{pkg_summary}\n"

    return (
        "You are a senior groundwater-modeling copilot for a MODFLOW 6 project.\n\n"

        "YOUR CAPABILITIES:\n"
        "- You have FULL READ ACCESS to every file in the workspace (text and binary).\n"
        "- Text file contents are provided directly in context when the router reads them.\n"
        "- Binary output files (.hds, .cbc) are FULLY EXTRACTED via FloPy. You receive:\n"
        "  * Per-layer head statistics (min, max, mean, median, std) for sampled timesteps\n"
        "  * Drawdown analysis (head change between first and last timestep)\n"
        "  * Cell budget breakdowns (inflow, outflow, net per component)\n"
        "  * Sample well/boundary entries from structured budget records\n"
        "  This data is ALREADY EXTRACTED and present in your context — you do NOT need\n"
        "  to ask the user to run FloPy or extract data themselves.\n"
        "- You can read .csv, .json, .txt, and any other text file in the workspace.\n"
        "- You understand MODFLOW 6 file formats deeply: NAM, DIS/DISV/DISU, TDIS,\n"
        "  IMS, NPF, STO, IC, OC, WEL, CHD, GHB, RIV, DRN, RCH, EVT, UZF, etc.\n\n"

        "TOOLS:\n"
        "You have access to tools that you can call during the conversation:\n"
        "- read_file(path) — Read any text file from the workspace\n"
        "- read_binary_output(path) — Extract numerical data from .hds or .cbc files\n"
        "- list_files(pattern?) — List workspace files matching a glob pattern\n"
        "- generate_plot(prompt, script?) — Generate a plot from workspace data\n"
        "- run_qa_check(check_name) — Run specialized QA/QC diagnostic checks\n\n"
        "Use these tools when:\n"
        "- You need data from a file not already in your context\n"
        "- The user asks for a plot or visualization\n"
        "- You want to verify or explore additional files\n"
        "- The user asks about model quality, mass balance, dry cells, convergence, etc.\n"
        "Do NOT use tools for data that is ALREADY in your context below.\n\n"

        "CRITICAL — ANSWER USING DATA IN CONTEXT:\n"
        "- When binary file data (heads, budgets) appears in your context, USE IT DIRECTLY.\n"
        "  Do NOT tell the user to 'extract data using FloPy' or 'run a script' — the data\n"
        "  is already extracted and available to you right now.\n"
        "- When the user asks about the HDS file, look for 'binary_output_data' or\n"
        "  'binary_extract' entries in your context — they contain the actual statistics.\n"
        "- Analyze and interpret the data: identify trends, compare layers, note anomalies.\n"
        "- If you need data NOT in context, use the read_file or read_binary_output tools.\n\n"

        "ANSWER QUALITY RULES:\n"
        "- ALWAYS respond with natural-language markdown. Never return raw JSON or code as the answer.\n"
        "- When file data is in context, ANALYZE it. Do not just echo the raw data back.\n"
        "- For well data: identify multi-screened wells (same row/col in multiple layers),\n"
        "  calculate total pumping rates, note spatial patterns.\n"
        "- For head data: describe distributions, identify highs/lows, note gradients.\n"
        "- For property data: provide statistics (min/max/mean), note uniformity or heterogeneity.\n"
        "- Support deep multi-turn conversations. Use conversation history to understand\n"
        "  follow-up questions. If the user previously discussed wells and now asks about\n"
        "  pumping rates, you should know they mean the WEL package data.\n"
        "- NEVER respond with 'Needs clarification' unless you truly cannot determine what\n"
        "  the user is asking. Instead, make your best interpretation and answer it.\n"
        "- NEVER tell the user to extract data themselves or run Python scripts when the\n"
        "  data is already in your context. You have the data — use it.\n\n"

        "PLOT GENERATION:\n"
        "- When the user asks for a plot, visualization, chart, or figure, use the\n"
        "  generate_plot tool.\n"
        "- Provide a clear prompt describing the plot. You can also provide a full\n"
        "  Python script if you want precise control.\n"
        "- The tool returns image URLs. Include them in your response as markdown:\n"
        "  ![Description](/plots/run/output?inputs_dir=...&run_id=...&path=...)\n"
        "- If the plot fails, explain the error and try to fix the script.\n\n"

        "QA/QC DIAGNOSTICS:\n"
        "When the user asks about model quality, mass balance, dry cells, convergence,\n"
        "pumping data issues, or any QA/QC analysis, use the run_qa_check tool.\n\n"
        "Available checks:\n"
        "- mass_balance: Parse listing file for volumetric budget, compute % discrepancy\n"
        "  per stress period. Good balance: <0.5%, marginal: 0.5-1%, poor: >1%.\n"
        "- dry_cells: Count cells with dry/inactive heads per layer and timestep.\n"
        "  Reports spatial clusters and trends over time.\n"
        "- convergence: Parse listing file for solver iteration counts and failures.\n"
        "  Flags non-convergence and high iteration counts.\n"
        "- pumping_summary: Analyze WEL package pumping rates by stress period.\n"
        "  Detects anomalous rate jumps (>2x between consecutive periods).\n"
        "- budget_timeseries: Extract IN/OUT per budget term across all timesteps.\n"
        "  Shows temporal trends in the water budget.\n"
        "- head_gradient: Compute cell-to-cell gradients per layer (final timestep).\n"
        "  Flags extreme gradients that may indicate grid resolution issues.\n"
        "- property_check: Check K, Kv/Kh, Ss, Sy ranges for unreasonable values.\n"
        "  Detects layer inversions (deeper layer more permeable than expected).\n\n"
        "For a comprehensive QA review, run multiple checks sequentially:\n"
        "1. mass_balance (overall model health)\n"
        "2. convergence (solver performance)\n"
        "3. dry_cells (stability issues)\n"
        "4. property_check (input data reasonableness)\n\n"
        "QA/QC DOMAIN KNOWLEDGE:\n"
        "- Hydraulic conductivity (K) typical ranges:\n"
        "  * Gravel: 10-1000 m/d, Sand: 0.1-100 m/d, Silt: 1e-4 to 1 m/d\n"
        "  * Clay: 1e-8 to 1e-3 m/d, Sandstone: 1e-4 to 10 m/d\n"
        "- Kv/Kh ratio: typically 0.01-1.0 (vertical always less than horizontal)\n"
        "- Specific storage (Ss): typically 1e-6 to 1e-4 per meter\n"
        "- Specific yield (Sy): sand 0.15-0.30, gravel 0.20-0.35, clay 0.01-0.05\n"
        "- Mass balance: <0.5% discrepancy is good, 0.5-1% needs attention, >1% is problematic\n"
        "- Convergence: IMS outer iterations >50 suggest difficulty; non-convergence is critical\n"
        "- Dry cells: >10% of a layer dry warrants investigation; growing dry zones are concerning\n\n"

        "GEOGRAPHIC LOCATION AWARENESS:\n"
        "If a MODEL GEOGRAPHIC LOCATION section appears in the context below, it means\n"
        "the model has been georeferenced and you know its real-world coordinates.\n"
        "When the user asks about geology, site conditions, or parameter reasonableness,\n"
        "use those coordinates to identify the region and provide site-specific context.\n"
        "NEVER say you don't know the location if coordinates are provided.\n\n"

        "DETERMINISTIC SPINE (the only hard constraints):\n"
        "- You do NOT modify files directly. For changes, respond with:\n"
        "  (1) what to change, (2) why, (3) which command to run.\n"
        "- For risky changes, ask the user to confirm before suggesting 'apply'.\n"
        "- Quick actions: validate | revalidate | suggest-fix | apply-fixes\n\n"

        "MODFLOW 6 DOMAIN KNOWLEDGE:\n"
        "- In WEL packages, a well with entries in multiple layers at the same (row,col)\n"
        "  or (node) position represents a multi-screened well.\n"
        "- Negative Q values = extraction (pumping), positive Q values = injection.\n"
        "- PERIOD blocks define stress-period-specific data; PACKAGEDATA defines defaults.\n"
        "- The .hds file contains computed heads per timestep; .cbc contains cell budgets.\n"
        "- The .lst file contains the solver convergence history and water budget.\n"
        f"{pkg_section}"
        f"{facts_section}"
        f"{model_brief_section}\n"
        "WORKSPACE CONTEXT (JSON):\n"
        f"{json.dumps(ctx, indent=2, default=str)}"
    )


def _openai_reply(
    *,
    messages: List[Dict[str, str]],
    inputs_dir: Optional[str],
    workspace: Optional[str] = None,
    message: str = "",
    model: Optional[str] = None,
    extra_ctx: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate reply using OpenAI."""
    client = _openai_client()
    if client is None:
        raise RuntimeError(
            "OpenAI not configured. Set OPENAI_API_KEY or configure in Settings."
        )

    ctx, model_facts_text = _build_context(inputs_dir, workspace, message, extra_ctx)
    system = _build_system_prompt(ctx, model_facts_text)

    # Build messages for OpenAI
    input_msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        *[{"role": m["role"], "content": m["content"]} for m in messages],
    ]

    # Use a model that matches the configured provider.
    configured_model = _get_configured_model()
    env_default = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    use_model = _select_model_for_provider("openai", model or configured_model, env_default)

    # Be resilient to transient network/rate-limit errors, and to accidental
    # provider/model mismatches (e.g., Claude model name while using OpenAI).
    import time

    resp = None
    for attempt in range(4):
        try:
            resp = client.responses.create(
                model=use_model,
                input=input_msgs,
            )
            break
        except Exception as e:
            # If the model name is invalid, retry once with a safe fallback.
            if _looks_like_model_not_found(e) and use_model != env_default:
                use_model = env_default
                continue

            # Exponential backoff for likely-transient errors.
            if attempt < 3:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise

    if resp is None:
        raise RuntimeError("OpenAI response was not created.")

    # Extract response text
    try:
        return resp.output_text  # type: ignore[attr-defined]
    except Exception:
        out = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in {"output_text", "text"}:
                    out.append(getattr(c, "text", ""))
        return "".join(out).strip() or "(no text returned)"


def _anthropic_reply(
    *,
    messages: List[Dict[str, str]],
    inputs_dir: Optional[str],
    workspace: Optional[str] = None,
    message: str = "",
    model: Optional[str] = None,
    extra_ctx: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate reply using Anthropic Claude."""
    client = _anthropic_client()
    if client is None:
        raise RuntimeError(
            "Anthropic not configured. Set ANTHROPIC_API_KEY or configure in Settings."
        )

    ctx, model_facts_text = _build_context(inputs_dir, workspace, message, extra_ctx)
    system = _build_system_prompt(ctx, model_facts_text)

    # Convert messages to Anthropic format
    anthropic_messages = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]

    # Use a model that matches the configured provider.
    configured_model = _get_configured_model()
    env_default = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    use_model = _select_model_for_provider(
        "anthropic", model or configured_model, env_default
    )

    # Resilience for transient failures and accidental provider/model mismatch.
    import time

    resp = None
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=use_model,
                max_tokens=4096,
                system=system,
                messages=anthropic_messages,
            )
            break
        except Exception as e:
            if _looks_like_model_not_found(e) and use_model != env_default:
                use_model = env_default
                continue

            if attempt < 3:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise

    if resp is None:
        raise RuntimeError("Anthropic response was not created.")

    # Extract response text
    try:
        if hasattr(resp, "content") and resp.content:
            text_parts = []
            for block in resp.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            return "".join(text_parts).strip() or "(no text returned)"
        return "(no text returned)"
    except Exception:
        return "(error extracting response)"


# -----------------------------
# Public entry point
# -----------------------------


def chat_reply(
    *,
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    inputs_dir: Optional[str] = None,
    workspace: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, List[str], List[Dict[str, str]]]:
    """Return (reply, suggestions, plot_outputs).

    plot_outputs is a list of {name, url} dicts for any plots generated
    during tool-use. The LLM also embeds these as markdown images in
    the reply text.

    - Handles command-like inputs deterministically
    - Falls back to helpful deterministic guidance if LLM unavailable
    - Uses LLM for natural explanations when available
    """

    text = (message or "").strip()
    if not text:
        return (
            "Type a question, or try: validate | revalidate | suggest-fix | apply-fixes",
            ["validate", "revalidate", "suggest-fix", "apply-fixes"],
            [],
        )

    def _remember(ws_root: Optional[Path], reply_text: str) -> None:
        """Update lightweight per-workspace state (best-effort)."""
        if not ws_root:
            return
        try:
            update_workspace_state(ws_root, user_message=text, assistant_reply=reply_text)
        except Exception:
            pass

    ws_root_local: Optional[Path] = None
    if inputs_dir:
        try:
            ws_root_local = resolve_workspace_root(inputs_dir, workspace)
        except Exception:
            ws_root_local = None

    # 1) If the user typed a bare command, respond with exactly what to run/click.
    cmd = _detect_direct_command(text)
    if cmd:
        if cmd == "validate":
            return (
                "To validate stress inputs, run:\n"
                "- CLI: `python gw_cli.py validate-stresses --config <path/to/config.json> --inputs-dir <runs/...>`\n"
                "- GUI: click **Validate** (calls /jobs/run?name=validate)\n\n"
                "If you want a diff vs last run, use **revalidate**.",
                ["validate", "revalidate"],
                [],
            )
        if cmd == "revalidate":
            return (
                "To revalidate and write artifacts (and optional diff), run:\n"
                "- CLI: `python gw_cli.py revalidate --inputs-dir <runs/...>`\n"
                "- GUI: click **Revalidate** (calls /jobs/run?name=revalidate)",
                ["revalidate"],
                [],
            )
        if cmd == "suggest-fix":
            gd = _gate_action("llm-suggest-fixes", severity="caution")
            return (
                "To generate a FixPlan suggestion, run:\n"
                "- CLI: `python gw_cli.py llm-suggest-fixes --config <path/to/config.json> --inputs-dir <runs/...>`\n\n"
                f"Gate: {gd.reason}",
                ["suggest-fix"],
                [],
            )
        if cmd == "apply-fixes":
            gd = _gate_action("apply-fixes", severity="manual")
            return (
                "Applying fixes modifies input CSVs and/or config, so it's **manual**.\n\n"
                "Recommended flow:\n"
                "1) `llm-suggest-fixes` to create fix_plan.json\n"
                "2) `apply-fixes` (dry run)\n"
                "3) `apply-fixes --apply` after you review\n\n"
                f"Confirmations needed to proceed safely: {', '.join(gd.confirmations_needed)}",
                ["apply-fixes"],
                [],
            )

    # 2) Common conceptual question we can answer deterministically.
    if re.search(r"validate\s+vs\s+stress\s+validate", text, re.I) or re.search(
        r"model\s+validate\s+vs\s+stress", text, re.I
    ):
        return (_explain_validate_vs_stress_validate(), [], [])

    # 3) If the user is asking for improvements/tuning, generate a patch plan automatically.
    if inputs_dir and _looks_like_patch_request(text):
        ws = workspace
        if not ws:
            try:
                ws = str((json.loads((Path(inputs_dir) / "config.json").read_text(encoding="utf-8")).get("workspace")) or "")
            except Exception:
                ws = workspace
        try:
            patch_ws_root = None
            try:
                patch_ws_root = resolve_workspace_root(inputs_dir, ws)
            except Exception:
                patch_ws_root = None
            plan = _plan_patch_from_chat(inputs_dir=inputs_dir, workspace=ws or None, goal=text)
            status = str(plan.get("status") or "ok") if isinstance(plan, dict) else "ok"
            edits = plan.get("edits") if isinstance(plan, dict) else None
            has_edits = isinstance(edits, list) and len(edits) > 0
            if has_edits:
                reply = _format_patch_plan_reply(plan)
                _remember(patch_ws_root, reply)
                return (reply, ["patch:validate", "patch:apply"], [])
            # If no concrete edits, fall through to LLM for a richer answer.
        except Exception:
            pass  # Fall through to LLM reply

    # 4) LLM-first: route ALL questions through the LLM with rich workspace context.
    messages = (history or []) + [{"role": "user", "content": text}]
    try:
        ws_root = None
        scan = None
        try:
            if inputs_dir:
                ws_root = resolve_workspace_root(inputs_dir, workspace)
                scan = ensure_workspace_scan(ws_root, force=False)
        except Exception:
            ws_root = None

        # Router → bounded reads: ask a lightweight LLM which files to read,
        # then execute those reads and inject into context.
        extra_ctx = None
        if ws_root is not None:
            try:
                t0 = _time.time()
                files, truncated = list_workspace_files(ws_root, max_files=300, include_hash=False)
                # Include recent conversation for follow-up understanding
                recent_history: list = []
                if history:
                    for h in history[-8:]:
                        role = h.get("role", "")
                        content = h.get("content", "")
                        if len(content) > 800:
                            content = content[:800] + "..."
                        recent_history.append({"role": role, "content": content})
                router_ctx = {
                    'question': text,
                    'workspace_root': ws_root.as_posix(),
                    'workspace_files': [f.path_rel for f in files],
                    'workspace_files_truncated': bool(truncated),
                    'snapshot': (scan or {}).get('snapshot') if isinstance(scan, dict) else None,
                    'recent_conversation': recent_history,
                }
                plan = llm_route_read_plan(question=text, router_context=router_ctx, model=model)
                if plan and isinstance(plan, dict):
                    extra_ctx = execute_read_plan(ws_root=ws_root, scan=scan, plan=plan,
                                                  max_total_chars=200_000)
                    # Audit trail: log what files were read
                    reads = extra_ctx.get("router_reads", []) if extra_ctx else []
                    budget = extra_ctx.get("router_read_budget", {}) if extra_ctx else {}
                    elapsed = _time.time() - t0
                    logger.info(
                        "chat_reply router: question=%r  reads=%d  chars_used=%s  elapsed=%.2fs  files=%s",
                        text[:80], len(reads), budget.get("used_chars", "?"), elapsed,
                        [r.get("path", r.get("key", "?")) for r in reads[:12]],
                    )
                else:
                    logger.info("chat_reply router: no plan returned for question=%r", text[:80])
            except Exception as router_err:
                logger.warning("chat_reply router failed: %s: %s", type(router_err).__name__, router_err)
                extra_ctx = None

        resp_text, tool_audit = _llm_reply(
            messages=messages,
            inputs_dir=inputs_dir,
            workspace=workspace,
            message=text,
            model=model,
            extra_ctx=extra_ctx,
            ws_root=ws_root,
            enable_tools=True,
        )

        # Log tool audit trail
        if tool_audit:
            logger.info(
                "chat_reply tool_audit: %d tool calls for question=%r",
                len(tool_audit), text[:80],
            )
            for entry in tool_audit:
                logger.info(
                    "  tool=%s ok=%s elapsed=%.2fs chars=%s",
                    entry.get("tool"), entry.get("ok"),
                    entry.get("elapsed_sec", 0), entry.get("result_chars", "?"),
                )

        # Guardrail: if the model returns a uselessly terse clarification marker, retry.
        if (resp_text or "").strip().lower() in {
            "needs clarification", "needs clarification.",
            "need clarification", "need clarification.",
        }:
            retry_messages = messages + [
                {"role": "assistant", "content": "Needs clarification."},
                {"role": "user", "content": (
                    "Please do your best to answer from the model files and context you already have. "
                    "If data was cited in earlier messages, use it. "
                    "If you truly cannot answer, explain what specific information is missing and why."
                )},
            ]
            try:
                resp_text, _ = _llm_reply(
                    messages=retry_messages,
                    inputs_dir=inputs_dir,
                    workspace=workspace,
                    message=text,
                    model=model,
                    extra_ctx=extra_ctx,
                    ws_root=ws_root,
                    enable_tools=False,
                )
            except Exception:
                pass
            if (resp_text or "").strip().lower() in {
                "needs clarification", "needs clarification.",
                "need clarification", "need clarification.",
            }:
                resp_text = (
                    "I wasn't able to fully answer that from the available context. "
                    "Could you try rephrasing or providing more detail? For example:\n"
                    "- Mention the specific file name (e.g., the .wel file)\n"
                    "- Reference the specific property or value you're looking for\n"
                    "- Ask about a specific layer, stress period, or cell"
                )
        # Extract plot outputs from tool audit log (if any generate_plot calls succeeded)
        plot_outputs: List[Dict[str, str]] = []
        for entry in tool_audit:
            if entry.get("tool") == "generate_plot" and entry.get("ok"):
                # The actual image data is in the tool result, not the audit.
                # We can't easily access it here, so we parse them from the response text.
                pass

        # Parse plot image URLs from the response markdown
        # Pattern: ![...](url) where url contains /plots/run/output
        import re as _re
        for match in _re.finditer(r"!\[([^\]]*)\]\((/plots/run/output[^)]+)\)", resp_text or ""):
            alt = match.group(1)
            url = match.group(2)
            # Extract filename from URL params (endpoint param is "path")
            fname_m = _re.search(r"path=([^&]+)", url)
            name = fname_m.group(1) if fname_m else alt or "plot.png"
            plot_outputs.append({"name": name, "url": url})

        _remember(ws_root, resp_text)
        return (resp_text, [], plot_outputs)
    except Exception as e:
        # Deterministic fallback when no LLM is available.
        hint = (
            "For deeper analysis you need an LLM configured.\n\n"
            "To enable:\n"
            "- OpenAI: `pip install openai` + set `OPENAI_API_KEY`\n"
            "- Anthropic: `pip install anthropic` + set `ANTHROPIC_API_KEY`\n\n"
            "Quick actions available without LLM: validate | revalidate | suggest-fix"
        )
        try:
            ws_root = resolve_workspace_root(inputs_dir, workspace) if inputs_dir else None
        except Exception:
            ws_root = None
        reply = f"(LLM unavailable: {type(e).__name__}: {e})\n\n{hint}"
        _remember(ws_root, reply)
        return (reply, [], [])
