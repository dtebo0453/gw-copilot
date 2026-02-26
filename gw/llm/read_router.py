from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gw.api.workspace_files import read_file_text
from gw.api.output_probes import probe_hds, probe_cbc, extract_hds_data, extract_cbc_data
from gw.llm.mf6_filetype_knowledge import package_property_summary


"""LLM router + deterministic read-plan executor.

This module provides a *capability* layer:

1) An LLM "router" that interprets a user question and emits a small JSON "read plan".
2) A deterministic executor that performs only safe, bounded reads *inside the workspace*.

Design constraints:
  - Read-only.
  - Bounded IO (bytes/lines/total chars).
  - No path traversal; no absolute/drive-qualified paths.
  - MF6-aware block extraction for large files.

The intent is to reduce brittleness from regex-only fast paths: if the user
misspells a word (e.g., "layrs"), the router can still propose the correct reads.
"""


# -----------------------------
# Plan schema
# -----------------------------


@dataclass(frozen=True)
class ReadRequest:
    kind: str
    file_hint: Optional[str] = None
    path: Optional[str] = None
    block: Optional[str] = None
    key: Optional[str] = None
    pattern: Optional[str] = None
    max_chars: int = 40_000


def _safe_relpath(p: str) -> Optional[str]:
    if not p:
        return None
    p = p.strip().strip('"').strip("'").replace('\\', '/')
    if p.startswith('/') or re.match(r'^[A-Za-z]:/', p):
        return None
    if '..' in p.split('/'):
        return None
    return p


def _mf6_extract_blocks(text: str, wanted: List[str], *, max_lines: int = 4000) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not text:
        return out
    want = {w.upper() for w in wanted}
    cur = None
    buf: List[str] = []
    for raw in text.splitlines():
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
                out[cur] = "\n".join(buf[:max_lines])
            cur = None
            buf = []
            continue
        if cur is not None:
            buf.append(raw.rstrip())
            if len(buf) >= max_lines:
                out[cur] = "\n".join(buf) + "\n... (truncated)"
                cur = None
                buf = []
    return out


# -----------------------------
# Router
# -----------------------------


def _get_active_provider_name() -> str:
    try:
        from gw.api.llm_config import get_active_provider
        return get_active_provider()
    except Exception:
        return "openai"


def _openai_client():
    try:
        from gw.api.llm_config import get_api_key
        api_key = get_api_key("openai")
    except Exception:
        api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _anthropic_client():
    try:
        from gw.api.llm_config import get_api_key
        api_key = get_api_key("anthropic")
    except Exception:
        api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    try:
        import anthropic  # type: ignore
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def _build_router_system_prompt(router_context: Dict[str, Any]) -> str:
    pkg_knowledge = package_property_summary()
    return (
        "You are a router for a groundwater modeling copilot. "
        "Given the user request and a workspace index + snapshot facts, "
        "output ONLY a JSON object describing what to read.\n\n"
        "Rules:\n"
        "- Output valid JSON only (no markdown).\n"
        "- Be GENEROUS with reads — the main LLM needs data in context to give good answers.\n"
        "- You can read ANY file in the workspace: MF6 packages, .csv, .json, .txt, .lst, etc.\n"
        "- For binary output files (.hds, .cbc), use binary_probe to get metadata via FloPy.\n"
        "- Prefer snapshot facts for basic grid/time info.\n"
        "- For MF6 structure (DIS/DISV/DISU/TDIS/IMS/NAM), request mf6_block reads.\n"
        "- For well data, pumping rates, boundary conditions — read the relevant package file.\n"
        "- The context may include 'recent_conversation' with the last few chat turns. "
        "Use this to understand follow-up questions. For example, if the user previously "
        "asked about wells and now asks about pumping rates, read the .wel file.\n"
        "- ALWAYS propose reads for follow-up questions that need file data.\n"
        "- Do NOT ask clarifying questions — use your knowledge to pick the right files.\n\n"
        "PACKAGE FILE KNOWLEDGE:\n"
        f"{pkg_knowledge}\n\n"
        "Common mappings:\n"
        "- Hydraulic conductivity → NPF (.npf) GRIDDATA block\n"
        "- Storage (Ss, Sy) → STO (.sto) GRIDDATA block\n"
        "- Initial/starting heads → IC (.ic) GRIDDATA block\n"
        "- Well pumping → WEL (.wel) PERIOD block\n"
        "- Solver settings → IMS (.ims) OPTIONS block\n"
        "- Output heads → .hds binary file (use binary_probe)\n"
        "- Cell budgets → .cbc binary file (use binary_probe)\n\n"
        "Supported read kinds:\n"
        "- snapshot_fact: {kind:'snapshot_fact', key:'NLAY'|'NPER'|...}\n"
        "- mf6_block: {kind:'mf6_block', file_hint:'*.disv|*.dis|*.tdis|*.ims|*.nam', block:'DIMENSIONS|GRIDDATA|OPTIONS|PERIOD|PACKAGES'}\n"
        "- file_peek: {kind:'file_peek', path:'relative/path.ext'}  (reads first ~30KB of any text file)\n"
        "- find_in_file: {kind:'find_in_file', path:'relative/path.ext', pattern:'regex or literal'}\n"
        "- binary_probe: {kind:'binary_probe', path:'relative/path.hds|.cbc'}  (probes binary output via FloPy)\n\n"
        "Your JSON schema:\n"
        "{\n"
        "  'task': 'answer_question' | 'plan_plot',\n"
        "  'reads': [ ... ],\n"
        "  'answer_style': 'concise'|'normal'\n"
        "}\n\n"
        "Workspace context (JSON):\n"
        f"{json.dumps(router_context, ensure_ascii=False)}"
    )


def _parse_plan_json(raw: str) -> Optional[Dict[str, Any]]:
    """Try to parse a JSON plan from raw LLM text."""
    if not raw:
        return None
    try:
        plan = json.loads(raw)
        if isinstance(plan, dict) and isinstance(plan.get('reads', []), list):
            return plan
    except Exception:
        pass
    # Try to salvage JSON object from a messy response
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        plan = json.loads(m.group(0))
        if isinstance(plan, dict) and isinstance(plan.get('reads', []), list):
            return plan
    except Exception:
        pass
    return None


def llm_route_read_plan(
    *,
    question: str,
    router_context: Dict[str, Any],
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Ask the LLM to propose a safe read plan.

    Returns parsed JSON dict or None on failure.
    Uses the currently-configured provider (OpenAI or Anthropic).
    """
    provider = _get_active_provider_name()
    sys_prompt = _build_router_system_prompt(router_context)

    if provider == "anthropic":
        client = _anthropic_client()
        if client is None:
            return None
        try:
            resp = client.messages.create(
                model=model or "claude-sonnet-4-20250514",
                max_tokens=2048,
                system=sys_prompt,
                messages=[{"role": "user", "content": question.strip()}],
            )
            raw = ""
            if hasattr(resp, "content") and resp.content:
                for block in resp.content:
                    if hasattr(block, "text"):
                        raw += block.text
            raw = raw.strip()
            # Strip code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```\s*$", "", raw)
            return _parse_plan_json(raw)
        except Exception:
            return None
    else:
        # OpenAI (default)
        client = _openai_client()
        if client is None:
            return None
        try:
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question.strip()},
            ]
            resp = client.responses.create(
                model=model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                input=msgs,
            )
            raw = getattr(resp, 'output_text', None)
            if not raw:
                parts: List[str] = []
                for item in getattr(resp, 'output', []) or []:
                    for c in getattr(item, 'content', []) or []:
                        if getattr(c, 'type', None) in {'output_text', 'text'}:
                            parts.append(getattr(c, 'text', ''))
                raw = ''.join(parts).strip()
            return _parse_plan_json(raw or "")
        except Exception:
            return None


# -----------------------------
# Executor
# -----------------------------


def _match_files(ws_root: Path, file_hint: str, *, limit: int = 8) -> List[Path]:
    """Match small number of files in workspace for a glob hint."""
    file_hint = (file_hint or '').strip()
    if not file_hint:
        return []
    # normalize hint
    hint = file_hint.replace('\\', '/')
    # block absolute hints
    if hint.startswith('/') or re.match(r'^[A-Za-z]:', hint):
        return []
    # accept simple patterns like *.disv or **/*.dis
    matches: List[Path] = []
    try:
        for p in ws_root.rglob(hint.replace('**/', '')) if hint.startswith('**/') else ws_root.glob(hint):
            if p.is_file():
                matches.append(p)
            if len(matches) >= limit:
                break
    except Exception:
        return []
    return matches


def _snapshot_get(scan: Dict[str, Any], key: str) -> Optional[Any]:
    if not scan or not isinstance(scan, dict):
        return None
    snap = scan.get('snapshot') if isinstance(scan.get('snapshot'), dict) else {}
    grid = snap.get('grid') if isinstance(snap.get('grid'), dict) else {}
    tdis = snap.get('tdis') if isinstance(snap.get('tdis'), dict) else {}
    facts = snap.get('facts') if isinstance(snap.get('facts'), list) else []

    k = (key or '').strip().upper()
    if k in {'NLAY', 'NROW', 'NCOL', 'NCPL', 'NODES'}:
        return grid.get(k.lower()) or grid.get(k)
    if k in {'NPER'}:
        return tdis.get('nper')

    for f in facts:
        if isinstance(f, dict) and str(f.get('key', '')).upper() == k:
            return f.get('value')
    return None


def execute_read_plan(
    *,
    ws_root: Path,
    scan: Optional[Dict[str, Any]],
    plan: Dict[str, Any],
    max_total_chars: int = 140_000,
) -> Dict[str, Any]:
    """Execute a validated read plan inside the workspace.

    Returns a dict with collected read results suitable for injecting into LLM context.
    """

    reads = plan.get('reads') if isinstance(plan, dict) else []
    if not isinstance(reads, list):
        reads = []

    results: List[Dict[str, Any]] = []
    total = 0

    for r in reads[:12]:
        if total >= max_total_chars:
            break
        if not isinstance(r, dict):
            continue
        kind = str(r.get('kind') or '').strip()
        if not kind:
            continue

        if kind == 'snapshot_fact':
            key = str(r.get('key') or '')
            val = _snapshot_get(scan or {}, key)
            results.append({'kind': 'snapshot_fact', 'key': key, 'value': val})
            continue

        if kind == 'file_peek':
            rel = _safe_relpath(str(r.get('path') or ''))
            if not rel:
                continue
            p = (ws_root / rel).resolve()
            try:
                p.relative_to(ws_root.resolve())
            except Exception:
                continue
            if not p.exists() or not p.is_file():
                continue
            text, truncated, sha, size, fkind = read_file_text(p, max_bytes=80_000)
            if fkind != 'text' or not text:
                results.append({'kind': 'file_peek', 'path': rel, 'note': 'binary/unreadable', 'sha256': sha, 'bytes': size})
                continue
            snippet = text[: min(len(text), 30_000)]
            total += len(snippet)
            results.append({'kind': 'file_peek', 'path': rel, 'content': snippet, 'truncated': bool(truncated), 'sha256': sha, 'bytes': size})
            continue

        if kind == 'mf6_block':
            file_hint = str(r.get('file_hint') or '')
            block = str(r.get('block') or '').strip()
            if not block:
                continue
            matches = _match_files(ws_root, file_hint or '*')
            for mp in matches[:4]:
                try:
                    txt, truncated, sha, size, fkind = read_file_text(mp, max_bytes=220_000)
                except Exception:
                    continue
                if fkind != 'text' or not txt:
                    continue
                blocks = _mf6_extract_blocks(txt, [block])
                if not blocks:
                    continue
                content = blocks.get(block.upper()) or next(iter(blocks.values()))
                if not content:
                    continue
                # ensure bounded
                content = content[: min(len(content), 45_000)]
                total += len(content)
                relp = mp.relative_to(ws_root).as_posix()
                results.append({'kind': 'mf6_block', 'path': relp, 'block': block.upper(), 'content': content, 'sha256': sha, 'bytes': size})
                break
            continue

        if kind == 'find_in_file':
            rel = _safe_relpath(str(r.get('path') or ''))
            pattern = str(r.get('pattern') or '')
            if not rel or not pattern:
                continue
            p = (ws_root / rel).resolve()
            try:
                p.relative_to(ws_root.resolve())
            except Exception:
                continue
            if not p.exists() or not p.is_file():
                continue
            txt, truncated, sha, size, fkind = read_file_text(p, max_bytes=240_000)
            if fkind != 'text' or not txt:
                continue
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except Exception:
                rx = re.compile(re.escape(pattern), re.IGNORECASE)
            hits: List[str] = []
            for line in txt.splitlines():
                if rx.search(line):
                    hits.append(line[:300])
                if len(hits) >= 40:
                    break
            if hits:
                content = "\n".join(hits)
                total += len(content)
                results.append({'kind': 'find_in_file', 'path': rel, 'pattern': pattern, 'content': content, 'sha256': sha, 'bytes': size})
            continue

        if kind == 'binary_probe':
            rel = _safe_relpath(str(r.get('path') or ''))
            if not rel:
                continue
            p = (ws_root / rel).resolve()
            try:
                p.relative_to(ws_root.resolve())
            except Exception:
                continue
            if not p.exists() or not p.is_file():
                continue
            ext = p.suffix.lower()
            # Calculate remaining char budget for this extraction
            remaining_chars = max(10_000, max_total_chars - total)
            try:
                if ext == '.hds':
                    # Full data extraction: per-layer stats + drawdown analysis
                    extraction = extract_hds_data(ws_root, rel, max_chars=remaining_chars)
                    if extraction.get('ok') and extraction.get('summary_text'):
                        content = extraction['summary_text']
                        total += len(content)
                        results.append({
                            'kind': 'binary_extract', 'path': rel, 'file_type': 'hds',
                            'content': content, 'metadata': extraction.get('metadata', {}),
                        })
                    else:
                        # Fallback to metadata-only probe
                        probe = probe_hds(ws_root, rel)
                        content = json.dumps(probe, default=str)
                        total += len(content)
                        results.append({'kind': 'binary_probe', 'path': rel, 'file_type': 'hds', 'content': content, 'probe': probe})
                elif ext == '.cbc':
                    # Full data extraction: per-component budget stats
                    extraction = extract_cbc_data(ws_root, rel, max_chars=remaining_chars)
                    if extraction.get('ok') and extraction.get('summary_text'):
                        content = extraction['summary_text']
                        total += len(content)
                        results.append({
                            'kind': 'binary_extract', 'path': rel, 'file_type': 'cbc',
                            'content': content, 'metadata': extraction.get('metadata', {}),
                        })
                    else:
                        # Fallback to metadata-only probe
                        probe = probe_cbc(ws_root, rel)
                        content = json.dumps(probe, default=str)
                        total += len(content)
                        results.append({'kind': 'binary_probe', 'path': rel, 'file_type': 'cbc', 'content': content, 'probe': probe})
                else:
                    results.append({'kind': 'binary_probe', 'path': rel, 'note': f'unsupported binary extension: {ext}'})
            except Exception as e:
                results.append({'kind': 'binary_probe', 'path': rel, 'error': f'{type(e).__name__}: {e}'})
            continue

    return {
        'router_plan': plan,
        'router_reads': results,
        'router_read_budget': {'max_total_chars': max_total_chars, 'used_chars': total},
    }
