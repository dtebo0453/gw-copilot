from __future__ import annotations

"""Workspace scan + cache.

This module provides a lightweight, deterministic scan of a workspace folder
that can be reused by chat and plotting.

It produces a cached JSON file under:

  <workspace>/.gw_copilot/cache_scan.json

Cache invalidation is based on (file_count, newest_mtime). The scan is read-only
and only enumerates files under the workspace root.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gw.api.model_snapshot import build_model_snapshot
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gw.api.workspace_files import resolve_workspace_root


SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
}

router = APIRouter()


class WorkspaceScanRequest(BaseModel):
    inputs_dir: str
    workspace: str | None = None
    force: bool = False


@router.post('/workspace/scan')
def scan_workspace(req: WorkspaceScanRequest):
    try:
        ws_root = resolve_workspace_root(req.inputs_dir, req.workspace)
    except ValueError as e:
        # Client-side path issue; don't crash the server with a 500.
        raise HTTPException(status_code=400, detail=str(e))
    return ensure_workspace_scan(ws_root, force=req.force)



def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.name


def _is_texty(path: Path, sniff_bytes: int = 4096) -> bool:
    try:
        b = path.open("rb").read(sniff_bytes)
    except Exception:
        return False
    # Heuristic: if NUL present, probably binary
    if b"\x00" in b:
        return False
    try:
        b.decode("utf-8")
        return True
    except Exception:
        return False


def _peek_header(path: Path, max_chars: int = 240) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            txt = f.read(max_chars)
        txt = " ".join(txt.replace("\r", "").split("\n")[:4]).strip()
        return txt[:max_chars] if txt else None
    except Exception:
        return None


def _compute_fingerprint(ws_root: Path, max_files: int = 50000) -> Tuple[int, float]:
    """Return (file_count, newest_mtime) for invalidation."""
    count = 0
    newest = 0.0
    for p in ws_root.rglob("*"):
        try:
            if p.is_dir():
                if p.name in SKIP_DIRS:
                    # Skip walking heavy dirs
                    # NOTE: rglob doesn't allow pruning easily; we just ignore contents.
                    continue
                continue
            if not p.is_file():
                continue
            st = p.stat()
        except Exception:
            continue
        count += 1
        newest = max(newest, float(st.st_mtime))
        if count >= max_files:
            break
    return count, newest


def build_file_index(ws_root: Path, max_files: int = 4000, max_peeks: int = 80) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    peeked = 0
    for p in ws_root.rglob("*"):
        if len(files) >= max_files:
            break
        try:
            if p.is_dir():
                if p.name in SKIP_DIRS:
                    continue
                continue
            if not p.is_file():
                continue
            st = p.stat()
        except Exception:
            continue

        rel = _safe_rel(p, ws_root)
        entry: Dict[str, Any] = {"path": rel, "bytes": int(st.st_size), "ext": p.suffix.lower()}
        if peeked < max_peeks and st.st_size <= 2_000_000 and _is_texty(p):
            hdr = _peek_header(p)
            if hdr:
                entry["peek"] = hdr
                peeked += 1
        files.append(entry)

    return {"workspace_root": str(ws_root), "files_count": len(files), "files": files}


def build_health_report(snapshot: Dict[str, Any], file_index: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    hints: List[str] = []

    if not snapshot.get("ok"):
        issues.append({"severity": "error", "code": "snapshot_failed", "message": snapshot.get("error") or "Snapshot failed"})
        return {"ok": False, "issues": issues, "hints": hints}

    outputs = snapshot.get("outputs_present") or {}
    if not outputs.get("hds"):
        issues.append({"severity": "warn", "code": "missing_hds", "message": "No .hds file found; head plots will be unavailable."})
        hints.append("Enable head saving in OC (SAVE HEAD) and rerun.")
    if not outputs.get("cbc"):
        issues.append({"severity": "warn", "code": "missing_cbc", "message": "No .cbc file found; budget plots will be limited."})
        hints.append("Enable budget saving in OC and rerun.")
    if not outputs.get("lst"):
        issues.append({"severity": "info", "code": "missing_lst", "message": "No .lst file found; solver diagnostics may be limited."})

    # Basic package presence hints
    pkgs = snapshot.get("packages") or {}
    if not pkgs:
        # Sometimes NAM isn't present; still ok
        hints.append("If .nam files exist, loading them improves package discovery.")

    # If file_index is truncated (approx by max_files), warn
    if isinstance(file_index, dict) and file_index.get("files_count", 0) >= 3999:
        issues.append({"severity": "info", "code": "file_index_truncated", "message": "File index may be truncated; increase scan limits if needed."})

    # Binary output probe health
    out_meta = snapshot.get("output_metadata", {})
    if out_meta.get("probed"):
        hds_info = out_meta.get("hds", {})
        if hds_info and not hds_info.get("ok") and outputs.get("hds"):
            issues.append({"severity": "warn", "code": "hds_probe_failed", "message": f"HDS file may be corrupt: {hds_info.get('error', 'unknown')}"})
        cbc_info = out_meta.get("cbc", {})
        if cbc_info and not cbc_info.get("ok") and outputs.get("cbc"):
            issues.append({"severity": "warn", "code": "cbc_probe_failed", "message": f"CBC file may be corrupt: {cbc_info.get('error', 'unknown')}"})
        if cbc_info.get("ok") and not cbc_info.get("record_names"):
            issues.append({"severity": "warn", "code": "cbc_empty", "message": "CBC file appears empty (no record names found)."})

    return {"ok": True, "issues": issues, "hints": hints}


def _cache_dir(ws_root: Path) -> Path:
    d = ws_root / ".gw_copilot"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_workspace_scan(ws_root: Path, force: bool = False) -> Dict[str, Any]:
    """Return cached scan dict; rebuild if stale or force=True."""
    cache_path = _cache_dir(ws_root) / "cache_scan.json"

    cur_count, cur_newest = _compute_fingerprint(ws_root)

    if not force and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            fp = data.get("fingerprint") or {}
            if int(fp.get("file_count", -1)) == int(cur_count) and float(fp.get("newest_mtime", -2.0)) == float(cur_newest):
                return data
        except Exception:
            pass

    file_index = build_file_index(ws_root)
    snapshot = build_model_snapshot(ws_root)
    health = build_health_report(snapshot, file_index)

    out = {
        "ok": True,
        "workspace_root": str(ws_root),
        "fingerprint": {"file_count": int(cur_count), "newest_mtime": float(cur_newest), "newest_mtime_iso": _utc_iso(cur_newest) if cur_newest else None},
        "file_index": file_index,
        "snapshot": snapshot,
        "health": health,
    }
    try:
        cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return out