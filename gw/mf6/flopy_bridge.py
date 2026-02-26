from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import threading

# FloPy is optional. We import lazily and fall back to deterministic text parsing.
try:
    import flopy  # type: ignore
    from flopy.mf6 import MFSimulation  # type: ignore
except Exception:  # pragma: no cover
    flopy = None
    MFSimulation = None  # type: ignore


_ALLOWED_EXTS = {
    ".nam", ".dis", ".tdis", ".ims", ".npf", ".sto", ".ic", ".oc",
    ".wel", ".chd", ".ghb", ".riv", ".drn", ".rch", ".evt", ".uzf",
    ".sfr", ".lak", ".maw", ".obs", ".lst", ".hds", ".cbc",
}

def flopy_is_available() -> bool:
    return flopy is not None and MFSimulation is not None


def _workspace_fingerprint(ws: Path) -> int:
    """
    Lightweight fingerprint for cache invalidation:
    max mtime_ns of common MF6-related files in the workspace root.
    """
    max_m = 0
    try:
        for p in ws.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in _ALLOWED_EXTS:
                continue
            try:
                m = p.stat().st_mtime_ns
                if m > max_m:
                    max_m = m
            except Exception:
                continue
    except Exception:
        return 0
    return max_m


def _validate_paths_under_workspace(ws: Path, paths: List[Path]) -> Tuple[bool, Optional[str]]:
    for p in paths:
        try:
            p.resolve().relative_to(ws.resolve())
        except Exception:
            return False, f"Referenced path outside workspace: {p}"
    return True, None


def _collect_namefiles(ws: Path) -> List[Path]:
    # Prefer mfsim.nam, else any *.nam.
    cand = []
    mfsim = ws / "mfsim.nam"
    if mfsim.exists():
        cand.append(mfsim)
    cand.extend(sorted(ws.glob("*.nam")))
    # De-dup while preserving order
    out = []
    seen = set()
    for p in cand:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _extract_file_references_from_nam(nam_path: Path) -> List[Path]:
    """
    Very lightweight NAME file reference extraction.
    Supports lines like: 'GWF6 model.nam' or 'DIS6 aoi.dis' etc.
    """
    refs: List[Path] = []
    ws = nam_path.parent
    try:
        txt = nam_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return refs

    for raw in txt:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # strip inline comment
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        parts = line.split()
        if len(parts) < 2:
            continue
        # first token is filetype, second is filename
        f = parts[1].strip('"').strip("'")
        p = (ws / f).resolve()
        refs.append(p)
    return refs


@dataclass
class FlopySimEntry:
    fingerprint: int
    sim: Any


_LOCK = threading.Lock()
_SIM_CACHE: Dict[str, FlopySimEntry] = {}


def get_simulation(ws: Path) -> Tuple[Optional[Any], Optional[str]]:
    """
    FloPy-first loader with caching.
    Returns (sim, error_message). If sim is None, caller should fall back.
    """
    if not flopy_is_available():
        return None, "flopy not installed"

    ws = ws.resolve()
    fp = _workspace_fingerprint(ws)
    key = str(ws)

    with _LOCK:
        ent = _SIM_CACHE.get(key)
        if ent and ent.fingerprint == fp:
            return ent.sim, None

    # Validate namefile references stay under workspace to avoid path escapes.
    namefiles = _collect_namefiles(ws)
    if not namefiles:
        return None, "no name file found"

    referenced: List[Path] = []
    for nf in namefiles:
        referenced.extend(_extract_file_references_from_nam(nf))
    ok, msg = _validate_paths_under_workspace(ws, referenced)
    if not ok:
        return None, msg

    try:
        sim = MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
    except Exception as e:
        return None, f"flopy load failed: {e}"

    with _LOCK:
        _SIM_CACHE[key] = FlopySimEntry(fingerprint=fp, sim=sim)

    return sim, None


def clear_cache(ws: Optional[Path] = None) -> None:
    with _LOCK:
        if ws is None:
            _SIM_CACHE.clear()
        else:
            _SIM_CACHE.pop(str(ws.resolve()), None)
