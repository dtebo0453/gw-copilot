from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple


def _split_roots(env_val: str | None) -> List[str]:
    """Split a roots string on commas OR semicolons (supports both)."""
    if not env_val:
        return []
    # Accept both commas and semicolons as separators
    import re
    parts = [p.strip().strip('"').strip("'") for p in re.split(r"[,;]", env_val)]
    return [p for p in parts if p]


def allowed_roots() -> List[Path]:
    """Return allowed filesystem roots for search.

    Controlled by ``GW_FS_ROOTS`` (comma or semicolon separated).
    If unset, defaults to:

    - User's home directory
    - Current working directory
    - ``./runs`` (if it exists)
    """

    roots: List[Path] = []
    env_roots = _split_roots(os.getenv("GW_FS_ROOTS"))

    for p in env_roots:
        try:
            roots.append(Path(p).expanduser().resolve())
        except Exception:
            pass

    # Sensible defaults when GW_FS_ROOTS is not set
    if not env_roots:
        try:
            roots.append(Path.home().resolve())
        except Exception:
            pass

    # Always include CWD
    try:
        roots.append(Path.cwd().resolve())
    except Exception:
        pass

    # Include ./runs if it exists (common convention)
    try:
        runs = Path("runs").resolve()
        if runs.exists():
            roots.append(runs)
    except Exception:
        pass

    # de-dupe
    uniq: List[Path] = []
    seen = set()
    for r in roots:
        s = str(r)
        if s not in seen:
            uniq.append(r)
            seen.add(s)
    return uniq


def _normalize_roots(requested: List[str] | None) -> List[Path]:
    allowed = allowed_roots()
    if not requested:
        return allowed

    req_paths: List[Path] = []
    for p in requested:
        try:
            req_paths.append(Path(p).expanduser().resolve())
        except Exception:
            continue

    # Only allow requested roots that are within an allowed root (or equal)
    allowed_strs = [str(a) for a in allowed]
    out: List[Path] = []
    for rp in req_paths:
        rps = str(rp)
        if any(rps == a or rps.startswith(a + os.sep) for a in allowed_strs):
            out.append(rp)
    return out or allowed


def find_paths(
    *,
    query: str,
    kind: str = "dir",
    max_results: int = 25,
    roots: List[str] | None = None,
) -> Tuple[List[str], List[str]]:
    """Find matching dirs/files under allowed roots.

    Returns (matches, roots_used)
    """

    q = (query or "").strip()
    if len(q) < 2:
        return [], [str(p) for p in _normalize_roots(roots)]

    kind = (kind or "dir").lower().strip()
    if kind not in {"dir", "file"}:
        kind = "dir"

    max_results = int(max(1, min(max_results or 25, 200)))
    max_depth = int(os.getenv("GW_FS_MAX_DEPTH", "7"))

    roots_p = _normalize_roots(roots)
    roots_used = [str(p) for p in roots_p]

    ql = q.lower()
    matches: List[str] = []

    def consider(p: Path):
        s = str(p)
        # match either basename or full path
        if ql in p.name.lower() or ql in s.lower():
            matches.append(s)

    for root in roots_p:
        try:
            root = root.resolve()
            if not root.exists():
                continue
        except Exception:
            continue

        root_depth = len(root.parts)

        for dirpath, dirnames, filenames in os.walk(root):
            try:
                dp = Path(dirpath)
                depth = len(dp.parts) - root_depth
                if depth > max_depth:
                    dirnames[:] = []
                    continue

                if kind == "dir":
                    consider(dp)
                else:
                    for fn in filenames:
                        consider(dp / fn)

                if len(matches) >= max_results:
                    return matches[:max_results], roots_used
            except Exception:
                continue

    return matches[:max_results], roots_used
