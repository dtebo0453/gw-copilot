from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

"""Plotting + Workspace Q&A routes.

IMPORTANT:
This module historically used `gw.artifacts.resolve_workspace()` to locate the
workspace directory. That resolver can legitimately return absolute,
drive-qualified paths on Windows (e.g. `C:\\...`). For *browsing* and *Q&A*
we should use the hardened resolver in `gw.api.workspace_files` so we get:

* consistent workspace resolution across tabs
* traversal protection
* Windows drive-path blocking (unless explicitly allowed)

Using the hardened resolver prevents situations where the UI can list files in
the Model Files tab but the Q&A index is empty because it is pointing at a
different directory.
"""

from gw.api.workspace_files import resolve_workspace_root
from gw.api.workspace_scan import ensure_workspace_scan
from gw.llm.read_router import llm_route_read_plan, execute_read_plan
from gw.api.workspace_state import load_workspace_state, workspace_state_summary

router = APIRouter()


# -----------------------------
# Utils
# -----------------------------


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _plots_root(inputs_dir: str, workspace: Optional[str]) -> Path:
    # GW_Copilot layout: plots go into <inputs_dir>/GW_Copilot/plots/
    gw_copilot_cfg = Path(inputs_dir) / "GW_Copilot" / "config.json"
    if gw_copilot_cfg.exists():
        p = (Path(inputs_dir) / "GW_Copilot" / "plots").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Legacy layout: use hardened resolver
    ws = resolve_workspace_root(inputs_dir, workspace)
    p = (ws / "run_artifacts" / "plots").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run_dir(inputs_dir: str, workspace: Optional[str], run_id: str) -> Path:
    p = _plots_root(inputs_dir, workspace) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _workspace_root(inputs_dir: str, workspace: Optional[str]) -> Path:
    return resolve_workspace_root(inputs_dir, workspace)


def _safe_rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return p.name


def _is_texty(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext in {
        ".txt",
        ".csv",
        ".tsv",
        ".dat",
        ".obs",
        ".json",
        ".yaml",
        ".yml",
        ".md",
        ".nam",
        ".dis",
        ".ic",
        ".npf",
        ".chd",
        ".wel",
        ".rch",
        ".gwe",
        ".lst",
        ".out",
        ".log",
    }


def _peek_header(path: Path, max_bytes: int = 4096) -> Optional[str]:
    try:
        with path.open("rb") as f:
            blob = f.read(max_bytes)
        txt = blob.decode("utf-8", errors="replace")
        for line in txt.splitlines():
            s = line.strip()
            if s:
                return s[:500]
        return None
    except Exception:
        return None


def _build_file_index(
    ws_root: Path,
    max_files: int = 4000,
    max_peeks: int = 80,
) -> Dict[str, Any]:
    skip_dirs = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
    }
    files: List[Dict[str, Any]] = []
    peeked = 0

    for p in ws_root.rglob("*"):
        if len(files) >= max_files:
            break
        try:
            if p.is_dir():
                if p.name in skip_dirs:
                    continue
                continue
            if not p.is_file():
                continue
        except Exception:
            continue

        try:
            st = p.stat()
        except Exception:
            continue

        rel = _safe_rel(p, ws_root)
        entry: Dict[str, Any] = {
            "path": rel,
            "bytes": int(st.st_size),
            "ext": p.suffix.lower(),
        }

        if peeked < max_peeks and _is_texty(p) and st.st_size <= 2_000_000:
            hdr = _peek_header(p)
            if hdr:
                entry["peek"] = hdr
                peeked += 1

        files.append(entry)

    return {
        "workspace_root": str(ws_root),
        "files_count": len(files),
        "files": files,
    }


def _try_load_json_relaxed(blob: str) -> Optional[Any]:
    """Best-effort JSON loader with minimal sanitization.

    Fixes invalid backslash escapes inside JSON strings by escaping the backslash
    only when the escape is not one of the JSON-valid sequences.
    """
    blob = (blob or "").strip()
    try:
        return json.loads(blob)
    except Exception:
        pass

    try:
        fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', blob)
        return json.loads(fixed)
    except Exception:
        return None


def _extract_json(s: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from model text output.

    The LLM sometimes emits backslashes inside the JSON string fields (e.g. '\\plt'),
    which makes strict json.loads fail with 'invalid escape'. We apply a small
    sanitization pass that only escapes *invalid* JSON escapes (\\p -> \\\\p),
    then retry parsing.
    """
    s = (s or "").strip()

    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        blob = m.group(1)
        obj = _try_load_json_relaxed(blob)
        if isinstance(obj, dict):
            return obj

    m2 = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if not m2:
        return None

    blob = m2.group(1).strip()
    for _ in range(6):
        obj = _try_load_json_relaxed(blob)
        if isinstance(obj, dict):
            return obj
        if blob.endswith("}"):
            blob = blob[:-1]
        else:
            break
    return None

def _extract_script_from_obj(obj: Any) -> Optional[str]:
    """
    Robustly extract a script string from a model response, even if schema drifts.
    Accepts:
      - {"script": "..."}
      - {"code": "..."}
      - {"fixed_script": "..."}
      - {"plot_script": "..."}
      - {"text": "```python ...```"} (fallback)
    """
    if not isinstance(obj, dict):
        return None

    for k in ("script", "code", "fixed_script", "plot_script"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # fallback: fenced python in any text-ish field
    for k in ("text", "content", "message"):
        v = obj.get(k)
        if isinstance(v, str) and "```" in v:
            m = re.search(r"```(?:python)?\s*\n(.*?)```", v, flags=re.DOTALL | re.IGNORECASE)
            if m:
                s = m.group(1).strip()
                return s or None

    return None

def _recommend_files(file_index: Dict[str, Any], hint: str = "") -> List[Dict[str, str]]:
    """
    Cheap heuristic recommendations to help the user + LLM pick the right files.
    """
    hint_l = (hint or "").lower()
    files = file_index.get("files") or []
    scored: List[Tuple[int, str, str]] = []

    for f in files:
        if not isinstance(f, dict):
            continue
        p = str(f.get("path") or "")
        ext = str(f.get("ext") or "")
        if not p:
            continue
        pl = p.lower()
        score = 0

        # Strong signals
        if "obs" in pl or "observation" in pl:
            score += 40
        if pl.endswith(".csv"):
            score += 20
        if pl.endswith(".lst") or pl.endswith(".out") or pl.endswith(".log"):
            score += 10
        if pl.endswith(".dis"):
            score += 15
        if pl.endswith(".nam"):
            score += 10
        if pl.endswith(".hds") or pl.endswith(".cbc"):
            score += 35

        # Hint matching
        if hint_l and hint_l in pl:
            score += 60
        if any(k in hint_l for k in ("head", "water level", "wl", "stage", "hds")) and (".hds" in pl or "head" in pl):
            score += 40
        if any(k in hint_l for k in ("budget", "cbc", "flow")) and (".cbc" in pl or "budget" in pl):
            score += 40
        if any(k in hint_l for k in ("grid", "dis", "dimensions")) and pl.endswith(".dis"):
            score += 50

        reason = f"{ext} file"
        if "peek" in f:
            reason = f"{ext} file (peek: {str(f.get('peek'))[:120]})"

        scored.append((score, p, reason))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, str]] = []
    for score, p, reason in scored[:12]:
        if score <= 0:
            break
        out.append({"path": p, "reason": reason})
    return out


def _default_script_template() -> str:
    return """#!/usr/bin/env python3
\"\"\"
GW Copilot Plot Script (reproducible)

Rules:
- Read inputs ONLY via gw_plot_io.safe_* helpers where possible.
- Write outputs ONLY into GW_PLOT_OUTDIR (use gw_plot_io.out_path / safe_savefig).

Environment:
  GW_WORKSPACE_ROOT: absolute path to workspace root
  GW_PLOT_OUTDIR: absolute path to output folder for this run
\"\"\"

from pathlib import Path
import os
import matplotlib.pyplot as plt

import gw_plot_io
import gw_mf6_io as io  # provided by the runner


def main():
    outdir = Path(os.environ["GW_PLOT_OUTDIR"])
    outdir.mkdir(parents=True, exist_ok=True)

    # Example / fallback: simple plot
    x = [0, 1, 2, 3, 4]
    y = [0, 1, 4, 9, 16]
    plt.figure()
    plt.plot(x, y)
    plt.title("Example plot (planner could not determine inputs)")
    plt.xlabel("x")
    plt.ylabel("y")

    io.safe_savefig("plot.png", dpi=160, bbox_inches="tight")
    print("Wrote", io.out_path("plot.png"))


if __name__ == "__main__":
    main()
"""


def _gw_plot_io_module() -> str:
    return r'''from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

# FloPy-first integration (optional). The plot sandbox may or may not have flopy
# installed; we fail-soft and fall back to text parsers.
try:
    import flopy  # type: ignore
    HAVE_FLOPY = True
except Exception:
    flopy = None  # type: ignore
    HAVE_FLOPY = False

WS_ROOT = Path(os.environ.get("GW_WORKSPACE_ROOT", ".")).resolve()
OUTDIR = Path(os.environ.get("GW_PLOT_OUTDIR", ".")).resolve()

def _resolve_inside_ws(rel: str) -> Path:
    p = (WS_ROOT / rel).resolve()
    if not str(p).lower().startswith(str(WS_ROOT).lower()):
        raise ValueError(f"path escapes workspace: {rel}")
    return p

def list_files() -> List[str]:
    out: List[str] = []
    for p in WS_ROOT.rglob("*"):
        try:
            if p.is_file():
                out.append(p.resolve().relative_to(WS_ROOT).as_posix())
        except Exception:
            continue
    return out

def find_one(pattern: str) -> str:
    """
    Return the first matching relative path based on case-insensitive substring match.
    Use this instead of trying to Path(list_files()).
    """
    pat = (pattern or "").lower().strip()
    if not pat:
        raise ValueError("find_one(pattern) requires a non-empty pattern")
    hits = [f for f in list_files() if pat in f.lower()]
    if not hits:
        raise FileNotFoundError(f"No workspace file matched: {pattern}")
    return hits[0]

def ws_path(rel: str) -> Path:
    """
    Safe absolute path inside workspace (for binary readers like flopy).
    """
    return _resolve_inside_ws(rel)

def safe_read_text(rel: str, max_bytes: int = 2_000_000) -> str:
    p = _resolve_inside_ws(rel)
    b = p.read_bytes()
    if len(b) > max_bytes:
        raise ValueError(f"file too large for safe_read_text: {rel} ({len(b)} bytes)")
    return b.decode("utf-8", errors="replace")

def safe_read_csv(rel: str, **kwargs) -> pd.DataFrame:
    p = _resolve_inside_ws(rel)
    return pd.read_csv(p, **kwargs)

def safe_read_tsv(rel: str, **kwargs) -> pd.DataFrame:
    p = _resolve_inside_ws(rel)
    return pd.read_csv(p, sep="\\t", **kwargs)

def safe_write_text(name: str, content: str) -> Path:
    p = (OUTDIR / name).resolve()
    if not str(p).lower().startswith(str(OUTDIR).lower()):
        raise ValueError("write outside output dir is not allowed")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p

def out_path(name: str) -> Path:
    p = (OUTDIR / name).resolve()
    if not str(p).lower().startswith(str(OUTDIR).lower()):
        raise ValueError("output path escapes OUTDIR")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def safe_savefig(name: str, fig: Optional[object] = None, **kwargs) -> Path:
    import matplotlib.pyplot as plt
    path = out_path(name)
    if fig is None:
        plt.savefig(path, **kwargs)
    else:
        fig.savefig(path, **kwargs)
    return path
'''




def _gw_mf6_io_module() -> str:
    # Deterministic, read-only MF6 text parsers for plot sandbox.
    return r'''from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

# FloPy integration (optional). The plot sandbox may or may not have flopy
# installed; we fail-soft and fall back to text parsers.
try:
    import flopy  # type: ignore
    HAVE_FLOPY = True
except Exception:
    flopy = None  # type: ignore
    HAVE_FLOPY = False

_BEGIN_RE = re.compile(r"^\s*BEGIN\s+([A-Z0-9_]+)(?:\s+(.+))?\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^\s*END\s+([A-Z0-9_]+)\s*$", re.IGNORECASE)
_PERIOD_RE = re.compile(r"^\s*BEGIN\s+PERIOD\s+(\d+)\s*$", re.IGNORECASE)
_ENDPERIOD_RE = re.compile(r"^\s*END\s+PERIOD\s*$", re.IGNORECASE)
_OPEN_CLOSE_RE = re.compile(r"^\s*OPEN/CLOSE\s+(.+?)\s*$", re.IGNORECASE)

_COMMENT_RE = re.compile(r"^\s*#")
_INLINE_COMMENT_RE = re.compile(r"(.*?)\s*(#.*)?$")

def _strip_inline_comment(line: str) -> str:
    m = _INLINE_COMMENT_RE.match(line)
    return (m.group(1) if m else line).strip()

def _ws_root() -> Path:
    return Path(os.environ.get("GW_WORKSPACE_ROOT", ".")).resolve()

def _resolve_rel(path: Union[str, Path]) -> Path:
    p = Path(path)
    ws = _ws_root()
    if p.is_absolute():
        try:
            p.resolve().relative_to(ws)
        except Exception:
            raise ValueError("absolute paths outside workspace are not allowed")
        return p.resolve()
    return (ws / p).resolve()

def _safe_read_lines(rel: Union[str, Path], max_bytes: int = 5_000_000) -> List[str]:
    p = _resolve_rel(rel)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = p.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    txt = data.decode("utf-8", errors="ignore")
    return txt.splitlines()

def _expand_open_close_in_lines(lines: Iterable[str], base_dir: Path, max_bytes: int = 5_000_000) -> List[str]:
    out: List[str] = []
    ws = _ws_root()
    for raw in lines:
        line = raw.strip()
        if not line or _COMMENT_RE.match(line):
            continue
        line = _strip_inline_comment(line)
        if not line:
            continue
        m = _OPEN_CLOSE_RE.match(line)
        if m:
            token = m.group(1).strip().split()[0].strip('"').strip("'")
            ref = (base_dir / token).resolve()
            try:
                ref.relative_to(ws)
            except Exception:
                continue
            if ref.exists():
                data = ref.read_bytes()
                if len(data) > max_bytes:
                    data = data[:max_bytes]
                txt = data.decode("utf-8", errors="ignore")
                out.extend(txt.splitlines())
            continue
        out.append(line)
    return out

@dataclass
class MF6TextFile:
    relpath: str
    blocks: Dict[str, List[str]]
    periods: Dict[int, List[str]]
    raw_lines: List[str]

def parse_mf6_text(rel: Union[str, Path], max_bytes: int = 5_000_000) -> MF6TextFile:
    p = _resolve_rel(rel)
    base_dir = p.parent
    raw_lines = _safe_read_lines(rel, max_bytes=max_bytes)

    blocks: Dict[str, List[str]] = {}
    periods: Dict[int, List[str]] = {}

    cur_block: Optional[str] = None
    cur_period: Optional[int] = None
    buf_block: List[str] = []
    buf_period: List[str] = []

    for raw in raw_lines:
        s = raw.strip()
        if not s or _COMMENT_RE.match(s):
            continue
        s = _strip_inline_comment(s)
        if not s:
            continue

        pm = _PERIOD_RE.match(s)
        if pm:
            if cur_block and buf_block:
                blocks.setdefault(cur_block.upper(), []).extend(_expand_open_close_in_lines(buf_block, base_dir, max_bytes=max_bytes))
                buf_block = []
            cur_block = None
            cur_period = int(pm.group(1))
            buf_period = []
            continue

        if _ENDPERIOD_RE.match(s):
            if cur_period is not None:
                periods[cur_period] = _expand_open_close_in_lines(buf_period, base_dir, max_bytes=max_bytes)
            cur_period = None
            buf_period = []
            continue

        bm = _BEGIN_RE.match(s)
        if bm and cur_period is None:
            if cur_block and buf_block:
                blocks.setdefault(cur_block.upper(), []).extend(_expand_open_close_in_lines(buf_block, base_dir, max_bytes=max_bytes))
                buf_block = []
            cur_block = bm.group(1)
            continue

        em = _END_RE.match(s)
        if em and cur_period is None:
            blk = em.group(1)
            if cur_block and cur_block.upper() == blk.upper():
                blocks.setdefault(cur_block.upper(), []).extend(_expand_open_close_in_lines(buf_block, base_dir, max_bytes=max_bytes))
            cur_block = None
            buf_block = []
            continue

        if cur_period is not None:
            buf_period.append(s)
        elif cur_block is not None:
            buf_block.append(s)
        else:
            blocks.setdefault("HEADER", []).append(s)

    if cur_block and buf_block:
        blocks.setdefault(cur_block.upper(), []).extend(_expand_open_close_in_lines(buf_block, base_dir, max_bytes=max_bytes))
    if cur_period is not None and buf_period:
        periods[cur_period] = _expand_open_close_in_lines(buf_period, base_dir, max_bytes=max_bytes)

    return MF6TextFile(relpath=str(rel), blocks=blocks, periods=periods, raw_lines=raw_lines)

def _try_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _is_number(x: str) -> bool:
    return _try_float(x) is not None

def parse_stress_package(rel: Union[str, Path], value_cols: int = 1, keep_aux: bool = True) -> pd.DataFrame:
    mf = parse_mf6_text(rel)
    rows: List[Dict[str, object]] = []
    for per, lines in mf.periods.items():
        for ln in lines:
            u = ln.upper()
            if u.startswith("BEGIN ") or u.startswith("END "):
                continue
            parts = ln.split()
            if len(parts) < 1 + value_cols:
                continue

            # Stress-package CELLID shape varies by discretization:
            #  - DIS:   (k, i, j)
            #  - DISV:  (k, icpl)
            #  - DISU:  (node)
            # We do NOT require a specific shape here; we just need to extract values robustly.
            cell_fields: Dict[str, int] = {}
            idx0 = 0
            parsed = False

            # 3-int (lay,row,col)
            if len(parts) >= 3 + value_cols:
                try:
                    cell_fields = {"lay": int(parts[0]), "row": int(parts[1]), "col": int(parts[2])}
                    idx0 = 3
                    parsed = True
                except Exception:
                    parsed = False

            # 2-int (lay,cell)
            if not parsed and len(parts) >= 2 + value_cols:
                try:
                    cell_fields = {"lay": int(parts[0]), "cell": int(parts[1])}
                    idx0 = 2
                    parsed = True
                except Exception:
                    parsed = False

            # 1-int (node)
            if not parsed and len(parts) >= 1 + value_cols:
                try:
                    cell_fields = {"node": int(parts[0])}
                    idx0 = 1
                    parsed = True
                except Exception:
                    parsed = False

            if not parsed:
                continue

            vals: List[float] = []
            ok = True
            for i in range(value_cols):
                f = _try_float(parts[idx0 + i])
                if f is None:
                    ok = False
                    break
                vals.append(f)
            if not ok:
                continue

            rest = parts[idx0 + value_cols :]
            boundname = None
            if rest and not _is_number(rest[-1]):
                boundname = rest[-1]
                rest = rest[:-1]

            rec: Dict[str, object] = {"per": per}
            rec.update(cell_fields)
            rec["cellid"] = ",".join(str(v) for v in cell_fields.values())
            for i, v in enumerate(vals, start=1):
                rec[f"v{i}"] = v
            if keep_aux:
                rec["aux"] = rest
            if boundname is not None:
                rec["boundname"] = boundname
            rows.append(rec)

    if not rows:
        return pd.DataFrame(rows)

    df = pd.DataFrame(rows)

    # MF6 period inheritance: periods without explicit BEGIN PERIOD blocks
    # inherit data from the previous defined period.  Fill those gaps so
    # callers see complete per-period data.
    try:
        defined_periods = sorted(df["per"].unique())
        # Determine total number of stress periods from TDIS
        total_periods = int(defined_periods[-1])  # default: highest defined
        try:
            for _f in gw_plot_io.list_files():
                if _f.lower().endswith(".tdis") or _f.lower() == "mfsim.tdis":
                    _tdf = read_tdis_times(_f)
                    if len(_tdf) > total_periods:
                        total_periods = len(_tdf)
                    break
        except Exception:
            pass

        need_fill = False
        for _p in range(1, total_periods + 1):
            if _p not in defined_periods:
                need_fill = True
                break

        if need_fill:
            _fill_parts = []
            _last_def = defined_periods[0]
            for _p in range(1, total_periods + 1):
                if _p in defined_periods:
                    _last_def = _p
                _chunk = df[df["per"] == _last_def].copy()
                _chunk["per"] = _p
                _fill_parts.append(_chunk)
            df = pd.concat(_fill_parts, ignore_index=True)
    except Exception:
        pass  # If inheritance logic fails, return the raw parsed data

    return df

def read_wel(rel: Union[str, Path]) -> pd.DataFrame:
    df = parse_stress_package(rel, value_cols=1, keep_aux=True)
    if not df.empty:
        df = df.rename(columns={"v1": "q"})
    return df

def read_chd(rel: Union[str, Path]) -> pd.DataFrame:
    df = parse_stress_package(rel, value_cols=1, keep_aux=True)
    if not df.empty:
        df = df.rename(columns={"v1": "head"})
    return df

def read_ghb(rel: Union[str, Path]) -> pd.DataFrame:
    df = parse_stress_package(rel, value_cols=2, keep_aux=True)
    if not df.empty:
        df = df.rename(columns={"v1": "bhead", "v2": "cond"})
    return df

def read_riv(rel: Union[str, Path]) -> pd.DataFrame:
    df = parse_stress_package(rel, value_cols=3, keep_aux=True)
    if not df.empty:
        df = df.rename(columns={"v1": "stage", "v2": "cond", "v3": "rbot"})
    return df

def read_drn(rel: Union[str, Path]) -> pd.DataFrame:
    df = parse_stress_package(rel, value_cols=2, keep_aux=True)
    if not df.empty:
        df = df.rename(columns={"v1": "elev", "v2": "cond"})
    return df

def read_evt(rel: Union[str, Path]) -> pd.DataFrame:
    return parse_stress_package(rel, value_cols=1, keep_aux=True)

def read_rch(rel: Union[str, Path]) -> pd.DataFrame:
    return parse_stress_package(rel, value_cols=1, keep_aux=True)

def read_tdis_times(rel: Union[str, Path] = "mfsim.tdis") -> pd.DataFrame:
    mf = parse_mf6_text(rel)
    perdata = mf.blocks.get("PERIODDATA", [])
    perlen: List[float] = []
    for ln in perdata:
        parts = ln.split()
        if not parts:
            continue
        f = _try_float(parts[0])
        if f is None:
            continue
        perlen.append(f)
    t = 0.0
    out = []
    for i, L in enumerate(perlen, start=1):
        t0, t1 = t, t + L
        out.append({"per": i, "perlen": L, "t_start": t0, "t_end": t1, "t_mid": 0.5*(t0+t1)})
        t = t1
    return pd.DataFrame(out)

def read_nam(rel: Union[str, Path]) -> pd.DataFrame:
    lines = _safe_read_lines(rel, max_bytes=2_000_000)
    rows=[]
    in_files=False
    for raw in lines:
        s=_strip_inline_comment(raw.strip())
        if not s:
            continue
        u=s.upper()
        if u.startswith("BEGIN FILES"):
            in_files=True
            continue
        if u.startswith("END FILES"):
            in_files=False
            continue
        if not in_files:
            continue
        parts=s.split()
        if len(parts) < 2:
            continue
        ftype=parts[0]
        fname=parts[1].strip('"').strip("'")
        pkgname=parts[2] if len(parts) > 2 else None
        rows.append({"ftype": ftype, "fname": fname, "pkgname": pkgname})
    return pd.DataFrame(rows)

def read_package_records(rel: Union[str, Path]) -> pd.DataFrame:
    """Generic MF6 record reader.

    Returns a DataFrame with columns: block, per, tokens (list[str]) and raw.
    Useful for quick exploration when a specialized parser is not available.
    """
    mf = parse_mf6_text(rel)
    rows: List[Dict[str, object]] = []
    for bname, blines in mf.blocks.items():
        if bname.upper() == "PERIODDATA":
            # Keep as a normal block; period parsing is handled elsewhere.
            pass
        for ln in blines:
            toks = ln.split()
            if not toks:
                continue
            rows.append({"block": bname, "per": None, "tokens": toks, "raw": ln})
    # Period blocks
    for per, plines in mf.periods.items():
        for ln in plines:
            toks = ln.split()
            if not toks:
                continue
            rows.append({"block": "PERIOD", "per": per, "tokens": toks, "raw": ln})
    return pd.DataFrame(rows)

def load_simulation(mfsim_rel: Optional[Union[str, Path]] = None):
    """Load a MODFLOW 6 simulation via FloPy (if available).

    - If mfsim_rel is None, this will search for 'mfsim.nam' in the workspace root.
    - Returns the loaded flopy.mf6.MFSimulation, or None if FloPy isn't available.
    """
    if not HAVE_FLOPY:
        return None
    ws = _ws_root()
    if mfsim_rel is None:
        cand = ws / "mfsim.nam"
        if not cand.exists():
            # Fall back: find first mfsim.nam anywhere under workspace (cheap scan)
            for p in ws.rglob("mfsim.nam"):
                cand = p
                break
        mfsim_path = cand
    else:
        mfsim_path = _resolve_rel(mfsim_rel)
    if not mfsim_path.exists():
        raise FileNotFoundError(str(mfsim_path))
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(mfsim_path.parent), sim_name="mfsim", exe_name=None)
    return sim
'''


def _sitecustomize_module() -> str:
    """
    Sandbox:
      - READ allowed from: workspace + python env (sys.prefix/base_prefix) + temp + user home
      - WRITE allowed only to OUTDIR
    """
    return r'''from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path

# Encourage a non-GUI backend (also set in env in plots_run)
os.environ.setdefault("MPLBACKEND", "Agg")

WS_ROOT = Path(os.environ.get("GW_WORKSPACE_ROOT", ".")).resolve()
OUTDIR = Path(os.environ.get("GW_PLOT_OUTDIR", ".")).resolve()

_real_open = builtins.open

def _norm(p: Path) -> str:
    try:
        return str(p.resolve()).lower()
    except Exception:
        return str(p).lower()

def _allowed_read_roots():
    roots = []
    roots.append(WS_ROOT)

    # Python environment roots
    try:
        roots.append(Path(sys.prefix).resolve())
    except Exception:
        pass
    try:
        roots.append(Path(sys.base_prefix).resolve())
    except Exception:
        pass

    # temp
    tmp = os.environ.get("TEMP") or os.environ.get("TMP")
    if tmp:
        try:
            roots.append(Path(tmp).resolve())
        except Exception:
            pass

    # user home (matplotlib config/cache reads are ok)
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if home:
        try:
            roots.append(Path(home).resolve())
        except Exception:
            pass

    return [r for r in roots if r]

ALLOWED_READ = _allowed_read_roots()

def _is_under(path: Path, root: Path) -> bool:
    try:
        return _norm(path).startswith(_norm(root))
    except Exception:
        return False

def _allowed_read(path: Path) -> bool:
    for root in ALLOWED_READ:
        if _is_under(path, root):
            return True
    return False

def _allowed_write(path: Path) -> bool:
    return _is_under(path, OUTDIR)

def guarded_open(file, mode="r", *args, **kwargs):
    if isinstance(file, int):
        return _real_open(file, mode, *args, **kwargs)

    p = Path(file)
    m = mode or "r"
    wants_write = any(ch in m for ch in ("w", "a", "x", "+"))

    try:
        rp = p.resolve()
    except Exception:
        rp = p

    if wants_write:
        if not _allowed_write(rp):
            raise PermissionError(f"write outside output dir is not allowed: {rp}")
    else:
        if not _allowed_read(rp):
            raise PermissionError(f"read outside allowed roots is not allowed: {rp}")

    return _real_open(file, mode, *args, **kwargs)

builtins.open = guarded_open
'''


def _call_openai_json(system: str, user: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    openai>=1.x compatible. Returns parsed JSON object.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(f"openai package not available: {e}")

    client = OpenAI()

    use_model = (
        model
        or os.environ.get("GW_PLOT_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4o-mini"
    )

    try:
        resp = client.responses.create(
            model=use_model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.output_text
    except Exception:
        chat = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        text = (chat.choices[0].message.content or "").strip()

    obj = _extract_json(text)
    if not obj:
        raise RuntimeError(f"Planner did not return valid JSON. Raw:\n{text[:1400]}")
    return obj


def _call_anthropic_json(system: str, user: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Anthropic Claude API. Returns parsed JSON object.
    """
    try:
        import anthropic  # type: ignore
    except Exception as e:
        raise RuntimeError(f"anthropic package not available: {e}")

    from gw.api.llm_config import get_api_key

    api_key = get_api_key("anthropic")
    if not api_key:
        raise RuntimeError("Anthropic API key not configured. Set ANTHROPIC_API_KEY or configure in Settings.")

    client = anthropic.Anthropic(api_key=api_key)

    use_model = (
        model
        or os.environ.get("GW_PLOT_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or "claude-sonnet-4-20250514"
    )

    resp = client.messages.create(
        model=use_model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    # Extract text from response content blocks
    text = ""
    if hasattr(resp, "content") and resp.content:
        parts = []
        for block in resp.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        text = "".join(parts).strip()

    # Strip markdown code fences if present
    text = _strip_code_fences_json(text)

    obj = _extract_json(text)
    if not obj:
        raise RuntimeError(f"Planner did not return valid JSON. Raw:\n{text[:1400]}")
    return obj


def _strip_code_fences_json(s: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` fences wrapping JSON."""
    s = (s or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _llm_is_configured() -> bool:
    """Check whether any LLM provider has a valid API key."""
    from gw.api.llm_config import get_api_key
    return bool(get_api_key("openai") or get_api_key("anthropic"))


def _call_llm_json(system: str, user: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Provider-agnostic LLM JSON call. Routes to OpenAI or Anthropic based on active config.
    Falls back to the other provider if the primary one's package is not installed.
    """
    from gw.api.llm_config import get_active_provider, get_api_key
    provider = get_active_provider()

    errors: List[str] = []

    # Try primary provider first
    if provider == "anthropic":
        try:
            return _call_anthropic_json(system=system, user=user, model=model)
        except RuntimeError as e:
            err_str = str(e)
            errors.append(f"Anthropic: {err_str}")
            # If the package is missing or key not set, try OpenAI fallback
            if "not available" in err_str or "not configured" in err_str:
                if get_api_key("openai"):
                    try:
                        return _call_openai_json(system=system, user=user, model=model)
                    except Exception as e2:
                        errors.append(f"OpenAI fallback: {e2}")
            raise RuntimeError(
                "LLM call failed. " + " | ".join(errors) + "\n\n"
                "To fix: install the anthropic package (pip install anthropic) "
                "and set ANTHROPIC_API_KEY, or configure OpenAI instead."
            )
    else:
        try:
            return _call_openai_json(system=system, user=user, model=model)
        except RuntimeError as e:
            err_str = str(e)
            errors.append(f"OpenAI: {err_str}")
            if "not available" in err_str or "not configured" in err_str:
                if get_api_key("anthropic"):
                    try:
                        return _call_anthropic_json(system=system, user=user, model=model)
                    except Exception as e2:
                        errors.append(f"Anthropic fallback: {e2}")
            raise RuntimeError(
                "LLM call failed. " + " | ".join(errors) + "\n\n"
                "To fix: install the openai package (pip install openai) "
                "and set OPENAI_API_KEY, or configure Anthropic instead."
            )


def _llm_understand(
    prompt: str,
    model_brief: str,
    compact_files: List[str],
    probe_results: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Phase 1 of two-phase planning: understand what the user wants.

    Returns a dict with approach, data_sources, plot_type, requires_flopy,
    layer, time.  Returns None on failure.
    """
    system = (
        "You are a groundwater modeling expert. Analyze the user's plot request "
        "and determine the best approach. Return ONLY a JSON object with:\n"
        "  approach: string describing the plotting approach in 1-2 sentences\n"
        "  data_sources: list of file types/names needed (e.g., ['.hds', '.wel'])\n"
        "  plot_type: one of 'contour_map', 'time_series', 'bar_chart', 'scatter', 'heatmap', 'other'\n"
        "  requires_flopy: boolean - does this plot need FloPy binary readers?\n"
        "  layer: integer layer number if spatial plot (null if not applicable)\n"
        "  time: 'last', 'first', 'all', or specific time value (null if not applicable)\n"
        "  complexity: 'simple', 'moderate', 'complex'\n"
    )
    if model_brief:
        system += f"\nMODEL CONTEXT:\n{model_brief}\n"

    user = json.dumps({
        "prompt": prompt,
        "available_files": compact_files[:500],
        "output_probes": probe_results,
    }, ensure_ascii=False)

    try:
        return _call_llm_json(system=system, user=user)
    except Exception:
        return None


def _llm_generate_script(
    understanding: Dict[str, Any],
    model_brief: str,
    compact_files: List[str],
    probe_results: Optional[Dict[str, Any]],
    full_system_prompt: str,
    full_user_payload: str,
) -> Dict[str, Any]:
    """Phase 2 of two-phase planning: generate the script with approach context.

    Takes the understanding from phase 1 and the full system/user prompts
    from the regular plan flow, but prepends the approach guidance.
    """
    approach_section = (
        f"\n\nAPPROACH (from analysis phase):\n"
        f"- Approach: {understanding.get('approach', 'N/A')}\n"
        f"- Data sources: {understanding.get('data_sources', [])}\n"
        f"- Plot type: {understanding.get('plot_type', 'other')}\n"
        f"- Requires FloPy: {understanding.get('requires_flopy', False)}\n"
        f"- Target layer: {understanding.get('layer', 'N/A')}\n"
        f"- Target time: {understanding.get('time', 'N/A')}\n"
    )

    enhanced_system = full_system_prompt + approach_section
    return _call_llm_json(system=enhanced_system, user=full_user_payload)



def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    # Remove ```python ... ``` or ``` ... ``` fences if present
    if s.startswith("```"):
        # drop first line
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        # drop trailing fence line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _coerce_script(obj: Dict[str, Any], fallback: str = "") -> str:
    """Try hard to extract a python script string from a model JSON response."""
    if not isinstance(obj, dict):
        return fallback

    # Preferred key
    cand = obj.get("script")

    # Common alternates from some models / prompt drift
    if not cand:
        for k in ("fixed_script", "corrected_script", "plot_script", "python_script", "code", "python"):
            if obj.get(k):
                cand = obj.get(k)
                break

    # Sometimes nested like {"code": {"python": "..."}}
    if isinstance(cand, dict):
        for k in ("script", "python", "code", "content", "text"):
            if cand.get(k):
                cand = cand.get(k)
                break

    # Sometimes list of blocks
    if isinstance(cand, list):
        parts = []
        for item in cand:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for k in ("python", "script", "text", "content"):
                    if isinstance(item.get(k), str):
                        parts.append(item.get(k))
                        break
        cand = "\n\n".join([p for p in parts if p]).strip()

    if isinstance(cand, str):
        cand = _strip_code_fences(cand).strip()

    return cand if isinstance(cand, str) and cand.strip() else fallback


def _validate_files_used(files_used: Any, file_index: Dict[str, Any]) -> List[str]:
    if not isinstance(files_used, list):
        return []
    existing = {
        f.get("path")
        for f in (file_index.get("files") or [])
        if isinstance(f, dict) and f.get("path")
    }
    return [p for p in files_used if isinstance(p, str) and p in existing]


def _is_missing_requested_file(text: str) -> Optional[str]:
    """
    Detect common "missing file" errors. If it looks like the script requested a file that doesn't exist,
    we should stop repair loops and ask for clarification.

    IMPORTANT: call this with stderr+stdout combined because some tracebacks appear in stdout.
    """
    if not text:
        return None

    m = re.search(r"No workspace file matched:\s*(.+)", text)
    if m:
        return m.group(1).strip()

    m2 = re.search(
        r"FileNotFoundError:\s*\[Errno 2\]\s*No such file or directory:\s*'([^']+)'",
        text,
    )
    if m2:
        return m2.group(1).strip()

    m3 = re.search(
        r'FileNotFoundError:\s*\[Errno 2\]\s*No such file or directory:\s*"([^"]+)"',
        text,
    )
    if m3:
        return m3.group(1).strip()

    return None





def read_cbc_term_timeseries(workspace_root: str, *, cbc_rel: str, term: str = 'WEL'):
    """Return (times, totals) for a budget term from a .cbc file.

    This is a pragmatic fallback when package stress data are driven by TS6 or OPEN/CLOSE
    files that are not easily expanded.

    Requires flopy at runtime.
    """
    import os
    try:
        import flopy  # type: ignore
        from flopy.utils import CellBudgetFile  # type: ignore
    except Exception as e:
        raise RuntimeError(f"FloPy is required to read CBC budgets (pip install flopy). {e}")

    cbc_path = os.path.join(workspace_root, cbc_rel)
    if not os.path.exists(cbc_path):
        raise FileNotFoundError(cbc_path)

    cb = CellBudgetFile(cbc_path, precision='double')
    times = cb.get_times()
    tot = []
    # Normalize term lookups
    term_u = term.upper()
    for t in times:
        # Try exact term, then common variants
        rec = None
        for key in (term_u, term_u + 'S', 'WEL', 'WELLS'):
            try:
                rec = cb.get_data(text=key, totim=t)
                if rec is not None:
                    break
            except Exception:
                continue
        if rec is None:
            tot.append(float('nan'))
            continue
        # rec can be list of arrays
        s = 0.0
        try:
            for arr in (rec if isinstance(rec, list) else [rec]):
                # arr may be ndarray or recarray with 'q'
                if hasattr(arr, 'dtype') and arr.dtype.fields and 'q' in arr.dtype.fields:
                    s += float(arr['q'].sum())
                else:
                    import numpy as np
                    s += float(np.asarray(arr).sum())
        except Exception:
            pass
        tot.append(s)
    return times, tot



# -----------------------------
# API endpoints
# -----------------------------


@router.post("/plots/plan-agentic")
def plots_plan_agentic(payload: Dict[str, Any]):
    """Agentic plot planning and execution — uses the chat agent's full tool loop.

    This endpoint gives the plot tab the same power as the chatbox: the LLM can
    iteratively read workspace files, inspect binary outputs, generate a plot
    script, execute it in the sandbox, and self-correct if it fails — all within
    a single request.

    Returns: same shape as plots_plan but also includes run results (images, stdout, stderr)
    when execution succeeds.
    """
    prompt = str(payload.get("prompt") or "").strip()
    inputs_dir = str(payload.get("inputs_dir") or "").strip()
    workspace = payload.get("workspace", None)

    if not inputs_dir:
        raise HTTPException(status_code=422, detail="inputs_dir is required")
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt is required")

    if not _llm_is_configured():
        raise HTTPException(
            status_code=400,
            detail="LLM not configured. Set an API key (OpenAI or Anthropic) in Settings.",
        )

    ws_root = _workspace_root(inputs_dir, workspace)

    # Load workspace scan for context
    try:
        scan = ensure_workspace_scan(ws_root, force=False)
    except Exception:
        scan = {}
    snapshot = scan.get("snapshot") or {}

    # Build a plot-focused system prompt that tells the agent to use the
    # generate_plot tool.  This reuses all the same domain knowledge the
    # chatbox gets but focuses the agent on plotting.
    from gw.api.model_snapshot import build_model_brief

    model_brief = build_model_brief(snapshot)
    file_index = scan.get("file_index") or _build_file_index(ws_root)
    compact_files = sorted(list(file_index.keys()))[:1200]

    # Binary data summaries so agent can validate data ranges
    binary_data_summary: Dict[str, Any] = {}
    try:
        from gw.api.output_probes import extract_hds_data, extract_cbc_data
        for fpath in sorted(file_index.keys()):
            ext = fpath.rsplit(".", 1)[-1].lower() if "." in fpath else ""
            if ext == "hds" and "hds" not in binary_data_summary:
                result = extract_hds_data(ws_root, fpath, max_times=3, max_chars=6000)
                if result.get("ok"):
                    binary_data_summary["hds"] = result["summary_text"]
            elif ext == "cbc" and "cbc" not in binary_data_summary:
                result = extract_cbc_data(ws_root, fpath, max_records=6, max_chars=6000)
                if result.get("ok"):
                    binary_data_summary["cbc"] = result["summary_text"]
    except Exception:
        pass

    system = (
        "You are a groundwater modeling plotting assistant with full workspace access.\n\n"
        "Your job is to create a plot based on the user's request. You have tools to:\n"
        "- list_files: discover workspace files\n"
        "- read_file: read any text file (MF6 packages, config, etc.)\n"
        "- read_binary_output: extract data from .hds/.cbc binary files\n"
        "- generate_plot: execute a Python plotting script in a sandbox\n\n"
        "WORKFLOW:\n"
        "1. First, use list_files and/or read_file to understand what data is available.\n"
        "2. Write a complete Python script using the sandbox helpers.\n"
        "3. Call generate_plot with both a prompt description AND the full script.\n"
        "4. If the plot fails, read the error, fix the script, and try again.\n"
        "5. When the plot succeeds, embed the image in your response as markdown.\n\n"
        "CRITICAL SCRIPT RULES:\n"
        "- NEVER use hardcoded absolute paths. ALWAYS use gw_plot_io helpers.\n"
        "- import gw_plot_io for workspace file access.\n"
        "- gw_plot_io.ws_path(rel) — get absolute path to a workspace file.\n"
        "- gw_plot_io.out_path('name.png') — get output path for saving plots.\n"
        "- gw_plot_io.find_one('.hds') — find a file by extension/name.\n"
        "- import gw_mf6_io for MF6 package readers (read_wel, read_chd, read_tdis_times, etc.).\n"
        "- gw_mf6_io is FloPy-first with text parsing fallback.\n"
        "- Do NOT treat MF6 package files as CSV. Use gw_mf6_io readers.\n"
        "- Use gw_plot_io.safe_read_csv ONLY for actual CSV files (MF6 OBS outputs, etc.).\n"
        "- Save plots with: plt.savefig(gw_plot_io.out_path('name.png'), dpi=150, bbox_inches='tight')\n"
        "  or: gw_plot_io.safe_savefig('name.png')\n"
        "- Output dir: Path(os.environ['GW_PLOT_OUTDIR']) or gw_plot_io.out_path()\n\n"
        "MF6 STRESS PERIOD INHERITANCE:\n"
        "In MODFLOW 6, if a stress period has no explicit BEGIN PERIOD block, it inherits\n"
        "data from the previous period. gw_mf6_io.read_wel() and similar functions handle\n"
        "this automatically — the DataFrame includes rows for ALL periods.\n\n"
        "COOKBOOK:\n"
        "1. Read heads: hds = flopy.utils.HeadFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.hds'))))\n"
        "2. Read budget: cbc = flopy.utils.CellBudgetFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.cbc'))), precision='double')\n"
        "3. Read WEL: df = gw_mf6_io.read_wel(gw_plot_io.find_one('.wel'))  # cols: per, lay, row, col, cellid, q\n"
        "4. Read TDIS: tdf = gw_mf6_io.read_tdis_times()  # cols: per, perlen, t_start, t_end, t_mid\n"
        "5. Load simulation: sim = gw_mf6_io.load_simulation(); gwf = sim.get_model(sim.model_names[0])\n"
        "6. Read OBS CSV: df = gw_plot_io.safe_read_csv(gw_plot_io.find_one('.obs.csv'))\n\n"
        "GRID TYPE NOTES:\n"
        "- DIS (structured): arrays shape (nlay, nrow, ncol). Use imshow/contourf.\n"
        "- DISV (vertex): arrays shape (nlay, ncpl). MUST use flopy.plot.PlotMapView.\n"
        "- DISU (unstructured): arrays shape (nodes,). MUST use flopy.plot.PlotMapView.\n\n"
        "When the plot is generated successfully, embed the image URL in your response:\n"
        "![Plot Description](url)\n\n"
        "IMPORTANT: You MUST call generate_plot with a complete script. Do not just return\n"
        "a script as text — actually execute it via the generate_plot tool.\n"
    )

    if model_brief:
        system += f"\nMODEL CONTEXT:\n{model_brief}\n"

    if binary_data_summary:
        system += "\nBINARY DATA SUMMARY (actual extracted values):\n"
        for key, val in binary_data_summary.items():
            summary_str = val if isinstance(val, str) else str(val)
            system += f"\n--- {key.upper()} ---\n{summary_str[:4000]}\n"

    system += f"\nWORKSPACE FILES ({len(compact_files)} files):\n"
    system += "\n".join(compact_files[:200])
    if len(compact_files) > 200:
        system += f"\n... and {len(compact_files) - 200} more files"
    system += "\n"

    # Build conversation messages
    messages = [{"role": "user", "content": prompt}]

    # Run through the agentic tool loop
    try:
        from gw.llm.tool_loop import tool_loop

        reply_text, audit = tool_loop(
            messages=messages,
            system=system,
            ws_root=ws_root,
            inputs_dir=inputs_dir,
            workspace=workspace,
            max_iterations=15,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agentic plot generation failed: {type(e).__name__}: {e}",
        )

    # Parse the reply to find any generated plot images
    # The generate_plot tool returns image URLs that the LLM embeds as markdown
    image_urls = re.findall(r'!\[.*?\]\((.*?)\)', reply_text)

    # Find the latest run_id from the audit log (if generate_plot was called)
    run_id = ""
    script = ""
    stdout = ""
    stderr = ""
    exit_code = None
    outputs: List[Dict[str, str]] = []

    for entry in reversed(audit):
        if entry.get("tool") == "generate_plot" and entry.get("ok"):
            # The tool was called successfully — we need to find the run_id
            # from the run directory
            break

    # Scan run_artifacts/plots for the most recent run
    try:
        plots_root = _plots_root(inputs_dir, workspace)
        recent_runs = sorted(plots_root.iterdir(), reverse=True)
        for run_dir in recent_runs[:3]:
            if not run_dir.is_dir():
                continue
            manifest = run_dir / "plot_manifest.json"
            if manifest.exists():
                mobj = json.loads(manifest.read_text(encoding="utf-8"))
                req_json = run_dir / "plot_request.json"
                if req_json.exists():
                    robj = json.loads(req_json.read_text(encoding="utf-8"))

                run_id = run_dir.name
                exit_code = mobj.get("exit_code")

                # Read script
                script_path = run_dir / "plot_script.py"
                if script_path.exists():
                    script = script_path.read_text(encoding="utf-8")

                # Read stdout/stderr
                stdout_path = run_dir / "plot_stdout.txt"
                stderr_path = run_dir / "plot_stderr.txt"
                if stdout_path.exists():
                    stdout = stdout_path.read_text(encoding="utf-8")
                if stderr_path.exists():
                    stderr = stderr_path.read_text(encoding="utf-8")

                # Collect output files
                ignore_files = {
                    "plot_request.json", "plot_script.py", "gw_plot_io.py",
                    "gw_mf6_io.py", "sitecustomize.py", "plot_stdout.txt",
                    "plot_stderr.txt", "plot_manifest.json", "mplconfig",
                }
                for p in sorted(run_dir.iterdir()):
                    if p.is_file() and p.name not in ignore_files:
                        outputs.append({"name": p.name, "path": str(p)})
                break
    except Exception:
        pass

    return {
        "status": "ok",
        "notes": reply_text,
        "questions": [],
        "recommendations": [],
        "files_used": [],
        "script": script,
        "context_hash": "",
        "validation_warnings": [],
        # Run results (so frontend can display immediately)
        "run_id": run_id,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "outputs": outputs,
    }


@router.post("/plots/plan")
def plots_plan(payload: Dict[str, Any]):
    """
    LLM-driven plot planning (NO execution). Strictly grounded in workspace files.
    """
    prompt = str(payload.get("prompt") or "").strip()
    inputs_dir = str(payload.get("inputs_dir") or "").strip()
    workspace = payload.get("workspace", None)
    selected_files = payload.get("selected_files") or []

    if not inputs_dir:
        raise HTTPException(status_code=422, detail="inputs_dir is required")
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt is required")
    if not isinstance(selected_files, list):
        raise HTTPException(status_code=422, detail="selected_files must be a list")

    ws_root = _workspace_root(inputs_dir, workspace)
    # Prefer cached workspace scan/index so planning is grounded and fast.
    scan = ensure_workspace_scan(ws_root, force=False)
    file_index = scan.get("file_index") or _build_file_index(ws_root)
    snapshot = scan.get("snapshot") or {}

    # Per-workspace memory helps planner stay consistent across turns.
    try:
        state = load_workspace_state(ws_root)
        state_summary = workspace_state_summary(state)
    except Exception:
        state = {}
        state_summary = ""

    ctx = {
        "prompt": prompt,
        "inputs_dir": inputs_dir,
        "workspace_root": str(ws_root),
        "selected_files": selected_files,
        "workspace_state_summary": state_summary,
        "file_index_hash": _sha256_text(json.dumps(file_index, sort_keys=True, ensure_ascii=False)[:20000]),
    }
    context_hash = _sha256_text(json.dumps(ctx, sort_keys=True, ensure_ascii=False))

    # Use the LLM router to propose small, bounded reads.
    # This makes planning resilient to typos / phrasing differences and helps the planner
    # generate MF6-aware scripts without asking the user to specify files.
    router_reads_ctx = None
    try:
        router_ctx = {
            "task": "plan_plot",
            "prompt": prompt,
            "selected_files": selected_files,
            "workspace_root": str(ws_root),
            "workspace_files": sorted(list(file_index.keys()))[:1200],
            "workspace_snapshot": snapshot,
            "workspace_health": (scan.get("health") or {}),
        }
        plan = llm_route_read_plan(question=prompt, router_context=router_ctx)
        if plan and isinstance(plan, dict):
            router_reads_ctx = execute_read_plan(ws_root=ws_root, scan=scan, plan=plan)
    except Exception:
        router_reads_ctx = None


    if not _llm_is_configured():
        return {
            "status": "needs_clarification",
            "notes": "LLM not configured. Set an API key (OpenAI or Anthropic) in Settings to enable LLM-driven plotting.",
            "questions": ["Configure an LLM provider in Settings, then retry Plan."],
            "recommendations": _recommend_files(file_index, hint=prompt),
            "files_used": [],
            "script": _default_script_template(),
            "context_hash": context_hash,
        }

    system = (
        "You are a groundwater modeling plotting assistant.\n"
        "You MUST only use files that exist in the provided workspace file index.\n"
        "You MUST generate a Python script that reads inputs ONLY from within the workspace\n"
        "and writes outputs ONLY to GW_PLOT_OUTDIR.\n"
        "Use matplotlib for plotting (Agg backend is fine). You may use pandas/numpy.\n\n"
        "CRITICAL:\n"
        "- Treat MODFLOW 6 input files as structured text (BEGIN/END blocks, PERIOD data).\n"
        "- You MUST use gw_mf6_io for reading ANY MODFLOW 6 package or control file when possible (.nam, .tdis, .dis, .ims, .wel, .chd, .ghb, .riv, .drn, .rcha, .oc, etc.).\n"
        "  - gw_mf6_io is FloPy-first: it will use FloPy loaders when available and fall back to robust text parsing.\n"
        "  - Do NOT treat MF6 package files as CSV.\n"
        "- Use gw_plot_io.safe_read_csv only for true delimited tables (e.g., MF6 OBS csv outputs) or known CSV inputs.\n"


        "- For binary readers (flopy HeadFile/CellBudgetFile), use gw_plot_io.ws_path(rel).\n"
        "- Prefer gw_plot_io.find_one('filename.ext') to locate specific files.\n"
        "- NEVER do Path(gw_plot_io.list_files()) or join on the list.\n"
        "- NEVER treat the literal string 'GW_PLOT_OUTDIR' as a folder name.\n"
        "  Always use Path(os.environ['GW_PLOT_OUTDIR']) or gw_plot_io.out_path().\n"
        "- If the requested file is not in the file index, you MUST return status='needs_clarification'.\n"
        "- Avoid using .lst for time-series heads unless there is no better source.\n"
        "- Prefer MF6 OBS CSV outputs first.\n\n"
        "IMPORTANT — MF6 STRESS PERIOD INHERITANCE:\n"
        "In MODFLOW 6, if a stress period has no explicit BEGIN PERIOD block in a package\n"
        "file, it inherits data from the previous stress period. The gw_mf6_io.read_wel()\n"
        "and similar functions handle this automatically — the returned DataFrame includes\n"
        "rows for ALL periods including inherited ones. You can safely group by or filter\n"
        "on the 'per' column and expect complete data for every stress period.\n\n"
        "IMPORTANT — USE ACTUAL DATA FROM binary_data_summary:\n"
        "The context includes binary_data_summary with actual extracted head statistics\n"
        "and budget data. Use these REAL values to set appropriate axis ranges, contour\n"
        "levels, and validate your script logic against actual model magnitudes.\n\n"
        "COOKBOOK (common patterns the script can use):\n"
        "1. Read heads from .hds binary file:\n"
        "   import flopy\n"
        "   hds_path = gw_plot_io.ws_path(gw_plot_io.find_one('.hds'))\n"
        "   hds = flopy.utils.HeadFile(str(hds_path))\n"
        "   times = hds.get_times()\n"
        "   head_array = hds.get_data(totim=times[-1])  # shape: (nlay, nrow, ncol)\n\n"
        "2. Read budget from .cbc binary file:\n"
        "   cbc_path = gw_plot_io.ws_path(gw_plot_io.find_one('.cbc'))\n"
        "   cbc = flopy.utils.CellBudgetFile(str(cbc_path), precision='double')\n"
        "   records = cbc.get_unique_record_names()\n\n"
        "3. Read WEL package data:\n"
        "   import gw_mf6_io\n"
        "   df = gw_mf6_io.read_wel(gw_plot_io.find_one('.wel'))\n"
        "   # df columns: per, lay, row, col, cellid, q, aux, boundname\n\n"
        "4. Read TDIS for time mapping:\n"
        "   tdf = gw_mf6_io.read_tdis_times()  # auto-finds mfsim.tdis\n"
        "   # tdf columns: per, perlen, t_start, t_end, t_mid\n\n"
        "5. Load full simulation via FloPy:\n"
        "   sim = gw_mf6_io.load_simulation()\n"
        "   if sim is not None:\n"
        "       gwf = sim.get_model(sim.model_names[0])\n"
        "       head = gwf.output.head().get_data()\n\n"
        "6. Read OBS CSV output:\n"
        "   df = gw_plot_io.safe_read_csv(gw_plot_io.find_one('.obs.csv'))\n\n"
        "7. Read DIS grid info:\n"
        "   dis_file = gw_mf6_io.parse_mf6_text(gw_plot_io.find_one('.dis'))\n"
        "   # dis_file.blocks has 'DIMENSIONS', 'GRIDDATA', etc.\n"
        "   # Or load via FloPy: sim = gw_mf6_io.load_simulation(); gwf = sim.get_model(sim.model_names[0]); dis = gwf.dis\n\n"
        "8. DISV/DISU head plotting via PlotMapView (when grid is NOT DIS):\n"
        "   sim = gw_mf6_io.load_simulation()\n"
        "   gwf = sim.get_model(sim.model_names[0])\n"
        "   hds = flopy.utils.HeadFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.hds'))))\n"
        "   data = hds.get_data(totim=hds.get_times()[-1])\n"
        "   fig, ax = plt.subplots()\n"
        "   pmv = flopy.plot.PlotMapView(model=gwf, layer=0, ax=ax)\n"
        "   contour = pmv.plot_array(data[0])  # DISV shape: (nlay, ncpl); DISU shape: (nodes,)\n"
        "   pmv.plot_grid(linewidth=0.3, color='grey', alpha=0.3)\n\n"
        "GRID TYPE NOTES:\n"
        "- DIS (structured): arrays have shape (nlay, nrow, ncol). Use imshow/contourf directly.\n"
        "- DISV (vertex): arrays have shape (nlay, ncpl). MUST use flopy.plot.PlotMapView.\n"
        "- DISU (unstructured): arrays have shape (nodes,). MUST use flopy.plot.PlotMapView.\n"
        "- Check the MODEL CONTEXT section below for the actual grid type of this model.\n\n"
        "Return ONLY a single JSON object with keys:\n"
        "  status: 'ok' or 'needs_clarification'\n"
        "  notes: short explanation\n"
        "  questions: array of clarification questions (empty if status='ok')\n"
        "  recommendations: array of {path, reason} suggestions (can be empty)\n"
        "  files_used: array of relative paths you intend to read (must exist)\n"
        "  script: full python script (as a string)\n"
    )

    # Inject model context brief so the LLM knows the model structure
    from gw.api.model_snapshot import build_model_brief
    model_brief = build_model_brief(snapshot)
    if model_brief:
        system += f"\n\nMODEL CONTEXT:\n{model_brief}\n"

    # Compact file list instead of full file index to save tokens
    compact_files = sorted(list(file_index.keys()))[:1200]

    # Probe binary outputs for LLM context
    output_probes = None
    try:
        from gw.api.output_probes import probe_workspace_outputs
        output_probes = probe_workspace_outputs(ws_root, file_index)
    except Exception:
        pass

    # Extract actual binary data summaries so the planner knows real value ranges,
    # budget magnitudes, and layer statistics (the chat agent gets this but the plot
    # planner previously only got metadata probes).
    binary_data_summary: Dict[str, Any] = {}
    try:
        from gw.api.output_probes import extract_hds_data, extract_cbc_data

        for fpath in sorted(file_index.keys()):
            ext = fpath.rsplit(".", 1)[-1].lower() if "." in fpath else ""
            if ext == "hds" and "hds" not in binary_data_summary:
                result = extract_hds_data(ws_root, fpath, max_times=3, max_chars=8000)
                if result.get("ok"):
                    binary_data_summary["hds"] = result["summary_text"]
            elif ext == "cbc" and "cbc" not in binary_data_summary:
                result = extract_cbc_data(ws_root, fpath, max_records=6, max_chars=8000)
                if result.get("ok"):
                    binary_data_summary["cbc"] = result["summary_text"]
    except Exception:
        pass

    # Cross-package analysis from model profile
    cross_pkg_analysis = None
    try:
        from gw.api.model_profile import get_model_profile
        profile = get_model_profile(ws_root)
        cross_pkg_analysis = profile.get("cross_package_analysis")
    except Exception:
        pass

    user = json.dumps(
        {
            "prompt": prompt,
            "selected_files": selected_files,
            "workspace_files": compact_files,
            "workspace_snapshot": snapshot,
            "workspace_health": (scan.get("health") or {}),
            "output_probes": output_probes,
            "binary_data_summary": binary_data_summary or None,
            "cross_package_analysis": cross_pkg_analysis,
            "router_reads": router_reads_ctx,
            "helper_api": {
                "gw_plot_io": [
                    "list_files()",
                    "find_one(pattern)",
                    "ws_path(rel)",
                    "safe_read_csv(rel, ...)",
                    "safe_read_text(rel, ...)",
                    "out_path(name)",
                    "safe_savefig(name, fig=None, **kwargs)",
                ],
                "gw_mf6_io": [
                    "read_wel(rel) -> DataFrame[per, lay, row, col, cellid, q, aux, boundname]",
                    "read_chd(rel) -> DataFrame[per, lay, row, col, cellid, head, aux, boundname]",
                    "read_ghb(rel) -> DataFrame[per, lay, row, col, cellid, bhead, cond, aux, boundname]",
                    "read_drn(rel) -> DataFrame[per, lay, row, col, cellid, elev, cond, aux, boundname]",
                    "read_riv(rel) -> DataFrame[per, lay, row, col, cellid, stage, cond, rbot, aux, boundname]",
                    "read_nam(rel) -> DataFrame[ftype, fname, pkgname]",
                    "read_tdis_times(rel='mfsim.tdis') -> DataFrame[per, perlen, t_start, t_end, t_mid]",
                    "read_package_records(rel) -> DataFrame[block, per, tokens, raw]",
                    "parse_mf6_text(rel) -> MF6TextFile with .blocks dict[str, list[str]] and .periods dict[int, list[str]]",
                    "load_simulation(mfsim_rel=None) -> flopy.mf6.MFSimulation or None (if FloPy unavailable)",
                ],
                "env": ["GW_WORKSPACE_ROOT", "GW_PLOT_OUTDIR"],
                "output_rule": "Write plots into GW_PLOT_OUTDIR using gw_plot_io.out_path/safe_savefig.",
            },
        },
        ensure_ascii=False,
    )

    # Two-phase planning: understand first, then generate script
    try:
        understanding = _llm_understand(
            prompt=prompt,
            model_brief=model_brief,
            compact_files=compact_files,
            probe_results=output_probes,
        )
        if understanding:
            obj = _llm_generate_script(
                understanding=understanding,
                model_brief=model_brief,
                compact_files=compact_files,
                probe_results=output_probes,
                full_system_prompt=system,
                full_user_payload=user,
            )
        else:
            # Phase 1 failed; fall back to single-call
            obj = _call_llm_json(system=system, user=user)
    except Exception as e:
        return {
            "status": "needs_clarification",
            "notes": f"Planner failed; using fallback template. Error: {type(e).__name__}: {e}",
            "questions": [
                "Which file contains the observed data (CSV)?",
                "Which file contains simulated output (MF6 OBS CSV, or another CSV)?",
                "What are the relevant column names (time and value)?",
            ],
            "recommendations": _recommend_files(file_index, hint=prompt),
            "files_used": [],
            "script": _default_script_template(),
            "context_hash": context_hash,
        }

    status = str(obj.get("status") or "ok").strip()
    notes = str(obj.get("notes") or "").strip()
    questions = obj.get("questions") or []
    recommendations = obj.get("recommendations") or []
    files_used = _validate_files_used(obj.get("files_used") or [], file_index)
    script = _coerce_script(obj, fallback="")

    if not script:
        status = "needs_clarification"
        notes = notes or "Planner returned no script; using fallback."
        script = _default_script_template()

    # Enforce MF6 readers: the planner must NOT treat MF6 package/control files as CSV.
    if _script_violates_mf6_io(script):
        status = "needs_clarification"
        notes = (notes + " " if notes else "") + "Planner script treated a MODFLOW 6 file as CSV. Please regenerate using gw_mf6_io readers."
        script = _default_script_template()

    if status not in {"ok", "needs_clarification"}:
        status = "ok"

    # Pre-validate script
    validation_warnings = _pre_validate_script(script, file_index, snapshot)

    # Store understanding for follow-up requests (e.g., "now show layer 2")
    if understanding:
        try:
            from gw.api.workspace_state import load_workspace_state, save_workspace_state
            ws_state = load_workspace_state(ws_root)
            ws_state["last_plot_understanding"] = understanding
            save_workspace_state(ws_root, ws_state)
        except Exception:
            pass

    return {
        "status": status,
        "notes": notes,
        "questions": questions if isinstance(questions, list) else [],
        "recommendations": recommendations if isinstance(recommendations, list) else [],
        "files_used": files_used,
        "script": script,
        "context_hash": context_hash,
        "validation_warnings": validation_warnings,
    }

def _extract_script_from_obj(obj: Any) -> Optional[str]:
    """Robustly extract a script from an LLM response dict, even if schema drifts."""
    if not isinstance(obj, dict):
        return None

    for k in ("script", "code", "fixed_script", "plot_script"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for k in ("text", "content", "message", "notes"):
        v = obj.get(k)
        if isinstance(v, str) and "```" in v:
            m = re.search(r"```(?:python)?\s*\n(.*?)```", v, flags=re.DOTALL | re.IGNORECASE)
            if m and m.group(1).strip():
                return m.group(1).strip()

    return None


# -----------------------------
# Error classification + repair hints
# -----------------------------


def _classify_error(stderr: str, stdout: str) -> Dict[str, Any]:
    """Classify a script error to guide repair hints."""
    combined = (stderr or "") + "\n" + (stdout or "")
    cl = combined.lower()

    if "filenotfounderror" in cl or "no such file" in cl:
        # Extract filename
        m = re.search(r"(?:FileNotFoundError|No such file).*?['\"]([^'\"]+)['\"]", combined, re.IGNORECASE)
        return {"type": "file_not_found", "detail": m.group(1) if m else None}

    if "modulenotfounderror" in cl or "no module named" in cl:
        m = re.search(r"No module named ['\"]?([^'\";\s]+)", combined, re.IGNORECASE)
        module = m.group(1) if m else None
        return {"type": "import_error", "detail": module, "is_flopy": module and "flopy" in module.lower()}

    if "cannot reshape array" in cl or "shape" in cl and "mismatch" in cl:
        return {"type": "shape_mismatch", "detail": None}

    if "flopy" in cl and ("error" in cl or "exception" in cl or "traceback" in cl):
        return {"type": "flopy_error", "detail": None}

    if "permissionerror" in cl or "permission denied" in cl:
        return {"type": "permission_error", "detail": None}

    if "keyerror" in cl:
        m = re.search(r"KeyError:\s*['\"]?([^'\";\s]+)", combined)
        return {"type": "key_error", "detail": m.group(1) if m else None}

    if "timeout" in cl or "timed out" in cl:
        return {"type": "timeout", "detail": None}

    return {"type": "unknown", "detail": None}


def _find_similar_files(file_index: Dict[str, Any], pattern: str, max_results: int = 5) -> List[str]:
    """Find files in the workspace similar to a given pattern."""
    if not pattern:
        return []

    pattern_lower = pattern.lower()
    pattern_ext = ""
    if "." in pattern:
        pattern_ext = pattern.rsplit(".", 1)[-1].lower()
    pattern_stem = pattern.rsplit(".", 1)[0].lower() if "." in pattern else pattern_lower

    scored: List[Tuple[float, str]] = []
    for f in (file_index.get("files") or []):
        if not isinstance(f, dict):
            continue
        p = (f.get("path") or "").strip()
        if not p:
            continue
        p_lower = p.lower()
        name = p_lower.rsplit("/", 1)[-1] if "/" in p_lower else p_lower

        score = 0.0
        # Extension match
        if pattern_ext and name.endswith("." + pattern_ext):
            score += 3.0
        # Name substring match
        if pattern_stem in name:
            score += 5.0
        elif any(part in name for part in pattern_stem.split("_")):
            score += 2.0
        # Exact name match
        if name == pattern_lower:
            score += 10.0

        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:max_results]]


def _build_repair_hints(error_class: Dict[str, Any], file_index: Dict[str, Any], snapshot: Dict[str, Any]) -> str:
    """Build contextual repair hints based on error classification."""
    hints: List[str] = []
    etype = error_class.get("type", "unknown")
    detail = error_class.get("detail")
    grid = snapshot.get("grid", {})

    if etype == "file_not_found" and detail:
        similar = _find_similar_files(file_index, detail)
        if similar:
            hints.append(f"File '{detail}' not found. Similar files available: {', '.join(similar)}")
        else:
            hints.append(f"File '{detail}' not found. No similar files found in workspace.")

    elif etype == "import_error":
        if error_class.get("is_flopy"):
            hints.append(
                "FloPy import failed. Use gw_mf6_io text parsers as fallback:\n"
                "  - gw_mf6_io.parse_mf6_text(rel) for package files\n"
                "  - gw_mf6_io.read_wel/read_chd/etc. for stress packages"
            )
        else:
            hints.append(
                f"Module '{detail}' not available. Only these packages are in the sandbox: "
                "stdlib, numpy, pandas, matplotlib, flopy, gw_plot_io, gw_mf6_io."
            )

    elif etype == "shape_mismatch":
        nlay = grid.get("nlay")
        nrow = grid.get("nrow")
        ncol = grid.get("ncol")
        gtype = (grid.get("type") or "").upper()
        if nlay and nrow and ncol:
            hints.append(f"Grid dimensions: {gtype} with NLAY={nlay}, NROW={nrow}, NCOL={ncol}. "
                         f"HDS array shape is ({nlay}, {nrow}, {ncol}).")
        ncpl = grid.get("ncpl")
        if ncpl:
            hints.append(f"Grid is DISV with NLAY={nlay}, NCPL={ncpl}. Array shape is ({nlay}, {ncpl}).")
        nodes = grid.get("nodes")
        if nodes:
            hints.append(f"Grid is DISU with NODES={nodes}. Array shape is ({nodes},).")

    elif etype == "flopy_error":
        flopy_ok = snapshot.get("flopy_available", False)
        if not flopy_ok:
            hints.append(
                "FloPy is not available. Rewrite the script using gw_mf6_io text parsers instead."
            )
        else:
            hints.append(
                "FloPy error occurred. Consider using gw_mf6_io text parsers as fallback."
            )

    elif etype == "key_error" and detail:
        hints.append(f"KeyError on '{detail}'. Check column/key names carefully.")

    elif etype == "permission_error":
        hints.append("Permission error. Ensure all file writes go to GW_PLOT_OUTDIR via gw_plot_io.out_path().")

    elif etype == "timeout":
        hints.append("Script timed out. Simplify the computation (e.g., use fewer timesteps, sample data).")

    return "\n".join(hints)


def _pre_validate_script(script: str, file_index: Dict[str, Any], snapshot: Dict[str, Any]) -> List[Dict[str, str]]:
    """Pre-validate a generated script for common issues before execution.

    Returns a list of {level: 'error'|'warning', message: str, fix_hint: str}.
    """
    warnings: List[Dict[str, str]] = []
    if not script or not script.strip():
        return warnings

    # Check for find_one() patterns that don't match any file
    find_one_calls = re.findall(r"find_one\(\s*['\"]([^'\"]+)['\"]\s*\)", script)
    known_files = set()
    for f in (file_index.get("files") or []):
        if isinstance(f, dict):
            p = (f.get("path") or "").strip().lower()
            if p:
                known_files.add(p)
                # Also add just the filename
                known_files.add(p.rsplit("/", 1)[-1] if "/" in p else p)

    for pattern in find_one_calls:
        pat_lower = pattern.lower()
        # Check if any file matches the pattern
        matched = any(pat_lower in fn for fn in known_files)
        if not matched:
            similar = _find_similar_files(file_index, pattern, max_results=3)
            hint = f"Similar files: {', '.join(similar)}" if similar else "No similar files found."
            warnings.append({
                "level": "error",
                "message": f"find_one('{pattern}') has no matching file in workspace.",
                "fix_hint": hint,
            })

    # Check for unguarded FloPy import when FloPy unavailable
    flopy_ok = snapshot.get("flopy_available", True)
    if not flopy_ok and "import flopy" in script:
        # Check if it's guarded with try/except
        if "try:" not in script or "import flopy" not in script.split("try:")[-1].split("except")[0] if "except" in script else script:
            warnings.append({
                "level": "warning",
                "message": "Script imports FloPy but FloPy is not available in this environment.",
                "fix_hint": "Guard with try/except or use gw_mf6_io text parsers.",
            })

    # Check output uses correct pattern
    if "GW_PLOT_OUTDIR" not in script and "out_path" not in script and "safe_savefig" not in script:
        warnings.append({
            "level": "warning",
            "message": "Script may not save output correctly (no GW_PLOT_OUTDIR/out_path/safe_savefig found).",
            "fix_hint": "Use gw_plot_io.safe_savefig() or gw_plot_io.out_path() for output.",
        })

    # Check for hardcoded absolute paths
    abs_path_patterns = [
        r'["\'][A-Z]:\\',  # Windows absolute path
        r'["\']/home/',    # Linux home
        r'["\']/Users/',   # macOS home
        r'["\']/tmp/',     # temp dir
    ]
    for pat in abs_path_patterns:
        if re.search(pat, script):
            warnings.append({
                "level": "error",
                "message": "Script contains hardcoded absolute paths.",
                "fix_hint": "Use gw_plot_io.ws_path() for workspace files and gw_plot_io.out_path() for output.",
            })
            break

    return warnings


@router.post("/plots/repair")
def plots_repair(payload: Dict[str, Any]):
    """
    LLM-driven script repair (NO execution). Uses traceback/stderr to correct the script.
    """
    prompt = str(payload.get("prompt") or "").strip()
    inputs_dir = str(payload.get("inputs_dir") or "").strip()
    workspace = payload.get("workspace", None)
    script = str(payload.get("script") or "")
    stderr = str(payload.get("stderr") or "")
    stdout = str(payload.get("stdout") or "")
    files_used_hint = payload.get("files_used") or []

    if not inputs_dir:
        raise HTTPException(status_code=422, detail="inputs_dir is required")
    if not script.strip():
        raise HTTPException(status_code=422, detail="script is required")
    if not _llm_is_configured():
        raise HTTPException(status_code=400, detail="LLM not configured. Set an API key (OpenAI or Anthropic) in Settings.")

    ws_root = _workspace_root(inputs_dir, workspace)

    # Local import to avoid startup-time circular imports
    from gw.api.model_snapshot import build_model_snapshot  # type: ignore
    # Use the scan cache if available so we can give the repair LLM better context
    # and keep behavior consistent with /plots/plan.
    try:
        scan = ensure_workspace_scan(ws_root, force=False)
    except Exception:
        scan = {}
    snapshot = scan.get("snapshot") or {}

    file_index = _build_file_index(ws_root)

    stderr_trim = stderr[-6000:] if len(stderr) > 6000 else stderr
    stdout_trim = stdout[-4000:] if len(stdout) > 4000 else stdout

    combined_err = (stderr_trim or "") + "\n" + (stdout_trim or "")
    missing = _is_missing_requested_file(combined_err)
    if missing:
        # Stop pointless repair loops if the data isn't in the workspace.
        return {
            "status": "needs_clarification",
            "notes": (
                f"It looks like the script requested a file that does not exist in the workspace: {missing}. "
                "Because reads are restricted to workspace files, I can't fix this by guessing paths."
            ),
            "questions": [
                "Is the requested file actually present somewhere else on disk (outside the workspace)?",
                "If yes, do you want to copy it into the workspace, or change the request to use an available file (e.g., OBS CSV)?",
            ],
            "recommendations": _recommend_files(file_index, hint=prompt or missing),
            "files_used": _validate_files_used(files_used_hint, file_index),
            "script": script,
            "context_hash": _sha256_text(json.dumps({"missing": missing, "inputs_dir": inputs_dir}, sort_keys=True)),
        }

    ctx = {
        "prompt": prompt,
        "inputs_dir": inputs_dir,
        "workspace_root": str(ws_root),
        "files_used_hint": files_used_hint,
        "stderr_sha": _sha256_text(stderr_trim),
        "script_sha": _sha256_text(script),
    }
    context_hash = _sha256_text(json.dumps(ctx, sort_keys=True, ensure_ascii=False))

    # Router-assisted context: propose minimal workspace reads to help repairs
    # (e.g., peek at the referenced MF6 package file or relevant blocks).
    router_reads_ctx = None
    try:
        router_ctx = {
            "task": "plan_plot",
            "prompt": prompt,
            "error": (combined_err or "")[:5000],
            "workspace_root": str(ws_root),
            "workspace_files": sorted(list(file_index.keys()))[:1200],
            "workspace_snapshot": snapshot,
            "workspace_health": (scan.get("health") or {}),
        }
        plan = llm_route_read_plan(question=(prompt + "\n\n" + (combined_err or ""))[:6000], router_context=router_ctx)
        if plan and isinstance(plan, dict):
            router_reads_ctx = execute_read_plan(ws_root=ws_root, scan=scan, plan=plan)
    except Exception:
        router_reads_ctx = None

    system = (
        "You are a groundwater modeling plotting assistant.\n"
        "Your job is to FIX the provided Python plotting script so it runs successfully.\n\n"
        "Hard constraints:\n"
        "- You MUST only use files that exist in the provided workspace file index.\n"
        "- The script MUST write outputs ONLY into GW_PLOT_OUTDIR.\n"
        "- You MUST use gw_mf6_io for reading ANY MODFLOW 6 package or control file (.nam, .tdis, .dis, .wel, .chd, .ghb, .riv, .drn, .rcha, .oc, etc.).\n"
        "  - gw_mf6_io is FloPy-first: it uses FloPy loaders when available and falls back to robust text parsing.\n"
        "  - Do NOT treat MF6 package files as CSV.\n"
        "- Use gw_plot_io.safe_read_csv only for true delimited tables (e.g., MF6 OBS csv outputs) or known CSV inputs.\n"
        "- For binary readers (flopy HeadFile/CellBudgetFile), use gw_plot_io.ws_path(rel).\n"
        "- Prefer locating files via gw_plot_io.find_one(pattern).\n"
        "- NEVER do Path(gw_plot_io.list_files()) or joinpath on the list.\n"
        "- NEVER treat the literal string 'GW_PLOT_OUTDIR' as a folder name.\n"
        "  Always use Path(os.environ['GW_PLOT_OUTDIR']) or gw_plot_io.out_path().\n"
        "- Do NOT introduce new dependencies beyond stdlib + pandas/numpy/matplotlib/flopy.\n"
        "- If the requested file is not present in the index, return status='needs_clarification'.\n\n"
        "COOKBOOK (common patterns):\n"
        "1. Read heads: hds = flopy.utils.HeadFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.hds'))))\n"
        "2. Read budget: cbc = flopy.utils.CellBudgetFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.cbc'))), precision='double')\n"
        "3. Read WEL: df = gw_mf6_io.read_wel(gw_plot_io.find_one('.wel'))  # cols: per, lay, row, col, cellid, q\n"
        "4. Read TDIS: tdf = gw_mf6_io.read_tdis_times()  # cols: per, perlen, t_start, t_end, t_mid\n"
        "5. Load simulation: sim = gw_mf6_io.load_simulation(); gwf = sim.get_model(sim.model_names[0])\n"
        "6. Read OBS CSV: df = gw_plot_io.safe_read_csv(gw_plot_io.find_one('.obs.csv'))\n"
        "7. Parse MF6 text file: mf = gw_mf6_io.parse_mf6_text(rel); mf.blocks['GRIDDATA']\n"
        "8. DISV/DISU: use flopy.plot.PlotMapView(model=gwf, layer=0) for spatial plots.\n"
        "   DIS arrays: (nlay,nrow,ncol). DISV: (nlay,ncpl). DISU: (nodes,).\n\n"
        "Return ONLY a single JSON object with keys:\n"
        "  status: 'ok' or 'needs_clarification'\n"
        "  notes: short explanation of the fix\n"
        "  questions: array of clarification questions (empty if status='ok')\n"
        "  recommendations: array of {path, reason} suggestions (can be empty)\n"
        "  files_used: array of relative paths you intend to read (must exist)\n"
        "  script: full corrected python script (as a string)\n"
    )

    # Inject model context brief for repair context
    from gw.api.model_snapshot import build_model_brief
    model_brief = build_model_brief(snapshot)
    if model_brief:
        system += f"\n\nMODEL CONTEXT:\n{model_brief}\n"

    # Classify error and inject repair hints
    error_class = _classify_error(stderr_trim, stdout_trim)
    repair_hints = _build_repair_hints(error_class, file_index, snapshot)
    if repair_hints:
        system += f"\n\nERROR ANALYSIS:\nError type: {error_class.get('type', 'unknown')}\n{repair_hints}\n"

    compact_files = sorted(list(file_index.keys()))[:1200]

    # Probe binary outputs for repair context
    output_probes = None
    try:
        from gw.api.output_probes import probe_workspace_outputs
        output_probes = probe_workspace_outputs(ws_root, file_index)
    except Exception:
        pass

    # Compact binary data summaries for repair context
    binary_data_summary: Dict[str, Any] = {}
    try:
        from gw.api.output_probes import extract_hds_data, extract_cbc_data

        for fpath in sorted(file_index.keys()):
            ext = fpath.rsplit(".", 1)[-1].lower() if "." in fpath else ""
            if ext == "hds" and "hds" not in binary_data_summary:
                result = extract_hds_data(ws_root, fpath, max_times=2, max_chars=4000)
                if result.get("ok"):
                    binary_data_summary["hds"] = result["summary_text"]
            elif ext == "cbc" and "cbc" not in binary_data_summary:
                result = extract_cbc_data(ws_root, fpath, max_records=4, max_chars=4000)
                if result.get("ok"):
                    binary_data_summary["cbc"] = result["summary_text"]
    except Exception:
        pass

    # Cross-package analysis from model profile
    cross_pkg_analysis = None
    try:
        from gw.api.model_profile import get_model_profile
        profile = get_model_profile(ws_root)
        cross_pkg_analysis = profile.get("cross_package_analysis")
    except Exception:
        pass

    user = json.dumps(
        {
            "prompt": prompt,
            "previous_script": script,
            "stderr": stderr_trim,
            "stdout_tail": stdout_trim,
            "files_used_hint": files_used_hint,
            "workspace_files": compact_files,
            "workspace_snapshot": snapshot,
            "output_probes": output_probes,
            "binary_data_summary": binary_data_summary or None,
            "cross_package_analysis": cross_pkg_analysis,
            "router_reads": router_reads_ctx,
            "helper_api": {
                "gw_plot_io": [
                    "list_files()",
                    "find_one(pattern)",
                    "ws_path(rel)",
                    "safe_read_csv(rel, ...)",
                    "safe_read_text(rel, ...)",
                    "out_path(name)",
                    "safe_savefig(name, fig=None, **kwargs)",
                ],
                "gw_mf6_io": [
                    "read_wel(rel) -> DataFrame[per, lay, row, col, cellid, q, aux, boundname]",
                    "read_chd(rel) -> DataFrame[per, lay, row, col, cellid, head, aux, boundname]",
                    "read_ghb(rel) -> DataFrame[per, lay, row, col, cellid, bhead, cond, aux, boundname]",
                    "read_drn(rel) -> DataFrame[per, lay, row, col, cellid, elev, cond, aux, boundname]",
                    "read_riv(rel) -> DataFrame[per, lay, row, col, cellid, stage, cond, rbot, aux, boundname]",
                    "read_nam(rel) -> DataFrame[ftype, fname, pkgname]",
                    "read_tdis_times(rel='mfsim.tdis') -> DataFrame[per, perlen, t_start, t_end, t_mid]",
                    "read_package_records(rel) -> DataFrame[block, per, tokens, raw]",
                    "parse_mf6_text(rel) -> MF6TextFile with .blocks dict and .periods dict",
                    "load_simulation(mfsim_rel=None) -> flopy.mf6.MFSimulation or None",
                ],
                "env": ["GW_WORKSPACE_ROOT", "GW_PLOT_OUTDIR"],
                "output_rule": "Write plots into GW_PLOT_OUTDIR using gw_plot_io.out_path/safe_savefig.",
            },
        },
        ensure_ascii=False,
    )

    try:
        obj = _call_llm_json(system=system, user=user)
    except Exception as e:
        return {
            "status": "needs_clarification",
            "notes": f"Repair failed. Error: {type(e).__name__}: {e}",
            "questions": ["Paste the key lines of the traceback and tell me which file(s) you expect contain the data."],
            "recommendations": _recommend_files(file_index, hint=prompt),
            "files_used": _validate_files_used(files_used_hint, file_index),
            "script": script,
            "context_hash": context_hash,
        }

    status = str(obj.get("status") or "ok").strip()
    notes = str(obj.get("notes") or "").strip()
    questions = obj.get("questions") or []
    recommendations = obj.get("recommendations") or []
    files_used = _validate_files_used(obj.get("files_used") or [], file_index)
    #new_script = _coerce_script(obj, fallback=script)
    extracted = _extract_script_from_obj(obj)
    new_script = extracted if extracted else script

    if status not in {"ok", "needs_clarification"}:
        status = "ok"

    # Pre-validate repaired script
    validation_warnings = _pre_validate_script(new_script, file_index, snapshot)

    return {
        "status": status,
        "notes": notes,
        "questions": questions if isinstance(questions, list) else [],
        "recommendations": recommendations if isinstance(recommendations, list) else [],
        "files_used": files_used,
        "script": new_script,
        "context_hash": context_hash,
        "validation_warnings": validation_warnings,
    }


# ---------------------------------------------------------------------------
# Security: AST-based script validation
# ---------------------------------------------------------------------------

_ALLOWED_IMPORT_ROOTS = frozenset({
    "matplotlib", "numpy", "np", "pandas", "pd", "flopy", "pathlib", "math",
    "json", "csv", "os", "sys", "datetime", "time", "warnings", "re",
    "collections", "itertools", "functools", "io", "textwrap", "copy",
    "gw_plot_io", "gw_mf6_io", "scipy", "shapely", "PIL", "mpl_toolkits",
})

_FORBIDDEN_BUILTINS = frozenset({
    "exec", "eval", "compile", "__import__", "breakpoint", "input",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
})

_FORBIDDEN_ATTRS = frozenset({
    "__code__", "__globals__", "__class__", "__bases__", "__subclasses__",
    "__dict__", "__loader__", "__builtins__",
})


def _validate_script_safety(script: str) -> List[str]:
    """AST-level safety check on a plot script before sandbox execution.

    Returns a list of error strings.  An empty list means the script passed.
    """
    import ast as _ast

    errors: List[str] = []
    try:
        tree = _ast.parse(script)
    except SyntaxError as exc:
        errors.append(f"SyntaxError: {exc}")
        return errors

    for node in _ast.walk(tree):
        # Block forbidden builtin calls
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            if node.func.id in _FORBIDDEN_BUILTINS:
                errors.append(f"Forbidden function call: {node.func.id}()")

        # Block forbidden imports
        if isinstance(node, _ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    errors.append(f"Forbidden import: {alias.name}")

        if isinstance(node, _ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                errors.append(f"Forbidden import: from {node.module}")

        # Block forbidden dunder attributes (e.g. obj.__class__)
        if isinstance(node, _ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRS:
                errors.append(f"Forbidden attribute: .{node.attr}")

    return errors


def execute_plot_in_sandbox(
    *,
    ws_root: Path,
    script: str,
    inputs_dir: str,
    prompt: str = "",
    workspace: Optional[str] = None,
    context_hash: str = "",
    files_used_hint: Optional[List[str]] = None,
    timeout_sec: int = 150,
) -> Dict[str, Any]:
    """Execute a plot script in a sandboxed subprocess.

    This is the core sandbox runner used by both the /plots/run endpoint and
    the generate_plot tool in the chat agent.  It does NOT require a confirmation
    token — callers are responsible for gating execution if needed.

    Returns dict with: run_id, run_dir, outputs, stdout, stderr, exit_code.
    Raises RuntimeError on timeout (instead of HTTPException).
    """
    # Security: AST-level safety check before execution
    safety_errors = _validate_script_safety(script)
    if safety_errors:
        logger.warning("Script blocked by safety validator: %s", safety_errors)
        raise RuntimeError(
            "Script failed safety validation:\n" + "\n".join(f"  - {e}" for e in safety_errors)
        )

    inputs_dir_abs = Path(inputs_dir).resolve()
    run_id = f"{_utc_now_compact()}_{_sha256_text(script)[:10]}"
    outdir = _run_dir(inputs_dir, workspace, run_id)

    (outdir / "gw_plot_io.py").write_text(_gw_plot_io_module(), encoding="utf-8")
    (outdir / "gw_mf6_io.py").write_text(_gw_mf6_io_module(), encoding="utf-8")
    (outdir / "sitecustomize.py").write_text(_sitecustomize_module(), encoding="utf-8")
    (outdir / "plot_script.py").write_text(script, encoding="utf-8")

    _write_json(
        outdir / "plot_request.json",
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "inputs_dir": inputs_dir,
            "inputs_dir_abs": str(inputs_dir_abs),
            "workspace_root": str(ws_root),
            "context_hash": context_hash,
            "files_used_hint": files_used_hint or [],
            "python": sys.version,
            "platform": sys.platform,
        },
    )

    mplcfg = outdir / "mplconfig"
    mplcfg.mkdir(parents=True, exist_ok=True)

    stdout_path = outdir / "plot_stdout.txt"
    stderr_path = outdir / "plot_stderr.txt"

    env = os.environ.copy()
    env["GW_WORKSPACE_ROOT"] = str(ws_root)
    env["GW_PLOT_OUTDIR"] = str(outdir)
    env["PYTHONUTF8"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(outdir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    env["MPLBACKEND"] = env.get("MPLBACKEND") or "Agg"
    env["MPLCONFIGDIR"] = str(mplcfg)

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str((outdir / "plot_script.py").resolve())],
            cwd=str(inputs_dir_abs),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        stdout_path.write_text(e.stdout or "", encoding="utf-8", errors="replace")
        stderr_path.write_text((e.stderr or "") + "\nTIMEOUT", encoding="utf-8", errors="replace")
        raise RuntimeError(f"Plot script timed out after {timeout_sec}s")

    dt = time.time() - t0
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(dt, 3),
        "exit_code": proc.returncode,
        "files": {},
    }
    for p in sorted(outdir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(outdir).as_posix()
            manifest["files"][rel] = {"bytes": p.stat().st_size, "sha256": _sha256_file(p)}
    _write_json(outdir / "plot_manifest.json", manifest)

    ignore = {
        "plot_request.json",
        "plot_script.py",
        "gw_plot_io.py",
        "gw_mf6_io.py",
        "sitecustomize.py",
        "plot_stdout.txt",
        "plot_stderr.txt",
        "plot_manifest.json",
        "mplconfig",
    }
    outputs: List[Dict[str, str]] = []
    for p in sorted(outdir.iterdir()):
        if p.is_file() and p.name not in ignore:
            outputs.append({"name": p.name, "path": str(p)})

    return {
        "run_id": run_id,
        "run_dir": str(outdir),
        "outputs": outputs,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "exit_code": proc.returncode,
    }


@router.post("/plots/run")
def plots_run(payload: Dict[str, Any]):
    """
    Execute a plot script (execution-gated), sandboxed to workspace root for reads
    and run output dir for writes.
    """
    confirm = str(payload.get("confirm") or "")
    if confirm != "confirm:run_plot":
        raise HTTPException(status_code=403, detail="missing or invalid confirm token")

    inputs_dir = str(payload.get("inputs_dir") or "").strip()
    workspace = payload.get("workspace", None)
    script = str(payload.get("script") or "")
    prompt = str(payload.get("prompt") or "").strip()
    context_hash = str(payload.get("context_hash") or "").strip()
    files_used_hint = payload.get("files_used") or []

    if not inputs_dir:
        raise HTTPException(status_code=422, detail="inputs_dir is required")
    if not script.strip():
        raise HTTPException(status_code=422, detail="script is required")

    ws_root = _workspace_root(inputs_dir, workspace)

    try:
        return execute_plot_in_sandbox(
            ws_root=ws_root,
            script=script,
            inputs_dir=inputs_dir,
            prompt=prompt,
            workspace=workspace,
            context_hash=context_hash,
            files_used_hint=files_used_hint,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/plots/runs")
def plots_runs(inputs_dir: str, workspace: Optional[str] = None, max: int = 50):
    root = _plots_root(inputs_dir, workspace)
    items = []
    for p in sorted(root.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        req = p / "plot_request.json"
        ts = None
        prompt = ""
        exit_code = None
        try:
            if req.exists():
                obj = json.loads(req.read_text(encoding="utf-8"))
                ts = obj.get("ts_utc")
                prompt = obj.get("prompt") or ""
        except Exception:
            pass
        man = p / "plot_manifest.json"
        try:
            if man.exists():
                mobj = json.loads(man.read_text(encoding="utf-8"))
                exit_code = mobj.get("exit_code")
        except Exception:
            pass
        items.append({"run_id": p.name, "ts_utc": ts, "prompt": prompt, "exit_code": exit_code})
        if len(items) >= max:
            break
    return {"runs": items}


@router.get("/plots/run/detail")
def plots_run_detail(inputs_dir: str, run_id: str, workspace: Optional[str] = None):
    """Return full detail for a single run: prompt, script, stdout, stderr, exit_code, outputs."""
    root = _plots_root(inputs_dir, workspace)
    run_dir = root / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")

    # Read plot_request.json
    prompt = ""
    ts_utc = None
    context_hash = ""
    files_used = []
    req_path = run_dir / "plot_request.json"
    try:
        if req_path.exists():
            obj = json.loads(req_path.read_text(encoding="utf-8"))
            prompt = obj.get("prompt") or ""
            ts_utc = obj.get("ts_utc")
            context_hash = obj.get("context_hash") or ""
            files_used = obj.get("files_used_hint") or []
    except Exception:
        pass

    # Read plot_script.py
    script = ""
    script_path = run_dir / "plot_script.py"
    try:
        if script_path.exists():
            script = script_path.read_text(encoding="utf-8")
    except Exception:
        pass

    # Read stdout / stderr
    stdout = ""
    stderr = ""
    try:
        sp = run_dir / "plot_stdout.txt"
        if sp.exists():
            stdout = sp.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        ep = run_dir / "plot_stderr.txt"
        if ep.exists():
            stderr = ep.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Read manifest for exit_code + duration
    exit_code = None
    duration_sec = None
    man_path = run_dir / "plot_manifest.json"
    try:
        if man_path.exists():
            mobj = json.loads(man_path.read_text(encoding="utf-8"))
            exit_code = mobj.get("exit_code")
            duration_sec = mobj.get("duration_sec")
    except Exception:
        pass

    # Collect output files (same logic as execute_plot_in_sandbox)
    ignore = {
        "plot_request.json", "plot_script.py", "gw_plot_io.py",
        "gw_mf6_io.py", "sitecustomize.py", "plot_stdout.txt",
        "plot_stderr.txt", "plot_manifest.json", "mplconfig",
    }
    outputs: List[Dict[str, str]] = []
    for p in sorted(run_dir.iterdir()):
        if p.is_file() and p.name not in ignore:
            outputs.append({"name": p.name, "path": str(p)})

    return {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "prompt": prompt,
        "script": script,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "duration_sec": duration_sec,
        "context_hash": context_hash,
        "files_used": files_used,
        "outputs": outputs,
    }


@router.delete("/plots/run/{run_id}")
def plots_run_delete(run_id: str, inputs_dir: str, workspace: Optional[str] = None):
    """Delete a single run directory."""
    root = _plots_root(inputs_dir, workspace)
    run_dir = root / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    import shutil
    shutil.rmtree(run_dir, ignore_errors=True)
    return {"status": "ok", "deleted": run_id}


@router.get("/plots/run/output")
def plots_run_output(inputs_dir: str, run_id: str, path: str, workspace: Optional[str] = None):
    """Serve an individual output file from a plot run (e.g. a PNG image).

    `path` can be:
      - an absolute path (as returned by /plots/run)
      - a relative filename (just the basename)
    The file must reside inside the run output directory.
    """
    root = _plots_root(inputs_dir, workspace)
    run_dir = root / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")

    # Accept absolute path (from run response) or just a filename
    requested = Path(path)
    if requested.is_absolute():
        # Verify the file is inside the run dir
        try:
            requested.resolve().relative_to(run_dir.resolve())
        except ValueError:
            # Fallback: use just the filename within the run dir
            requested = run_dir / requested.name
    else:
        # Relative path / bare filename
        requested = run_dir / requested

    resolved = requested.resolve()
    if not resolved.exists() or not resolved.is_file():
        # One more fallback: search for the filename anywhere in the run dir
        fname = Path(path).name
        found = list(run_dir.rglob(fname))
        if found:
            resolved = found[0].resolve()
        else:
            raise HTTPException(status_code=404, detail=f"output file not found: {fname}")

    # Guess MIME type
    import mimetypes as _mt
    mime, _ = _mt.guess_type(str(resolved))
    if not mime:
        mime = "application/octet-stream"

    return FileResponse(path=str(resolved), media_type=mime)


@router.get("/plots/run/download")
def plots_run_download(inputs_dir: str, run_id: str, workspace: Optional[str] = None):
    root = _plots_root(inputs_dir, workspace)
    run_dir = root / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="run not found")

    zip_path = root / f"{run_id}.zip"
    if zip_path.exists():
        try:
            zip_path.unlink()
        except Exception:
            pass

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(run_dir).as_posix())

    return FileResponse(
        path=str(zip_path),
        filename=f"{run_id}.zip",
        media_type="application/zip",
    )

# -----------------------------
# Workspace Q&A (read-only)
# -----------------------------

def _qa_root(inputs_dir: str, workspace: Optional[str]) -> Path:
    ws = resolve_workspace_root(inputs_dir, workspace)
    p = (ws / "run_artifacts" / "qa").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def _qa_run_dir(inputs_dir: str, workspace: Optional[str], run_id: str) -> Path:
    p = _qa_root(inputs_dir, workspace) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _resolve_ws_file(ws_root: Path, rel: str) -> Path:
    # safe resolve inside workspace (no traversal)
    rel = (rel or "").replace("\\", "/").strip()
    if not rel or rel.startswith("/") or rel.startswith("~") or ".." in rel.split("/"):
        raise ValueError("invalid path")
    p = (ws_root / rel).resolve()
    if not str(p).lower().startswith(str(ws_root).lower()):
        raise ValueError("path escapes workspace")
    if not p.exists() or not p.is_file():
        raise FileNotFoundError("file not found")
    return p

def _read_excerpt_text(path: Path, max_bytes: int) -> str:
    b = path.read_bytes()
    if len(b) > max_bytes:
        b = b[:max_bytes]
    return b.decode("utf-8", errors="replace")

def _call_llm_json_maybe(system: str, user: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Wrapper around _call_llm_json so we can use a separate model env var for Q&A if desired.
    """
    use_model = model or os.environ.get("GW_QA_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("GW_PLOT_MODEL")
    return _call_llm_json(system=system, user=user, model=use_model)


@router.post("/workspace/ask")
def workspace_ask(payload: Dict[str, Any]):
    """
    Read-only, workspace-grounded Q&A.
    The LLM never receives arbitrary filesystem access; it only sees:
      - a workspace file index (paths/sizes/ext + limited peeks)
      - excerpts from files that the server reads safely (text only, capped bytes)
    Produces audit artifacts under run_artifacts/qa/<run_id>/.
    """
    question = str(payload.get("question") or "").strip()
    inputs_dir = str(payload.get("inputs_dir") or "").strip()
    workspace = payload.get("workspace", None)
    # Conversation history for follow-up context
    raw_history = payload.get("history") or []
    conversation_history: List[Dict[str, str]] = []
    if isinstance(raw_history, list):
        for h in raw_history[-10:]:
            if isinstance(h, dict) and h.get("role") and h.get("content"):
                content = str(h["content"])
                if len(content) > 800:
                    content = content[:800] + "..."
                conversation_history.append({"role": str(h["role"]), "content": content})

    # safety caps
    max_files = int(payload.get("max_files") or 4000)
    max_peeks = int(payload.get("max_peeks") or 80)
    max_read_files = int(payload.get("max_read_files") or 12)
    max_bytes_each = int(payload.get("max_bytes_each") or 120_000)  # ~120KB per file
    total_byte_cap = int(payload.get("total_byte_cap") or 600_000)  # ~600KB total text

    if not inputs_dir:
        raise HTTPException(status_code=422, detail="inputs_dir is required")
    if not question:
        raise HTTPException(status_code=422, detail="question is required")
    if not _llm_is_configured():
        raise HTTPException(status_code=400, detail="LLM not configured. Set an API key (OpenAI or Anthropic) in Settings.")

    ws_root = _workspace_root(inputs_dir, workspace)

    # build a token-safe index (your existing helper already skips heavy dirs + peeks a few text files)
    file_index = _build_file_index(ws_root, max_files=max_files, max_peeks=max_peeks)

    # Heuristic intent inference: if the user is asking for "how to improve/diagnose/optimize"
    # the model (e.g., "improve sensitivity in layer 2"), we deterministically compute a
    # compact model snapshot to ground the answer. This does NOT depend on the word
    # "recommendation".
    def _looks_like_improvement_question(q: str) -> bool:
        ql = (q or "").lower()
        # verbs that imply advice/diagnosis
        intent_words = [
            "improve", "optimize", "increase", "decrease", "reduce", "enhance",
            "make it", "tune", "adjust", "calibrate", "sensitivity", "sensitive",
            "why is", "why isn't", "why isnt", "troubleshoot", "diagnose", "fix",
            "stabilize", "converge", "convergence", "unstable", "oscillat",
            "match", "fit", "residual", "misfit",
        ]
        # signals that the advice is about a specific part of the model
        scope_words = [
            "layer ", "lay ", "hk", "k33", "vk", "sy", "ss", "idomain",
            "chd", "wel", "ghb", "riv", "drn", "rch", "evt", "uzf",
            "heads", "drawdown", "budget", "flux",
        ]
        return any(w in ql for w in intent_words) and any(w in ql for w in scope_words)

    model_snapshot: Optional[Dict[str, Any]] = None

    # Some questions are conceptual and only refer to a MODFLOW 6 file *type* (e.g., ".chd")
    # rather than a specific file present in the workspace. In these cases we should answer
    # naturally using documentation + the offline filetype knowledge base, without forcing a
    # file-read plan that often leads to "needs_clarification".
    def _ext_mentions(q: str) -> List[str]:
        ql = (q or "").lower()
        # minimal set of common MF6-ish extensions we support in mf6_filetype_knowledge
        exts = [
            ".nam", ".dis", ".disv", ".disu", ".tdis", ".ims", ".ic", ".npf", ".sto", ".oc",
            ".chd", ".wel", ".ghb", ".riv", ".drn", ".rcha", ".rch", ".evt", ".uzf",
            ".lst", ".hds", ".cbc",
        ]
        hits: List[str] = []
        for e in exts:
            if e in ql:
                hits.append(e)
        return sorted(set(hits))[:3]

    ext_hits = _ext_mentions(question)

    def _mentions_specific_file(q: str, idx: Dict[str, Any]) -> bool:
        """Return True if q appears to reference a specific file in the workspace."""
        ql = (q or "").lower()
        files = idx.get("files") or []
        for f in files:
            if not isinstance(f, dict):
                continue
            p = str(f.get("path") or "")
            if not p:
                continue
            bn = os.path.basename(p).lower()
            if bn and bn in ql:
                return True
        return False

    def _looks_like_filetype_question(q: str) -> bool:
        ql = (q or "").lower()
        # Intent signals that the user is asking conceptually about a file type/package.
        signals = [
            "what is", "what's", "purpose", "used for", "advantage", "why use",
            "difference", "when would", "how does", "explain", "meaning of",
        ]
        return any(s in ql for s in signals)

    # If the user explicitly mentions a file that exists in the index, we should
    # always read it. This prevents an LLM planning miss where it recommends
    # other files but forgets the one asked about.
    def _explicit_mentions(q: str, existing_paths: List[str]) -> List[str]:
        ql = (q or "").lower()
        hits: List[str] = []
        for p in existing_paths:
            bn = os.path.basename(p).lower()
            if bn and bn in ql:
                hits.append(p)
        # Prefer shortest paths (usually top-level), de-dup
        out: List[str] = []
        for p in sorted(set(hits), key=lambda x: (len(x), x)):
            out.append(p)
        return out[:3]

    run_id = f"{_utc_now_compact()}_{_sha256_text(question)[:10]}"
    outdir = _qa_run_dir(inputs_dir, workspace, run_id)

    # If the question is asking for advice/diagnosis (inferred by heuristics),
    # compute a compact deterministic snapshot and persist it as part of the QA
    # artifacts so results are reproducible.
    if _looks_like_improvement_question(question):
        try:
            from gw.api.model_snapshot import build_model_snapshot  # type: ignore
            model_snapshot = build_model_snapshot(ws_root)
            _write_json(outdir / "model_snapshot.json", model_snapshot)
        except Exception:
            model_snapshot = None

    _write_json(outdir / "qa_request.json", {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "inputs_dir": inputs_dir,
        "workspace_root": str(ws_root),
        "question": question,
        "caps": {
            "max_files": max_files,
            "max_peeks": max_peeks,
            "max_read_files": max_read_files,
            "max_bytes_each": max_bytes_each,
            "total_byte_cap": total_byte_cap,
        },
    })

    # Fast-path: conceptual questions about a file type/extension (e.g., ".chd")
    # should be answered using docs + the offline knowledge base, even if we don't
    # read any workspace files. This makes the chat feel more natural.
    skip_plan = False
    if ext_hits and (not _mentions_specific_file(question, file_index)) and _looks_like_filetype_question(question):
        skip_plan = True

    # -------- Stage 1: Ask LLM which files to read --------
    # If skip_plan=True, we fabricate an empty plan and proceed directly to Stage 2.
    if skip_plan:
        plan_obj = {
            "status": "ok",
            "notes": "conceptual file-type question; no workspace file excerpts required",
            "read_requests": [],
        }
    else:
        plan_obj = None

    system_plan = (
        "You are a groundwater model workspace analyst.\n"
        "You will be given a question and a workspace file index.\n"
        "You MUST ground your plan in the file index ONLY.\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "status": "ok" | "needs_clarification",\n'
        '  "notes": "short",\n'
        '  "questions": ["..."] (only if needs_clarification),\n'
        '  "read_requests": [\n'
        '     {"path": "relative/path.ext", "reason": "why", "max_bytes": 50000}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        f"- Choose at most {max_read_files} files.\n"
        "- Prefer MODFLOW control/input files first (e.g., .nam .tdis .dis .ic .npf .sto .wel .chd etc.).\n"
        "- Prefer text sources. Do NOT request binaries (e.g., .hds/.cbc) unless absolutely necessary.\n"
        "- Keep max_bytes modest; you can request more in a follow-up question if needed.\n"
        "- ALWAYS prefer status='ok' with read_requests. Only use 'needs_clarification' when "
        "no relevant files exist in the workspace at all.\n"
        "- If recent_conversation is provided, use it to understand follow-up questions. "
        "For example, if the user previously asked about wells and the .wel file was cited, "
        "and now they ask about pumping rates or multi-screened wells, you should read the .wel file.\n"
        "- For spatial or analytical questions (e.g., 'are there clustered wells?', 'what is the "
        "drawdown pattern?'), read the relevant package file (e.g., .wel for wells, .dis for grid) "
        "AND the .dis file (for grid dimensions/spacing). The answering LLM can then analyze "
        "spatial relationships from the cell indices and grid structure.\n"
        "- For questions about model results (heads, drawdown, water budgets), include .lst file.\n"
        "- When in doubt, read more files rather than fewer — it is better to over-read than to "
        "force a 'needs_clarification' response.\n"
    )

    user_plan_data: Dict[str, Any] = {
        "question": question,
        "workspace_file_index": file_index,
    }
    if conversation_history:
        user_plan_data["recent_conversation"] = conversation_history
    user_plan = json.dumps(user_plan_data, ensure_ascii=False)

    if plan_obj is None:
        try:
            plan_obj = _call_llm_json_maybe(system=system_plan, user=user_plan)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM plan failed: {type(e).__name__}: {e}")

    status = str(plan_obj.get("status") or "ok").strip()
    notes = str(plan_obj.get("notes") or "").strip()
    q_clar = plan_obj.get("questions") or []
    read_requests = plan_obj.get("read_requests") or []

    if status not in {"ok", "needs_clarification"}:
        status = "ok"

    # validate read requests against index + safe caps
    existing = {
        f.get("path")
        for f in (file_index.get("files") or [])
        if isinstance(f, dict) and f.get("path")
    }

    # When the planner says needs_clarification but we have conversation history
    # (user is in a multi-turn conversation), try to proceed with files from
    # conversation context rather than giving up immediately.
    if status == "needs_clarification":
        # Check if conversation history references files we can read
        history_explicit: List[str] = []
        if conversation_history:
            history_text = " ".join(h.get("content", "") for h in conversation_history)
            history_explicit = _explicit_mentions(history_text, sorted(existing))
        if history_explicit:
            # Override: proceed with files from conversation context
            status = "ok"
            read_requests = [
                {"path": p, "reason": "referenced in conversation history", "max_bytes": min(max_bytes_each, 120_000)}
                for p in history_explicit
            ]
        else:
            _write_json(outdir / "qa_plan.json", plan_obj)
            return {
                "status": "needs_clarification",
                "notes": notes,
                "questions": q_clar if isinstance(q_clar, list) else [],
                "recommendations": _recommend_files(file_index, hint=question),
                "run_id": run_id,
            }

    # Force-include explicitly mentioned files (if present in index).
    # Also check conversation history for previously cited files to support follow-ups.
    explicit = _explicit_mentions(question, sorted(existing))
    if not explicit and conversation_history:
        # Look for file mentions in recent conversation (files cited in prior answers)
        history_text = " ".join(h.get("content", "") for h in conversation_history)
        explicit = _explicit_mentions(history_text, sorted(existing))
    if explicit:
        forced = [
            {"path": p, "reason": "explicitly requested by user", "max_bytes": min(max_bytes_each, 120_000)}
            for p in explicit
        ]
        if isinstance(read_requests, list):
            read_requests = forced + read_requests
        else:
            read_requests = forced

    cleaned: List[Dict[str, Any]] = []
    byte_budget = total_byte_cap

    if isinstance(read_requests, list):
        for rr in read_requests[:max_read_files]:
            if not isinstance(rr, dict):
                continue
            p = str(rr.get("path") or "").strip()
            if not p or p not in existing:
                continue
            try:
                req_bytes = int(rr.get("max_bytes") or max_bytes_each)
            except Exception:
                req_bytes = max_bytes_each
            req_bytes = max(2000, min(req_bytes, max_bytes_each))
            if req_bytes > byte_budget:
                req_bytes = max(0, byte_budget)
            if req_bytes <= 0:
                break
            cleaned.append({"path": p, "reason": str(rr.get("reason") or ""), "max_bytes": req_bytes})
            byte_budget -= req_bytes
            if byte_budget <= 0:
                break

    # -------- Read excerpts safely (text only) --------
    excerpts: List[Dict[str, Any]] = []
    files_cited: List[Dict[str, Any]] = []
    for rr in cleaned:
        rel = rr["path"]
        try:
            abs_path = _resolve_ws_file(ws_root, rel)
        except Exception:
            continue

        # only feed the model text-ish files (we'll still cite binaries by metadata if needed later)
        if not _is_texty(abs_path):
            excerpts.append({
                "path": rel,
                "note": "binary_or_unsupported_for_excerpt",
                "bytes": int(abs_path.stat().st_size),
            })
            files_cited.append({
                "path": rel,
                "sha256": _sha256_file(abs_path),
                "bytes": int(abs_path.stat().st_size),
                "excerpt_bytes": 0,
            })
            continue

        try:
            text = _read_excerpt_text(abs_path, int(rr["max_bytes"]))
        except Exception as e:
            excerpts.append({"path": rel, "note": f"read_failed: {type(e).__name__}: {e}"})
            files_cited.append({
                "path": rel,
                "sha256": _sha256_file(abs_path),
                "bytes": int(abs_path.stat().st_size),
                "excerpt_bytes": 0,
            })
            continue

        excerpts.append({
            "path": rel,
            "reason": rr.get("reason") or "",
            "excerpt": text,
            "excerpt_bytes": len(text.encode("utf-8", errors="replace")),
        })
        files_cited.append({
            "path": rel,
            "sha256": _sha256_file(abs_path),
            "bytes": int(abs_path.stat().st_size),
            "excerpt_bytes": len(text.encode("utf-8", errors="replace")),
        })

    _write_json(outdir / "qa_plan.json", plan_obj)
    _write_json(outdir / "qa_excerpts.json", excerpts)
    _write_json(outdir / "qa_files_cited.json", files_cited)

    # -------- Stage 2: Answer grounded in excerpts --------
    # Optional: retrieve documentation snippets.
    # 1) Prefer local workspace docs (./docs or ./documentation) or built-in quickref.
    # 2) If enabled, fall back to controlled web retrieval from allowlisted official sources.
    retrieved_docs: List[Dict[str, Any]] = []
    q_docs = question
    if explicit:
        try:
            from gw.llm.mf6_filetype_knowledge import guess_filetype  # type: ignore
            info0 = guess_filetype(explicit[0])
            if info0 and info0.kind:
                q_docs = f"{question} {info0.kind}"
        except Exception:
            pass
    elif ext_hits:
        # Extension-only question: still enrich the docs query with the likely package kind
        try:
            from gw.llm.mf6_filetype_knowledge import guess_filetype  # type: ignore
            info0 = guess_filetype(f"dummy{ext_hits[0]}")
            if info0 and info0.kind:
                q_docs = f"{question} {info0.kind}"
        except Exception:
            pass

    try:
        from gw.llm.docs_retriever import search_workspace_docs  # type: ignore
        retrieved_docs = search_workspace_docs(ws_root, q_docs, k=6)
    except Exception:
        retrieved_docs = []

    # If we only have the tiny built-in quickref (or nothing), optionally augment with web docs.
    try:
        from gw.llm.docs_retriever import search_web_docs  # type: ignore
        sources = [str(d.get("source") or "") for d in (retrieved_docs or []) if isinstance(d, dict)]
        has_strong_local = any(s.startswith("docs/") or s.startswith("documentation/") for s in sources)
        has_web = any(s.startswith("WEB:") for s in sources)
        if not has_strong_local and not has_web:
            web_hits = search_web_docs(ws_root, q_docs, k=6, explicit_files=explicit)
            # merge (web hits first so they are more likely to be cited)
            if web_hits:
                retrieved_docs = list(web_hits) + list(retrieved_docs or [])
    except Exception:
        pass

    # Optional: add lightweight MF6 filetype hints for the explicitly requested file.
    filetype_hints: List[Dict[str, str]] = []
    try:
        from gw.llm.mf6_filetype_knowledge import guess_filetype  # type: ignore
        for p in explicit:
            info = guess_filetype(p)
            if info:
                filetype_hints.append({
                    "path": p,
                    "kind": info.kind,
                    "purpose": info.purpose,
                    "what_to_look_for": info.what_to_look_for,
                })

        # If the question only mentioned an extension (e.g., ".chd") without naming a
        # specific workspace file, add a hint for the extension as well.
        if not explicit and ext_hits:
            for ext in ext_hits:
                info = guess_filetype(f"dummy{ext}")
                if info:
                    filetype_hints.append({
                        "path": f"EXT:{ext}",
                        "kind": info.kind,
                        "purpose": info.purpose,
                        "what_to_look_for": info.what_to_look_for,
                    })
    except Exception:
        filetype_hints = []

    system_answer = (
        "You are a groundwater model workspace analyst.\n"
        "Answer the user's question using ONLY the provided excerpts, the workspace index, and (if provided) retrieved documentation snippets, filetype hints, and a deterministic model snapshot.\n"
        "If information is missing, say so and ask a concrete follow-up question.\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "status": "ok" | "needs_clarification",\n'
        '  "answer": "string (MUST be natural-language markdown, NOT raw JSON or code)",\n'
        '  "followups": ["..."],\n'
        '  "citations": [\n'
        '     {"path": "relative/path.ext", "quote": "short snippet (<=200 chars)"}\n'
        "  ]\n"
        "}\n\n"
        "CRITICAL — Answer format rules:\n"
        "- The 'answer' field MUST contain well-written, natural-language prose using Markdown formatting.\n"
        "- Use bullet points, bold text, and clear headings to organize the answer.\n"
        "- NEVER put raw JSON, Python dicts, or code blocks as the answer. The answer is displayed directly to a human user.\n"
        "- When listing items (cells, values, etc.), format them as readable bullet points or tables, not JSON arrays.\n"
        "- Provide context and explanation, not just raw data. Explain what the data means.\n\n"
        "Rules:\n"
        "- ALWAYS prefer status='ok'. Use 'needs_clarification' ONLY when the excerpts/index genuinely lack the needed data AND you cannot provide any useful analysis.\n"
        "- Do NOT invent file contents.\n"
        "- You MAY explain general MF6 concepts if they are supported by retrieved_docs or filetype_hints; otherwise label them as general background.\n"
        "- When making a claim, cite the file path. Keep quotes short.\n"
        "- If you use the model_snapshot, cite it using path 'SNAPSHOT' with a short quote of the relevant field/value.\n"
        "- If you cite documentation snippets, use citations with path like 'DOCS:<source>' or 'WEB:<url>' and include a short quote from the snippet.\n\n"
        "MODFLOW 6 domain knowledge for analysis:\n"
        "- In MODFLOW 6 well (WEL) files, each entry is (layer, row, col, pumping_rate). "
        "Wells at the SAME (row, col) but DIFFERENT layers represent a single physical well screened across multiple layers (multi-screened well). "
        "Group by (row, col) to identify multi-screened wells.\n"
        "- Wells that share similar (row, col) values (close together) form well clusters that may cause localized drawdown cones.\n"
        "- Negative pumping rates indicate extraction; positive rates indicate injection.\n"
        "- When asked about spatial patterns, analyze (row, col) coordinates to identify clustering, spacing, and proximity.\n"
        "- When asked about multi-layer or multi-screened wells, group entries by (row, col) and check if multiple layers share the same (row, col).\n\n"
        "Conversation context:\n"
        "- If recent_conversation is provided, use it to understand follow-up questions. "
        "The user may refer to information from earlier in the conversation. "
        "Answer follow-up questions using the excerpts and your understanding of the conversation context.\n"
        "- Analyze the file data provided in the excerpts thoroughly — look for patterns, "
        "compare values, compute statistics (min/max/mean), identify spatial relationships, "
        "and provide insightful analysis.\n"
    )

    user_answer_data: Dict[str, Any] = {
        "question": question,
        "workspace_file_index": file_index,
        "excerpts": excerpts,
        "retrieved_docs": retrieved_docs,
        "filetype_hints": filetype_hints,
        "model_snapshot": model_snapshot,
    }
    if conversation_history:
        user_answer_data["recent_conversation"] = conversation_history
    user_answer = json.dumps(user_answer_data, ensure_ascii=False)

    try:
        ans_obj = _call_llm_json_maybe(system=system_answer, user=user_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM answer failed: {type(e).__name__}: {e}")

    ans_status = str(ans_obj.get("status") or "ok").strip()
    if ans_status not in {"ok", "needs_clarification"}:
        ans_status = "ok"

    answer = str(ans_obj.get("answer") or "").strip()
    followups = ans_obj.get("followups") or []
    citations = ans_obj.get("citations") or []

    _write_json(outdir / "qa_response.json", ans_obj)

    # manifest
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "question_sha256": _sha256_text(question),
        "files_cited": files_cited,
        "artifacts": {},
    }
    for p in sorted(outdir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(outdir).as_posix()
            manifest["artifacts"][rel] = {"bytes": p.stat().st_size, "sha256": _sha256_file(p)}
    _write_json(outdir / "qa_manifest.json", manifest)

    return {
        "status": ans_status,
        "answer": answer,
        "followups": followups if isinstance(followups, list) else [],
        "citations": citations if isinstance(citations, list) else [],
        "files_cited": files_cited,
        "run_id": run_id,
        "run_dir": str(outdir),
    }

def _script_violates_mf6_io(script: str) -> bool:
    """Detect common anti-pattern: using generic CSV readers on MF6 package/name files."""
    s = script or ''
    bad_ext = ('.wel','.chd','.ghb','.drn','.riv','.rcha','.evt','.nam','.dis','.disv','.disu','.npf','.sto','.ims','.tdis')
    # Quick heuristic: pd.read_csv / safe_read_csv called with any of these extensions in a literal.
    for ext in bad_ext:
        if re.search(rf"(read_csv|safe_read_csv)\([^\)]*{re.escape(ext)}", s):
            return True
    return False
