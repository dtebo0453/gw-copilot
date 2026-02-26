from __future__ import annotations

import json
import logging
import math
from enum import Enum
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from gw.api.workspace_files import resolve_workspace_root
from gw.api.model_session import ModelSessionCache
from gw.llm.mf6_filetype_knowledge import (
    PACKAGE_PROPERTIES,
    DIS_PROPERTIES,
    property_to_package,
    get_all_property_keys,
)

router = APIRouter()

# Small in-memory cache for derived products (mesh/scalars) per workspace model
_MODEL_SESSIONS = ModelSessionCache(max_sessions=4)


# ============================================================
# Grid type abstraction — supports DIS, DISV, and DISU
# ============================================================


class GridType(str, Enum):
    DIS = "dis"
    DISV = "disv"
    DISU = "disu"


@dataclass
class GridInfo:
    """Unified grid metadata for all grid types."""
    grid_type: GridType
    nlay: int
    ncpl: int              # cells per layer (nrow*ncol for DIS, ncpl for DISV)
    total_cells: int       # nlay * ncpl  (or nodes for DISU)
    xorigin: float = 0.0
    yorigin: float = 0.0
    angrot: float = 0.0
    # DIS-specific (None for DISV/DISU)
    nrow: Optional[int] = None
    ncol: Optional[int] = None
    # DISV-specific
    nvert: Optional[int] = None
    # DISU-specific
    nodes: Optional[int] = None
    nja: Optional[int] = None
    # Source
    dis_path: Optional[Path] = None
    source: str = "text_parser"  # or "flopy"


# Size safeguards — prevent browser crashes on large models
MAX_SURFACE_CELLS = 250_000       # single-layer polygon mesh
MAX_BLOCK_MODEL_CELLS = 100_000   # 3D block model (8+ pts per cell)
MAX_JSON_POINTS = 3_000_000       # absolute cap on points array length
WARN_CELL_THRESHOLD = 50_000      # include warning in response


# ============================================================
# Robust MF6 grid reader — DIS text parser (Tier 2 fallback)
# ============================================================
#
# Supports:
# - DIS GRIDDATA: DELR, DELC, TOP, BOTM, IDOMAIN
# - Encodings: CONSTANT, INTERNAL (FACTOR/IPRN), OPEN/CLOSE (FACTOR), LAYERED
# - Basic OPTIONS: XORIGIN, YORIGIN, ANGROT
#
# For DISV/DISU, FloPy is the primary (Tier 1) loader; this text parser
# only handles DIS grids.

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
}


@dataclass
class DisInfo:
    dis_path: Path
    nlay: int
    nrow: int
    ncol: int
    delr: np.ndarray  # (ncol,)
    delc: np.ndarray  # (nrow,)
    top: np.ndarray  # (nrow*ncol,)
    botm: np.ndarray  # (nlay*nrow*ncol,)
    idomain: Optional[np.ndarray] = None  # (nlay*nrow*ncol,)
    xorigin: float = 0.0
    yorigin: float = 0.0
    angrot: float = 0.0
    deps: Tuple[Path, ...] = ()  # referenced files (OPEN/CLOSE)
    extras: Dict[str, np.ndarray] = field(default_factory=dict)  # any additional GRIDDATA arrays


# -----------------------------
# Workspace + discovery
# -----------------------------

def _ws_root(inputs_dir: str, workspace: Optional[str]) -> Path:
    try:
        return resolve_workspace_root(inputs_dir, workspace)
    except Exception as e:
        # Keep failures in workspace resolution as a clean 400 instead of a 500.
        raise HTTPException(status_code=400, detail=str(e))


def _scan_files(ws: Path) -> List[Path]:
    out: List[Path] = []
    for p in ws.rglob("*"):
        try:
            if not p.is_file():
                continue
            parts = p.relative_to(ws).parts
            if any(part in SKIP_DIRS for part in parts):
                continue
            out.append(p)
        except Exception:
            continue
    return out


def _score_path(p: Path) -> int:
    s = p.as_posix().lower()
    name = p.name.lower()
    score = 0
    if name.endswith(".dis"):
        score += 10
    if "aoi_model" in name:
        score += 80
    if "model" in s:
        score += 10
    if "/run_artifacts/" in s:
        score -= 10
    if "/artifacts/" in s:
        score -= 5
    if any(x in s for x in ["/old/", "/backup/", "/archive/"]):
        score -= 50
    return score


def _find_dis_candidates(ws: Path) -> List[Path]:
    """Legacy helper — finds only .dis files (for DIS text-parser path)."""
    files = _scan_files(ws)
    cands = [p for p in files if p.suffix.lower() == ".dis"]
    cands.sort(key=lambda p: (-_score_path(p), str(p).lower()))
    return cands


def _find_grid_candidates(ws: Path) -> List[Tuple[Path, GridType]]:
    """Find all discretization files (.dis, .disv, .disu), scored by priority."""
    files = _scan_files(ws)
    _EXT_MAP = {".dis": GridType.DIS, ".disv": GridType.DISV, ".disu": GridType.DISU}
    cands: List[Tuple[Path, GridType]] = []
    for p in files:
        gt = _EXT_MAP.get(p.suffix.lower())
        if gt is not None:
            cands.append((p, gt))
    # DIS files score higher than DISV/DISU by default (for backwards compat)
    def _sort_key(t: Tuple[Path, GridType]) -> Tuple[int, int, str]:
        type_prio = 0 if t[1] == GridType.DIS else 1
        return (-_score_path(t[0]), type_prio, str(t[0]).lower())
    cands.sort(key=_sort_key)
    return cands


def _find_one(ws: Path, exts: Tuple[str, ...]) -> Optional[Path]:
    exts_l = tuple(e.lower() for e in exts)
    files = _scan_files(ws)
    hits = [p for p in files if p.suffix.lower() in exts_l]
    hits.sort(key=lambda p: str(p).lower())
    return hits[0] if hits else None


# -----------------------------
# Text parsing helpers
# -----------------------------

def _strip_comments(line: str) -> str:
    s = line
    for ch in ("#", "!"):
        if ch in s:
            s = s.split(ch, 1)[0]
    return s.strip()


def _tokenize(text: str) -> List[str]:
    toks: List[str] = []
    deps: set[Path] = set()
    for ln in text.splitlines():
        s = _strip_comments(ln)
        if not s:
            continue
        toks.extend(s.split())
    return toks


def _find_block_lines(text: str, block_name: str) -> List[str]:
    lines = text.splitlines()
    inside = False
    out: List[str] = []
    bn = block_name.upper()
    for ln in lines:
        s = _strip_comments(ln)
        if not s:
            continue
        u = s.upper()
        if u.startswith("BEGIN") and bn in u:
            inside = True
            continue
        if inside and u.startswith("END") and bn in u:
            break
        if inside:
            out.append(s)
    return out


def _parse_dimensions(text: str) -> Tuple[int, int, int]:
    lines = _find_block_lines(text, "DIMENSIONS")
    if not lines:
        raise ValueError("DIS file missing DIMENSIONS block")
    nlay = nrow = ncol = None
    for ln in lines:
        parts = ln.split()
        if len(parts) < 2:
            continue
        k = parts[0].upper()
        v = parts[1]
        if k == "NLAY":
            nlay = int(float(v))
        elif k == "NROW":
            nrow = int(float(v))
        elif k == "NCOL":
            ncol = int(float(v))
    if not (nlay and nrow and ncol):
        raise ValueError("Could not parse NLAY/NROW/NCOL from DIS DIMENSIONS")
    return int(nlay), int(nrow), int(ncol)


def _parse_options(text: str) -> Tuple[float, float, float]:
    xorigin = 0.0
    yorigin = 0.0
    angrot = 0.0
    lines = _find_block_lines(text, "OPTIONS")
    for ln in lines:
        parts = ln.split()
        if len(parts) < 2:
            continue
        k = parts[0].upper()
        if k == "XORIGIN":
            xorigin = float(parts[1])
        elif k == "YORIGIN":
            yorigin = float(parts[1])
        elif k == "ANGROT":
            angrot = float(parts[1])
    return xorigin, yorigin, angrot


def _read_numbers_from_tokens(tokens: List[str], n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.float64)
    if len(tokens) < n:
        raise ValueError(f"Not enough numeric tokens for array: need {n}, got {len(tokens)}")
    vals = [float(x) for x in tokens[:n]]
    del tokens[:n]
    return np.array(vals, dtype=np.float64)


def _read_openclose_file(ws_root: Path, filename: str) -> Tuple[List[str], Path]:
    fn = filename.strip().strip('"').strip("'")
    p = (ws_root / fn).resolve()
    # Disallow escaping the workspace
    try:
        p.relative_to(ws_root.resolve())
    except Exception:
        raise ValueError("OPEN/CLOSE path escapes workspace")
    if not p.exists():
        raise FileNotFoundError(f"OPEN/CLOSE file not found: {fn}")
    return _tokenize(p.read_text(encoding="utf-8", errors="replace")), p


def _parse_griddata_arrays(ws_root: Path, text: str, nlay: int, nrow: int, ncol: int) -> Tuple[Dict[str, np.ndarray], Tuple[Path, ...]]:
    lines = _find_block_lines(text, "GRIDDATA")
    if not lines:
        raise ValueError("DIS file missing GRIDDATA block")

    toks: List[str] = []
    deps: set[Path] = set()
    for ln in lines:
        toks.extend(ln.split())

    def read_array(name: str, count: int) -> np.ndarray:
        nonlocal toks
        if not toks:
            raise ValueError(f"Expected array directive for {name}")

        kind = toks.pop(0).upper()

        if kind == "LAYERED":
            per_layer = nrow * ncol
            expected = nlay * per_layer
            if count != expected:
                raise ValueError(
                    f"LAYERED used for {name} but expected count={count} does not match nlay*nrow*ncol={expected}"
                )
            layers: List[np.ndarray] = []
            for k in range(nlay):
                if not toks:
                    raise ValueError(f"LAYERED missing directive for {name} layer {k+1}/{nlay}")
                subkind = toks.pop(0).upper()
                toks = [subkind] + toks
                layers.append(read_array(f"{name}[{k+1}]", per_layer))
            return np.concatenate(layers, axis=0)

        if kind == "CONSTANT":
            if not toks:
                raise ValueError(f"CONSTANT missing value for {name}")
            v = float(toks.pop(0))
            return np.full(count, v, dtype=np.float64)

        if kind == "INTERNAL":
            factor = 1.0
            # Consume optional metadata: FACTOR <f>, IPRN <n>
            while toks and toks[0].upper() in {"FACTOR", "IPRN"}:
                k = toks.pop(0).upper()
                if not toks:
                    break
                if k == "FACTOR":
                    factor = float(toks.pop(0))
                elif k == "IPRN":
                    _ = toks.pop(0)
            arr = _read_numbers_from_tokens(toks, count)
            if factor != 1.0:
                arr = arr * factor
            return arr

        if kind in {"OPEN/CLOSE", "OPENCLOSE"}:
            if not toks:
                raise ValueError(f"OPEN/CLOSE missing filename for {name}")
            filename = toks.pop(0)
            factor = 1.0
            if toks and toks[0].upper() == "FACTOR":
                toks.pop(0)
                if toks:
                    factor = float(toks.pop(0))
            ftoks, dep_path = _read_openclose_file(ws_root, filename)
            deps.add(dep_path)
            arr = _read_numbers_from_tokens(ftoks, count)
            if factor != 1.0:
                arr = arr * factor
            return arr

        # If kind is numeric, treat as implicit INTERNAL with the first number already read
        try:
            float(kind)
            toks = [kind] + toks
            return _read_numbers_from_tokens(toks, count)
        except Exception:
            raise ValueError(f"Unsupported array encoding '{kind}' for {name}")

    want = {
        "DELR": ncol,
        "DELC": nrow,
        "TOP": nrow * ncol,
        "BOTM": nlay * nrow * ncol,
        "IDOMAIN": nlay * nrow * ncol,
    }

    arrays: Dict[str, np.ndarray] = {}
    i = 0
    while i < len(toks):
        key = toks[i].upper()
        if key in want:
            toks.pop(i)
            arrays[key] = read_array(key, want[key])
            continue
        i += 1

    missing = [k for k in ("DELR", "DELC", "TOP", "BOTM") if k not in arrays]
    if missing:
        raise ValueError(f"DIS GRIDDATA missing required arrays: {', '.join(missing)}")

    return arrays, tuple(sorted(deps))



@lru_cache(maxsize=8)
def _load_dis_cached(ws_root_str: str, dis_path_str: str, dis_mtime_ns: int) -> "DisInfo":
    """Parse a DIS file with a small in-memory cache.

    Caching avoids re-reading/parsing large grids on every UI interaction.
    Cache invalidates automatically when the DIS file mtime changes.

    Note: if OPEN/CLOSE references change without updating the DIS mtime,
    you may need to touch the DIS file to invalidate the cache.
    """
    ws_root = Path(ws_root_str)
    dis = Path(dis_path_str)
    txt = dis.read_text(encoding="utf-8", errors="replace")
    nlay, nrow, ncol = _parse_dimensions(txt)
    xorigin, yorigin, angrot = _parse_options(txt)
    arrays, deps = _parse_griddata_arrays(ws_root, txt, nlay, nrow, ncol)

    idom = arrays.get("IDOMAIN")
    return DisInfo(
        dis_path=dis,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=arrays["DELR"].astype(np.float64),
        delc=arrays["DELC"].astype(np.float64),
        top=arrays["TOP"].astype(np.float64),
        botm=arrays["BOTM"].astype(np.float64),
        idomain=(idom.astype(np.float64) if idom is not None else None),
        xorigin=float(xorigin),
        yorigin=float(yorigin),
        angrot=float(angrot),
        extras={k: v for k, v in arrays.items() if k not in ("DELR", "DELC", "TOP", "BOTM", "IDOMAIN")},
    )


def _load_dis(ws_root: Path) -> DisInfo:
    cands = _find_dis_candidates(ws_root)
    dis = cands[0] if cands else None
    if not dis:
        raise FileNotFoundError("No .dis file found in workspace")
    try:
        mtime_ns = int(dis.stat().st_mtime_ns)
    except Exception:
        # fallback if filesystem doesn't support ns
        mtime_ns = int(dis.stat().st_mtime * 1e9)
    return _load_dis_cached(str(ws_root), str(dis), mtime_ns)


def _mtime_ns(p: Path) -> int:
    try:
        return int(p.stat().st_mtime_ns)
    except Exception:
        return int(p.stat().st_mtime * 1e9)


def _dep_signature(ws_root: Path, deps: Tuple[Path, ...]) -> Tuple[Tuple[str, int], ...]:
    sig = []
    for p in deps:
        try:
            rel = str(p.resolve().relative_to(ws_root.resolve()))
        except Exception:
            rel = str(p)
        try:
            sig.append((rel, _mtime_ns(p)))
        except Exception:
            # if a dep disappears, force mismatch
            sig.append((rel, -1))
    return tuple(sorted(sig))


def _load_grid(ws_root: Path) -> Tuple[GridInfo, Optional["DisInfo"]]:
    """Unified grid loader: DIS via text parser, DISV/DISU via FloPy.

    Returns (grid_info, dis_info_or_none).
    """
    cands = _find_grid_candidates(ws_root)
    if not cands:
        raise FileNotFoundError(
            "No discretization file (.dis, .disv, or .disu) found in workspace"
        )

    grid_path, grid_type = cands[0]

    # --- DIS: use existing text parser ---
    if grid_type == GridType.DIS:
        dis_info = _load_dis(ws_root)
        gi = GridInfo(
            grid_type=GridType.DIS,
            nlay=dis_info.nlay,
            ncpl=dis_info.nrow * dis_info.ncol,
            total_cells=dis_info.nlay * dis_info.nrow * dis_info.ncol,
            nrow=dis_info.nrow,
            ncol=dis_info.ncol,
            xorigin=dis_info.xorigin,
            yorigin=dis_info.yorigin,
            angrot=dis_info.angrot,
            dis_path=dis_info.dis_path,
            source="text_parser",
        )
        return gi, dis_info

    # --- DISV / DISU: require FloPy ---
    from gw.mf6.flopy_bridge import flopy_is_available, get_simulation

    if not flopy_is_available():
        raise FileNotFoundError(
            f"{grid_type.value.upper()} model detected ({grid_path.name}) but FloPy "
            f"is not installed. Install flopy to view DISV/DISU models."
        )

    sim, err = get_simulation(ws_root)
    if sim is None:
        raise FileNotFoundError(
            f"{grid_type.value.upper()} model detected but FloPy failed to load it: {err}"
        )

    # Find primary GWF model
    model_names = list(getattr(sim, "model_names", []))
    model_name = model_names[0] if model_names else None
    if model_name is None:
        raise FileNotFoundError("FloPy loaded simulation but found no models")

    model = sim.get_model(model_name)
    mg = model.modelgrid

    if grid_type == GridType.DISV:
        disv_pkg = getattr(model, "disv", None)
        nlay = 1
        ncpl = 1
        nvert = 0
        try:
            nlay_val = getattr(disv_pkg, "nlay", None)
            if hasattr(nlay_val, "get_data"):
                nlay_val = nlay_val.get_data()
            nlay = int(nlay_val) if nlay_val else 1

            ncpl_val = getattr(disv_pkg, "ncpl", None)
            if hasattr(ncpl_val, "get_data"):
                ncpl_val = ncpl_val.get_data()
            ncpl = int(ncpl_val) if ncpl_val else 1

            nvert_val = getattr(disv_pkg, "nvert", None)
            if hasattr(nvert_val, "get_data"):
                nvert_val = nvert_val.get_data()
            nvert = int(nvert_val) if nvert_val else 0
        except Exception:
            # Fallback: infer from modelgrid
            try:
                nlay = int(mg.nlay)
            except Exception:
                pass
            try:
                ncpl = int(mg.ncpl)
            except Exception:
                pass

        # Try to get origin from modelgrid
        xo = float(getattr(mg, "xoffset", 0.0) or 0.0)
        yo = float(getattr(mg, "yoffset", 0.0) or 0.0)
        ang = float(getattr(mg, "angrot", 0.0) or 0.0)

        gi = GridInfo(
            grid_type=GridType.DISV,
            nlay=nlay,
            ncpl=ncpl,
            total_cells=nlay * ncpl,
            nvert=nvert,
            xorigin=xo,
            yorigin=yo,
            angrot=ang,
            dis_path=grid_path,
            source="flopy",
        )
        return gi, None

    # --- DISU ---
    disu_pkg = getattr(model, "disu", None)
    nodes = 1
    nja = 0
    try:
        nodes_val = getattr(disu_pkg, "nodes", None)
        if hasattr(nodes_val, "get_data"):
            nodes_val = nodes_val.get_data()
        nodes = int(nodes_val) if nodes_val else 1

        nja_val = getattr(disu_pkg, "nja", None)
        if hasattr(nja_val, "get_data"):
            nja_val = nja_val.get_data()
        nja = int(nja_val) if nja_val else 0
    except Exception:
        pass

    xo = float(getattr(mg, "xoffset", 0.0) or 0.0)
    yo = float(getattr(mg, "yoffset", 0.0) or 0.0)
    ang = float(getattr(mg, "angrot", 0.0) or 0.0)

    gi = GridInfo(
        grid_type=GridType.DISU,
        nlay=1,
        ncpl=nodes,
        total_cells=nodes,
        nodes=nodes,
        nja=nja,
        xorigin=xo,
        yorigin=yo,
        angrot=ang,
        dis_path=grid_path,
        source="flopy",
    )
    return gi, None


def _get_session(ws_root: Path) -> Tuple[GridInfo, Any, Optional["DisInfo"]]:
    """Load grid and return (grid_info, session, dis_info_or_none).

    For DIS models, dis_info is the parsed DisInfo from the text parser.
    For DISV/DISU, dis_info is None and the FloPy sim is stored in the session.
    """
    grid_info, dis_info = _load_grid(ws_root)

    dis_path = grid_info.dis_path or Path("unknown")
    dis_mtime = _mtime_ns(dis_path) if dis_path.exists() else 0

    if dis_info is not None:
        dep_sig = _dep_signature(ws_root, getattr(dis_info, "deps", ()))
    else:
        dep_sig = ()

    sess = _MODEL_SESSIONS.get(
        ws_root, dis_path,
        dis_mtime_ns=dis_mtime, dep_sig=dep_sig, dis_obj=dis_info,
    )

    # Store FloPy sim in session for DISV/DISU
    if grid_info.grid_type in (GridType.DISV, GridType.DISU):
        if sess.flopy_sim is None:
            from gw.mf6.flopy_bridge import get_simulation
            sim, _ = get_simulation(ws_root)
            sess.flopy_sim = sim
            model_names = list(getattr(sim, "model_names", []))
            sess.flopy_model_name = model_names[0] if model_names else ""
        sess.grid_type = grid_info.grid_type.value

    return grid_info, sess, dis_info




def _rot_xy(x: float, y: float, ang_deg: float) -> Tuple[float, float]:
    if not ang_deg:
        return x, y
    a = math.radians(ang_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    return (x * ca - y * sa), (x * sa + y * ca)


def _check_size(grid_info: GridInfo, mode: str) -> None:
    """Raise HTTPException if the requested mesh would exceed size limits."""
    if mode == "block_model":
        if grid_info.total_cells > MAX_BLOCK_MODEL_CELLS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Block model too large: {grid_info.total_cells:,} cells "
                    f"(limit {MAX_BLOCK_MODEL_CELLS:,}). "
                    f"Try 'Layer Surface' or 'All Layers' mode instead."
                ),
            )
    elif mode == "top_surface":
        if grid_info.ncpl > MAX_SURFACE_CELLS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Surface mesh too large: {grid_info.ncpl:,} cells per layer "
                    f"(limit {MAX_SURFACE_CELLS:,})."
                ),
            )
    elif mode == "all_layers_surface":
        if grid_info.total_cells > MAX_SURFACE_CELLS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"All-layers surface too large: {grid_info.total_cells:,} cells "
                    f"(limit {MAX_SURFACE_CELLS:,}). "
                    f"Try 'Layer Surface' mode for a single layer."
                ),
            )


def _size_class(grid_info: GridInfo) -> str:
    """Classify model size for UI hints."""
    tc = grid_info.total_cells
    if tc <= WARN_CELL_THRESHOLD:
        return "small"
    elif tc <= MAX_SURFACE_CELLS:
        return "medium"
    else:
        return "large"


def _build_top_surface_mesh(dis: DisInfo, layer: int) -> Dict[str, Any]:
    if layer < 0 or layer >= dis.nlay:
        raise ValueError("layer out of range")

    nrow, ncol = dis.nrow, dis.ncol

    if layer == 0:
        cell_z = dis.top.reshape((nrow, ncol))
    else:
        cell_z = dis.botm.reshape((dis.nlay, nrow, ncol))[layer - 1]

    # Vertex Z = average of adjacent cells
    zv = np.zeros((nrow + 1, ncol + 1), dtype=np.float64)
    cnt = np.zeros((nrow + 1, ncol + 1), dtype=np.float64)
    for r in range(nrow):
        for c in range(ncol):
            z = float(cell_z[r, c])
            for dr, dc in ((0, 0), (0, 1), (1, 1), (1, 0)):
                zv[r + dr, c + dc] += z
                cnt[r + dr, c + dc] += 1.0
    cnt[cnt == 0] = 1.0
    zv = zv / cnt

    x_edges = np.concatenate(([0.0], np.cumsum(dis.delr)))
    y_edges = np.concatenate(([0.0], np.cumsum(dis.delc)))

    points: List[float] = []
    for r in range(nrow + 1):
        for c in range(ncol + 1):
            x0 = float(x_edges[c])
            y0 = float(y_edges[r])
            xr, yr = _rot_xy(x0, y0, dis.angrot)
            points.extend([dis.xorigin + xr, dis.yorigin + yr, float(zv[r, c])])

    def vid(rr: int, cc: int) -> int:
        return rr * (ncol + 1) + cc

    polys: List[int] = []
    for r in range(nrow):
        for c in range(ncol):
            polys.extend([4, vid(r, c), vid(r, c + 1), vid(r + 1, c + 1), vid(r + 1, c)])

    return {
        "mode": "top_surface",
        "layer": layer,
        "nrow": nrow,
        "ncol": ncol,
        "points": points,
        "polys": polys,
        "cell_count": nrow * ncol,
        "point_count": (nrow + 1) * (ncol + 1),
    }


# -----------------------------
# Generic MF6 package array parser
# -----------------------------


def _find_package_file(ws_root: Path, pkg_type: str) -> Optional[Path]:
    """Locate a package file in the workspace using snapshot inventory or extension scan."""
    pkg_info = PACKAGE_PROPERTIES.get(pkg_type.upper())
    if pkg_info is None:
        return None
    ext = pkg_info.file_ext.lower()

    # Try to find via cached scan / snapshot (fastest)
    try:
        from gw.api.workspace_scan import ensure_workspace_scan
        scan = ensure_workspace_scan(ws_root, force=False)
        snapshot = (scan or {}).get("snapshot", {}) if isinstance(scan, dict) else {}
        pkg_files = snapshot.get("package_files", {}) if isinstance(snapshot, dict) else {}
        rel = pkg_files.get(pkg_type.upper())
        if rel:
            p = (ws_root / rel).resolve()
            if p.exists() and p.is_file():
                return p
    except Exception:
        pass

    # Fallback: scan for files with the expected extension
    files = _scan_files(ws_root)
    hits = [p for p in files if p.suffix.lower() == ext]
    hits.sort(key=lambda p: (-_score_path(p), str(p).lower()))
    return hits[0] if hits else None


def _parse_package_griddata(ws_root: Path, pkg_path: Path, nlay: int, nrow: int, ncol: int, block_name: str = "GRIDDATA") -> Dict[str, np.ndarray]:
    """Parse all arrays from a GRIDDATA (or similar) block in an MF6 package file.

    Uses the same CONSTANT/INTERNAL/OPEN/CLOSE/LAYERED encoding logic as the DIS parser.
    Returns {array_name_lower: flat_ndarray}.
    """
    txt = pkg_path.read_text(encoding="utf-8", errors="replace")
    lines = _find_block_lines(txt, block_name)
    if not lines:
        return {}

    toks: List[str] = []
    deps: set[Path] = set()
    for ln in lines:
        toks.extend(ln.split())

    per_layer = nlay * nrow * ncol
    per_surface = nrow * ncol

    def read_array(name: str, count: int) -> np.ndarray:
        nonlocal toks
        if not toks:
            raise ValueError(f"Expected array directive for {name}")
        kind = toks.pop(0).upper()

        if kind == "LAYERED":
            if count != per_layer:
                raise ValueError(f"LAYERED used for {name} but count={count} != nlay*nrow*ncol={per_layer}")
            layers: List[np.ndarray] = []
            for k in range(nlay):
                if not toks:
                    raise ValueError(f"LAYERED missing directive for {name} layer {k+1}")
                subkind = toks.pop(0).upper()
                toks = [subkind] + toks
                layers.append(read_array(f"{name}[{k+1}]", per_surface))
            return np.concatenate(layers, axis=0)

        if kind == "CONSTANT":
            if not toks:
                raise ValueError(f"CONSTANT missing value for {name}")
            v = float(toks.pop(0))
            return np.full(count, v, dtype=np.float64)

        if kind == "INTERNAL":
            factor = 1.0
            while toks and toks[0].upper() in {"FACTOR", "IPRN"}:
                k2 = toks.pop(0).upper()
                if not toks:
                    break
                if k2 == "FACTOR":
                    factor = float(toks.pop(0))
                elif k2 == "IPRN":
                    _ = toks.pop(0)
            arr = _read_numbers_from_tokens(toks, count)
            if factor != 1.0:
                arr = arr * factor
            return arr

        if kind in {"OPEN/CLOSE", "OPENCLOSE"}:
            if not toks:
                raise ValueError(f"OPEN/CLOSE missing filename for {name}")
            filename = toks.pop(0)
            factor = 1.0
            if toks and toks[0].upper() == "FACTOR":
                toks.pop(0)
                if toks:
                    factor = float(toks.pop(0))
            ftoks, dep_path = _read_openclose_file(ws_root, filename)
            deps.add(dep_path)
            arr = _read_numbers_from_tokens(ftoks, count)
            if factor != 1.0:
                arr = arr * factor
            return arr

        # Implicit numeric (no keyword)
        try:
            float(kind)
            toks = [kind] + toks
            return _read_numbers_from_tokens(toks, count)
        except Exception:
            raise ValueError(f"Unsupported array encoding '{kind}' for {name}")

    arrays: Dict[str, np.ndarray] = {}
    i = 0
    while i < len(toks):
        key = toks[i].upper()
        # Check if this token is a known array name for this package
        # Arrays can be per-layer (nlay*nrow*ncol) or per-surface (nrow*ncol)
        known_per_layer = {"K", "K22", "K33", "ICELLTYPE", "SS", "SY", "ICONVERT", "STRT", "RECHARGE"}
        if key in known_per_layer:
            toks.pop(i)
            arrays[key.lower()] = read_array(key, per_layer)
            continue
        i += 1

    return arrays


def _load_package_arrays(ws_root: Path, dis: DisInfo, sess: Any, pkg_type: str) -> Dict[str, np.ndarray]:
    """Load and cache package arrays. Returns {array_name: flat_ndarray}."""
    pkg_upper = pkg_type.upper()
    if pkg_upper in sess.pkg_arrays_cache:
        return sess.pkg_arrays_cache[pkg_upper]

    pkg_path = _find_package_file(ws_root, pkg_upper)
    if pkg_path is None:
        sess.pkg_arrays_cache[pkg_upper] = {}
        return {}

    pkg_info = PACKAGE_PROPERTIES.get(pkg_upper)
    block = pkg_info.block if pkg_info else "GRIDDATA"

    try:
        arrays = _parse_package_griddata(ws_root, pkg_path, dis.nlay, dis.nrow, dis.ncol, block)
    except Exception:
        arrays = {}

    sess.pkg_arrays_cache[pkg_upper] = arrays
    return arrays


# -----------------------------
# Mesh builders
# -----------------------------


def _build_block_model_mesh(dis: DisInfo) -> Dict[str, Any]:
    """Build a full 3D block model mesh with hexahedral cells rendered as 6 quads each.

    All active cells (idomain > 0) across all layers are included.
    """
    nlay, nrow, ncol = dis.nlay, dis.nrow, dis.ncol

    x_edges = np.concatenate(([0.0], np.cumsum(dis.delr)))
    y_edges = np.concatenate(([0.0], np.cumsum(dis.delc)))

    top_2d = dis.top.reshape((nrow, ncol))
    botm_3d = dis.botm.reshape((nlay, nrow, ncol))
    idomain_3d = dis.idomain.reshape((nlay, nrow, ncol)) if dis.idomain is not None else np.ones((nlay, nrow, ncol))

    points: List[float] = []
    polys: List[int] = []
    cell_layer_map: List[int] = []
    point_offset = 0

    for k in range(nlay):
        cell_top = top_2d if k == 0 else botm_3d[k - 1]
        cell_bot = botm_3d[k]

        for r in range(nrow):
            for c in range(ncol):
                if idomain_3d[k, r, c] <= 0:
                    continue

                x0, x1 = float(x_edges[c]), float(x_edges[c + 1])
                y0, y1 = float(y_edges[r]), float(y_edges[r + 1])
                zt = float(cell_top[r, c])
                zb = float(cell_bot[r, c])

                # Apply rotation
                corners_xy = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                rotated = [_rot_xy(x, y, dis.angrot) for x, y in corners_xy]

                # 8 vertices: 4 top + 4 bottom
                base = point_offset
                for rx, ry in rotated:
                    points.extend([dis.xorigin + rx, dis.yorigin + ry, zt])
                for rx, ry in rotated:
                    points.extend([dis.xorigin + rx, dis.yorigin + ry, zb])

                # 6 quads (faces): top, bottom, 4 sides
                v = base
                # Top face
                polys.extend([4, v + 0, v + 1, v + 2, v + 3])
                # Bottom face
                polys.extend([4, v + 4, v + 7, v + 6, v + 5])
                # Front face (y0)
                polys.extend([4, v + 0, v + 1, v + 5, v + 4])
                # Back face (y1)
                polys.extend([4, v + 2, v + 3, v + 7, v + 6])
                # Left face (x0)
                polys.extend([4, v + 0, v + 3, v + 7, v + 4])
                # Right face (x1)
                polys.extend([4, v + 1, v + 2, v + 6, v + 5])

                cell_layer_map.append(k)
                point_offset += 8

    cell_count = len(cell_layer_map)
    return {
        "mode": "block_model",
        "nrow": nrow,
        "ncol": ncol,
        "nlay": nlay,
        "points": points,
        "polys": polys,
        "cell_count": cell_count,
        "point_count": point_offset,
        "cell_layer_map": cell_layer_map,
        "faces_per_cell": 6,
    }


def _build_all_layers_mesh(dis: DisInfo) -> Dict[str, Any]:
    """Build stacked surface meshes for all layers (top + each botm).

    Returns combined points/polys and a cell_layer_map.
    """
    nrow, ncol, nlay = dis.nrow, dis.ncol, dis.nlay

    x_edges = np.concatenate(([0.0], np.cumsum(dis.delr)))
    y_edges = np.concatenate(([0.0], np.cumsum(dis.delc)))

    all_points: List[float] = []
    all_polys: List[int] = []
    cell_layer_map: List[int] = []
    total_pts = 0

    for layer in range(nlay):
        if layer == 0:
            cell_z = dis.top.reshape((nrow, ncol))
        else:
            cell_z = dis.botm.reshape((nlay, nrow, ncol))[layer - 1]

        # Vertex Z = average of adjacent cells
        zv = np.zeros((nrow + 1, ncol + 1), dtype=np.float64)
        cnt = np.zeros((nrow + 1, ncol + 1), dtype=np.float64)
        for r2 in range(nrow):
            for c2 in range(ncol):
                z = float(cell_z[r2, c2])
                for dr, dc in ((0, 0), (0, 1), (1, 1), (1, 0)):
                    zv[r2 + dr, c2 + dc] += z
                    cnt[r2 + dr, c2 + dc] += 1.0
        cnt[cnt == 0] = 1.0
        zv = zv / cnt

        base = total_pts
        for r2 in range(nrow + 1):
            for c2 in range(ncol + 1):
                x0 = float(x_edges[c2])
                y0 = float(y_edges[r2])
                xr, yr = _rot_xy(x0, y0, dis.angrot)
                all_points.extend([dis.xorigin + xr, dis.yorigin + yr, float(zv[r2, c2])])

        def vid(rr: int, cc: int) -> int:
            return base + rr * (ncol + 1) + cc

        for r2 in range(nrow):
            for c2 in range(ncol):
                all_polys.extend([4, vid(r2, c2), vid(r2, c2 + 1), vid(r2 + 1, c2 + 1), vid(r2 + 1, c2)])
                cell_layer_map.append(layer)

        total_pts += (nrow + 1) * (ncol + 1)

    return {
        "mode": "all_layers_surface",
        "nrow": nrow,
        "ncol": ncol,
        "nlay": nlay,
        "points": all_points,
        "polys": all_polys,
        "cell_count": len(cell_layer_map),
        "point_count": total_pts,
        "cell_layer_map": cell_layer_map,
    }


def _cell_scalars_for_surface(dis: DisInfo, key: str, layer: int) -> np.ndarray:
    key_l = (key or "").lower().strip()
    nrow, ncol = dis.nrow, dis.ncol

    if key_l == "top":
        if layer == 0:
            return dis.top.reshape((nrow, ncol)).ravel()
        return dis.botm.reshape((dis.nlay, nrow, ncol))[layer - 1].ravel()

    if key_l in {"botm", "bottom"}:
        return dis.botm.reshape((dis.nlay, nrow, ncol))[layer].ravel()

    if key_l in {"idomain", "ibound"}:
        if dis.idomain is None:
            return np.ones((nrow * ncol,), dtype=np.float64)
        return dis.idomain.reshape((dis.nlay, nrow, ncol))[layer].ravel()

    # Package property lookup
    pkg_info = property_to_package(key_l)
    if pkg_info is not None:
        raise KeyError(f"Package property '{key}' requires ws_root/session context; use _cell_scalars_for_surface_pkg instead")

    raise KeyError(f"Unknown or unavailable scalar key: {key}")


def _cell_scalars_for_surface_pkg(dis: DisInfo, ws_root: Path, sess: Any, key: str, layer: int) -> np.ndarray:
    """Extended scalar extractor that also handles package-level properties."""
    key_l = (key or "").lower().strip()
    nrow, ncol = dis.nrow, dis.ncol

    # Try DIS-native properties first
    if key_l in {"top", "botm", "bottom", "idomain", "ibound"}:
        return _cell_scalars_for_surface(dis, key, layer)

    # Package property
    pkg_info = property_to_package(key_l)
    if pkg_info is None:
        raise KeyError(f"Unknown or unavailable scalar key: {key}")

    pkg_type, arr_name, _label = pkg_info
    arrays = _load_package_arrays(ws_root, dis, sess, pkg_type)
    if arr_name not in arrays:
        raise KeyError(f"Array '{arr_name}' not found in {pkg_type} package file")

    arr = arrays[arr_name]
    expected = dis.nlay * nrow * ncol
    if arr.size == expected:
        return arr.reshape((dis.nlay, nrow, ncol))[layer].ravel()
    elif arr.size == nrow * ncol:
        # Single-layer array
        return arr.reshape((nrow, ncol)).ravel()
    else:
        raise KeyError(f"Array '{arr_name}' has unexpected size {arr.size} (expected {expected} or {nrow*ncol})")


def _cell_scalars_all_layers(dis: DisInfo, ws_root: Path, sess: Any, key: str) -> np.ndarray:
    """Extract scalars for all layers (for block_model / all_layers_surface modes)."""
    nrow, ncol, nlay = dis.nrow, dis.ncol, dis.nlay
    key_l = (key or "").lower().strip()

    parts: List[np.ndarray] = []
    idomain_3d = dis.idomain.reshape((nlay, nrow, ncol)) if dis.idomain is not None else np.ones((nlay, nrow, ncol))

    for k in range(nlay):
        layer_vals = _cell_scalars_for_surface_pkg(dis, ws_root, sess, key, k)
        parts.append(layer_vals)

    return np.concatenate(parts, axis=0)


def _cell_scalars_block_model(dis: DisInfo, ws_root: Path, sess: Any, key: str) -> np.ndarray:
    """Extract scalars for the block model (only active cells, ordered to match mesh).

    Each active model cell is rendered as 6 quad faces (top, bottom, 4 sides),
    so the scalar value is repeated 6 times per cell to align with VTK's
    cell-data mode (one scalar per polygon).
    """
    nrow, ncol, nlay = dis.nrow, dis.ncol, dis.nlay
    idomain_3d = dis.idomain.reshape((nlay, nrow, ncol)) if dis.idomain is not None else np.ones((nlay, nrow, ncol))

    all_scalars = _cell_scalars_all_layers(dis, ws_root, sess, key).reshape((nlay, nrow, ncol))

    # Filter to active cells in the same order as the mesh builder.
    # Repeat each value 6 times (one for each face/quad of the hexahedral cell).
    active_vals: List[float] = []
    for k in range(nlay):
        for r in range(nrow):
            for c in range(ncol):
                if idomain_3d[k, r, c] > 0:
                    v = float(all_scalars[k, r, c])
                    active_vals.extend([v, v, v, v, v, v])

    return np.array(active_vals, dtype=np.float64)


# ============================================================
# DISV / DISU mesh builders (FloPy-based)
# ============================================================


def _get_flopy_model(sess: Any):
    """Get the FloPy model object from the session."""
    sim = sess.flopy_sim
    if sim is None:
        raise ValueError("FloPy simulation not available in session")
    model_name = sess.flopy_model_name
    if not model_name:
        model_names = list(getattr(sim, "model_names", []))
        if not model_names:
            raise ValueError("No models found in FloPy simulation")
        model_name = model_names[0]
    return sim.get_model(model_name)


def _get_cell_vertices_2d(mg) -> List[List[Tuple[float, float]]]:
    """Extract 2D cell vertex coordinates from FloPy modelgrid.

    Returns list of cells, each cell is a list of (x, y) vertices.
    """
    ncpl = mg.ncpl
    cells: List[List[Tuple[float, float]]] = []
    for i in range(ncpl):
        try:
            verts = mg.get_cell_vertices(i)
            cells.append([(float(v[0]), float(v[1])) for v in verts])
        except Exception:
            cells.append([])
    return cells


def _build_disv_surface_mesh(sess: Any, grid_info: GridInfo, layer: int) -> Dict[str, Any]:
    """Build a surface mesh for one layer of a DISV model."""
    model = _get_flopy_model(sess)
    mg = model.modelgrid

    ncpl = grid_info.ncpl
    nlay = grid_info.nlay

    if layer < 0 or layer >= nlay:
        raise ValueError(f"layer {layer} out of range (0..{nlay - 1})")

    # Get top/botm for Z values
    try:
        top_arr = np.array(mg.top, dtype=np.float64).ravel()
    except Exception:
        top_arr = np.zeros(ncpl, dtype=np.float64)

    try:
        botm_arr = np.array(mg.botm, dtype=np.float64)
        if botm_arr.ndim == 2:
            # shape (nlay, ncpl)
            pass
        else:
            botm_arr = botm_arr.reshape((nlay, ncpl))
    except Exception:
        botm_arr = np.zeros((nlay, ncpl), dtype=np.float64)

    # Z for this layer: layer 0 uses top, layer k>0 uses botm[k-1]
    if layer == 0:
        cell_z = top_arr[:ncpl]
    else:
        cell_z = botm_arr[layer - 1, :ncpl] if botm_arr.shape[0] > layer - 1 else np.zeros(ncpl)

    # Get cell vertices
    cell_verts = _get_cell_vertices_2d(mg)

    points: List[float] = []
    polys: List[int] = []
    pt_offset = 0

    for i in range(ncpl):
        verts = cell_verts[i]
        if len(verts) < 3:
            continue
        z = float(cell_z[i]) if i < len(cell_z) else 0.0
        n = len(verts)
        # Emit vertices
        for vx, vy in verts:
            points.extend([vx, vy, z])
        # Emit polygon
        polys.append(n)
        for j in range(n):
            polys.append(pt_offset + j)
        pt_offset += n

    return {
        "mode": "top_surface",
        "layer": layer,
        "ncpl": ncpl,
        "nlay": nlay,
        "grid_type": "disv",
        "points": points,
        "polys": polys,
        "cell_count": ncpl,
        "point_count": pt_offset,
    }


def _build_disv_all_layers(sess: Any, grid_info: GridInfo) -> Dict[str, Any]:
    """Build stacked surface meshes for all layers of a DISV model."""
    model = _get_flopy_model(sess)
    mg = model.modelgrid
    ncpl = grid_info.ncpl
    nlay = grid_info.nlay

    try:
        top_arr = np.array(mg.top, dtype=np.float64).ravel()
    except Exception:
        top_arr = np.zeros(ncpl, dtype=np.float64)
    try:
        botm_arr = np.array(mg.botm, dtype=np.float64).reshape((nlay, ncpl))
    except Exception:
        botm_arr = np.zeros((nlay, ncpl), dtype=np.float64)

    cell_verts = _get_cell_vertices_2d(mg)

    all_points: List[float] = []
    all_polys: List[int] = []
    cell_layer_map: List[int] = []
    pt_offset = 0

    for k in range(nlay):
        cell_z = top_arr[:ncpl] if k == 0 else botm_arr[k - 1, :ncpl]
        for i in range(ncpl):
            verts = cell_verts[i]
            if len(verts) < 3:
                continue
            z = float(cell_z[i]) if i < len(cell_z) else 0.0
            n = len(verts)
            for vx, vy in verts:
                all_points.extend([vx, vy, z])
            all_polys.append(n)
            for j in range(n):
                all_polys.append(pt_offset + j)
            pt_offset += n
            cell_layer_map.append(k)

    return {
        "mode": "all_layers_surface",
        "ncpl": ncpl,
        "nlay": nlay,
        "grid_type": "disv",
        "points": all_points,
        "polys": all_polys,
        "cell_count": len(cell_layer_map),
        "point_count": pt_offset,
        "cell_layer_map": cell_layer_map,
    }


def _build_disv_block_model(sess: Any, grid_info: GridInfo) -> Dict[str, Any]:
    """Build a 3D block model for a DISV grid by extruding cells between top/botm."""
    model = _get_flopy_model(sess)
    mg = model.modelgrid
    ncpl = grid_info.ncpl
    nlay = grid_info.nlay

    try:
        top_arr = np.array(mg.top, dtype=np.float64).ravel()
    except Exception:
        top_arr = np.zeros(ncpl, dtype=np.float64)
    try:
        botm_arr = np.array(mg.botm, dtype=np.float64).reshape((nlay, ncpl))
    except Exception:
        botm_arr = np.zeros((nlay, ncpl), dtype=np.float64)

    # IDOMAIN
    try:
        idom = np.array(mg.idomain, dtype=np.float64).reshape((nlay, ncpl))
    except Exception:
        idom = np.ones((nlay, ncpl), dtype=np.float64)

    cell_verts = _get_cell_vertices_2d(mg)

    points: List[float] = []
    polys: List[int] = []
    cell_layer_map: List[int] = []
    pt_offset = 0

    for k in range(nlay):
        cell_top = top_arr[:ncpl] if k == 0 else botm_arr[k - 1, :ncpl]
        cell_bot = botm_arr[k, :ncpl]

        for i in range(ncpl):
            if idom[k, i] <= 0:
                continue
            verts = cell_verts[i]
            if len(verts) < 3:
                continue

            zt = float(cell_top[i])
            zb = float(cell_bot[i])
            n = len(verts)

            # Top face vertices, then bottom face vertices
            for vx, vy in verts:
                points.extend([vx, vy, zt])
            for vx, vy in verts:
                points.extend([vx, vy, zb])

            base = pt_offset
            # Top face
            polys.append(n)
            for j in range(n):
                polys.append(base + j)
            # Bottom face (reversed winding)
            polys.append(n)
            for j in range(n - 1, -1, -1):
                polys.append(base + n + j)
            # Side quads
            for j in range(n):
                j2 = (j + 1) % n
                polys.extend([4, base + j, base + j2, base + n + j2, base + n + j])

            cell_layer_map.append(k)
            pt_offset += 2 * n

    return {
        "mode": "block_model",
        "ncpl": ncpl,
        "nlay": nlay,
        "grid_type": "disv",
        "points": points,
        "polys": polys,
        "cell_count": len(cell_layer_map),
        "point_count": pt_offset,
        "cell_layer_map": cell_layer_map,
        "faces_per_cell": -1,  # variable for DISV
    }


def _cell_scalars_disv(sess: Any, grid_info: GridInfo, key: str, layer: int) -> np.ndarray:
    """Extract scalar values for one layer of a DISV/DISU model via FloPy."""
    model = _get_flopy_model(sess)
    mg = model.modelgrid
    ncpl = grid_info.ncpl
    nlay = grid_info.nlay
    key_l = (key or "").lower().strip()

    if key_l == "top":
        try:
            arr = np.array(mg.top, dtype=np.float64).ravel()
        except Exception:
            arr = np.zeros(ncpl, dtype=np.float64)
        if layer == 0:
            return arr[:ncpl]
        else:
            try:
                botm = np.array(mg.botm, dtype=np.float64).reshape((nlay, ncpl))
                return botm[layer - 1, :ncpl]
            except Exception:
                return arr[:ncpl]

    if key_l in {"botm", "bottom"}:
        try:
            botm = np.array(mg.botm, dtype=np.float64).reshape((nlay, ncpl))
            return botm[layer, :ncpl]
        except Exception:
            return np.zeros(ncpl, dtype=np.float64)

    if key_l in {"idomain", "ibound"}:
        try:
            idom = np.array(mg.idomain, dtype=np.float64).reshape((nlay, ncpl))
            return idom[layer, :ncpl]
        except Exception:
            return np.ones(ncpl, dtype=np.float64)

    if key_l == "thickness":
        try:
            top_arr = np.array(mg.top, dtype=np.float64).ravel()[:ncpl]
            botm_arr = np.array(mg.botm, dtype=np.float64).reshape((nlay, ncpl))
            cell_top = top_arr if layer == 0 else botm_arr[layer - 1, :ncpl]
            cell_bot = botm_arr[layer, :ncpl]
            return cell_top - cell_bot
        except Exception:
            return np.zeros(ncpl, dtype=np.float64)

    # Try FloPy package arrays
    try:
        for pkg_name in model.package_names:
            pkg = model.get_package(pkg_name)
            if pkg is None:
                continue
            if hasattr(pkg, key_l):
                data_obj = getattr(pkg, key_l)
                if hasattr(data_obj, "get_data"):
                    data_obj = data_obj.get_data()
                arr = np.array(data_obj, dtype=np.float64)
                if arr.size == nlay * ncpl:
                    return arr.reshape((nlay, ncpl))[layer, :ncpl]
                elif arr.size == ncpl:
                    return arr.ravel()[:ncpl]
    except Exception:
        pass

    raise KeyError(f"Unknown or unavailable scalar key '{key}' for {grid_info.grid_type.value} model")


def _cell_scalars_disv_all_layers(sess: Any, grid_info: GridInfo, key: str) -> np.ndarray:
    """Scalars for all layers (DISV), concatenated."""
    parts = []
    for k in range(grid_info.nlay):
        parts.append(_cell_scalars_disv(sess, grid_info, key, k))
    return np.concatenate(parts, axis=0)


def _cell_scalars_disv_block_model(sess: Any, grid_info: GridInfo, key: str) -> np.ndarray:
    """Scalars for block model (only active cells, matching mesh order).

    Each active DISV cell is extruded into (2 + n) polygon faces (top, bottom,
    n side quads) where n = number of vertices.  The scalar value is repeated
    for each face to align with VTK's cell-data mode.
    """
    model = _get_flopy_model(sess)
    mg = model.modelgrid
    ncpl = grid_info.ncpl
    nlay = grid_info.nlay

    try:
        idom = np.array(mg.idomain, dtype=np.float64).reshape((nlay, ncpl))
    except Exception:
        idom = np.ones((nlay, ncpl), dtype=np.float64)

    cell_verts = _get_cell_vertices_2d(mg)
    all_vals = _cell_scalars_disv_all_layers(sess, grid_info, key).reshape((nlay, ncpl))

    active_vals: List[float] = []
    for k in range(nlay):
        for i in range(ncpl):
            if idom[k, i] <= 0:
                continue
            verts = cell_verts[i]
            if len(verts) < 3:
                continue
            v = float(all_vals[k, i])
            # Repeat for each polygon face: top + bottom + n side quads
            n_faces = 2 + len(verts)
            active_vals.extend([v] * n_faces)

    return np.array(active_vals, dtype=np.float64)


def _build_boundary_disv(sess: Any, grid_info: GridInfo) -> Dict[str, Any]:
    """Build boundary response for a DISV model."""
    model = _get_flopy_model(sess)
    mg = model.modelgrid

    try:
        extent = mg.extent  # (xmin, xmax, ymin, ymax)
        xmin, xmax, ymin, ymax = float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])
    except Exception:
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0

    corners: List[List[float]] = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ]

    x_total = xmax - xmin
    y_total = ymax - ymin
    has_real_coords = abs(xmin) > 1e-6 or abs(ymin) > 1e-6

    result: Dict[str, Any] = {
        "corners": corners,
        "xorigin": float(grid_info.xorigin),
        "yorigin": float(grid_info.yorigin),
        "angrot": float(grid_info.angrot),
        "x_total": x_total,
        "y_total": y_total,
        "has_real_coords": has_real_coords,
        "nrow": 0,
        "ncol": 0,
        "nlay": int(grid_info.nlay),
        "ncpl": int(grid_info.ncpl),
        "grid_type": str(grid_info.grid_type.value),
        "delr_range": [0.0, 0.0],
        "delc_range": [0.0, 0.0],
        "delr": [],
        "delc": [],
    }

    # For DISV models, include cell polygons for map display (up to 10k cells)
    if grid_info.ncpl <= 10_000:
        cell_verts = _get_cell_vertices_2d(mg)
        cell_polys: List[List[List[float]]] = []
        for verts in cell_verts:
            if len(verts) >= 3:
                # Ensure native Python float (not numpy)
                cell_polys.append([[float(v[0]), float(v[1])] for v in verts])
        result["cell_polygons"] = cell_polys

    return result


def _build_bounds_disv(sess: Any, grid_info: GridInfo) -> Dict[str, Any]:
    """Build 3D bounding box for a DISV model."""
    model = _get_flopy_model(sess)
    mg = model.modelgrid

    try:
        extent = mg.extent
        xmin, xmax, ymin, ymax = float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])
    except Exception:
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0

    try:
        top_arr = np.array(mg.top, dtype=np.float64).ravel()
        botm_arr = np.array(mg.botm, dtype=np.float64).ravel()
        zmax = float(np.nanmax(top_arr))
        zmin = float(np.nanmin(botm_arr))
    except Exception:
        zmax, zmin = 1.0, 0.0

    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
    }


# ============================================================
# DISU stubs
# ============================================================

def _build_disu_surface_mesh(sess: Any, grid_info: GridInfo, layer: int = 0) -> Dict[str, Any]:
    """Build surface mesh for a DISU model (treated as single-layer)."""
    # DISU models may not have a layer concept. We reuse the DISV approach.
    model = _get_flopy_model(sess)
    mg = model.modelgrid
    nodes = grid_info.nodes or grid_info.ncpl

    try:
        top_arr = np.array(mg.top, dtype=np.float64).ravel()
    except Exception:
        top_arr = np.zeros(nodes, dtype=np.float64)

    points: List[float] = []
    polys: List[int] = []
    pt_offset = 0

    for i in range(nodes):
        try:
            verts = mg.get_cell_vertices(i)
            verts = [(float(v[0]), float(v[1])) for v in verts]
        except Exception:
            continue
        if len(verts) < 3:
            continue
        z = float(top_arr[i]) if i < len(top_arr) else 0.0
        n = len(verts)
        for vx, vy in verts:
            points.extend([vx, vy, z])
        polys.append(n)
        for j in range(n):
            polys.append(pt_offset + j)
        pt_offset += n

    return {
        "mode": "top_surface",
        "layer": 0,
        "grid_type": "disu",
        "nodes": nodes,
        "points": points,
        "polys": polys,
        "cell_count": nodes,
        "point_count": pt_offset,
    }


# -----------------------------
# API
# -----------------------------

@router.get("/viz/summary")
def viz_summary(
    inputs_dir: str,
    workspace: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    ws = _ws_root(inputs_dir, workspace)

    files = _scan_files(ws) if ws.exists() else []
    grid_candidates = _find_grid_candidates(ws) if ws.exists() else []

    diagnostics: Dict[str, Any] = {
        "workspace_root": str(ws),
        "workspace_exists": bool(ws.exists()),
        "cwd": str(Path.cwd().resolve()),
        "scan_file_count": len(files),
        "grid_candidates": [
            f"{p.relative_to(ws).as_posix()} ({gt.value})"
            for p, gt in grid_candidates[:50]
        ],
        "skipped_dirs": sorted(SKIP_DIRS),
    }

    try:
        grid_info, sess, dis_info = _get_session(ws)
        diagnostics["grid_file"] = str(grid_info.dis_path.relative_to(ws).as_posix()) if grid_info.dis_path else "unknown"
        diagnostics["grid_type"] = grid_info.grid_type.value
        diagnostics["grid"] = {
            "nlay": grid_info.nlay,
            "ncpl": grid_info.ncpl,
            "total_cells": grid_info.total_cells,
        }
        if grid_info.nrow is not None:
            diagnostics["grid"]["nrow"] = grid_info.nrow
            diagnostics["grid"]["ncol"] = grid_info.ncol
    except Exception as e:
        diagnostics["error"] = f"{type(e).__name__}: {e}"
        if not debug:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"3D viz grid parse failed: {type(e).__name__}: {e}. "
                    f"workspace_root={diagnostics['workspace_root']}; "
                    f"grid_candidates={diagnostics.get('grid_candidates', [])[:5]}"
                ),
            )
        return {"ok": False, **diagnostics}

    if debug:
        return {"ok": True, **diagnostics}

    # Build available properties list
    properties = [
        {"key": "top", "kind": "cell", "label": "Top Elevation", "source": grid_info.grid_type.value.upper()},
        {"key": "botm", "kind": "cell", "label": "Bottom Elevation", "source": grid_info.grid_type.value.upper()},
        {"key": "idomain", "kind": "cell", "label": "IDOMAIN", "source": grid_info.grid_type.value.upper()},
    ]

    # For DIS models, scan for package files
    if grid_info.grid_type == GridType.DIS:
        for pkg_type, pkg_info in PACKAGE_PROPERTIES.items():
            pkg_path = _find_package_file(ws, pkg_type)
            if pkg_path is not None:
                for arr_name, arr_info in pkg_info.arrays.items():
                    properties.append({
                        "key": arr_name,
                        "kind": "cell",
                        "label": arr_info.label,
                        "source": pkg_type,
                    })

    # Determine available mesh modes based on grid type and size
    sc = _size_class(grid_info)
    block_available = grid_info.total_cells <= MAX_BLOCK_MODEL_CELLS
    if grid_info.grid_type == GridType.DISU:
        mesh_modes = ["top_surface"]
        block_available = False
    elif block_available:
        mesh_modes = ["top_surface", "all_layers_surface", "block_model"]
    else:
        mesh_modes = ["top_surface", "all_layers_surface"]

    result: Dict[str, Any] = {
        "grid_type": str(grid_info.grid_type.value),
        "grid_file": diagnostics.get("grid_file", "unknown"),
        "nlay": int(grid_info.nlay),
        "ncpl": int(grid_info.ncpl),
        "total_cells": int(grid_info.total_cells),
        "size_class": sc,
        "block_model_available": bool(block_available),
        "xorigin": float(grid_info.xorigin),
        "yorigin": float(grid_info.yorigin),
        "angrot": float(grid_info.angrot),
        "properties": properties,
        "mesh_modes": mesh_modes,
        "mvp": {"mesh_modes": mesh_modes, "supports_slicing": True},
    }

    # Include DIS-specific fields for backwards compat
    if grid_info.grid_type == GridType.DIS and dis_info:
        result["nrow"] = int(dis_info.nrow)
        result["ncol"] = int(dis_info.ncol)
        result["dis_file"] = result["grid_file"]
    elif grid_info.grid_type == GridType.DISV:
        result["nvert"] = int(grid_info.nvert or 0)
    elif grid_info.grid_type == GridType.DISU:
        result["nodes"] = int(grid_info.nodes or 0)

    return result


@router.get("/viz/mesh")
def viz_mesh(
    inputs_dir: str,
    workspace: Optional[str] = None,
    mode: str = "top_surface",
    layer: int = 0,
) -> Dict[str, Any]:
    ws = _ws_root(inputs_dir, workspace)
    try:
        grid_info, sess, dis_info = _get_session(ws)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load grid: {type(e).__name__}: {e}")

    valid_modes = {"top_surface", "block_model", "all_layers_surface"}
    if mode not in valid_modes:
        raise HTTPException(status_code=422, detail=f"Unsupported mode '{mode}'. Valid: {', '.join(sorted(valid_modes))}")

    # Size guard
    _check_size(grid_info, mode)

    # DISU only supports top_surface
    if grid_info.grid_type == GridType.DISU and mode != "top_surface":
        raise HTTPException(
            status_code=422,
            detail=f"Mode '{mode}' is not supported for DISU grids. Use 'top_surface'.",
        )

    try:
        ck = (mode, int(layer))
        if ck not in sess.mesh_cache:
            if grid_info.grid_type == GridType.DIS:
                if dis_info is None:
                    raise ValueError("DIS info not available")
                if mode == "top_surface":
                    sess.mesh_cache[ck] = _build_top_surface_mesh(dis_info, layer=layer)
                elif mode == "block_model":
                    sess.mesh_cache[ck] = _build_block_model_mesh(dis_info)
                elif mode == "all_layers_surface":
                    sess.mesh_cache[ck] = _build_all_layers_mesh(dis_info)

            elif grid_info.grid_type == GridType.DISV:
                if mode == "top_surface":
                    sess.mesh_cache[ck] = _build_disv_surface_mesh(sess, grid_info, layer=layer)
                elif mode == "block_model":
                    sess.mesh_cache[ck] = _build_disv_block_model(sess, grid_info)
                elif mode == "all_layers_surface":
                    sess.mesh_cache[ck] = _build_disv_all_layers(sess, grid_info)

            elif grid_info.grid_type == GridType.DISU:
                sess.mesh_cache[ck] = _build_disu_surface_mesh(sess, grid_info, layer=0)

        result = sess.mesh_cache[ck]

        # Final safety check on payload size
        if len(result.get("points", [])) > MAX_JSON_POINTS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Mesh payload too large ({len(result['points']):,} points, "
                    f"limit {MAX_JSON_POINTS:,}). Try a simpler view mode."
                ),
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build mesh: {type(e).__name__}: {e}")


@router.get("/viz/scalars")
def viz_scalars(
    inputs_dir: str,
    key: str,
    workspace: Optional[str] = None,
    layer: int = 0,
    mode: str = "top_surface",
) -> Dict[str, Any]:
    ws = _ws_root(inputs_dir, workspace)
    try:
        grid_info, sess, dis_info = _get_session(ws)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load grid: {type(e).__name__}: {e}")

    try:
        ck = (key, int(layer), mode)
        if ck in sess.scalars_cache:
            arr = sess.scalars_cache[ck]
        else:
            if grid_info.grid_type == GridType.DIS and dis_info is not None:
                # DIS path — use text-parsed arrays
                if mode == "block_model":
                    arr = _cell_scalars_block_model(dis_info, ws, sess, key=key)
                elif mode == "all_layers_surface":
                    arr = _cell_scalars_all_layers(dis_info, ws, sess, key=key)
                else:
                    arr = _cell_scalars_for_surface_pkg(dis_info, ws, sess, key=key, layer=layer)
            elif grid_info.grid_type in (GridType.DISV, GridType.DISU):
                # DISV/DISU path — use FloPy
                if mode == "block_model":
                    arr = _cell_scalars_disv_block_model(sess, grid_info, key=key)
                elif mode == "all_layers_surface":
                    arr = _cell_scalars_disv_all_layers(sess, grid_info, key=key)
                else:
                    arr = _cell_scalars_disv(sess, grid_info, key=key, layer=layer)
            else:
                raise ValueError(f"Unsupported grid type: {grid_info.grid_type}")
            sess.scalars_cache[ck] = arr
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build scalars: {type(e).__name__}: {e}")

    a = arr.astype(np.float32)
    mn = float(np.nanmin(a)) if a.size else 0.0
    mx = float(np.nanmax(a)) if a.size else 0.0

    # Resolve label for the property
    label = key
    pkg_info = property_to_package(key)
    if pkg_info:
        label = pkg_info[2]
    elif key.lower() in DIS_PROPERTIES:
        label = DIS_PROPERTIES[key.lower()]

    return {"key": key, "layer": layer, "mode": mode, "count": int(a.size), "min": mn, "max": mx, "label": label, "values": a.tolist()}


# ============================================================
# Spatial reference management
# ============================================================

class SpatialRefPayload(BaseModel):
    """User-defined spatial reference for a model."""
    epsg: Optional[int] = None        # EPSG code (e.g. 32614 for UTM Zone 14N)
    xorigin: Optional[float] = None   # easting of model grid origin
    yorigin: Optional[float] = None   # northing of model grid origin
    angrot: Optional[float] = None    # grid rotation angle (degrees)
    crs_name: Optional[str] = None    # human-readable CRS name (auto-populated)
    centroid_lat: Optional[float] = None  # centroid latitude (computed by frontend)
    centroid_lon: Optional[float] = None  # centroid longitude (computed by frontend)


def _gw_copilot_config_path(inputs_dir: str) -> Optional[Path]:
    """Return the path to GW_Copilot/config.json if it exists."""
    p = Path(inputs_dir)
    cfg = p / "GW_Copilot" / "config.json"
    if cfg.exists():
        return cfg
    return None


def _load_spatial_ref(inputs_dir: str) -> Optional[Dict[str, Any]]:
    """Load user-defined spatial reference from GW_Copilot/config.json."""
    cfg_path = _gw_copilot_config_path(inputs_dir)
    if cfg_path is None:
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return cfg.get("spatial_ref")
    except Exception:
        return None


def _save_spatial_ref(inputs_dir: str, ref: Dict[str, Any]) -> None:
    """Save user-defined spatial reference into GW_Copilot/config.json."""
    p = Path(inputs_dir)
    gw_dir = p / "GW_Copilot"
    gw_dir.mkdir(exist_ok=True)
    cfg_path = gw_dir / "config.json"

    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    cfg["spatial_ref"] = ref
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _save_location_context(inputs_dir: str, loc: Dict[str, Any]) -> None:
    """Save location context (centroid lat/lon) into GW_Copilot/config.json."""
    p = Path(inputs_dir)
    gw_dir = p / "GW_Copilot"
    gw_dir.mkdir(exist_ok=True)
    cfg_path = gw_dir / "config.json"

    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    cfg["location_context"] = loc
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# Built-in proj4 strings for common US State Plane and other projections.
# This avoids network requests for the most commonly used CRS codes.
_BUILTIN_PROJ4: Dict[int, str] = {
    # NAD83 State Plane California zones (US feet)
    2225: "+proj=lcc +lat_0=39.3333333333333 +lon_0=-122 +lat_1=41.6666666666667 +lat_2=40 +x_0=2000000.0001016 +y_0=500000.0001016 +ellps=GRS80 +units=us-ft +no_defs",
    2226: "+proj=lcc +lat_0=37.6666666666667 +lon_0=-122 +lat_1=39.8333333333333 +lat_2=38.3333333333333 +x_0=2000000.0001016 +y_0=500000.0001016 +ellps=GRS80 +units=us-ft +no_defs",
    2227: "+proj=lcc +lat_0=36.5 +lon_0=-120.5 +lat_1=38.4333333333333 +lat_2=37.0666666666667 +x_0=2000000.0001016 +y_0=500000.0001016 +ellps=GRS80 +units=us-ft +no_defs",
    2228: "+proj=lcc +lat_0=35.3333333333333 +lon_0=-119 +lat_1=37.25 +lat_2=36 +x_0=2000000.0001016 +y_0=500000.0001016 +ellps=GRS80 +units=us-ft +no_defs",
    2229: "+proj=lcc +lat_0=33.5 +lon_0=-118 +lat_1=35.4666666666667 +lat_2=34.0333333333333 +x_0=2000000.0001016 +y_0=500000.0001016 +ellps=GRS80 +units=us-ft +no_defs",
    2230: "+proj=lcc +lat_0=32.1666666666667 +lon_0=-116.25 +lat_1=33.8833333333333 +lat_2=32.7833333333333 +x_0=2000000.0001016 +y_0=500000.0001016 +ellps=GRS80 +units=us-ft +no_defs",
    # NAD83 State Plane — other common states (US feet)
    2254: "+proj=tmerc +lat_0=29.5 +lon_0=-88.8333333333333 +k=0.99995 +x_0=300000.0001016 +y_0=0 +ellps=GRS80 +units=us-ft +no_defs",  # Mississippi East
    2256: "+proj=tmerc +lat_0=29.5 +lon_0=-90.3333333333333 +k=0.99995 +x_0=700000.0001016 +y_0=0 +ellps=GRS80 +units=us-ft +no_defs",  # Mississippi West
    2264: "+proj=lcc +lat_0=33.75 +lon_0=-79 +lat_1=36.1666666666667 +lat_2=34.3333333333333 +x_0=609601.2192024384 +y_0=0 +ellps=GRS80 +units=us-ft +no_defs",  # NC
    2274: "+proj=lcc +lat_0=34.3333333333333 +lon_0=-86 +lat_1=36.4166666666667 +lat_2=35.25 +x_0=600000 +y_0=0 +ellps=GRS80 +units=us-ft +no_defs",  # Tennessee
    2277: "+proj=lcc +lat_0=31.6666666666667 +lon_0=-98.5 +lat_1=33.9666666666667 +lat_2=32.1333333333333 +x_0=600000 +y_0=2000000.0001016 +ellps=GRS80 +units=us-ft +no_defs",  # TX N Central
    2278: "+proj=lcc +lat_0=29.6666666666667 +lon_0=-100.333333333333 +lat_1=31.8833333333333 +lat_2=30.1166666666667 +x_0=600000 +y_0=4000000.0001016 +ellps=GRS80 +units=us-ft +no_defs",  # TX Central
    2285: "+proj=lcc +lat_0=47 +lon_0=-120.833333333333 +lat_1=48.7333333333333 +lat_2=47.5 +x_0=500000.0001016 +y_0=0 +ellps=GRS80 +units=us-ft +no_defs",  # WA North
    2286: "+proj=lcc +lat_0=45.3333333333333 +lon_0=-120.5 +lat_1=47.3333333333333 +lat_2=45.8333333333333 +x_0=500000.0001016 +y_0=0 +ellps=GRS80 +units=us-ft +no_defs",  # WA South
    # NAD83 State Plane (meters)
    26941: "+proj=lcc +lat_0=39.3333333333333 +lon_0=-122 +lat_1=41.6666666666667 +lat_2=40 +x_0=2000000 +y_0=500000 +ellps=GRS80 +units=m +no_defs",  # CA 1
    26942: "+proj=lcc +lat_0=37.6666666666667 +lon_0=-122 +lat_1=39.8333333333333 +lat_2=38.3333333333333 +x_0=2000000 +y_0=500000 +ellps=GRS80 +units=m +no_defs",  # CA 2
    26943: "+proj=lcc +lat_0=36.5 +lon_0=-120.5 +lat_1=38.4333333333333 +lat_2=37.0666666666667 +x_0=2000000 +y_0=500000 +ellps=GRS80 +units=m +no_defs",  # CA 3
    26944: "+proj=lcc +lat_0=35.3333333333333 +lon_0=-119 +lat_1=37.25 +lat_2=36 +x_0=2000000 +y_0=500000 +ellps=GRS80 +units=m +no_defs",  # CA 4
    26945: "+proj=lcc +lat_0=33.5 +lon_0=-118 +lat_1=35.4666666666667 +lat_2=34.0333333333333 +x_0=2000000 +y_0=500000 +ellps=GRS80 +units=m +no_defs",  # CA 5
    26946: "+proj=lcc +lat_0=32.1666666666667 +lon_0=-116.25 +lat_1=33.8833333333333 +lat_2=32.7833333333333 +x_0=2000000 +y_0=500000 +ellps=GRS80 +units=m +no_defs",  # CA 6
    # Common Florida
    2236: "+proj=tmerc +lat_0=24.3333333333333 +lon_0=-81 +k=0.999941177 +x_0=200000.0001016 +y_0=0 +ellps=GRS80 +units=us-ft +no_defs",  # FL East
    2237: "+proj=tmerc +lat_0=24.3333333333333 +lon_0=-82 +k=0.999941177 +x_0=200000.0001016 +y_0=0 +ellps=GRS80 +units=us-ft +no_defs",  # FL West
    2238: "+proj=lcc +lat_0=29 +lon_0=-84.5 +lat_1=30.75 +lat_2=29.5833333333333 +x_0=600000 +y_0=0 +ellps=GRS80 +units=us-ft +no_defs",  # FL North
    # WGS84 geographic
    4326: "+proj=longlat +datum=WGS84 +no_defs",
}


def _fetch_proj4_string(epsg: int) -> Optional[str]:
    """Get proj4 string for an EPSG code. Uses built-in dict first, then epsg.io."""
    # Check built-in first (no network needed)
    if epsg in _BUILTIN_PROJ4:
        logger.info("_fetch_proj4_string: EPSG:%s found in built-in database", epsg)
        return _BUILTIN_PROJ4[epsg]

    # UTM zones can be computed directly (no lookup needed)
    if 32601 <= epsg <= 32660:
        zone = epsg - 32600
        return f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
    if 32701 <= epsg <= 32760:
        zone = epsg - 32700
        return f"+proj=utm +zone={zone} +south +datum=WGS84 +units=m +no_defs"
    # NAD83 UTM
    if 26901 <= epsg <= 26923:
        zone = epsg - 26900
        return f"+proj=utm +zone={zone} +ellps=GRS80 +units=m +no_defs"

    # Fetch from epsg.io with proper User-Agent header
    try:
        import urllib.request
        url = f"https://epsg.io/{epsg}.proj4"
        req = urllib.request.Request(url, headers={"User-Agent": "GW-Copilot/1.0"})
        logger.info("_fetch_proj4_string: fetching from %s", url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            proj_str = resp.read().decode("utf-8").strip()
        if proj_str and len(proj_str) >= 10 and not proj_str.startswith("<"):
            logger.info("_fetch_proj4_string: got proj4: %s", proj_str[:100])
            return proj_str
        logger.warning("_fetch_proj4_string: invalid response from epsg.io: %s", proj_str[:50])
    except Exception as exc:
        logger.warning("_fetch_proj4_string: fetch failed for EPSG:%s: %s: %s",
                       epsg, type(exc).__name__, exc)

    return None


def _compute_centroid_latlon(
    epsg: int, xorigin: float, yorigin: float, angrot: float,
    x_total: float, y_total: float
) -> Optional[Tuple[float, float]]:
    """Compute centroid lat/lon from model grid parameters.

    Tries pyproj first (fast, no network), falls back to epsg.io + formula.
    Returns (lat, lon) or None.
    """
    # Centroid in model-local coordinates
    cx = x_total / 2.0
    cy = y_total / 2.0
    # Apply rotation
    a = math.radians(angrot)
    ca = math.cos(a)
    sa = math.sin(a)
    wx = xorigin + cx * ca - cy * sa
    wy = yorigin + cx * sa + cy * ca

    logger.info("_compute_centroid_latlon: EPSG:%s centroid_projected=(%.1f, %.1f)", epsg, wx, wy)

    # Try pyproj first (preferred — no network needed)
    try:
        from pyproj import Transformer  # type: ignore
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(wx, wy)
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            logger.info("_compute_centroid_latlon: pyproj succeeded: lat=%.6f lon=%.6f", lat, lon)
            return (float(lat), float(lon))
    except Exception as exc:
        logger.debug("_compute_centroid_latlon: pyproj not available: %s", exc)

    # Fall back to epsg.io proj4 string + simple projection formula
    try:
        proj_str = _fetch_proj4_string(epsg)
        if not proj_str:
            logger.warning("_compute_centroid_latlon: could not get proj4 string for EPSG:%s", epsg)
            return None

        # Parse key proj4 params for manual transform
        params: Dict[str, str] = {}
        for token in proj_str.split():
            if token.startswith("+") and "=" in token:
                k, v = token[1:].split("=", 1)
                params[k] = v

        proj_type = params.get("proj", "")
        logger.info("_compute_centroid_latlon: proj_type=%s units=%s", proj_type, params.get("units", "m"))

        if proj_type == "utm":
            zone = int(params.get("zone", "0"))
            south = "+south" in proj_str
            if zone < 1 or zone > 60:
                return None
            # UTM inverse formula (simplified via pyproj-less math)
            # Use the standard UTM parameters
            k0 = 0.9996
            a_wgs = 6378137.0
            f_wgs = 1 / 298.257223563
            e2 = 2 * f_wgs - f_wgs ** 2
            e_prime2 = e2 / (1 - e2)

            # Convert from false easting/northing
            lon0 = math.radians((zone - 1) * 6 - 180 + 3)

            # Check units
            units = params.get("units", "m")
            x_m = wx
            y_m = wy
            if units == "us-ft":
                x_m = wx * 0.3048006096012192
                y_m = wy * 0.3048006096012192
            elif units == "ft":
                x_m = wx * 0.3048
                y_m = wy * 0.3048

            x_m -= 500000.0  # remove false easting
            if south:
                y_m -= 10000000.0  # remove false northing for southern hemisphere

            M = y_m / k0
            mu = M / (a_wgs * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))

            e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
            phi1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu) + \
                   (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu) + \
                   (151 * e1 ** 3 / 96) * math.sin(6 * mu)

            sin_phi1 = math.sin(phi1)
            cos_phi1 = math.cos(phi1)
            tan_phi1 = math.tan(phi1)
            N1 = a_wgs / math.sqrt(1 - e2 * sin_phi1 ** 2)
            T1 = tan_phi1 ** 2
            C1 = e_prime2 * cos_phi1 ** 2
            R1 = a_wgs * (1 - e2) / ((1 - e2 * sin_phi1 ** 2) ** 1.5)
            D = x_m / (N1 * k0)

            lat = phi1 - (N1 * tan_phi1 / R1) * (
                D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * e_prime2) * D ** 4 / 24 +
                (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * e_prime2 - 3 * C1 ** 2) * D ** 6 / 720
            )
            lon = lon0 + (
                D - (1 + 2 * T1 + C1) * D ** 3 / 6 +
                (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * e_prime2 + 24 * T1 ** 2) * D ** 5 / 120
            ) / cos_phi1

            lat_deg = math.degrees(lat)
            lon_deg = math.degrees(lon)
            if -180 <= lon_deg <= 180 and -90 <= lat_deg <= 90:
                return (lat_deg, lon_deg)

        elif proj_type == "tmerc":
            # Transverse Mercator (State Plane etc.) — approximate inverse
            # For non-UTM TM, we need lat_0, lon_0, k, x_0, y_0
            lat_0 = math.radians(float(params.get("lat_0", "0")))
            lon_0 = math.radians(float(params.get("lon_0", "0")))
            k_0 = float(params.get("k", params.get("k_0", "1")))
            x_0 = float(params.get("x_0", "0"))
            y_0 = float(params.get("y_0", "0"))

            units = params.get("units", "m")
            x_m = wx
            y_m = wy
            if units == "us-ft":
                x_m = wx * 0.3048006096012192
                y_m = wy * 0.3048006096012192
            elif units == "ft":
                x_m = wx * 0.3048
                y_m = wy * 0.3048

            x_m -= x_0
            y_m -= y_0

            a_wgs = 6378137.0
            f_wgs = 1 / 298.257223563
            e2 = 2 * f_wgs - f_wgs ** 2
            e_prime2 = e2 / (1 - e2)

            # Footpoint latitude
            M = y_m / k_0 + _meridional_arc(a_wgs, e2, lat_0)
            mu = M / (a_wgs * (1 - e2 / 4 - 3 * e2 ** 2 / 64))
            e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
            phi1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu) + \
                   (21 * e1 ** 2 / 16) * math.sin(4 * mu)

            sin_phi1 = math.sin(phi1)
            cos_phi1 = math.cos(phi1)
            tan_phi1 = math.tan(phi1)
            N1 = a_wgs / math.sqrt(1 - e2 * sin_phi1 ** 2)
            R1 = a_wgs * (1 - e2) / ((1 - e2 * sin_phi1 ** 2) ** 1.5)
            D = x_m / (N1 * k_0)
            T1 = tan_phi1 ** 2

            lat_deg = math.degrees(phi1 - (N1 * tan_phi1 / R1) * D ** 2 / 2)
            lon_deg = math.degrees(lon_0 + D / cos_phi1)

            if -180 <= lon_deg <= 180 and -90 <= lat_deg <= 90:
                return (lat_deg, lon_deg)

        elif proj_type == "lcc":
            # Lambert Conformal Conic — full inverse projection
            lat_0 = math.radians(float(params.get("lat_0", "0")))
            lon_0 = math.radians(float(params.get("lon_0", "0")))
            lat_1 = math.radians(float(params.get("lat_1", params.get("lat_0", "0"))))
            lat_2 = math.radians(float(params.get("lat_2", params.get("lat_1", params.get("lat_0", "0")))))
            x_0 = float(params.get("x_0", "0"))
            y_0 = float(params.get("y_0", "0"))

            units = params.get("units", "m")
            x_m = wx
            y_m = wy
            if units == "us-ft":
                x_m = wx * 0.3048006096012192
                y_m = wy * 0.3048006096012192
            elif units == "ft":
                x_m = wx * 0.3048
                y_m = wy * 0.3048

            x_m -= x_0
            y_m -= y_0

            # Ellipsoid params — use GRS80 for NAD83 (nearly identical to WGS84)
            a_e = 6378137.0
            f_e = 1 / 298.257222101  # GRS80
            ellps = params.get("ellps", "GRS80")
            if ellps in ("WGS84", "wgs84"):
                f_e = 1 / 298.257223563
            e2 = 2 * f_e - f_e ** 2
            e = math.sqrt(e2)

            def _lcc_m(phi: float) -> float:
                return math.cos(phi) / math.sqrt(1 - e2 * math.sin(phi) ** 2)

            def _lcc_t(phi: float) -> float:
                sp = e * math.sin(phi)
                return math.tan(math.pi / 4 - phi / 2) / ((1 - sp) / (1 + sp)) ** (e / 2)

            m1 = _lcc_m(lat_1)
            m2 = _lcc_m(lat_2)
            t0 = _lcc_t(lat_0)
            t1 = _lcc_t(lat_1)
            t2 = _lcc_t(lat_2)

            if abs(lat_1 - lat_2) > 1e-10:
                n = (math.log(m1) - math.log(m2)) / (math.log(t1) - math.log(t2))
            else:
                n = math.sin(lat_1)

            F = m1 / (n * t1 ** n)
            rho0 = a_e * F * t0 ** n

            # Inverse
            rho_sign = 1 if n > 0 else -1
            x_s = rho_sign * x_m
            y_s = rho_sign * (rho0 - y_m)
            rho = rho_sign * math.sqrt(x_s ** 2 + y_s ** 2)
            theta = math.atan2(x_s, y_s)

            lon_deg = math.degrees(theta / n + lon_0)

            if abs(rho) < 1e-10:
                lat_deg = math.degrees(math.copysign(math.pi / 2, n))
            else:
                t = (rho / (a_e * F)) ** (1 / n)
                # Iterative lat from t
                phi = math.pi / 2 - 2 * math.atan(t)
                for _ in range(10):
                    sp = e * math.sin(phi)
                    phi_new = math.pi / 2 - 2 * math.atan(t * ((1 - sp) / (1 + sp)) ** (e / 2))
                    if abs(phi_new - phi) < 1e-12:
                        break
                    phi = phi_new
                lat_deg = math.degrees(phi)

            if -180 <= lon_deg <= 180 and -90 <= lat_deg <= 90:
                return (lat_deg, lon_deg)

        elif proj_type == "longlat":
            # Already geographic
            if -180 <= wx <= 180 and -90 <= wy <= 90:
                return (wy, wx)

    except Exception:
        pass

    return None


def _meridional_arc(a: float, e2: float, phi: float) -> float:
    """Compute the meridional arc distance from equator to latitude phi."""
    e4 = e2 ** 2
    e6 = e2 ** 3
    return a * (
        (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi -
        (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi) +
        (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi) -
        (35 * e6 / 3072) * math.sin(6 * phi)
    )


def load_location_context(inputs_dir: str) -> Optional[Dict[str, Any]]:
    """Load location context from GW_Copilot/config.json.

    This is a public function, also used by chat_agent for LLM grounding.
    Checks both inputs_dir/GW_Copilot/config.json and parent directories.

    If spatial_ref exists but location_context doesn't, auto-computes
    the centroid lat/lon from the spatial ref + model grid.
    """
    if not inputs_dir:
        logger.debug("load_location_context: no inputs_dir")
        return None

    # Find config file
    cfg_path = _gw_copilot_config_path(inputs_dir)
    if cfg_path is None:
        parent = Path(inputs_dir).parent
        if parent != Path(inputs_dir):
            parent_cfg = parent / "GW_Copilot" / "config.json"
            if parent_cfg.exists():
                cfg_path = parent_cfg

    if cfg_path is None:
        logger.debug("load_location_context: no config.json found for %s", inputs_dir)
        return None

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("load_location_context: failed to read %s: %s", cfg_path, exc)
        return None

    # If location_context already exists, return it
    loc = cfg.get("location_context")
    if loc and loc.get("centroid_lat") is not None:
        logger.info("load_location_context: returning cached location: lat=%s lon=%s",
                     loc.get("centroid_lat"), loc.get("centroid_lon"))
        return loc

    # Auto-compute from spatial_ref if available
    sr = cfg.get("spatial_ref")
    if not sr or not sr.get("epsg"):
        logger.debug("load_location_context: no spatial_ref with EPSG in config")
        return None

    epsg = sr["epsg"]
    xorigin = float(sr.get("xorigin", 0))
    yorigin = float(sr.get("yorigin", 0))
    angrot = float(sr.get("angrot", 0))
    crs_name = sr.get("crs_name", f"EPSG:{epsg}")
    logger.info("load_location_context: auto-computing centroid for EPSG:%s origin=(%.1f, %.1f)",
                epsg, xorigin, yorigin)

    # We need model extent — try to load from the grid
    try:
        ws = _ws_root(inputs_dir, None)
        _grid_info, _sess, dis_info = _get_session(ws)
        if dis_info is None:
            logger.warning("load_location_context: dis_info is None (DISV/DISU grid?)")
            return None
        x_total = float(np.sum(dis_info.delr))
        y_total = float(np.sum(dis_info.delc))
        logger.info("load_location_context: grid extent: x_total=%.1f y_total=%.1f", x_total, y_total)
    except Exception as exc:
        logger.warning("load_location_context: failed to load grid: %s: %s",
                       type(exc).__name__, exc)
        return None

    result = _compute_centroid_latlon(epsg, xorigin, yorigin, angrot, x_total, y_total)
    if result is None:
        logger.warning("load_location_context: _compute_centroid_latlon returned None")
        return None

    lat, lon = result
    logger.info("load_location_context: computed centroid lat=%.6f lon=%.6f", lat, lon)
    loc = {
        "centroid_lat": round(lat, 6),
        "centroid_lon": round(lon, 6),
        "epsg": epsg,
        "crs_name": crs_name,
    }

    # Persist so we don't recompute next time
    try:
        cfg["location_context"] = loc
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        logger.info("load_location_context: persisted location_context to %s", cfg_path)
    except Exception as exc:
        logger.warning("load_location_context: failed to persist: %s", exc)

    return loc


@router.get("/viz/spatial-ref")
def get_spatial_ref(inputs_dir: str) -> Dict[str, Any]:
    """Return the current spatial reference (user-defined or from model files)."""
    # Check user-defined first
    user_ref = _load_spatial_ref(inputs_dir)
    if user_ref:
        return {"source": "user", **user_ref}

    # Try model files (DIS options)
    ws = _ws_root(inputs_dir, None)
    try:
        grid_info, sess, dis_info = _get_session(ws)
        xo = grid_info.xorigin
        yo = grid_info.yorigin
        ang = grid_info.angrot
        if abs(xo) > 1e-6 or abs(yo) > 1e-6:
            return {
                "source": "model",
                "xorigin": xo,
                "yorigin": yo,
                "angrot": ang,
                "epsg": None,
            }
    except Exception:
        pass

    return {"source": "none"}


@router.post("/viz/spatial-ref")
def set_spatial_ref(inputs_dir: str, payload: SpatialRefPayload) -> Dict[str, Any]:
    """Save a user-defined spatial reference for the model."""
    ref = {
        "epsg": payload.epsg,
        "xorigin": payload.xorigin if payload.xorigin is not None else 0.0,
        "yorigin": payload.yorigin if payload.yorigin is not None else 0.0,
        "angrot": payload.angrot if payload.angrot is not None else 0.0,
        "crs_name": payload.crs_name,
    }
    _save_spatial_ref(inputs_dir, ref)

    # Store location context if centroid lat/lon was provided by the frontend
    location_ctx: Optional[Dict[str, Any]] = None
    if payload.centroid_lat is not None and payload.centroid_lon is not None:
        location_ctx = {
            "centroid_lat": round(payload.centroid_lat, 6),
            "centroid_lon": round(payload.centroid_lon, 6),
            "epsg": payload.epsg,
            "crs_name": payload.crs_name,
        }
        _save_location_context(inputs_dir, location_ctx)

    return {"ok": True, "spatial_ref": ref, "location_context": location_ctx}


@router.delete("/viz/spatial-ref")
def clear_spatial_ref(inputs_dir: str) -> Dict[str, Any]:
    """Remove user-defined spatial reference and location context."""
    cfg_path = _gw_copilot_config_path(inputs_dir)
    if cfg_path:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg.pop("spatial_ref", None)
            cfg.pop("location_context", None)
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except Exception:
            pass
    return {"ok": True}


@router.get("/viz/boundary")
def viz_boundary(
    inputs_dir: str,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """Return model domain boundary polygon as model coordinates."""
    ws = _ws_root(inputs_dir, workspace)
    try:
        grid_info, sess, dis_info = _get_session(ws)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load grid: {type(e).__name__}: {e}")

    # Check for user-defined spatial reference
    user_ref = _load_spatial_ref(inputs_dir)

    # DISV/DISU: delegate to FloPy-based boundary builder
    if grid_info.grid_type in (GridType.DISV, GridType.DISU):
        result = _build_boundary_disv(sess, grid_info)
        if user_ref:
            result["spatial_ref"] = user_ref
        return result

    # DIS path
    if dis_info is None:
        raise HTTPException(status_code=400, detail="DIS info not available")

    # Use user-defined origin/rotation if available, otherwise model defaults
    xorigin = dis_info.xorigin
    yorigin = dis_info.yorigin
    angrot = dis_info.angrot

    if user_ref:
        if user_ref.get("xorigin") is not None:
            xorigin = float(user_ref["xorigin"])
        if user_ref.get("yorigin") is not None:
            yorigin = float(user_ref["yorigin"])
        if user_ref.get("angrot") is not None:
            angrot = float(user_ref["angrot"])

    # Compute total model extent in model coordinates
    x_total = float(np.sum(dis_info.delr))
    y_total = float(np.sum(dis_info.delc))

    raw_corners = [
        (0.0, 0.0),
        (x_total, 0.0),
        (x_total, y_total),
        (0.0, y_total),
    ]

    corners: List[List[float]] = []
    for x, y in raw_corners:
        xr, yr = _rot_xy(x, y, angrot)
        corners.append([xorigin + xr, yorigin + yr])

    has_real_coords = abs(xorigin) > 1e-6 or abs(yorigin) > 1e-6

    delr_min = float(np.min(dis_info.delr))
    delr_max = float(np.max(dis_info.delr))
    delc_min = float(np.min(dis_info.delc))
    delc_max = float(np.max(dis_info.delc))

    max_grid = 500
    delr_list = dis_info.delr.tolist() if dis_info.ncol <= max_grid else []
    delc_list = dis_info.delc.tolist() if dis_info.nrow <= max_grid else []

    result: Dict[str, Any] = {
        "corners": corners,
        "xorigin": xorigin,
        "yorigin": yorigin,
        "angrot": angrot,
        "x_total": x_total,
        "y_total": y_total,
        "has_real_coords": has_real_coords,
        "nrow": int(dis_info.nrow),
        "ncol": int(dis_info.ncol),
        "nlay": int(dis_info.nlay),
        "grid_type": "dis",
        "delr_range": [delr_min, delr_max],
        "delc_range": [delc_min, delc_max],
        "delr": delr_list,
        "delc": delc_list,
    }

    if user_ref:
        result["spatial_ref"] = user_ref

    # Include location context if available
    loc_ctx = load_location_context(inputs_dir)
    if loc_ctx:
        result["location_context"] = loc_ctx

    return result


@router.get("/viz/bounds")
def viz_bounds(
    inputs_dir: str,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """Return 3D bounding box of the model in world coordinates (for clip plane ranges)."""
    ws = _ws_root(inputs_dir, workspace)
    try:
        grid_info, sess, dis_info = _get_session(ws)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load grid: {type(e).__name__}: {e}")

    # DISV/DISU: delegate to FloPy-based bounds
    if grid_info.grid_type in (GridType.DISV, GridType.DISU):
        return _build_bounds_disv(sess, grid_info)

    # DIS path
    if dis_info is None:
        raise HTTPException(status_code=400, detail="DIS info not available")

    x_total = float(np.sum(dis_info.delr))
    y_total = float(np.sum(dis_info.delc))

    zmax = float(np.nanmax(dis_info.top))
    zmin = float(np.nanmin(dis_info.botm))

    raw_corners = [(0.0, 0.0), (x_total, 0.0), (x_total, y_total), (0.0, y_total)]
    xs = []
    ys = []
    for x, y in raw_corners:
        xr, yr = _rot_xy(x, y, dis_info.angrot)
        xs.append(dis_info.xorigin + xr)
        ys.append(dis_info.yorigin + yr)

    return {
        "xmin": min(xs),
        "xmax": max(xs),
        "ymin": min(ys),
        "ymax": max(ys),
        "zmin": zmin,
        "zmax": zmax,
    }