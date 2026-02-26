from __future__ import annotations

"""Deterministic, read-only model snapshot.

This snapshot supports LLM-assisted model improvement queries by extracting
key facts about a MODFLOW 6 workspace without executing the model.

Design goals:
- Deterministic and audit-friendly (pure filesystem reads)
- FloPy-first for reliability, with fallback to lightweight text parsing
- Rich fact extraction for confident LLM grounding

The snapshot extracts:
- Grid type and dimensions (DIS/DISV/DISU)
- Model types in simulation (GWF, GWT, GWE, etc.)
- Package inventory with file paths
- TDIS summary (stress periods, time units)
- Output configuration (HDS, CBC, LST locations)
- Basic solver info (IMS)
- Layer thickness statistics (when available)
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException

from gw.api.workspace_files import resolve_workspace_root
from gw.mf6.flopy_bridge import get_simulation, flopy_is_available

router = APIRouter()


SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".gw_copilot",
}


# =============================================================================
# File scanning utilities
# =============================================================================

def _scan(ws: Path) -> List[Path]:
    """Scan workspace for files, excluding common non-model directories."""
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


def _read_text_capped(path: Path, max_bytes: int = 500_000) -> str:
    """Read text file with size cap."""
    try:
        raw = path.read_bytes()[:max_bytes]
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return raw.decode("latin-1", errors="replace")
    except Exception:
        return ""


def _rel_path(ws: Path, p: Path) -> str:
    """Get relative path as posix string."""
    try:
        return p.relative_to(ws).as_posix()
    except Exception:
        return p.name


# =============================================================================
# FloPy-based snapshot extraction (Tier 1)
# =============================================================================

def _extract_flopy_snapshot(ws: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extract model snapshot using FloPy (primary method).
    
    Returns (snapshot_dict, None) on success, or (None, error_message) on failure.
    """
    if not flopy_is_available():
        return None, "FloPy not available"
    
    sim, err = get_simulation(ws)
    if sim is None:
        return None, err or "Failed to load simulation"
    
    snapshot: Dict[str, Any] = {
        "extraction_method": "flopy",
        "ok": True,
    }
    
    try:
        # Simulation-level info
        snapshot["sim_name"] = getattr(sim, "name", None)
        snapshot["sim_path"] = str(ws)
        
        # TDIS (time discretization)
        tdis_info = _extract_flopy_tdis(sim)
        snapshot["tdis"] = tdis_info
        
        # Models in simulation
        models_info = []
        model_names = list(getattr(sim, "model_names", []))
        
        for mn in model_names:
            try:
                model = sim.get_model(mn)
                model_info = _extract_flopy_model(ws, model)
                models_info.append(model_info)
            except Exception as e:
                models_info.append({
                    "name": mn,
                    "error": f"{type(e).__name__}: {e}"
                })
        
        snapshot["models"] = models_info
        
        # Pick primary model (usually GWF) for top-level grid info
        primary = next((m for m in models_info if m.get("model_type") == "gwf"), None)
        if primary is None and models_info:
            primary = models_info[0]
        
        if primary and "grid" in primary:
            snapshot["grid"] = primary["grid"]
            snapshot["packages"] = primary.get("packages", {})
            snapshot["package_files"] = primary.get("package_files", {})
            snapshot["layers"] = primary.get("layers", [])
        
        # IMS (solver) info
        ims_info = _extract_flopy_ims(sim)
        if ims_info:
            snapshot["ims"] = ims_info
        
        # Output files present
        snapshot["outputs_present"] = _scan_output_files(ws)
        
        return snapshot, None
        
    except Exception as e:
        return None, f"FloPy extraction failed: {type(e).__name__}: {e}"


def _extract_flopy_tdis(sim) -> Dict[str, Any]:
    """Extract TDIS info from FloPy simulation."""
    info: Dict[str, Any] = {}
    
    try:
        tdis = sim.tdis
        if tdis is None:
            return info
        
        # Number of periods
        try:
            nper = int(tdis.nper.get_data())
        except Exception:
            nper = getattr(tdis, "nper", None)
            if hasattr(nper, "get_data"):
                nper = nper.get_data()
            nper = int(nper) if nper is not None else None
        
        info["nper"] = nper
        
        # Time units
        try:
            time_units = tdis.time_units.get_data()
        except Exception:
            time_units = getattr(tdis, "time_units", None)
            if hasattr(time_units, "get_data"):
                time_units = time_units.get_data()
        
        if time_units:
            info["time_units"] = str(time_units)
        
        # Period data (perlen, nstp, tsmult)
        try:
            perioddata = tdis.perioddata.get_data()
            if perioddata is not None:
                periods = []
                totim = 0.0
                for i, row in enumerate(perioddata):
                    perlen = float(row[0]) if len(row) > 0 else 0.0
                    nstp = int(row[1]) if len(row) > 1 else 1
                    tsmult = float(row[2]) if len(row) > 2 else 1.0
                    
                    periods.append({
                        "per": i + 1,
                        "perlen": perlen,
                        "nstp": nstp,
                        "tsmult": tsmult,
                        "t_start": totim,
                        "t_end": totim + perlen,
                    })
                    totim += perlen
                
                info["periods"] = periods
                info["total_time"] = totim
        except Exception:
            pass
        
        # TDIS filename
        try:
            fn = getattr(tdis, "filename", None)
            if fn:
                if isinstance(fn, (list, tuple)):
                    fn = fn[0] if fn else None
                info["file"] = str(fn) if fn else None
        except Exception:
            pass
        
    except Exception:
        pass
    
    return info


def _extract_flopy_model(ws: Path, model) -> Dict[str, Any]:
    """Extract info from a single FloPy model (GWF, GWT, etc.)."""
    info: Dict[str, Any] = {}
    
    try:
        info["name"] = getattr(model, "name", None)
        
        # Model type
        model_type = getattr(model, "model_type", None)
        if model_type is None:
            # Infer from class name
            cls_name = type(model).__name__.lower()
            if "gwf" in cls_name:
                model_type = "gwf"
            elif "gwt" in cls_name:
                model_type = "gwt"
            elif "gwe" in cls_name:
                model_type = "gwe"
        info["model_type"] = model_type
        
        # Grid info
        grid_info = _extract_flopy_grid(model)
        if grid_info:
            info["grid"] = grid_info
        
        # Package inventory
        packages: Dict[str, str] = {}
        package_files: Dict[str, str] = {}
        
        for pkg in getattr(model, "packagelist", []):
            try:
                pkg_type = type(pkg).__name__.upper()
                # Clean up type name (e.g., "ModflowGwfnpf" -> "NPF")
                for prefix in ("MODFLOWGWF", "MODFLOWGWT", "MODFLOWGWE", "MODFLOW"):
                    if pkg_type.startswith(prefix):
                        pkg_type = pkg_type[len(prefix):]
                        break
                
                fn = getattr(pkg, "filename", None)
                if fn:
                    if isinstance(fn, (list, tuple)):
                        fn = fn[0] if fn else None
                    fn = str(fn) if fn else None
                
                if fn:
                    packages[pkg_type] = fn
                    # Store relative path
                    fp = ws / fn
                    if fp.exists():
                        package_files[pkg_type] = _rel_path(ws, fp)
                    else:
                        package_files[pkg_type] = fn
                        
            except Exception:
                continue
        
        info["packages"] = packages
        info["package_files"] = package_files
        
        # Layer thickness stats (for structured grids)
        layers = _extract_flopy_layer_stats(model)
        if layers:
            info["layers"] = layers
        
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    
    return info


def _extract_flopy_grid(model) -> Optional[Dict[str, Any]]:
    """Extract grid info from FloPy model."""
    try:
        # Try to get the discretization package
        dis = None
        grid_type = None
        
        for attr in ("dis", "disv", "disu"):
            dis = getattr(model, attr, None)
            if dis is not None:
                grid_type = attr
                break
        
        if dis is None:
            return None
        
        info: Dict[str, Any] = {"type": grid_type}
        
        # Common attributes
        def get_val(obj, attr):
            val = getattr(obj, attr, None)
            if val is None:
                return None
            if hasattr(val, "get_data"):
                val = val.get_data()
            if hasattr(val, "array"):
                val = val.array
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    return int(val.flat[0])
            return int(val) if val is not None else None
        
        if grid_type == "dis":
            info["nlay"] = get_val(dis, "nlay")
            info["nrow"] = get_val(dis, "nrow")
            info["ncol"] = get_val(dis, "ncol")
            
            # Cell sizes
            try:
                delr = getattr(dis, "delr", None)
                if delr is not None:
                    delr_data = delr.get_data() if hasattr(delr, "get_data") else delr
                    if isinstance(delr_data, np.ndarray):
                        info["delr_range"] = [float(np.min(delr_data)), float(np.max(delr_data))]
                
                delc = getattr(dis, "delc", None)
                if delc is not None:
                    delc_data = delc.get_data() if hasattr(delc, "get_data") else delc
                    if isinstance(delc_data, np.ndarray):
                        info["delc_range"] = [float(np.min(delc_data)), float(np.max(delc_data))]
            except Exception:
                pass
                
        elif grid_type == "disv":
            info["nlay"] = get_val(dis, "nlay")
            info["ncpl"] = get_val(dis, "ncpl")
            info["nvert"] = get_val(dis, "nvert")
            
        elif grid_type == "disu":
            info["nodes"] = get_val(dis, "nodes")
            info["nja"] = get_val(dis, "nja")
        
        # Grid origin/rotation if available
        for attr in ("xorigin", "yorigin", "angrot"):
            val = get_val(dis, attr)
            if val is not None:
                info[attr] = float(val)
        
        # DIS filename
        try:
            fn = getattr(dis, "filename", None)
            if fn:
                if isinstance(fn, (list, tuple)):
                    fn = fn[0] if fn else None
                info["file"] = str(fn) if fn else None
        except Exception:
            pass
        
        return info
        
    except Exception:
        return None


def _extract_flopy_layer_stats(model) -> List[Dict[str, Any]]:
    """Compute layer thickness statistics from FloPy model."""
    layers: List[Dict[str, Any]] = []
    
    try:
        dis = getattr(model, "dis", None)
        if dis is None:
            return layers
        
        # Get arrays
        def get_array(obj, attr):
            val = getattr(obj, attr, None)
            if val is None:
                return None
            if hasattr(val, "get_data"):
                val = val.get_data()
            if hasattr(val, "array"):
                val = val.array
            return np.array(val, dtype=np.float64) if val is not None else None
        
        nlay = getattr(dis, "nlay", None)
        if hasattr(nlay, "get_data"):
            nlay = nlay.get_data()
        nlay = int(nlay) if nlay else 0
        
        nrow = getattr(dis, "nrow", None)
        if hasattr(nrow, "get_data"):
            nrow = nrow.get_data()
        nrow = int(nrow) if nrow else 0
        
        ncol = getattr(dis, "ncol", None)
        if hasattr(ncol, "get_data"):
            ncol = ncol.get_data()
        ncol = int(ncol) if ncol else 0
        
        if not (nlay and nrow and ncol):
            return layers
        
        top = get_array(dis, "top")
        botm = get_array(dis, "botm")
        idomain = get_array(dis, "idomain")
        
        if top is None or botm is None:
            return layers
        
        top = top.reshape((nrow, ncol))
        botm = botm.reshape((nlay, nrow, ncol))
        if idomain is not None:
            idomain = idomain.reshape((nlay, nrow, ncol))
        
        for k in range(nlay):
            if k == 0:
                th = top - botm[k]
            else:
                th = botm[k - 1] - botm[k]
            
            mask = np.isfinite(th)
            if idomain is not None:
                mask = mask & (idomain[k] != 0)
            
            thv = th[mask]
            
            if thv.size > 0:
                stats = {
                    "min": float(np.min(thv)),
                    "mean": float(np.mean(thv)),
                    "max": float(np.max(thv)),
                }
            else:
                stats = {"min": None, "mean": None, "max": None}
            
            active_frac = float(np.mean(idomain[k] != 0)) if idomain is not None else 1.0
            
            layers.append({
                "layer": k + 1,
                "thickness": stats,
                "active_frac": active_frac,
            })
        
    except Exception:
        pass
    
    return layers


def _extract_flopy_ims(sim) -> Optional[Dict[str, Any]]:
    """Extract IMS solver info from simulation."""
    try:
        # Look for IMS in simulation packages
        for pkg in getattr(sim, "packagelist", []):
            pkg_type = type(pkg).__name__.lower()
            if "ims" in pkg_type:
                info: Dict[str, Any] = {}
                
                # Key solver parameters
                for attr in ("outer_dvclose", "inner_dvclose", "outer_maximum", "inner_maximum",
                             "linear_acceleration", "preconditioner_levels"):
                    val = getattr(pkg, attr, None)
                    if val is not None:
                        if hasattr(val, "get_data"):
                            val = val.get_data()
                        info[attr] = val
                
                # IMS filename
                fn = getattr(pkg, "filename", None)
                if fn:
                    if isinstance(fn, (list, tuple)):
                        fn = fn[0] if fn else None
                    info["file"] = str(fn) if fn else None
                
                return info
                
    except Exception:
        pass
    
    return None


# =============================================================================
# Fallback text-based snapshot extraction (Tier 2)
# =============================================================================

_RE_WS = re.compile(r"\s+")
_DIM_BLOCK_RE = re.compile(r"BEGIN\s+DIMENSIONS(.*?)END\s+DIMENSIONS", re.IGNORECASE | re.DOTALL)


def _extract_fallback_snapshot(ws: Path) -> Dict[str, Any]:
    """
    Extract model snapshot using text parsing (fallback when FloPy fails).
    """
    snapshot: Dict[str, Any] = {
        "extraction_method": "text_fallback",
        "ok": True,
    }
    
    files = _scan(ws)
    
    # Grid dimensions from DIS/DISV/DISU file
    grid_info = _parse_grid_dims_fallback(ws, files)
    if grid_info:
        snapshot["grid"] = grid_info
    else:
        snapshot["ok"] = False
        snapshot["error"] = "Could not parse grid dimensions"
    
    # TDIS info
    tdis_info = _parse_tdis_fallback(ws, files)
    if tdis_info:
        snapshot["tdis"] = tdis_info
    
    # NAM packages
    nam_path = _choose_nam(ws, files)
    if nam_path:
        snapshot["nam_file"] = _rel_path(ws, nam_path)
        packages, package_files = _parse_nam_packages(ws, nam_path)
        snapshot["packages"] = packages
        snapshot["package_files"] = package_files
    
    # Output files
    snapshot["outputs_present"] = _scan_output_files(ws)
    
    # Stress counts
    stress_counts = _count_stress_records(ws, files, snapshot.get("grid", {}))
    if stress_counts:
        snapshot["stress_counts_by_layer"] = stress_counts
    
    return snapshot


def _choose_nam(ws: Path, files: Optional[List[Path]] = None) -> Optional[Path]:
    """Choose the most likely model namefile."""
    if files is None:
        files = _scan(ws)
    
    nams = [p for p in files if p.suffix.lower() == ".nam"]
    if not nams:
        return None
    
    def score(p: Path) -> Tuple[int, str]:
        n = p.name.lower()
        s = 0
        if n == "aoi_model.nam":
            s += 100
        if n.startswith("mfsim"):
            s -= 50  # Prefer model NAM over sim NAM
        if "model" in n:
            s += 10
        return (-s, str(p).lower())
    
    nams.sort(key=score)
    return nams[0]


def _parse_grid_dims_fallback(ws: Path, files: List[Path]) -> Optional[Dict[str, Any]]:
    """Parse grid dimensions from DIS/DISV/DISU file using regex."""
    
    # Find discretization file
    dis_path: Optional[Path] = None
    
    # First try to find it via GWF namefile
    gwf_nams = sorted(
        [p for p in files if p.name.lower().startswith("gwf") and p.suffix.lower() == ".nam"],
        key=lambda x: (len(str(x)), str(x).lower())
    )
    
    if gwf_nams:
        nam_txt = _read_text_capped(gwf_nams[0], max_bytes=200_000)
        m = re.search(r"^\s*(DISV?6?|DISU6?)\s+([^\s#;]+)", nam_txt, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            rel = m.group(2).strip().strip('"')
            cand = ws / rel
            if cand.exists() and cand.is_file():
                dis_path = cand
    
    # Fallback: find by extension
    if dis_path is None:
        for ext in (".dis", ".disv", ".disu"):
            cands = sorted(
                [p for p in files if p.suffix.lower() == ext],
                key=lambda x: (len(str(x)), str(x).lower())
            )
            if cands:
                dis_path = cands[0]
                break
    
    if dis_path is None:
        return None
    
    txt = _read_text_capped(dis_path, max_bytes=2_000_000)
    if not txt:
        return None
    
    # Extract DIMENSIONS block
    m = _DIM_BLOCK_RE.search(txt)
    blob = m.group(1) if m else txt[:20000]
    
    def grab_int(key: str) -> Optional[int]:
        mm = re.search(rf"\b{re.escape(key)}\b\s*[=:]?\s*(\d+)", blob, flags=re.IGNORECASE)
        return int(mm.group(1)) if mm else None
    
    # Determine grid type from extension or keys
    ext = dis_path.suffix.lower().lstrip(".")
    grid_type = ext
    
    dims: Dict[str, Any] = {
        "type": grid_type,
        "file": _rel_path(ws, dis_path),
        "nlay": grab_int("NLAY"),
        "nrow": grab_int("NROW"),
        "ncol": grab_int("NCOL"),
        "ncpl": grab_int("NCPL"),
        "nodes": grab_int("NODES"),
    }
    
    # Clean up None values
    dims = {k: v for k, v in dims.items() if v is not None}
    
    return dims if len(dims) > 2 else None


def _parse_tdis_fallback(ws: Path, files: List[Path]) -> Dict[str, Any]:
    """Parse TDIS info using text parsing."""
    info: Dict[str, Any] = {}
    
    tdis_files = [p for p in files if p.suffix.lower() == ".tdis"]
    if not tdis_files:
        return info
    
    tdis_path = sorted(tdis_files, key=lambda p: len(str(p)))[0]
    info["file"] = _rel_path(ws, tdis_path)
    
    txt = _read_text_capped(tdis_path, max_bytes=500_000)
    
    # Extract DIMENSIONS
    dim_m = re.search(r"BEGIN\s+DIMENSIONS(.*?)END\s+DIMENSIONS", txt, re.IGNORECASE | re.DOTALL)
    if dim_m:
        nper_m = re.search(r"\bNPER\s+(\d+)", dim_m.group(1), re.IGNORECASE)
        if nper_m:
            info["nper"] = int(nper_m.group(1))
    
    # Time units
    opts_m = re.search(r"BEGIN\s+OPTIONS(.*?)END\s+OPTIONS", txt, re.IGNORECASE | re.DOTALL)
    if opts_m:
        tu_m = re.search(r"TIME_UNITS\s+(\w+)", opts_m.group(1), re.IGNORECASE)
        if tu_m:
            info["time_units"] = tu_m.group(1)
    
    # Period data
    pd_m = re.search(r"BEGIN\s+PERIODDATA(.*?)END\s+PERIODDATA", txt, re.IGNORECASE | re.DOTALL)
    if pd_m:
        periods = []
        totim = 0.0
        for line in pd_m.group(1).strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 1:
                try:
                    perlen = float(parts[0])
                    nstp = int(parts[1]) if len(parts) > 1 else 1
                    tsmult = float(parts[2]) if len(parts) > 2 else 1.0
                    
                    periods.append({
                        "per": len(periods) + 1,
                        "perlen": perlen,
                        "nstp": nstp,
                        "tsmult": tsmult,
                        "t_start": totim,
                        "t_end": totim + perlen,
                    })
                    totim += perlen
                except ValueError:
                    continue
        
        if periods:
            info["periods"] = periods
            info["total_time"] = totim
            if "nper" not in info:
                info["nper"] = len(periods)
    
    return info


def _parse_nam_packages(ws: Path, nam_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Parse package inventory from NAM file."""
    packages: Dict[str, str] = {}
    package_files: Dict[str, str] = {}
    
    txt = _read_text_capped(nam_path, max_bytes=350_000)
    
    for line in txt.splitlines():
        l = line.strip()
        if not l or l.startswith("#") or l.startswith("!"):
            continue
        
        u = l.upper()
        if u.startswith("BEGIN") or u.startswith("END"):
            continue
        
        # Strip inline comments
        l = l.split("#", 1)[0].split("!", 1)[0].strip()
        toks = _RE_WS.split(l)
        
        if len(toks) < 2:
            continue
        
        pkg = toks[0].strip().upper()
        f = toks[1].strip().strip('"')
        
        if pkg in {"LIST", "OPTIONS", "PACKAGES"}:
            continue
        
        packages[pkg] = f
        
        # Resolve file path
        fp = (nam_path.parent / f).resolve()
        try:
            fp.relative_to(ws.resolve())
            if fp.exists():
                package_files[pkg] = _rel_path(ws, fp)
            else:
                package_files[pkg] = f
        except Exception:
            package_files[pkg] = f
    
    return packages, package_files


def _scan_output_files(ws: Path) -> Dict[str, bool]:
    """Check for presence of common output files."""
    files = _scan(ws)
    
    return {
        "lst": any(p.suffix.lower() == ".lst" for p in files),
        "hds": any(p.suffix.lower() == ".hds" for p in files),
        "cbc": any(p.suffix.lower() == ".cbc" for p in files),
        "obs": any(p.suffix.lower() in (".obs", ".ob_gw") for p in files),
    }


def _count_stress_records(ws: Path, files: List[Path], grid: Dict[str, Any]) -> Dict[str, Dict[int, int]]:
    """Count stress records by layer for common stress packages."""
    nlay = int(grid.get("nlay") or 1)
    
    stress_exts = {
        "CHD": ".chd",
        "WEL": ".wel",
        "GHB": ".ghb",
        "RIV": ".riv",
        "DRN": ".drn",
        "RCH": ".rch",
        "EVT": ".evt",
    }
    
    counts: Dict[str, Dict[int, int]] = {}
    cellid_re = re.compile(r"^\(?\s*(\d+)(?:\s+(\d+))?(?:\s+(\d+))?\s*\)?")
    
    for label, ext in stress_exts.items():
        hits = [p for p in files if p.suffix.lower() == ext]
        if not hits:
            continue
        
        hits.sort(key=lambda p: (len(str(p)), str(p).lower()))
        pkg_file = hits[0]
        
        txt = _read_text_capped(pkg_file, max_bytes=600_000)
        layer_counts: Dict[int, int] = {}
        
        for line in txt.splitlines():
            l = line.strip()
            if not l or l.startswith("#") or l.startswith("!"):
                continue
            u = l.upper()
            if u.startswith("BEGIN") or u.startswith("END"):
                continue
            
            m = cellid_re.match(l)
            if m:
                k = int(m.group(1))
                if 1 <= k <= nlay:
                    layer_counts[k] = layer_counts.get(k, 0) + 1
        
        if layer_counts:
            counts[label] = layer_counts
    
    return counts


# =============================================================================
# Facts builder
# =============================================================================

def _build_model_facts(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create a compact, human-friendly table of model facts for LLM grounding.
    
    Each fact is a dict: {key, value, source}
    """
    facts: List[Dict[str, Any]] = []
    
    def add(key: str, value: Any, source: str) -> None:
        if value is None:
            return
        facts.append({"key": key, "value": value, "source": source})
    
    # Grid info
    grid = snapshot.get("grid", {})
    if grid:
        gtype = grid.get("type")
        if gtype:
            add("Grid type", gtype.upper(), "snapshot:grid")
        
        for k in ("nlay", "nrow", "ncol", "ncpl", "nodes"):
            v = grid.get(k)
            if v is not None:
                add(k.upper(), int(v), "snapshot:grid")

        # Cell sizes (DELR/DELC) with uniform/variable indicator
        delr_range = grid.get("delr_range")
        if delr_range and len(delr_range) == 2:
            if abs(delr_range[0] - delr_range[1]) < 1e-6:
                add("Cell size (DELR)", f"Uniform: {delr_range[0]:.2f}", "snapshot:grid")
            else:
                add("Cell size (DELR)", f"Variable: {delr_range[0]:.2f} – {delr_range[1]:.2f}", "snapshot:grid")

        delc_range = grid.get("delc_range")
        if delc_range and len(delc_range) == 2:
            if abs(delc_range[0] - delc_range[1]) < 1e-6:
                add("Cell size (DELC)", f"Uniform: {delc_range[0]:.2f}", "snapshot:grid")
            else:
                add("Cell size (DELC)", f"Variable: {delc_range[0]:.2f} – {delc_range[1]:.2f}", "snapshot:grid")

        if grid.get("file"):
            add("Discretization file", grid.get("file"), "snapshot:grid")
    
    # TDIS info
    tdis = snapshot.get("tdis", {})
    if tdis:
        nper = tdis.get("nper")
        if nper:
            add("NPER (stress periods)", int(nper), "snapshot:tdis")
        
        time_units = tdis.get("time_units")
        if time_units:
            add("Time units", time_units, "snapshot:tdis")
        
        total_time = tdis.get("total_time")
        if total_time:
            add("Total simulation time", float(total_time), "snapshot:tdis")
        
        if tdis.get("file"):
            add("TDIS file", tdis.get("file"), "snapshot:tdis")
    
    # Package inventory
    packages = snapshot.get("packages", {})
    package_files = snapshot.get("package_files", {})
    
    if packages:
        pkg_list = sorted(packages.keys())
        add("Packages", ", ".join(pkg_list), "snapshot:packages")
    
    for pkg, f in sorted(package_files.items()):
        add(f"{pkg} file", f, "snapshot:package_files")
    
    # Outputs
    outputs = snapshot.get("outputs_present", {})
    for outk, present in sorted(outputs.items()):
        add(f"Has {outk.upper()}", bool(present), "snapshot:outputs")
    
    # Stress counts
    stress_counts = snapshot.get("stress_counts_by_layer", {})
    for pkg, counts in sorted(stress_counts.items()):
        try:
            total = sum(int(x) for x in counts.values())
            add(f"{pkg} stress records", int(total), "snapshot:stress_counts")
        except Exception:
            continue
    
    # IMS info
    ims = snapshot.get("ims", {})
    if ims:
        for k in ("outer_dvclose", "inner_dvclose"):
            v = ims.get(k)
            if v is not None:
                add(f"IMS {k}", v, "snapshot:ims")
    
    # Models in simulation
    models = snapshot.get("models", [])
    if models:
        model_names = [m.get("name") for m in models if m.get("name")]
        model_types = [m.get("model_type") for m in models if m.get("model_type")]
        if model_types:
            add("Model types", ", ".join(t.upper() for t in model_types), "snapshot:models")

    # FloPy availability
    flopy_avail = snapshot.get("flopy_available")
    if flopy_avail is not None:
        add("FloPy available", bool(flopy_avail), "snapshot:flopy")

    # Output metadata (binary probing)
    out_meta = snapshot.get("output_metadata", {})
    if out_meta.get("probed"):
        hds_info = out_meta.get("hds", {})
        if hds_info.get("ok"):
            add("HDS times", hds_info.get("ntimes"), "snapshot:output_metadata")
            shape = hds_info.get("shape")
            if shape:
                add("HDS shape", "x".join(str(s) for s in shape), "snapshot:output_metadata")
            vr = hds_info.get("value_range")
            if vr:
                add("HDS value range", f"{vr[0]:.4g} – {vr[1]:.4g}", "snapshot:output_metadata")
        elif hds_info:
            add("HDS probe", f"FAILED: {hds_info.get('error', 'unknown')}", "snapshot:output_metadata")

        cbc_info = out_meta.get("cbc", {})
        if cbc_info.get("ok"):
            add("CBC records", ", ".join(cbc_info.get("record_names", [])), "snapshot:output_metadata")
            add("CBC times", cbc_info.get("ntimes"), "snapshot:output_metadata")
            add("CBC precision", cbc_info.get("precision"), "snapshot:output_metadata")
        elif cbc_info:
            add("CBC probe", f"FAILED: {cbc_info.get('error', 'unknown')}", "snapshot:output_metadata")

    # Stress package summaries
    stress_sums = snapshot.get("stress_summaries", {})
    for pkg, info in sorted(stress_sums.items()):
        total = info.get("total_records")
        periods = info.get("periods_with_data")
        vr = info.get("value_range")
        parts = []
        if total is not None:
            parts.append(f"{total} records")
        if periods is not None:
            parts.append(f"{periods} periods")
        if vr:
            parts.append(f"range [{vr[0]:.4g}, {vr[1]:.4g}]")
        if parts:
            add(f"{pkg} summary", ", ".join(parts), "snapshot:stress_summaries")

    return facts


def build_model_brief(snapshot: Dict[str, Any]) -> str:
    """Build a compact model context brief (~300-500 chars) for LLM prompt injection.

    Summarises the grid, time discretisation, packages (with stress counts),
    binary output metadata, and FloPy availability in a dense, LLM-friendly
    format.
    """
    lines: List[str] = []
    grid = snapshot.get("grid", {})
    gtype = (grid.get("type") or "").upper()

    # Grid line
    nlay = grid.get("nlay")
    if gtype == "DIS":
        nrow = grid.get("nrow")
        ncol = grid.get("ncol")
        lines.append(f"Grid: {nlay or '?'}-layer DIS ({nrow or '?'}x{ncol or '?'})")
    elif gtype == "DISV":
        ncpl = grid.get("ncpl")
        lines.append(f"Grid: {nlay or '?'}-layer DISV ({ncpl or '?'} cells/layer)")
    elif gtype == "DISU":
        nodes = grid.get("nodes")
        lines.append(f"Grid: DISU ({nodes or '?'} nodes)")
    else:
        lines.append(f"Grid: {gtype or 'unknown'}")

    # TDIS line
    tdis = snapshot.get("tdis", {})
    nper = tdis.get("nper")
    total_time = tdis.get("total_time")
    time_units = tdis.get("time_units", "")
    if nper:
        parts = [f"{nper} stress periods"]
        if total_time:
            parts.append(f"over {total_time:.6g} {time_units}".rstrip())
        lines.append("TDIS: " + " ".join(parts))

    # Packages line with stress counts
    packages = snapshot.get("packages", {})
    stress_sums = snapshot.get("stress_summaries", {})
    if packages:
        pkg_parts: List[str] = []
        for pkg in sorted(packages.keys()):
            ss = stress_sums.get(pkg)
            if ss and ss.get("total_records"):
                pkg_parts.append(f"{pkg} ({ss['total_records']} recs)")
            else:
                pkg_parts.append(pkg)
        lines.append("Packages: " + ", ".join(pkg_parts))

    # Outputs line
    out_meta = snapshot.get("output_metadata", {})
    out_parts: List[str] = []
    hds_info = out_meta.get("hds", {})
    if hds_info.get("ok"):
        shape_str = "x".join(str(s) for s in hds_info.get("shape", []))
        out_parts.append(f"HDS ({hds_info.get('ntimes', '?')} times, shape {shape_str})")
    cbc_info = out_meta.get("cbc", {})
    if cbc_info.get("ok"):
        rec_names = cbc_info.get("record_names", [])
        out_parts.append(f"CBC (records: {', '.join(rec_names[:8])})")
    if out_parts:
        lines.append("Outputs: " + "; ".join(out_parts))

    # FloPy availability
    if snapshot.get("flopy_available"):
        lines.append("FloPy: available")
    else:
        lines.append("FloPy: NOT available (use text parsers only)")

    return "\n".join(lines)


# =============================================================================
# Binary output probing
# =============================================================================

def _probe_binary_outputs(ws: Path) -> Dict[str, Any]:
    """Probe HDS and CBC binary output files for metadata.

    Returns a dict keyed by file type ('hds', 'cbc') with timing, shape, and
    record-name information.  Everything is wrapped in try/except so a corrupt
    file never crashes the snapshot.
    """
    result: Dict[str, Any] = {"probed": False}

    if not flopy_is_available():
        result["reason"] = "FloPy not available"
        return result

    import flopy  # type: ignore

    files = _scan(ws)

    # --- HDS ---
    hds_files = sorted(
        [p for p in files if p.suffix.lower() == ".hds"],
        key=lambda p: (len(str(p)), str(p).lower()),
    )
    if hds_files:
        hds_path = hds_files[0]
        try:
            hf = flopy.utils.HeadFile(str(hds_path))
            times = hf.get_times()
            # Truncate to first 20 + last 5 for compact representation
            if len(times) > 25:
                sample_times = times[:20] + times[-5:]
            else:
                sample_times = list(times)
            # Get shape from first timestep
            first_data = hf.get_data(totim=times[0])
            result["hds"] = {
                "file": _rel_path(ws, hds_path),
                "ntimes": len(times),
                "times_sample": [float(t) for t in sample_times],
                "shape": list(first_data.shape),
                "ok": True,
            }
            # Value range from first timestep (finite values only)
            finite = first_data[np.isfinite(first_data)]
            if finite.size > 0:
                result["hds"]["value_range"] = [float(np.min(finite)), float(np.max(finite))]
        except Exception as e:
            result["hds"] = {
                "file": _rel_path(ws, hds_path),
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }

    # --- CBC ---
    cbc_files = sorted(
        [p for p in files if p.suffix.lower() == ".cbc"],
        key=lambda p: (len(str(p)), str(p).lower()),
    )
    if cbc_files:
        cbc_path = cbc_files[0]
        cbc_opened = False
        cbc_obj = None
        for precision in ("double", "single"):
            try:
                cbc_obj = flopy.utils.CellBudgetFile(str(cbc_path), precision=precision)
                cbc_opened = True
                break
            except Exception:
                continue

        if cbc_opened and cbc_obj is not None:
            try:
                record_names = [n.strip() if isinstance(n, (str, bytes)) else str(n)
                                for n in cbc_obj.get_unique_record_names()]
                # Decode bytes if needed
                record_names = [n.decode() if isinstance(n, bytes) else n for n in record_names]
                record_names = [n.strip() for n in record_names]
                times = cbc_obj.get_times()
                if len(times) > 25:
                    sample_times = times[:20] + times[-5:]
                else:
                    sample_times = list(times)
                result["cbc"] = {
                    "file": _rel_path(ws, cbc_path),
                    "record_names": record_names,
                    "ntimes": len(times),
                    "times_sample": [float(t) for t in sample_times],
                    "precision": precision,
                    "ok": True,
                }
            except Exception as e:
                result["cbc"] = {
                    "file": _rel_path(ws, cbc_path),
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }
        elif cbc_files:
            result["cbc"] = {
                "file": _rel_path(ws, cbc_path),
                "ok": False,
                "error": "Could not open CBC file with either precision",
            }

    if result.get("hds") or result.get("cbc"):
        result["probed"] = True

    return result


# =============================================================================
# Stress package summaries
# =============================================================================

def _summarize_stress_packages(ws: Path, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize stress package data (record counts, value ranges).

    Tries FloPy first for structured extraction; falls back to text regex
    for basic counting and range extraction.
    """
    packages = snapshot.get("packages", {})
    package_files = snapshot.get("package_files", {})
    grid = snapshot.get("grid", {})
    nlay = int(grid.get("nlay") or 1)

    stress_types = {"WEL", "CHD", "GHB", "RIV", "DRN", "RCH", "EVT"}
    summaries: Dict[str, Any] = {}

    for pkg_type in sorted(stress_types):
        if pkg_type not in packages and pkg_type not in package_files:
            continue

        rel = package_files.get(pkg_type) or packages.get(pkg_type, "")
        if not rel:
            continue

        pkg_path = ws / rel
        if not pkg_path.exists():
            continue

        summary: Dict[str, Any] = {"file": rel}

        # Text-based extraction (works with or without FloPy)
        try:
            txt = _read_text_capped(pkg_path, max_bytes=1_000_000)
            # Count records in PERIOD blocks
            total_records = 0
            periods_with_data = 0
            float_values: List[float] = []

            in_period = False
            for line in txt.splitlines():
                l = line.strip()
                if not l or l.startswith("#") or l.startswith("!"):
                    continue
                upper = l.upper()
                if upper.startswith("BEGIN PERIOD"):
                    in_period = True
                    period_count = 0
                    continue
                if upper.startswith("END PERIOD"):
                    if in_period and period_count > 0:
                        periods_with_data += 1
                    in_period = False
                    continue
                if in_period:
                    # Try to count as a data record (starts with cellid or number)
                    toks = l.split()
                    if toks and (toks[0].isdigit() or toks[0].startswith("(")):
                        total_records += 1
                        period_count = 1  # at least one record
                        # Extract float values (skip cellid tokens)
                        for tok in toks:
                            tok_clean = tok.strip("()")
                            try:
                                v = float(tok_clean)
                                # Skip small integers that are likely cellid components
                                if abs(v) > nlay and not tok_clean.isdigit():
                                    float_values.append(v)
                                elif "." in tok_clean or "e" in tok_clean.lower():
                                    float_values.append(v)
                            except ValueError:
                                continue

            summary["total_records"] = total_records
            summary["periods_with_data"] = periods_with_data

            if float_values:
                summary["value_range"] = [float(min(float_values)), float(max(float_values))]

        except Exception:
            pass

        summaries[pkg_type] = summary

    return summaries


# =============================================================================
# Main entry point
# =============================================================================

def build_model_snapshot(ws_root: Path) -> Dict[str, Any]:
    """
    Build a comprehensive model snapshot for the given workspace.

    Uses FloPy as the primary extraction method, falling back to text parsing
    if FloPy fails or is unavailable.
    """
    if not ws_root.exists():
        return {
            "ok": False,
            "error": "workspace does not exist",
            "workspace_root": str(ws_root),
        }

    # Try FloPy first
    snapshot, flopy_err = _extract_flopy_snapshot(ws_root)

    if snapshot is None:
        # Fall back to text parsing
        snapshot = _extract_fallback_snapshot(ws_root)
        snapshot["flopy_error"] = flopy_err

    # Add workspace root
    snapshot["workspace_root"] = str(ws_root)

    # FloPy availability flag
    snapshot["flopy_available"] = flopy_is_available()

    # Binary output probing
    try:
        snapshot["output_metadata"] = _probe_binary_outputs(ws_root)
    except Exception:
        snapshot["output_metadata"] = {"probed": False, "reason": "probe exception"}

    # Stress package summaries
    try:
        snapshot["stress_summaries"] = _summarize_stress_packages(ws_root, snapshot)
    except Exception:
        snapshot["stress_summaries"] = {}

    # Build facts table (after enrichment so facts include new data)
    snapshot["facts"] = _build_model_facts(snapshot)

    # File counts
    files = _scan(ws_root)
    ext_counts: Dict[str, int] = {}
    for p in files:
        ext = p.suffix.lower()
        if ext:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    snapshot["file_ext_counts"] = ext_counts
    snapshot["file_count"] = len(files)

    return snapshot


@router.get("/model/snapshot")
def model_snapshot(inputs_dir: str, workspace: Optional[str] = None) -> Dict[str, Any]:
    """API endpoint to get model snapshot."""
    if not inputs_dir:
        raise HTTPException(status_code=422, detail="inputs_dir is required")
    
    try:
        ws_root = resolve_workspace_root(inputs_dir, workspace)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return build_model_snapshot(ws_root)


@router.get("/workspace/facts")
def workspace_facts(inputs_dir: str, workspace: Optional[str] = None) -> Dict[str, Any]:
    """
    API endpoint to get compact model facts for LLM grounding.
    
    Returns a lightweight JSON suitable for injection into chat/plot contexts.
    """
    if not inputs_dir:
        raise HTTPException(status_code=422, detail="inputs_dir is required")
    
    try:
        ws_root = resolve_workspace_root(inputs_dir, workspace)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    snapshot = build_model_snapshot(ws_root)
    
    return {
        "ok": snapshot.get("ok", False),
        "workspace_root": str(ws_root),
        "grid": snapshot.get("grid"),
        "tdis": {
            "nper": snapshot.get("tdis", {}).get("nper"),
            "time_units": snapshot.get("tdis", {}).get("time_units"),
            "total_time": snapshot.get("tdis", {}).get("total_time"),
        },
        "packages": list(snapshot.get("packages", {}).keys()),
        "outputs_present": snapshot.get("outputs_present", {}),
        "facts": snapshot.get("facts", []),
        "extraction_method": snapshot.get("extraction_method"),
        "layers": snapshot.get("layers", []),
    }
