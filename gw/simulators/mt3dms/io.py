"""MT3DMS / MT3D-USGS file I/O helpers.

Parsers for the MT3D name file, mass-balance (.mas) file, UCN file
discovery, variant detection, and linked MODFLOW model lookup.
These are used by the MT3DMS adapter when FloPy is not available or
when lightweight text parsing is sufficient.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MT3D namefile parser
# ---------------------------------------------------------------------------

def read_mt3d_nam(path: Path) -> "pd.DataFrame":
    """Parse an MT3D / MT3D-USGS name file.

    The format is the same flat layout as MODFLOW-2005::

        # comment
        FTYPE UNIT FNAME

    e.g.::

        LIST   6   mt3d.lst
        BTN    1   mt3d.btn
        ADV    2   mt3d.adv
        DSP    3   mt3d.dsp
        SSM    4   mt3d.ssm
        GCG    9   mt3d.gcg

    Returns a DataFrame with columns ``ftype``, ``unit``, ``fname``, ``status``.
    """
    import pandas as pd

    rows: List[Dict[str, Any]] = []

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("read_mt3d_nam: cannot read %s: %s", path, exc)
        return pd.DataFrame(columns=["ftype", "unit", "fname", "status"])

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        ftype = parts[0].upper()
        try:
            unit = int(parts[1])
        except ValueError:
            continue
        fname = parts[2]
        status = parts[3].upper() if len(parts) > 3 else ""
        rows.append({"ftype": ftype, "unit": unit, "fname": fname, "status": status})

    return pd.DataFrame(rows, columns=["ftype", "unit", "fname", "status"])


# ---------------------------------------------------------------------------
# Variant detection
# ---------------------------------------------------------------------------

def detect_mt3d_variant(ws_root: Path) -> str:
    """Distinguish between MT3DMS and MT3D-USGS.

    Returns ``"mt3d-usgs"`` if MT3D-USGS-specific packages (SFT, LKT, UZT,
    CTS) are found or a ``.mtnam`` file exists; otherwise ``"mt3dms"``.
    """
    # Check for .mtnam extension (MT3D-USGS convention)
    if list(ws_root.glob("*.mtnam")):
        return "mt3d-usgs"

    usgs_exts = {".sft", ".lkt", ".uzt", ".cts"}
    try:
        for p in ws_root.iterdir():
            if p.is_file() and p.suffix.lower() in usgs_exts:
                return "mt3d-usgs"
    except OSError:
        pass

    # Also check inside .nam / .mtnam for USGS-specific package types
    nam_files = list(ws_root.glob("*.nam")) + list(ws_root.glob("*.mtnam"))
    usgs_types = {"SFT", "LKT", "UZT", "CTS"}
    for nf in nam_files:
        try:
            txt = nf.read_text(encoding="utf-8", errors="replace")[:4000].upper()
            for utype in usgs_types:
                if utype in txt:
                    return "mt3d-usgs"
        except OSError:
            pass

    return "mt3dms"


# ---------------------------------------------------------------------------
# UCN file discovery
# ---------------------------------------------------------------------------

def find_ucn_files(ws_root: Path) -> List[Path]:
    """Find MT3D concentration output files (UCN) in the workspace.

    MT3DMS writes files named ``MT3D001.UCN``, ``MT3D002.UCN``, etc. — one
    per species.  Also catches lowercase variants and any ``*.ucn`` files.
    """
    ucn_files: List[Path] = []
    seen_lower: set = set()

    try:
        for p in ws_root.iterdir():
            if p.is_file() and p.suffix.lower() == ".ucn":
                low = p.name.lower()
                if low not in seen_lower:
                    seen_lower.add(low)
                    ucn_files.append(p)
    except OSError:
        pass

    # Sort numerically if the names follow MT3D###.UCN pattern
    def _sort_key(fp: Path) -> tuple:
        m = re.match(r"MT3D(\d+)", fp.stem, re.IGNORECASE)
        return (int(m.group(1)),) if m else (9999, fp.name.lower())

    ucn_files.sort(key=_sort_key)
    return ucn_files


# ---------------------------------------------------------------------------
# Mass-balance (.mas) file parser
# ---------------------------------------------------------------------------

def parse_mas_file(path: Path) -> Dict[str, Any]:
    """Parse a MT3D mass balance summary (.mas) file.

    The ``.mas`` file typically contains columnar data with headers like::

        TIME       TOTAL IN       TOTAL OUT      SOURCES       SINKS         ...
        0.0000     1234.5         -1234.0        100.0         -100.0        ...

    Returns a dict with ``ok``, ``records`` (list of row dicts), and
    ``summary`` (overall mass balance statistics).
    """
    import pandas as pd

    if not path.exists():
        return {"ok": False, "error": f"File not found: {path.name}"}

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"ok": False, "error": f"Cannot read {path.name}: {exc}"}

    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return {"ok": False, "error": "Empty .mas file"}

    # Find the header line (first non-comment line with text)
    header_idx = 0
    for i, line in enumerate(lines):
        if not line.strip().startswith("#"):
            header_idx = i
            break

    # Try to parse as whitespace-delimited columnar data
    header_parts = lines[header_idx].split()
    records: List[Dict[str, Any]] = []

    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        row: Dict[str, Any] = {}
        for j, hp in enumerate(header_parts):
            if j < len(parts):
                try:
                    row[hp] = float(parts[j])
                except ValueError:
                    row[hp] = parts[j]
        records.append(row)

    # Compute summary
    summary: Dict[str, Any] = {"num_records": len(records)}
    if records:
        # Look for common column names
        for col_name in ["TOTAL_IN", "TOTAL IN", "IN", "TOTAL_OUT", "TOTAL OUT", "OUT"]:
            matching_key = None
            for k in records[0]:
                if k.replace("_", " ").upper() == col_name.upper():
                    matching_key = k
                    break
            if matching_key:
                vals = [r[matching_key] for r in records
                        if isinstance(r.get(matching_key), (int, float))]
                if vals:
                    summary[f"{col_name.lower().replace(' ', '_')}_range"] = {
                        "min": min(vals),
                        "max": max(vals),
                    }

    return {
        "ok": True,
        "file": path.name,
        "records": records[:50],  # cap at 50 rows
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Linked MODFLOW model discovery
# ---------------------------------------------------------------------------

def find_linked_modflow_model(ws_root: Path) -> Optional[Path]:
    """Find the MODFLOW .nam file that this MT3D model is linked to.

    MT3D models require a flow solution via the Flow-Transport Link (FTL)
    file.  This function looks for:
    1. An FTL file reference inside the MT3D .nam file pointing back to
       a MODFLOW name file
    2. A .nam file in the same directory that is NOT an MT3D name file
       (i.e., contains MODFLOW packages like DIS, BAS6, LPF)
    """
    # Strategy 1: look for a MODFLOW .nam file in the same workspace
    modflow_indicators = {"LIST", "DIS", "BAS6", "BAS", "LPF", "BCF6",
                          "UPW", "PCG", "NWT", "SIP", "GMG", "DE4",
                          "WEL", "CHD", "RIV", "DRN", "GHB", "RCH"}

    mt3d_indicators = {"BTN", "ADV", "DSP", "SSM", "RCT", "GCG", "TOB"}

    for nf in ws_root.glob("*.nam"):
        if nf.name.lower() == "mfsim.nam":
            continue
        try:
            txt = nf.read_text(encoding="utf-8", errors="replace")[:4000].upper()
            has_modflow = any(ind in txt for ind in modflow_indicators)
            has_mt3d = any(ind in txt for ind in mt3d_indicators)
            if has_modflow and not has_mt3d:
                return nf
        except OSError:
            pass

    return None
