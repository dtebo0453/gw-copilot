"""MODFLOW-USG file I/O helpers.

Parsers for the free-format namefile, DIS/DISU time discretisation, and
stress-package files.  These are used by the MFUSG adapter when FloPy
is not available or when lightweight text parsing is sufficient.

MODFLOW-USG shares the same free-format namefile and DIS conventions as
MODFLOW-2005, but adds DISU (unstructured discretisation), SMS (Sparse
Matrix Solver), CLN (Connected Linear Networks), GNC (Ghost-Node
Corrections), and BCT (Block-Centered Transport).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Namefile parser
# ---------------------------------------------------------------------------

def read_mfusg_nam(path: Path) -> pd.DataFrame:
    """Parse a MODFLOW-USG free-format name file.

    Returns a DataFrame with columns ``ftype``, ``unit``, ``fname``, ``status``
    (one row per entry).

    MFUSG name files have the same format as MF2005::

        # comment
        FTYPE UNIT FNAME [STATUS]

    e.g.::

        LIST   6  model.lst
        DISU   7  model.disu
        BAS6   8  model.bas
        SMS   14  model.sms
        CLN   15  model.cln
    """
    rows: List[Dict[str, Any]] = []

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("read_mfusg_nam: cannot read %s: %s", path, exc)
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
# DIS time-discretisation parser
# ---------------------------------------------------------------------------

def read_mfusg_dis_times(path: Path) -> pd.DataFrame:
    """Parse DIS or DISU file for stress-period timing.

    Returns a DataFrame with columns ``per``, ``perlen``, ``nstp``, ``tsmult``,
    ``t_start``, ``t_end``, ``t_mid`` (0-indexed periods).

    Both DIS and DISU share the same stress-period data format at the end
    of the file: PERLEN NSTP TSMULT Ss/Tr (one line per stress period).

    For DIS, the header is:  NLAY NROW NCOL NPER ITMUNI LENUNI
    For DISU, the header is: NODES NLAY NJAG IVSD NPER ITMUNI LENUNI IDSYMRD
    """
    rows: List[Dict[str, Any]] = []

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("read_mfusg_dis_times: cannot read %s: %s", path, exc)
        return pd.DataFrame(columns=["per", "perlen", "nstp", "tsmult",
                                      "t_start", "t_end", "t_mid"])

    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.strip().startswith("#")]

    if not lines:
        return pd.DataFrame(columns=["per", "perlen", "nstp", "tsmult",
                                      "t_start", "t_end", "t_mid"])

    # Parse the header line to determine NPER
    # DIS:  NLAY NROW NCOL NPER ITMUNI LENUNI
    # DISU: NODES NLAY NJAG IVSD NPER ITMUNI LENUNI IDSYMRD
    header_parts = lines[0].split()
    nper = None
    ext_lower = path.suffix.lower()

    if ext_lower == ".disu":
        # DISU header: NODES NLAY NJAG IVSD NPER ...
        try:
            nper = int(header_parts[4])
        except (IndexError, ValueError):
            pass
    else:
        # DIS header: NLAY NROW NCOL NPER ...
        try:
            nper = int(header_parts[3])
        except (IndexError, ValueError):
            pass

    if nper is None or nper <= 0:
        logger.warning("read_mfusg_dis_times: could not parse NPER from header in %s", path)
        return pd.DataFrame(columns=["per", "perlen", "nstp", "tsmult",
                                      "t_start", "t_end", "t_mid"])

    # The stress-period data is at the end of the file -- last NPER lines
    sp_lines = lines[-nper:] if len(lines) >= nper else []

    t = 0.0
    for i, sp_line in enumerate(sp_lines):
        parts = sp_line.split()
        try:
            perlen = float(parts[0])
            nstp = int(parts[1])
            tsmult = float(parts[2])
        except (IndexError, ValueError):
            continue

        t_start = t
        t_end = t + perlen
        t_mid = t + perlen / 2.0
        rows.append({
            "per": i,
            "perlen": perlen,
            "nstp": nstp,
            "tsmult": tsmult,
            "t_start": t_start,
            "t_end": t_end,
            "t_mid": t_mid,
        })
        t = t_end

    return pd.DataFrame(rows, columns=["per", "perlen", "nstp", "tsmult",
                                         "t_start", "t_end", "t_mid"])


# ---------------------------------------------------------------------------
# Stress-package parser (via FloPy when available, text fallback)
# ---------------------------------------------------------------------------

def parse_mfusg_stress_package(
    path: Path,
    pkg_type: str = "WEL",
    value_cols: int = 1,
) -> pd.DataFrame:
    """Parse a MODFLOW-USG stress package file.

    Attempts FloPy-based loading first, falls back to regex-based text
    parsing for common free-format packages (WEL, DRN, RIV, GHB, CHD).

    For structured grids (DIS), columns are ``per``, ``layer``, ``row``,
    ``col``, plus value columns.  For unstructured grids (DISU), the
    cell index column is ``node`` instead of layer/row/col.

    Parameters
    ----------
    path : Path
        Full path to the stress package file.
    pkg_type : str
        Package type (e.g. "WEL", "DRN").
    value_cols : int
        Number of value columns (used only by text fallback).
    """
    # Try FloPy first
    try:
        return _parse_stress_via_flopy(path, pkg_type=pkg_type)
    except Exception:
        pass

    # Fallback: simple text parser
    return _parse_stress_text(path, pkg_type=pkg_type, value_cols=value_cols)


def _parse_stress_via_flopy(path: Path, *, pkg_type: str) -> pd.DataFrame:
    """Use FloPy to load the stress package via MfUsg."""
    import flopy

    ws_root = path.parent
    # Find the name file
    nam_files = list(ws_root.glob("*.nam"))
    if not nam_files:
        raise FileNotFoundError("No .nam file found")

    # Exclude mfsim.nam (that's MF6)
    nam_files = [f for f in nam_files if f.name.lower() != "mfsim.nam"]
    if not nam_files:
        raise FileNotFoundError("No MFUSG .nam file found")

    m = flopy.mfusg.MfUsg.load(
        nam_files[0].name,
        model_ws=str(ws_root),
        load_only=[pkg_type.lower()],
        check=False,
        verbose=False,
    )

    pkg = getattr(m, pkg_type.lower(), None)
    if pkg is None:
        raise AttributeError(f"Package {pkg_type} not found in model")

    rows: List[Dict[str, Any]] = []
    stress_data = pkg.stress_period_data
    if stress_data is None:
        return pd.DataFrame()

    for iper in range(m.nper):
        try:
            spd = stress_data[iper]
        except (KeyError, IndexError):
            continue
        if spd is None:
            continue
        for rec in spd:
            row_dict: Dict[str, Any] = {"per": iper}
            # Structured array fields
            if hasattr(rec, "dtype"):
                for name in rec.dtype.names:
                    val = rec[name]
                    if name == "k":
                        row_dict["layer"] = int(val) + 1
                    elif name == "i":
                        row_dict["row"] = int(val) + 1
                    elif name == "j":
                        row_dict["col"] = int(val) + 1
                    elif name == "node":
                        row_dict["node"] = int(val) + 1
                    else:
                        row_dict[name] = float(val) if hasattr(val, "__float__") else val
            rows.append(row_dict)

    return pd.DataFrame(rows)


def _parse_stress_text(
    path: Path,
    *,
    pkg_type: str,
    value_cols: int = 1,
) -> pd.DataFrame:
    """Fallback text parser for free-format stress packages.

    Handles the common pattern::

        NREC IPER
        LAYER ROW COL VALUE [AUX...]  (structured)
        -- or --
        NODE VALUE [AUX...]            (unstructured)
    """
    rows: List[Dict[str, Any]] = []
    val_names = {
        "WEL": ["flux"],
        "DRN": ["elev", "cond"],
        "RIV": ["stage", "cond", "rbot"],
        "GHB": ["bhead", "cond"],
        "CHD": ["shead", "ehead"],
    }
    vnames = val_names.get(pkg_type.upper(), [f"v{i}" for i in range(value_cols)])

    # Detect if this is an unstructured model
    ws_root = path.parent
    unstructured = is_unstructured(ws_root)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return pd.DataFrame()

    lines = text.splitlines()
    per = -1
    expect_data = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()

        if expect_data > 0:
            try:
                if unstructured:
                    # NODE VALUE [AUX...]
                    node = int(parts[0])
                    rec: Dict[str, Any] = {"per": per, "node": node}
                    for vi, vn in enumerate(vnames):
                        if 1 + vi < len(parts):
                            try:
                                rec[vn] = float(parts[1 + vi])
                            except ValueError:
                                pass
                else:
                    # LAYER ROW COL VALUE [AUX...]
                    layer = int(parts[0])
                    row = int(parts[1])
                    col = int(parts[2])
                    rec = {"per": per, "layer": layer, "row": row, "col": col}
                    for vi, vn in enumerate(vnames):
                        if 3 + vi < len(parts):
                            try:
                                rec[vn] = float(parts[3 + vi])
                            except ValueError:
                                pass
                rows.append(rec)
                expect_data -= 1
            except (IndexError, ValueError):
                expect_data -= 1
        else:
            # Check if this is a header line with ITMP (number of records)
            try:
                itmp = int(parts[0])
                if itmp >= 0:
                    per += 1
                    expect_data = itmp
                elif itmp < 0:
                    per += 1  # reuse previous period data
            except ValueError:
                pass

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grid type detection
# ---------------------------------------------------------------------------

def detect_grid_type(ws_root: Path) -> str:
    """Detect whether the MFUSG model uses DIS or DISU grid.

    Returns ``"disu"`` if a DISU package file is found or the namefile
    references a DISU package; otherwise returns ``"dis"`` (structured).
    """
    # Check for .disu file
    if list(ws_root.glob("*.disu")):
        return "disu"

    # Check namefile for DISU entry
    nam_files = [f for f in ws_root.glob("*.nam")
                 if f.name.lower() != "mfsim.nam"]
    for nam in nam_files:
        try:
            txt = nam.read_text(encoding="utf-8", errors="replace").upper()
            for line in txt.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts and parts[0] == "DISU":
                    return "disu"
        except OSError:
            pass

    return "dis"


def is_unstructured(ws_root: Path) -> bool:
    """Return ``True`` if the model uses a DISU (unstructured) grid."""
    return detect_grid_type(ws_root) == "disu"
