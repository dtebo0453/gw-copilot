from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

from . import flopy_bridge

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


def _try_flopy_stress(rel: Union[str, Path], value_cols: int, keep_aux: bool) -> Optional[pd.DataFrame]:
    """
    Attempt to read a stress-package file using FloPy (Tier-1). Returns None on failure.
    The returned dataframe matches parse_stress_package() output schema:
      per, lay, row, col, v1..vN, aux(optional), boundname(optional)
    """
    if os.environ.get("GW_MF6_USE_FLOPY", "1") == "0":
        return None
    ws = _ws_root()
    sim, _err = flopy_bridge.get_simulation(ws)
    if sim is None:
        return None

    target = Path(rel).name.lower()
    suffix = Path(rel).suffix.lower()

    # Preferred field order by package type
    field_map = {
        ".wel": ["q"],
        ".chd": ["head"],
        ".ghb": ["bhead", "cond"],
        ".riv": ["stage", "cond", "rbot"],
        ".drn": ["elev", "cond"],
    }
    preferred = field_map.get(suffix)

    # Find matching package by filename
    try:
        model_names = list(getattr(sim, "model_names", []))
    except Exception:
        model_names = []

    pkg_obj = None
    for mn in model_names:
        try:
            model = sim.get_model(mn)
        except Exception:
            continue
        for pkg in getattr(model, "packagelist", []):
            fn = getattr(pkg, "filename", None)
            if fn is None:
                continue
            fn0 = None
            if isinstance(fn, (list, tuple)):
                if fn:
                    fn0 = str(fn[0])
            else:
                fn0 = str(fn)
            if not fn0:
                continue
            if Path(fn0).name.lower() == target:
                pkg_obj = pkg
                break
        if pkg_obj is not None:
            break

    if pkg_obj is None:
        return None

    # Determine number of periods
    nper = None
    try:
        nper = int(sim.tdis.nper.get_data())
    except Exception:
        try:
            nper = int(getattr(sim.tdis, "nper", 0))
        except Exception:
            nper = None
    if not nper or nper <= 0:
        # If unknown, try a small reasonable default
        nper = 1

    rows: List[Dict[str, object]] = []
    spd = getattr(pkg_obj, "stress_period_data", None)
    if spd is None:
        return None

    for kper in range(nper):
        try:
            recs = spd.get_data(kper=kper)
        except Exception:
            recs = None
        if recs is None:
            continue

        try:
            df0 = pd.DataFrame.from_records(recs)
        except Exception:
            try:
                df0 = pd.DataFrame(recs)
            except Exception:
                continue

        if df0.empty:
            continue

        # cellid may be a tuple in a 'cellid' column
        cell_col = None
        for c in df0.columns:
            if c.lower() == "cellid":
                cell_col = c
                break
        if cell_col is None:
            # Sometimes it appears as "cellid1/cellid2/cellid3"
            continue

        # Split cellid into lay,row,col (FloPy typically 0-based internally)
        def _split_cellid(x):
            try:
                a, b, c = x
                return int(a), int(b), int(c)
            except Exception:
                return None

        cell = df0[cell_col].apply(_split_cellid)
        cell = cell.dropna()
        if cell.empty:
            continue
        lays = cell.apply(lambda t: t[0])
        rows_ = cell.apply(lambda t: t[1])
        cols_ = cell.apply(lambda t: t[2])

        # Convert to 1-based if it looks 0-based
        if lays.min() == 0 or rows_.min() == 0 or cols_.min() == 0:
            lays = lays + 1
            rows_ = rows_ + 1
            cols_ = cols_ + 1

        df0 = df0.loc[cell.index].copy()
        df0["lay"] = lays.values
        df0["row"] = rows_.values
        df0["col"] = cols_.values
        df0["per"] = kper + 1

        # Boundname if present
        bname_col = None
        for c in df0.columns:
            if c.lower() in ("boundname", "boundnames"):
                bname_col = c
                break

        # Determine value columns
        value_fields: List[str] = []
        if preferred:
            for f in preferred:
                for c in df0.columns:
                    if c.lower() == f.lower():
                        value_fields.append(c)
                        break
        if not value_fields:
            # fallback: pick numeric columns excluding metadata
            skip = {cell_col, "per", "lay", "row", "col"}
            if bname_col:
                skip.add(bname_col)
            for c in df0.columns:
                if c in skip:
                    continue
                if pd.api.types.is_numeric_dtype(df0[c]):
                    value_fields.append(c)
            value_fields = value_fields[:value_cols]

        if len(value_fields) < value_cols:
            return None

        # AUX fields are anything not used
        aux_fields = []
        if keep_aux:
            used = {cell_col, "per", "lay", "row", "col"}
            used.update(value_fields)
            if bname_col:
                used.add(bname_col)
            for c in df0.columns:
                if c in used:
                    continue
                # keep scalars only
                aux_fields.append(c)

        for _, r in df0.iterrows():
            rec: Dict[str, object] = {"per": int(r["per"]), "lay": int(r["lay"]), "row": int(r["row"]), "col": int(r["col"])}
            for i, vf in enumerate(value_fields, start=1):
                rec[f"v{i}"] = float(r[vf])
            if keep_aux:
                rec["aux"] = [r[c] for c in aux_fields] if aux_fields else []
            if bname_col and pd.notna(r.get(bname_col)):
                rec["boundname"] = str(r.get(bname_col))
            rows.append(rec)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def parse_stress_package(rel: Union[str, Path], value_cols: int = 1, keep_aux: bool = True) -> pd.DataFrame:
    df_flopy = _try_flopy_stress(rel, value_cols=value_cols, keep_aux=keep_aux)
    if df_flopy is not None and not df_flopy.empty:
        return df_flopy
    mf = parse_mf6_text(rel)
    rows: List[Dict[str, object]] = []

    # Stress-package CELLID shape varies by discretization:
    #  - DIS:   (k, i, j)
    #  - DISV:  (k, icpl)
    #  - DISU:  (node)
    # We do NOT require a specific shape here; we just need to extract values robustly.
    for per, lines in mf.periods.items():
        for ln in lines:
            u = ln.upper()
            if u.startswith("BEGIN ") or u.startswith("END "):
                continue

            parts = ln.split()
            if len(parts) < 1 + value_cols:
                continue

            # Try parse CELLID (3-int, then 2-int, then 1-int)
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
            # Always keep a normalized cellid string for consumers that don't care about shape
            rec["cellid"] = ",".join(str(v) for v in cell_fields.values())

            for i, v in enumerate(vals, start=1):
                rec[f"v{i}"] = v
            if keep_aux:
                rec["aux"] = rest
            if boundname is not None:
                rec["boundname"] = boundname
            rows.append(rec)

    return pd.DataFrame(rows)

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
