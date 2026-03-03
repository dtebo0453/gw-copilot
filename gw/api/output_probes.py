from __future__ import annotations

"""Probes and data extraction for MODFLOW 6 binary output files.

Probe functions extract metadata (times, shapes, record names, value ranges)
from HDS and CBC files for lightweight context.

Extract functions read actual numerical data and produce text summaries
suitable for LLM context injection — per-layer statistics, drawdown
analysis, budget component summaries, etc.

All functions are wrapped in try/except so a corrupt file never crashes
the caller.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from gw.mf6.flopy_bridge import flopy_is_available


def probe_hds(ws_root: Path, hds_rel: str) -> Dict[str, Any]:
    """Probe an HDS (head) binary file for metadata.

    Returns dict with keys: ok, file, ntimes, times_sample, shape,
    value_range.  On failure: ok=False, error=str.
    """
    hds_path = ws_root / hds_rel
    if not hds_path.exists():
        return {"ok": False, "error": f"File not found: {hds_rel}"}

    if not flopy_is_available():
        return {"ok": False, "error": "FloPy not available"}

    try:
        import flopy  # type: ignore

        hf = flopy.utils.HeadFile(str(hds_path))
        times = hf.get_times()

        # Sample times (first 20 + last 5)
        if len(times) > 25:
            sample_times = times[:20] + times[-5:]
        else:
            sample_times = list(times)

        # Shape and value range from first timestep
        first_data = hf.get_data(totim=times[0])
        result: Dict[str, Any] = {
            "ok": True,
            "file": hds_rel,
            "ntimes": len(times),
            "times_sample": [float(t) for t in sample_times],
            "shape": list(first_data.shape),
        }

        finite = first_data[np.isfinite(first_data) & (first_data < 1e29)]
        if finite.size > 0:
            result["value_range"] = [float(np.min(finite)), float(np.max(finite))]

        return result

    except Exception as e:
        return {"ok": False, "file": hds_rel, "error": f"{type(e).__name__}: {e}"}


def probe_cbc(ws_root: Path, cbc_rel: str) -> Dict[str, Any]:
    """Probe a CBC (cell budget) binary file for metadata.

    Returns dict with keys: ok, file, record_names, ntimes, times_sample,
    precision.  On failure: ok=False, error=str.
    """
    cbc_path = ws_root / cbc_rel
    if not cbc_path.exists():
        return {"ok": False, "error": f"File not found: {cbc_rel}"}

    if not flopy_is_available():
        return {"ok": False, "error": "FloPy not available"}

    try:
        import flopy  # type: ignore

        cbc_obj = None
        used_precision = None
        for precision in ("double", "single"):
            try:
                cbc_obj = flopy.utils.CellBudgetFile(str(cbc_path), precision=precision)
                used_precision = precision
                break
            except Exception:
                continue

        if cbc_obj is None:
            return {"ok": False, "file": cbc_rel, "error": "Could not open with either precision"}

        record_names_raw = cbc_obj.get_unique_record_names()
        record_names: List[str] = []
        for n in record_names_raw:
            if isinstance(n, bytes):
                record_names.append(n.decode().strip())
            else:
                record_names.append(str(n).strip())

        times = cbc_obj.get_times()
        if len(times) > 25:
            sample_times = times[:20] + times[-5:]
        else:
            sample_times = list(times)

        return {
            "ok": True,
            "file": cbc_rel,
            "record_names": record_names,
            "ntimes": len(times),
            "times_sample": [float(t) for t in sample_times],
            "precision": used_precision,
        }

    except Exception as e:
        return {"ok": False, "file": cbc_rel, "error": f"{type(e).__name__}: {e}"}


def probe_workspace_outputs(ws_root: Path, file_index: Dict[str, Any]) -> Dict[str, Any]:
    """Probe all HDS and CBC files found in the file index.

    Returns a dict with 'hds' and 'cbc' sub-dicts.
    """
    result: Dict[str, Any] = {}

    for f in (file_index.get("files") or []):
        if not isinstance(f, dict):
            continue
        p = (f.get("path") or "").strip()
        if not p:
            continue

        ext = p.rsplit(".", 1)[-1].lower() if "." in p else ""
        if ext == "hds" and "hds" not in result:
            result["hds"] = probe_hds(ws_root, p)
        elif ext == "cbc" and "cbc" not in result:
            result["cbc"] = probe_cbc(ws_root, p)

    return result


# -----------------------------------------------------------------------
# Data extraction — returns text summaries with actual numerical data
# -----------------------------------------------------------------------

def _layer_stats(arr_2d: np.ndarray, nodata_thresh: float = 1e29) -> Dict[str, Any]:
    """Compute statistics for a 2D array (single layer), ignoring nodata."""
    finite = arr_2d[np.isfinite(arr_2d) & (np.abs(arr_2d) < nodata_thresh)]
    if finite.size == 0:
        return {"cells": int(arr_2d.size), "active": 0}
    return {
        "cells": int(arr_2d.size),
        "active": int(finite.size),
        "min": round(float(np.min(finite)), 4),
        "max": round(float(np.max(finite)), 4),
        "mean": round(float(np.mean(finite)), 4),
        "median": round(float(np.median(finite)), 4),
        "std": round(float(np.std(finite)), 4),
    }


def extract_hds_data(
    ws_root: Path,
    hds_rel: str,
    *,
    max_times: int = 5,
    max_chars: int = 40_000,
) -> Dict[str, Any]:
    """Extract actual head data from an HDS file and return a text summary.

    Reads data for a sample of timesteps (first, last, and evenly spaced in between)
    and produces per-layer statistics plus drawdown analysis (last - first).

    Returns dict with:
        ok: bool
        file: str
        summary_text: str   (markdown-formatted text for LLM context)
        metadata: dict       (ntimes, shape, etc.)
    """
    hds_path = ws_root / hds_rel
    if not hds_path.exists():
        return {"ok": False, "error": f"File not found: {hds_rel}"}

    if not flopy_is_available():
        return {"ok": False, "error": "FloPy not available"}

    try:
        import flopy  # type: ignore

        hf = flopy.utils.HeadFile(str(hds_path))
        times = hf.get_times()
        if not times:
            return {"ok": False, "file": hds_rel, "error": "No timesteps in HDS file"}

        # Select sample timesteps: first, last, and evenly-spaced middle ones
        if len(times) <= max_times:
            sample_indices = list(range(len(times)))
        else:
            sample_indices = [0]
            step = (len(times) - 1) / (max_times - 1)
            for i in range(1, max_times - 1):
                sample_indices.append(round(i * step))
            sample_indices.append(len(times) - 1)
            # De-duplicate while preserving order
            seen = set()
            unique = []
            for idx in sample_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique.append(idx)
            sample_indices = unique

        sample_times = [times[i] for i in sample_indices]

        # Read data for each sample time
        lines: List[str] = []
        lines.append(f"## Head Data from `{hds_rel}`")
        lines.append(f"- Total timesteps: {len(times)}")
        lines.append(f"- Time range: {times[0]} to {times[-1]}")

        first_data = None
        last_data = None

        for ti, totim in enumerate(sample_times):
            data = hf.get_data(totim=totim)
            if first_data is None:
                first_data = data.copy()
            last_data = data.copy()

            shape = data.shape
            nlay = shape[0] if len(shape) == 3 else 1

            lines.append(f"\n### Time = {totim} (step {sample_indices[ti]+1} of {len(times)})")
            lines.append(f"Array shape: {list(shape)}")

            if len(shape) == 3:
                for k in range(nlay):
                    stats = _layer_stats(data[k])
                    if stats.get("active", 0) == 0:
                        lines.append(f"- Layer {k+1}: no active cells")
                    else:
                        lines.append(
                            f"- Layer {k+1}: min={stats['min']}, max={stats['max']}, "
                            f"mean={stats['mean']}, median={stats['median']}, "
                            f"std={stats['std']}, active_cells={stats['active']}"
                        )
            else:
                stats = _layer_stats(data.ravel().reshape(1, -1)[0])
                if stats.get("active", 0) > 0:
                    lines.append(
                        f"- min={stats['min']}, max={stats['max']}, "
                        f"mean={stats['mean']}, active_cells={stats['active']}"
                    )

            # Budget check: don't exceed character limit
            if len("\n".join(lines)) > max_chars * 0.8:
                lines.append("\n*(Truncated to stay within context limits)*")
                break

        # Drawdown analysis: last timestep minus first timestep
        if first_data is not None and last_data is not None and len(times) > 1:
            lines.append(f"\n### Drawdown Analysis (Time {times[-1]} minus Time {times[0]})")
            drawdown = first_data - last_data  # positive = head decline
            shape = drawdown.shape
            nlay = shape[0] if len(shape) == 3 else 1

            if len(shape) == 3:
                for k in range(nlay):
                    dd_layer = drawdown[k]
                    finite = dd_layer[np.isfinite(dd_layer) & (np.abs(dd_layer) < 1e29)]
                    if finite.size == 0:
                        lines.append(f"- Layer {k+1}: no active cells for drawdown")
                    else:
                        lines.append(
                            f"- Layer {k+1}: min_drawdown={round(float(np.min(finite)), 4)}, "
                            f"max_drawdown={round(float(np.max(finite)), 4)}, "
                            f"mean_drawdown={round(float(np.mean(finite)), 4)}, "
                            f"cells_with_decline={int(np.sum(finite > 0.001))}, "
                            f"cells_with_rise={int(np.sum(finite < -0.001))}"
                        )
            else:
                finite = drawdown[np.isfinite(drawdown) & (np.abs(drawdown) < 1e29)]
                if finite.size > 0:
                    lines.append(
                        f"- min_drawdown={round(float(np.min(finite)), 4)}, "
                        f"max_drawdown={round(float(np.max(finite)), 4)}, "
                        f"mean_drawdown={round(float(np.mean(finite)), 4)}"
                    )

        # QA diagnostics: dry cells, head gradients, head range warnings
        if last_data is not None and len(last_data.shape) == 3 and len("\n".join(lines)) < max_chars * 0.85:
            nlay_qa = last_data.shape[0]
            _nodata = 1e20

            lines.append(f"\n### QA Diagnostics (Final Timestep, Time {times[-1]})")

            for k in range(nlay_qa):
                layer = last_data[k]
                dry_mask = (layer >= _nodata) | (~np.isfinite(layer))
                dry_count = int(np.sum(dry_mask))
                total_cells = layer.size
                dry_pct = (dry_count / total_cells * 100) if total_cells > 0 else 0

                active = layer[~dry_mask]
                if active.size < 2:
                    lines.append(
                        f"- Layer {k+1}: {dry_count} dry cells ({dry_pct:.1f}%), "
                        f"insufficient active cells for gradient analysis"
                    )
                    continue

                head_range = float(np.max(active) - np.min(active))

                # Cell-to-cell gradient (row and col directions)
                layer_clean = layer.copy().astype(float)
                layer_clean[dry_mask] = np.nan
                nrow, ncol = layer_clean.shape
                max_grad = 0.0
                if nrow > 1 and ncol > 1:
                    grad_r = np.abs(np.diff(layer_clean, axis=0))
                    grad_c = np.abs(np.diff(layer_clean, axis=1))
                    finite_r = grad_r[np.isfinite(grad_r)]
                    finite_c = grad_c[np.isfinite(grad_c)]
                    if finite_r.size > 0 or finite_c.size > 0:
                        all_grad = np.concatenate([finite_r, finite_c]) if finite_r.size > 0 and finite_c.size > 0 else (finite_r if finite_r.size > 0 else finite_c)
                        max_grad = float(np.max(all_grad))

                parts = [f"- Layer {k+1}: dry_cells={dry_count} ({dry_pct:.1f}%)"]
                parts.append(f"head_range={head_range:.4f}")
                parts.append(f"max_gradient={max_grad:.6f}")

                # Warnings
                warnings = []
                if dry_pct > 25:
                    warnings.append("HIGH_DRY")
                if head_range > 500:
                    warnings.append("LARGE_HEAD_RANGE")
                if max_grad > 100:
                    warnings.append("EXTREME_GRADIENT")

                if warnings:
                    parts.append(f"**WARNINGS: {', '.join(warnings)}**")

                lines.append(", ".join(parts))

        summary_text = "\n".join(lines)

        return {
            "ok": True,
            "file": hds_rel,
            "summary_text": summary_text,
            "metadata": {
                "ntimes": len(times),
                "shape": list(first_data.shape) if first_data is not None else [],
                "times_sampled": sample_times,
            },
        }

    except Exception as e:
        return {"ok": False, "file": hds_rel, "error": f"{type(e).__name__}: {e}"}


def extract_cbc_data(
    ws_root: Path,
    cbc_rel: str,
    *,
    max_records: int = 8,
    max_chars: int = 30_000,
) -> Dict[str, Any]:
    """Extract actual budget data from a CBC file and return a text summary.

    Reads the last timestep's budget records and produces per-component statistics
    (flow in, flow out, net).

    Returns dict with:
        ok: bool
        file: str
        summary_text: str   (markdown-formatted text for LLM context)
        metadata: dict
    """
    cbc_path = ws_root / cbc_rel
    if not cbc_path.exists():
        return {"ok": False, "error": f"File not found: {cbc_rel}"}

    if not flopy_is_available():
        return {"ok": False, "error": "FloPy not available"}

    try:
        import flopy  # type: ignore

        cbc_obj = None
        used_precision = None
        for precision in ("double", "single"):
            try:
                cbc_obj = flopy.utils.CellBudgetFile(str(cbc_path), precision=precision)
                used_precision = precision
                break
            except Exception:
                continue

        if cbc_obj is None:
            return {"ok": False, "file": cbc_rel, "error": "Could not open with either precision"}

        record_names_raw = cbc_obj.get_unique_record_names()
        record_names: List[str] = []
        for n in record_names_raw:
            if isinstance(n, bytes):
                record_names.append(n.decode().strip())
            else:
                record_names.append(str(n).strip())

        times = cbc_obj.get_times()
        if not times:
            return {"ok": False, "file": cbc_rel, "error": "No timesteps in CBC file"}

        # Read the last timestep
        last_time = times[-1]
        lines: List[str] = []
        lines.append(f"## Cell Budget Data from `{cbc_rel}`")
        lines.append(f"- Total timesteps: {len(times)}")
        lines.append(f"- Time range: {times[0]} to {times[-1]}")
        lines.append(f"- Precision: {used_precision}")
        lines.append(f"- Budget components: {', '.join(record_names)}")
        lines.append(f"\n### Budget at Time = {last_time} (last timestep)")

        for rname in record_names[:max_records]:
            try:
                rname_b = rname.encode() if isinstance(rname, str) else rname
                data = cbc_obj.get_data(totim=last_time, text=rname_b)
                if not data:
                    lines.append(f"\n**{rname}**: no data")
                    continue

                # data is a list of arrays (one per entry)
                for di, arr in enumerate(data):
                    if hasattr(arr, 'dtype') and arr.dtype.names:
                        # Structured array (e.g., list-based budget like WEL)
                        lines.append(f"\n**{rname}** (record {di+1}): structured, {len(arr)} entries")
                        if 'q' in arr.dtype.names:
                            q_vals = arr['q']
                            finite = q_vals[np.isfinite(q_vals)]
                            if finite.size > 0:
                                total_in = float(np.sum(finite[finite > 0]))
                                total_out = float(np.sum(finite[finite < 0]))
                                lines.append(
                                    f"  - Total inflow: {round(total_in, 4)}, "
                                    f"Total outflow: {round(total_out, 4)}, "
                                    f"Net: {round(total_in + total_out, 4)}, "
                                    f"Entries: {finite.size}"
                                )
                        # Show a few sample entries
                        sample_count = min(10, len(arr))
                        if sample_count > 0:
                            field_names = list(arr.dtype.names)
                            header = " | ".join(field_names)
                            lines.append(f"  - Sample entries ({sample_count} of {len(arr)}): {header}")
                            for si in range(sample_count):
                                row = arr[si]
                                vals = " | ".join(str(row[fn]) for fn in field_names)
                                lines.append(f"    {vals}")
                    else:
                        # 3D array (grid-based budget)
                        arr_np = np.asarray(arr)
                        finite = arr_np[np.isfinite(arr_np) & (np.abs(arr_np) < 1e29)]
                        if finite.size > 0:
                            total_in = float(np.sum(finite[finite > 0]))
                            total_out = float(np.sum(finite[finite < 0]))
                            lines.append(
                                f"\n**{rname}** (record {di+1}): shape={list(arr_np.shape)}"
                            )
                            lines.append(
                                f"  - Total inflow: {round(total_in, 4)}, "
                                f"Total outflow: {round(total_out, 4)}, "
                                f"Net: {round(total_in + total_out, 4)}, "
                                f"Active cells: {finite.size}"
                            )
                        else:
                            lines.append(f"\n**{rname}** (record {di+1}): no active data")

            except Exception as rec_err:
                lines.append(f"\n**{rname}**: error reading — {type(rec_err).__name__}: {rec_err}")

            if len("\n".join(lines)) > max_chars * 0.8:
                lines.append("\n*(Truncated to stay within context limits)*")
                break

        # QA: Overall budget balance and percent discrepancy
        if len("\n".join(lines)) < max_chars * 0.9:
            lines.append(f"\n### QA Budget Summary (Time = {last_time})")
            grand_in = 0.0
            grand_out = 0.0
            term_balances: List[Dict[str, Any]] = []

            for rname in record_names[:max_records]:
                try:
                    rname_b = rname.encode() if isinstance(rname, str) else rname
                    data_qa = cbc_obj.get_data(totim=last_time, text=rname_b)
                    term_in = 0.0
                    term_out = 0.0
                    for arr in (data_qa or []):
                        if hasattr(arr, 'dtype') and arr.dtype.names and 'q' in arr.dtype.names:
                            q = arr['q']
                            finite = q[np.isfinite(q)]
                            term_in += float(np.sum(finite[finite > 0]))
                            term_out += float(np.sum(finite[finite < 0]))
                        else:
                            arr_np = np.asarray(arr)
                            finite = arr_np[np.isfinite(arr_np) & (np.abs(arr_np) < 1e29)]
                            term_in += float(np.sum(finite[finite > 0]))
                            term_out += float(np.sum(finite[finite < 0]))
                    grand_in += term_in
                    grand_out += term_out
                    denom = max(term_in, abs(term_out), 1e-30)
                    pct = abs(term_in + term_out) / denom * 100
                    term_balances.append({
                        "name": rname, "in": term_in, "out": term_out,
                        "net": term_in + term_out, "pct_disc": pct,
                    })
                except Exception:
                    continue

            if term_balances:
                lines.append("| Term | IN | OUT | Net | % Disc |")
                lines.append("|---|---:|---:|---:|---:|")
                for tb in term_balances:
                    flag = " **" if tb["pct_disc"] > 1.0 else ""
                    lines.append(
                        f"| {tb['name']} | {tb['in']:.4f} | {tb['out']:.4f} "
                        f"| {tb['net']:.4f} | {tb['pct_disc']:.4f}%{flag} |"
                    )
                grand_net = grand_in + grand_out
                grand_denom = max(grand_in, abs(grand_out), 1e-30)
                grand_pct = abs(grand_net) / grand_denom * 100
                lines.append(f"\n**Overall:** IN={grand_in:.4f}, OUT={grand_out:.4f}, "
                            f"Net={grand_net:.4f}, Discrepancy={grand_pct:.6f}%")
                if grand_pct > 1.0:
                    lines.append("**WARNING:** Overall budget discrepancy > 1%")
                elif grand_pct > 0.5:
                    lines.append("**Note:** Overall budget discrepancy is marginal (0.5-1%)")

        summary_text = "\n".join(lines)

        return {
            "ok": True,
            "file": cbc_rel,
            "summary_text": summary_text,
            "metadata": {
                "ntimes": len(times),
                "record_names": record_names,
                "precision": used_precision,
            },
        }

    except Exception as e:
        return {"ok": False, "file": cbc_rel, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Per-cell head time-series
# ---------------------------------------------------------------------------

def extract_hds_timeseries(
    ws_root: Path,
    hds_rel: str,
    *,
    cells: List[Dict[str, int]],
    max_chars: int = 50_000,
) -> Dict[str, Any]:
    """Extract head time-series at specific cells across all saved timesteps.

    Parameters
    ----------
    ws_root : workspace root
    hds_rel : relative path to the .hds file
    cells : list of {"layer": int, "row": int, "col": int} (all 1-based)
    max_chars : truncation limit for the summary text

    Returns
    -------
    dict with ok, summary_text, metadata (same shape as extract_hds_data)
    """
    hds_path = ws_root / hds_rel
    if not hds_path.exists():
        return {"ok": False, "error": f"File not found: {hds_rel}"}
    if not flopy_is_available():
        return {"ok": False, "error": "FloPy not available"}
    if not cells:
        return {"ok": False, "error": "No cells specified"}

    try:
        import flopy  # type: ignore

        hf = flopy.utils.HeadFile(str(hds_path))
        times = hf.get_times()
        if not times:
            return {"ok": False, "error": "HDS file contains no timesteps"}

        # Validate and convert cells to 0-based tuples
        cell_tuples: List[tuple] = []
        cell_labels: List[str] = []
        for c in cells[:10]:  # cap at 10 cells
            lay = int(c.get("layer", 0)) - 1
            row = int(c.get("row", 0)) - 1
            col = int(c.get("col", 0)) - 1
            if lay < 0 or row < 0 or col < 0:
                continue
            cell_tuples.append((lay, row, col))
            cell_labels.append(f"L{lay+1}R{row+1}C{col+1}")

        if not cell_tuples:
            return {"ok": False, "error": "No valid cells after validation"}

        # Extract time-series for each cell
        # FloPy get_ts returns shape (ntimes, 2) with [time, head]
        all_ts: Dict[str, List[float]] = {}
        ts_times: Optional[List[float]] = None
        for ct, label in zip(cell_tuples, cell_labels):
            try:
                ts = hf.get_ts(ct)  # (ntimes, 2)
                if ts_times is None:
                    ts_times = [float(t) for t in ts[:, 0]]
                all_ts[label] = [float(v) for v in ts[:, 1]]
            except Exception:
                all_ts[label] = []

        if ts_times is None:
            return {"ok": False, "error": "Could not extract any time-series data"}

        ntimes_total = len(ts_times)

        # Sample timesteps if too many (keep first, last, evenly spaced middle)
        MAX_ROWS = 50
        if ntimes_total > MAX_ROWS:
            indices = [0]
            step = (ntimes_total - 1) / (MAX_ROWS - 1)
            for i in range(1, MAX_ROWS - 1):
                indices.append(int(round(i * step)))
            indices.append(ntimes_total - 1)
            indices = sorted(set(indices))
        else:
            indices = list(range(ntimes_total))

        # Build markdown output
        lines: List[str] = []
        lines.append(f"## Head Time-Series at {len(cell_tuples)} cell(s)")
        lines.append(f"File: {hds_rel} | Total timesteps: {ntimes_total} | "
                      f"Showing: {len(indices)} samples\n")

        # Table header
        active_labels = [l for l in cell_labels if all_ts.get(l)]
        header = "| Time |"
        sep = "|---:|"
        for label in active_labels:
            header += f" {label} |"
            sep += "---:|"
        lines.append(header)
        lines.append(sep)

        # Table rows
        DRY_THRESHOLD = 1e20
        for idx in indices:
            t = ts_times[idx]
            row_parts = [f"| {t:.4g} |"]
            for label in active_labels:
                vals = all_ts.get(label, [])
                if idx < len(vals):
                    v = vals[idx]
                    if abs(v) >= DRY_THRESHOLD or not np.isfinite(v):
                        row_parts.append(" DRY |")
                    else:
                        row_parts.append(f" {v:.4f} |")
                else:
                    row_parts.append(" N/A |")
            lines.append("".join(row_parts))

        # Per-cell statistics
        lines.append("\n### Per-Cell Statistics\n")
        lines.append("| Cell | Min | Max | Mean | Range | First | Last | Change | Trend |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")

        for label in cell_labels:
            vals = all_ts.get(label, [])
            if not vals:
                lines.append(f"| {label} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | error |")
                continue
            arr = np.array(vals, dtype=float)
            valid = arr[np.isfinite(arr) & (np.abs(arr) < DRY_THRESHOLD)]
            if len(valid) == 0:
                lines.append(f"| {label} | — | — | — | — | — | — | — | always dry |")
                continue
            vmin = float(np.min(valid))
            vmax = float(np.max(valid))
            vmean = float(np.mean(valid))
            vrange = vmax - vmin
            first_val = float(valid[0])
            last_val = float(valid[-1])
            change = last_val - first_val
            if abs(change) < 0.001:
                trend = "stable"
            elif change < 0:
                trend = f"declining ({change:.3f})"
            else:
                trend = f"rising (+{change:.3f})"
            dry_count = int(np.sum(~np.isfinite(arr) | (np.abs(arr) >= DRY_THRESHOLD)))
            dry_note = f" ({dry_count} dry)" if dry_count > 0 else ""
            lines.append(
                f"| {label}{dry_note} | {vmin:.4f} | {vmax:.4f} | {vmean:.4f} | "
                f"{vrange:.4f} | {first_val:.4f} | {last_val:.4f} | {change:+.4f} | {trend} |"
            )

        summary_text = "\n".join(lines)
        if len(summary_text) > max_chars:
            summary_text = summary_text[:max_chars] + "\n... (truncated)"

        return {
            "ok": True,
            "file": hds_rel,
            "summary_text": summary_text,
            "metadata": {
                "ntimes": ntimes_total,
                "cells_queried": len(cell_tuples),
                "times_sampled": len(indices),
            },
        }

    except Exception as e:
        return {"ok": False, "file": hds_rel, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Per-cell budget flows
# ---------------------------------------------------------------------------

def extract_cbc_cell_flows(
    ws_root: Path,
    cbc_rel: str,
    *,
    component: str,
    max_cells: int = 50,
    max_chars: int = 40_000,
) -> Dict[str, Any]:
    """Extract per-cell flow data for a specific budget component.

    Reads the final timestep and returns the top cells by absolute flow
    magnitude, so the LLM can see which cells are the biggest sources/sinks.

    Parameters
    ----------
    ws_root : workspace root
    cbc_rel : relative path to the .cbc file
    component : budget record name (e.g. "WEL", "CHD", "RCH", "STO-SS")
    max_cells : maximum number of cells to return
    max_chars : truncation limit

    Returns
    -------
    dict with ok, summary_text, metadata
    """
    cbc_path = ws_root / cbc_rel
    if not cbc_path.exists():
        return {"ok": False, "error": f"File not found: {cbc_rel}"}
    if not flopy_is_available():
        return {"ok": False, "error": "FloPy not available"}
    if not component:
        return {"ok": False, "error": "component is required"}

    try:
        import flopy  # type: ignore

        cbc_obj = None
        used_precision = None
        for precision in ("double", "single"):
            try:
                cbc_obj = flopy.utils.CellBudgetFile(str(cbc_path), precision=precision)
                _ = cbc_obj.get_times()
                used_precision = precision
                break
            except Exception:
                cbc_obj = None
                continue

        if cbc_obj is None:
            return {"ok": False, "error": "Cannot read CBC file (tried both single and double precision)"}

        times = cbc_obj.get_times()
        if not times:
            return {"ok": False, "error": "CBC file contains no timesteps"}

        record_names = [r.decode().strip() if isinstance(r, bytes) else str(r).strip()
                        for r in cbc_obj.get_unique_record_names()]

        # Find matching component (case-insensitive, partial match)
        comp_upper = component.upper().strip()
        matched_name = None
        for rn in record_names:
            if rn.upper() == comp_upper:
                matched_name = rn
                break
        if matched_name is None:
            for rn in record_names:
                if comp_upper in rn.upper():
                    matched_name = rn
                    break
        if matched_name is None:
            return {
                "ok": False,
                "error": f"Component '{component}' not found in CBC. "
                         f"Available: {', '.join(record_names)}",
            }

        # Read data at the last timestep
        last_time = times[-1]
        data = cbc_obj.get_data(text=matched_name, totim=last_time)
        if not data:
            return {"ok": False, "error": f"No data for component '{matched_name}' at time {last_time}"}

        lines: List[str] = []
        lines.append(f"## Per-Cell Budget: {matched_name}")
        lines.append(f"File: {cbc_rel} | Time: {last_time:.4g} | Precision: {used_precision}\n")

        # Process each data array returned
        all_cells: List[Dict[str, Any]] = []

        for d in data:
            if hasattr(d, 'dtype') and d.dtype.names:
                # Structured array (e.g., WEL, MAW with node/q fields)
                q_field = None
                for fname in d.dtype.names:
                    if fname.lower() in ('q', 'flow', 'rate'):
                        q_field = fname
                        break
                if q_field is None:
                    for fname in d.dtype.names:
                        if np.issubdtype(d[fname].dtype, np.floating):
                            q_field = fname
                if q_field is None:
                    continue

                for row_rec in d:
                    q_val = float(row_rec[q_field])
                    cell_info: Dict[str, Any] = {"q": q_val, "abs_q": abs(q_val)}
                    for fname in d.dtype.names:
                        if fname.lower() in ('node', 'cellid', 'node2'):
                            cell_info[fname] = int(row_rec[fname])
                    all_cells.append(cell_info)

            elif isinstance(d, np.ndarray) and d.ndim == 3:
                # 3D array (nlay, nrow, ncol) — e.g. CHD, STO
                nlay, nrow, ncol = d.shape
                for k in range(nlay):
                    for i in range(nrow):
                        for j in range(ncol):
                            q_val = float(d[k, i, j])
                            if abs(q_val) > 0 and np.isfinite(q_val):
                                all_cells.append({
                                    "layer": k + 1, "row": i + 1, "col": j + 1,
                                    "q": q_val, "abs_q": abs(q_val),
                                })

        if not all_cells:
            lines.append("No non-zero flow cells found for this component.")
            return {
                "ok": True,
                "file": cbc_rel,
                "summary_text": "\n".join(lines),
                "metadata": {"component": matched_name, "cell_count": 0},
            }

        # Sort by absolute flow (largest first)
        all_cells.sort(key=lambda c: c["abs_q"], reverse=True)
        top_cells = all_cells[:max_cells]

        # Summary statistics
        q_vals = np.array([c["q"] for c in all_cells])
        total_in = float(np.sum(q_vals[q_vals > 0]))
        total_out = float(np.sum(q_vals[q_vals < 0]))
        total_net = total_in + total_out

        lines.append(f"**Total cells with flow:** {len(all_cells)}")
        lines.append(f"**Total IN:** {total_in:.4f} | **Total OUT:** {total_out:.4f} | "
                      f"**Net:** {total_net:.4f}")
        lines.append(f"**Showing top {len(top_cells)} cells by flow magnitude**\n")

        # Determine which ID fields to show
        has_node = any("node" in c for c in top_cells)
        has_lrc = any("layer" in c for c in top_cells)

        if has_lrc:
            lines.append("| Rank | Layer | Row | Col | Flow (q) | Direction |")
            lines.append("|---:|---:|---:|---:|---:|---|")
            for rank, c in enumerate(top_cells, 1):
                q = c["q"]
                direction = "IN" if q > 0 else "OUT"
                lines.append(
                    f"| {rank} | {c.get('layer', '?')} | {c.get('row', '?')} | "
                    f"{c.get('col', '?')} | {q:.4f} | {direction} |"
                )
        elif has_node:
            lines.append("| Rank | Node | Flow (q) | Direction |")
            lines.append("|---:|---:|---:|---|")
            for rank, c in enumerate(top_cells, 1):
                q = c["q"]
                direction = "IN" if q > 0 else "OUT"
                node = c.get("node", c.get("cellid", "?"))
                lines.append(f"| {rank} | {node} | {q:.4f} | {direction} |")
        else:
            lines.append("| Rank | Flow (q) | Direction |")
            lines.append("|---:|---:|---|")
            for rank, c in enumerate(top_cells, 1):
                q = c["q"]
                direction = "IN" if q > 0 else "OUT"
                lines.append(f"| {rank} | {q:.4f} | {direction} |")

        summary_text = "\n".join(lines)
        if len(summary_text) > max_chars:
            summary_text = summary_text[:max_chars] + "\n... (truncated)"

        return {
            "ok": True,
            "file": cbc_rel,
            "summary_text": summary_text,
            "metadata": {
                "component": matched_name,
                "cell_count": len(all_cells),
                "total_in": total_in,
                "total_out": total_out,
                "cells_shown": len(top_cells),
            },
        }

    except Exception as e:
        return {"ok": False, "file": cbc_rel, "error": f"{type(e).__name__}: {e}"}


# ===========================================================================
# UCN (Concentration) file probes — for MT3DMS / SEAWAT
# ===========================================================================

def probe_ucn(ws_root: Path, ucn_rel: str) -> Dict[str, Any]:
    """Probe a UCN (concentration) binary file for metadata.

    Returns dict with keys: ok, file, ntimes, times_sample, shape,
    value_range.  On failure: ok=False, error=str.
    """
    ucn_path = ws_root / ucn_rel
    if not ucn_path.exists():
        return {"ok": False, "file": ucn_rel, "error": "file not found"}
    try:
        import flopy
        uf = flopy.utils.UcnFile(str(ucn_path))
        times = uf.get_times()
        data0 = uf.get_data(totim=times[0]) if times else None
        shape = list(data0.shape) if data0 is not None else None

        # Value range across all times (sample first + last)
        vmin, vmax = np.inf, -np.inf
        sample_times = [times[0], times[-1]] if len(times) > 1 else times[:1]
        for t in sample_times:
            arr = uf.get_data(totim=t)
            valid = arr[np.isfinite(arr)]
            if valid.size:
                vmin = min(vmin, float(valid.min()))
                vmax = max(vmax, float(valid.max()))

        return {
            "ok": True,
            "file": ucn_rel,
            "ntimes": len(times),
            "times_sample": times[:5] + (times[-2:] if len(times) > 5 else []),
            "shape": shape,
            "value_range": [vmin, vmax] if np.isfinite(vmin) else None,
        }
    except Exception as e:
        return {"ok": False, "file": ucn_rel, "error": f"{type(e).__name__}: {e}"}


def extract_ucn_data(
    ws_root: Path,
    ucn_rel: str,
    *,
    max_times: int = 5,
    max_chars: int = 40_000,
) -> Dict[str, Any]:
    """Extract concentration data with per-layer statistics.

    Mirrors the pattern of extract_hds_data but for UCN files.
    """
    ucn_path = ws_root / ucn_rel
    if not ucn_path.exists():
        return {"ok": False, "file": ucn_rel, "error": "file not found"}
    try:
        import flopy
        uf = flopy.utils.UcnFile(str(ucn_path))
        times = uf.get_times()
        if not times:
            return {"ok": False, "file": ucn_rel, "error": "no timesteps"}

        # Sample evenly spaced times
        if len(times) <= max_times:
            sel_times = times
        else:
            indices = np.linspace(0, len(times) - 1, max_times, dtype=int)
            sel_times = [times[i] for i in indices]

        lines: List[str] = [f"## Concentration Summary — {ucn_rel}",
                            f"Total timesteps: {len(times)}",
                            f"Sampled: {len(sel_times)} timesteps", ""]

        for t in sel_times:
            arr = uf.get_data(totim=t)
            lines.append(f"### Time = {t}")
            nlay = arr.shape[0] if arr.ndim >= 3 else 1
            for lay in range(nlay):
                layer_data = arr[lay] if arr.ndim >= 3 else arr
                valid = layer_data[np.isfinite(layer_data)]
                if valid.size == 0:
                    lines.append(f"  Layer {lay + 1}: no valid data")
                    continue
                neg_count = int(np.sum(valid < 0))
                neg_note = f" ({neg_count} negative)" if neg_count else ""
                lines.append(
                    f"  Layer {lay + 1}: min={float(valid.min()):.4g}, "
                    f"mean={float(valid.mean()):.4g}, "
                    f"max={float(valid.max()):.4g}, "
                    f"std={float(valid.std()):.4g}{neg_note}"
                )
            lines.append("")

        summary_text = "\n".join(lines)
        if len(summary_text) > max_chars:
            summary_text = summary_text[:max_chars] + "\n... (truncated)"

        return {
            "ok": True,
            "file": ucn_rel,
            "summary_text": summary_text,
            "metadata": {"ntimes": len(times), "sampled": len(sel_times)},
        }
    except Exception as e:
        return {"ok": False, "file": ucn_rel, "error": f"{type(e).__name__}: {e}"}


def extract_ucn_timeseries(
    ws_root: Path,
    ucn_rel: str,
    *,
    cells: List[Dict[str, int]],
    max_chars: int = 50_000,
) -> Dict[str, Any]:
    """Extract concentration time-series at specific cells.

    *cells*: list of {"layer": int, "row": int, "col": int} (1-based).
    """
    ucn_path = ws_root / ucn_rel
    if not ucn_path.exists():
        return {"ok": False, "file": ucn_rel, "error": "file not found"}
    try:
        import flopy
        uf = flopy.utils.UcnFile(str(ucn_path))
        times = uf.get_times()
        if not times:
            return {"ok": False, "file": ucn_rel, "error": "no timesteps"}

        lines: List[str] = [f"## Concentration Time-Series — {ucn_rel}", ""]
        for cell in cells:
            lay = cell.get("layer", 1) - 1
            row = cell.get("row", 1) - 1
            col = cell.get("col", 1) - 1
            ts = uf.get_ts((lay, row, col))
            lines.append(f"### Cell (L{lay+1}, R{row+1}, C{col+1})")
            lines.append(f"{'Time':>12}  {'Concentration':>14}")
            for t_val, c_val in ts:
                lines.append(f"{t_val:12.4g}  {c_val:14.6g}")
            lines.append("")

        summary_text = "\n".join(lines)
        if len(summary_text) > max_chars:
            summary_text = summary_text[:max_chars] + "\n... (truncated)"

        return {
            "ok": True,
            "file": ucn_rel,
            "summary_text": summary_text,
            "metadata": {"cells": len(cells), "ntimes": len(times)},
        }
    except Exception as e:
        return {"ok": False, "file": ucn_rel, "error": f"{type(e).__name__}: {e}"}


# ===========================================================================
# Unstructured head file probes — for MODFLOW-USG (HeadUFile)
# ===========================================================================

def probe_hds_unstructured(ws_root: Path, hds_rel: str) -> Dict[str, Any]:
    """Probe an unstructured HDS file (MODFLOW-USG) for metadata.

    Uses FloPy's HeadUFile instead of HeadFile.
    """
    hds_path = ws_root / hds_rel
    if not hds_path.exists():
        return {"ok": False, "file": hds_rel, "error": "file not found"}
    try:
        import flopy
        huf = flopy.utils.HeadUFile(str(hds_path))
        times = huf.get_times()

        # Sample value range from first and last time
        vmin, vmax = np.inf, -np.inf
        sample_times = [times[0], times[-1]] if len(times) > 1 else times[:1]
        total_nodes = 0
        for t in sample_times:
            data_list = huf.get_data(totim=t)  # list of 1D arrays per layer
            for layer_arr in data_list:
                if layer_arr is None:
                    continue
                total_nodes = max(total_nodes, layer_arr.size)
                valid = layer_arr[np.isfinite(layer_arr) & (np.abs(layer_arr) < 1e15)]
                if valid.size:
                    vmin = min(vmin, float(valid.min()))
                    vmax = max(vmax, float(valid.max()))

        nlay = len(huf.get_data(totim=times[0])) if times else 0

        return {
            "ok": True,
            "file": hds_rel,
            "ntimes": len(times),
            "times_sample": times[:5] + (times[-2:] if len(times) > 5 else []),
            "shape": [nlay, total_nodes],
            "value_range": [vmin, vmax] if np.isfinite(vmin) else None,
            "unstructured": True,
        }
    except Exception as e:
        return {"ok": False, "file": hds_rel, "error": f"{type(e).__name__}: {e}"}


def extract_hds_data_unstructured(
    ws_root: Path,
    hds_rel: str,
    *,
    max_times: int = 5,
    max_chars: int = 40_000,
) -> Dict[str, Any]:
    """Extract head data from unstructured HDS file (MODFLOW-USG).

    HeadUFile returns list of 1D arrays per layer instead of a 3D array.
    """
    hds_path = ws_root / hds_rel
    if not hds_path.exists():
        return {"ok": False, "file": hds_rel, "error": "file not found"}
    try:
        import flopy
        huf = flopy.utils.HeadUFile(str(hds_path))
        times = huf.get_times()
        if not times:
            return {"ok": False, "file": hds_rel, "error": "no timesteps"}

        if len(times) <= max_times:
            sel_times = times
        else:
            indices = np.linspace(0, len(times) - 1, max_times, dtype=int)
            sel_times = [times[i] for i in indices]

        lines: List[str] = [f"## Head Summary (Unstructured) — {hds_rel}",
                            f"Total timesteps: {len(times)}",
                            f"Sampled: {len(sel_times)} timesteps", ""]

        for t in sel_times:
            data_list = huf.get_data(totim=t)
            lines.append(f"### Time = {t}")
            for lay_idx, layer_arr in enumerate(data_list):
                if layer_arr is None:
                    lines.append(f"  Layer {lay_idx + 1}: not saved")
                    continue
                valid = layer_arr[np.isfinite(layer_arr) & (np.abs(layer_arr) < 1e15)]
                if valid.size == 0:
                    lines.append(f"  Layer {lay_idx + 1}: no valid data")
                    continue
                lines.append(
                    f"  Layer {lay_idx + 1} ({layer_arr.size} nodes): "
                    f"min={float(valid.min()):.4g}, "
                    f"mean={float(valid.mean()):.4g}, "
                    f"max={float(valid.max()):.4g}"
                )
            lines.append("")

        summary_text = "\n".join(lines)
        if len(summary_text) > max_chars:
            summary_text = summary_text[:max_chars] + "\n... (truncated)"

        return {
            "ok": True,
            "file": hds_rel,
            "summary_text": summary_text,
            "metadata": {"ntimes": len(times), "sampled": len(sel_times),
                         "unstructured": True},
        }
    except Exception as e:
        return {"ok": False, "file": hds_rel, "error": f"{type(e).__name__}: {e}"}
