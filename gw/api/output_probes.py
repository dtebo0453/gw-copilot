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
