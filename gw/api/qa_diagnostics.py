from __future__ import annotations

"""QA/QC diagnostic checks for MODFLOW 6 models.

Each check function returns a markdown-formatted string that provides
senior-modeler-level analysis. These are designed to be called by the
LLM tool loop via the `run_qa_check` tool.

Available checks:
  - mass_balance      — Parse listing file for volumetric budget, compute % discrepancy
  - dry_cells         — Count cells with HDRY/1e30 per layer per time step
  - convergence       — Parse listing file for solver iterations and failures
  - pumping_summary   — Analyze WEL package rates by stress period
  - budget_timeseries — Extract IN/OUT per budget term across timesteps
  - head_gradient     — Compute cell-to-cell gradients, flag extremes
  - property_check    — Check K/SS/SY ranges for unreasonable values
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

NODATA_THRESH = 1e20  # heads above this are dry/inactive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_listing_file(ws_root: Path) -> Optional[Path]:
    """Find the .lst listing file in the workspace."""
    candidates = list(ws_root.glob("*.lst")) + list(ws_root.glob("*.list"))
    if not candidates:
        # Try one level deeper
        candidates = list(ws_root.glob("*/*.lst")) + list(ws_root.glob("*/*.list"))
    if not candidates:
        return None
    # Prefer the largest .lst file (usually the main model listing)
    return max(candidates, key=lambda p: p.stat().st_size)


def _find_binary(ws_root: Path, ext: str) -> Optional[Path]:
    """Find a binary output file (.hds or .cbc) in the workspace."""
    candidates = list(ws_root.glob(f"*.{ext}"))
    if not candidates:
        candidates = list(ws_root.glob(f"*/*.{ext}"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def _open_head_file(ws_root: Path, hds_path: Optional[Path] = None):
    """Open a HeadFile via FloPy. Returns (hf, path) or raises."""
    from gw.mf6.flopy_bridge import flopy_is_available
    if not flopy_is_available():
        raise RuntimeError("FloPy is not available")
    import flopy
    if hds_path is None:
        hds_path = _find_binary(ws_root, "hds")
    if hds_path is None:
        raise FileNotFoundError("No .hds file found in workspace")
    return flopy.utils.HeadFile(str(hds_path)), hds_path


def _open_budget_file(ws_root: Path, cbc_path: Optional[Path] = None):
    """Open a CellBudgetFile via FloPy. Returns (cbf, path, precision) or raises."""
    from gw.mf6.flopy_bridge import flopy_is_available
    if not flopy_is_available():
        raise RuntimeError("FloPy is not available")
    import flopy
    if cbc_path is None:
        cbc_path = _find_binary(ws_root, "cbc")
    if cbc_path is None:
        raise FileNotFoundError("No .cbc file found in workspace")
    for precision in ("double", "single"):
        try:
            cbf = flopy.utils.CellBudgetFile(str(cbc_path), precision=precision)
            return cbf, cbc_path, precision
        except Exception:
            continue
    raise RuntimeError(f"Could not open {cbc_path.name} with either precision")


# ---------------------------------------------------------------------------
# Check: mass_balance
# ---------------------------------------------------------------------------

def check_mass_balance(ws_root: Path, **kwargs) -> str:
    """Parse the listing file for volumetric budget data.

    Reports percent discrepancy per stress period and flags periods
    where the discrepancy exceeds acceptable thresholds.
    """
    lst_path = _find_listing_file(ws_root)
    if lst_path is None:
        return "## Mass Balance Check\n\nNo listing file (.lst) found in the workspace."

    try:
        text = lst_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"## Mass Balance Check\n\nError reading listing file: {e}"

    lines: List[str] = ["## Mass Balance Check", f"**Listing file:** `{lst_path.name}`\n"]

    # Parse volumetric budget sections
    # Look for "VOLUME BUDGET FOR ENTIRE MODEL" blocks
    budget_pattern = re.compile(
        r"VOLUME\s+BUDGET\s+FOR\s+ENTIRE\s+MODEL.*?"
        r"STRESS\s+PERIOD\s+(\d+).*?"
        r"TIME\s+STEP\s+(\d+)",
        re.IGNORECASE | re.DOTALL,
    )

    # Percent discrepancy pattern
    pct_disc_pattern = re.compile(
        r"PERCENT\s+DISCREPANCY\s*=?\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )

    # Total IN/OUT patterns
    total_in_pattern = re.compile(
        r"TOTAL\s+IN\s*=?\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    total_out_pattern = re.compile(
        r"TOTAL\s+OUT\s*=?\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )

    # Split by budget blocks
    budget_blocks = re.split(r"(?=VOLUME\s+BUDGET\s+FOR\s+ENTIRE\s+MODEL)", text, flags=re.IGNORECASE)

    results: List[Dict[str, Any]] = []
    for block in budget_blocks:
        sp_match = re.search(r"STRESS\s+PERIOD\s+(\d+)", block, re.IGNORECASE)
        ts_match = re.search(r"TIME\s+STEP\s+(\d+)", block, re.IGNORECASE)
        if not sp_match:
            continue

        sp = int(sp_match.group(1))
        ts = int(ts_match.group(1)) if ts_match else 0

        pct_matches = pct_disc_pattern.findall(block)
        in_matches = total_in_pattern.findall(block)
        out_matches = total_out_pattern.findall(block)

        pct_disc = float(pct_matches[-1]) if pct_matches else None
        total_in = float(in_matches[-1]) if in_matches else None
        total_out = float(out_matches[-1]) if out_matches else None

        results.append({
            "sp": sp,
            "ts": ts,
            "pct_disc": pct_disc,
            "total_in": total_in,
            "total_out": total_out,
        })

    if not results:
        # Try a simpler approach: look for percent discrepancy anywhere
        all_pct = pct_disc_pattern.findall(text)
        if all_pct:
            lines.append(f"Found {len(all_pct)} percent discrepancy values in listing file.\n")
            values = [float(v) for v in all_pct]
            lines.append(f"- Range: {min(values):.6f}% to {max(values):.6f}%")
            lines.append(f"- Mean absolute: {np.mean(np.abs(values)):.6f}%")
            bad = [v for v in values if abs(v) > 0.5]
            if bad:
                lines.append(f"\n**WARNING:** {len(bad)} values exceed 0.5% discrepancy (max: {max(abs(v) for v in bad):.4f}%)")
            else:
                lines.append("\nAll discrepancy values are within acceptable range (< 0.5%).")
        else:
            lines.append("Could not parse volumetric budget from the listing file.")
            lines.append("The listing file may use a different format or the model may not have run yet.")
        return "\n".join(lines)

    # Report results as a table
    lines.append("| Stress Period | Time Step | Total IN | Total OUT | Net | % Discrepancy | Status |")
    lines.append("|:---:|:---:|---:|---:|---:|---:|:---:|")

    flagged = []
    for r in results:
        pct = r["pct_disc"]
        t_in = r["total_in"]
        t_out = r["total_out"]
        net = (t_in - t_out) if (t_in is not None and t_out is not None) else None

        if pct is not None:
            if abs(pct) > 1.0:
                status = "**POOR**"
                flagged.append(r)
            elif abs(pct) > 0.5:
                status = "MARGINAL"
                flagged.append(r)
            else:
                status = "OK"
        else:
            status = "N/A"

        in_str = f"{t_in:.2f}" if t_in is not None else "N/A"
        out_str = f"{t_out:.2f}" if t_out is not None else "N/A"
        net_str = f"{net:.2f}" if net is not None else "N/A"
        pct_str = f"{pct:.6f}%" if pct is not None else "N/A"
        lines.append(
            f"| {r['sp']} | {r['ts']} | {in_str} | {out_str} | {net_str} | {pct_str} | {status} |"
        )

    # Summary
    lines.append("")
    if flagged:
        lines.append(f"### Issues Found")
        lines.append(f"- **{len(flagged)}** stress periods have mass balance discrepancy > 0.5%")
        worst = max(flagged, key=lambda r: abs(r["pct_disc"] or 0))
        lines.append(f"- Worst: SP {worst['sp']} with {worst['pct_disc']:.6f}% discrepancy")
        lines.append("")
        lines.append("**Recommendations:**")
        lines.append("- Check solver settings (IMS) — tighter inner/outer convergence criteria may help")
        lines.append("- Verify boundary condition rates are not too large for the time step size")
        lines.append("- Check for cells that are wetting/drying frequently")
    else:
        pct_values = [r["pct_disc"] for r in results if r["pct_disc"] is not None]
        if pct_values:
            lines.append(f"### Mass Balance Summary")
            lines.append(f"- All {len(results)} budget records have acceptable mass balance (< 0.5%)")
            lines.append(f"- Max absolute discrepancy: {max(abs(v) for v in pct_values):.6f}%")
            lines.append(f"- Mean absolute discrepancy: {np.mean(np.abs(pct_values)):.6f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check: dry_cells
# ---------------------------------------------------------------------------

def check_dry_cells(ws_root: Path, **kwargs) -> str:
    """Count dry/inactive cells per layer per timestep from HDS file."""
    try:
        hf, hds_path = _open_head_file(ws_root)
    except (FileNotFoundError, RuntimeError) as e:
        return f"## Dry Cell Check\n\n{e}"

    times = hf.get_times()
    if not times:
        return "## Dry Cell Check\n\nNo timesteps found in HDS file."

    lines: List[str] = [
        "## Dry Cell Check",
        f"**HDS file:** `{hds_path.name}`",
        f"**Timesteps:** {len(times)}\n",
    ]

    # Sample timesteps: first, middle, last (plus a few more if many)
    if len(times) <= 5:
        sample_indices = list(range(len(times)))
    else:
        sample_indices = [0, len(times) // 4, len(times) // 2, 3 * len(times) // 4, len(times) - 1]
        sample_indices = sorted(set(sample_indices))

    first_data = hf.get_data(totim=times[0])
    nlay = first_data.shape[0] if len(first_data.shape) == 3 else 1
    total_cells_per_layer = first_data.shape[1] * first_data.shape[2] if len(first_data.shape) == 3 else first_data.size

    lines.append(f"**Grid:** {nlay} layers, {total_cells_per_layer} cells per layer\n")

    # Track dry cell counts over time for trend analysis
    dry_history: Dict[int, List[int]] = {k: [] for k in range(nlay)}

    lines.append("| Time | " + " | ".join(f"Layer {k+1}" for k in range(min(nlay, 20))) + " | Total Dry |")
    lines.append("|---:|" + "---:|" * min(nlay, 20) + "---:|")

    for idx in sample_indices:
        t = times[idx]
        data = hf.get_data(totim=t)

        row_parts = [f"| {t:.4g}"]
        total_dry = 0

        if len(data.shape) == 3:
            for k in range(min(nlay, 20)):
                layer = data[k]
                dry_count = int(np.sum(
                    (layer >= NODATA_THRESH) | (~np.isfinite(layer))
                ))
                total_dry += dry_count
                pct = (dry_count / total_cells_per_layer * 100) if total_cells_per_layer > 0 else 0
                dry_history[k].append(dry_count)
                if pct > 10:
                    row_parts.append(f" **{dry_count}** ({pct:.1f}%)")
                else:
                    row_parts.append(f" {dry_count} ({pct:.1f}%)")
        else:
            dry_count = int(np.sum(
                (data >= NODATA_THRESH) | (~np.isfinite(data))
            ))
            total_dry = dry_count
            pct = (dry_count / data.size * 100) if data.size > 0 else 0
            row_parts.append(f" {dry_count} ({pct:.1f}%)")
            dry_history[0].append(dry_count)

        row_parts.append(f" {total_dry}")
        lines.append(" |".join(row_parts) + " |")

    # Trend analysis
    lines.append("")
    lines.append("### Dry Cell Analysis")

    issues_found = False
    for k in range(nlay):
        counts = dry_history.get(k, [])
        if not counts:
            continue
        if max(counts) == 0:
            continue

        issues_found = True
        first_dry = counts[0]
        last_dry = counts[-1]
        max_dry = max(counts)
        pct_max = (max_dry / total_cells_per_layer * 100) if total_cells_per_layer > 0 else 0

        lines.append(f"\n**Layer {k+1}:**")
        lines.append(f"- Max dry cells: {max_dry} ({pct_max:.1f}% of layer)")
        if len(counts) > 1 and last_dry > first_dry:
            increase = last_dry - first_dry
            lines.append(f"- Dry cells INCREASING over time: +{increase} cells (from {first_dry} to {last_dry})")
            lines.append("  - This may indicate insufficient recharge, over-pumping, or inappropriate boundary conditions")
        elif len(counts) > 1 and last_dry < first_dry:
            lines.append(f"- Dry cells decreasing over time: from {first_dry} to {last_dry}")
        if pct_max > 25:
            lines.append(f"  - **WARNING:** >25% of layer is dry — review layer thickness and permeability")

    if not issues_found:
        lines.append("No dry cells detected in any layer at any sampled timestep.")
        lines.append("This is a good sign for model stability.")

    # Spatial clustering analysis for the final timestep
    if nlay > 0 and len(data.shape) == 3:
        lines.append("\n### Spatial Distribution (Final Timestep)")
        final_data = hf.get_data(totim=times[-1])
        for k in range(min(nlay, 10)):
            layer = final_data[k]
            dry_mask = (layer >= NODATA_THRESH) | (~np.isfinite(layer))
            dry_count = int(np.sum(dry_mask))
            if dry_count == 0:
                continue

            # Identify spatial extent of dry cells
            dry_rows, dry_cols = np.where(dry_mask)
            if len(dry_rows) > 0:
                lines.append(
                    f"- Layer {k+1}: {dry_count} dry cells, "
                    f"row range [{dry_rows.min()}-{dry_rows.max()}], "
                    f"col range [{dry_cols.min()}-{dry_cols.max()}]"
                )
                # Check if clustered or scattered
                row_range = dry_rows.max() - dry_rows.min() + 1
                col_range = dry_cols.max() - dry_cols.min() + 1
                bbox_area = row_range * col_range
                density = dry_count / bbox_area if bbox_area > 0 else 0
                if density > 0.5:
                    lines.append(f"  - Clustered pattern (density={density:.2f} in bounding box)")
                else:
                    lines.append(f"  - Scattered pattern (density={density:.2f} in bounding box)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check: convergence
# ---------------------------------------------------------------------------

def check_convergence(ws_root: Path, **kwargs) -> str:
    """Parse listing file for solver convergence information."""
    lst_path = _find_listing_file(ws_root)
    if lst_path is None:
        return "## Convergence Check\n\nNo listing file (.lst) found in the workspace."

    try:
        text = lst_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"## Convergence Check\n\nError reading listing file: {e}"

    lines: List[str] = ["## Convergence Check", f"**Listing file:** `{lst_path.name}`\n"]

    # Look for solver convergence info
    # Pattern 1: "OUTER ITERATION SUMMARY" blocks
    outer_iter_pattern = re.compile(
        r"(?:OUTER|NONLINEAR)\s+ITERATION\s+SUMMARY.*?"
        r"TOTAL\s+ITERATIONS\s*[:=]\s*(\d+)",
        re.IGNORECASE | re.DOTALL,
    )

    # Pattern 2: "FAILED TO CONVERGE" or "CONVERGENCE FAILURE"
    failure_pattern = re.compile(
        r"(?:FAILED?\s+TO\s+CONVERGE|CONVERGENCE\s+FAILURE|"
        r"DID\s+NOT\s+CONVERGE|NO\s+CONVERGENCE)",
        re.IGNORECASE,
    )

    # Pattern 3: Generic iteration counts
    iter_pattern = re.compile(
        r"(?:TOTAL\s+(?:OUTER\s+)?ITERATIONS|NUMBER\s+OF\s+ITERATIONS)\s*[:=]\s*(\d+)",
        re.IGNORECASE,
    )

    # Pattern 4: "INNER ITERATIONS" for solver performance
    inner_iter_pattern = re.compile(
        r"INNER\s+ITERATIONS\s*[:=]\s*(\d+)",
        re.IGNORECASE,
    )

    # Check for convergence failures
    failures = failure_pattern.findall(text)
    if failures:
        lines.append(f"### **CONVERGENCE FAILURES DETECTED: {len(failures)}**\n")
        lines.append("The model has convergence failures. This means the solver could not")
        lines.append("find a solution within the allowed iterations for some time steps.\n")

        # Find which stress periods / time steps failed
        failure_locations = []
        for m in failure_pattern.finditer(text):
            # Look backward for stress period/time step info
            context_start = max(0, m.start() - 500)
            context = text[context_start:m.start()]
            sp_m = re.search(r"STRESS\s+PERIOD\s+(\d+)", context, re.IGNORECASE)
            ts_m = re.search(r"TIME\s+STEP\s+(\d+)", context, re.IGNORECASE)
            if sp_m:
                sp = int(sp_m.group(1))
                ts = int(ts_m.group(1)) if ts_m else 0
                failure_locations.append((sp, ts))

        if failure_locations:
            lines.append("**Failed at:**")
            for sp, ts in failure_locations[:20]:  # limit to first 20
                lines.append(f"- Stress Period {sp}, Time Step {ts}")
            if len(failure_locations) > 20:
                lines.append(f"- ... and {len(failure_locations) - 20} more")
    else:
        lines.append("No convergence failures detected.\n")

    # Iteration statistics
    all_iters = [int(m) for m in iter_pattern.findall(text)]
    inner_iters = [int(m) for m in inner_iter_pattern.findall(text)]

    if all_iters:
        lines.append("\n### Outer Iteration Statistics")
        lines.append(f"- Number of records: {len(all_iters)}")
        lines.append(f"- Min iterations: {min(all_iters)}")
        lines.append(f"- Max iterations: {max(all_iters)}")
        lines.append(f"- Mean iterations: {np.mean(all_iters):.1f}")
        lines.append(f"- Median iterations: {np.median(all_iters):.1f}")

        # Flag time steps with high iteration counts
        high_iter = [i for i in all_iters if i > 100]
        if high_iter:
            lines.append(f"\n**WARNING:** {len(high_iter)} time steps required >100 outer iterations")
            lines.append("This may indicate difficulty converging. Consider:")
            lines.append("- Reducing time step size")
            lines.append("- Increasing solver tolerance")
            lines.append("- Checking for unrealistic boundary conditions")

    if inner_iters:
        lines.append("\n### Inner Iteration Statistics")
        lines.append(f"- Total inner iterations across all time steps: {sum(inner_iters)}")
        lines.append(f"- Max inner iterations in a single solve: {max(inner_iters)}")

    if not all_iters and not failures:
        lines.append("Could not parse iteration statistics from the listing file.")
        lines.append("The model may not have run yet, or the listing format may differ.")

    # Recommendations
    if failures or (all_iters and max(all_iters) > 50):
        lines.append("\n### Recommendations")
        lines.append("- Review IMS (Iterative Model Solution) settings:")
        lines.append("  - OUTER_MAXIMUM: increase if solver needs more iterations")
        lines.append("  - INNER_MAXIMUM: increase for complex flow patterns")
        lines.append("  - OUTER_DVCLOSE: may need to be loosened slightly")
        lines.append("  - UNDER_RELAXATION: try SIMPLE or DBD for Newton solver")
        lines.append("- Check for abrupt changes in boundary conditions between stress periods")
        lines.append("- Look for thin layers or extreme K contrasts that challenge the solver")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check: pumping_summary
# ---------------------------------------------------------------------------

def check_pumping_summary(ws_root: Path, **kwargs) -> str:
    """Analyze WEL package pumping data by stress period."""
    # Look for WEL package file
    wel_files = list(ws_root.glob("*.wel")) + list(ws_root.glob("*/*.wel"))
    if not wel_files:
        # Try via FloPy simulation
        return _pumping_summary_via_flopy(ws_root)

    wel_path = wel_files[0]
    lines: List[str] = ["## Pumping Summary", f"**WEL file:** `{wel_path.name}`\n"]

    try:
        text = wel_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"## Pumping Summary\n\nError reading WEL file: {e}"

    # Parse PERIOD blocks
    # MF6 WEL format: BEGIN PERIOD n ... END PERIOD
    period_pattern = re.compile(
        r"BEGIN\s+PERIOD\s+(\d+)(.*?)END\s+PERIOD",
        re.IGNORECASE | re.DOTALL,
    )

    periods: Dict[int, List[float]] = {}
    for m in period_pattern.finditer(text):
        sp = int(m.group(1))
        block = m.group(2)

        rates = []
        for line in block.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            parts = line.split()
            # Find the rate value — typically the last numeric column
            for part in reversed(parts):
                try:
                    val = float(part)
                    rates.append(val)
                    break
                except ValueError:
                    continue
        if rates:
            periods[sp] = rates

    if not periods:
        lines.append("Could not parse WEL PERIOD blocks. Trying FloPy approach...")
        return _pumping_summary_via_flopy(ws_root)

    # Summarize by stress period
    lines.append("| SP | Wells | Total Q (extraction) | Total Q (injection) | Net Q | Max Single Q |")
    lines.append("|:---:|:---:|---:|---:|---:|---:|")

    sp_totals: List[Dict[str, float]] = []
    for sp in sorted(periods.keys()):
        rates = periods[sp]
        extraction = sum(r for r in rates if r < 0)
        injection = sum(r for r in rates if r > 0)
        net = sum(rates)
        max_single = min(rates) if rates else 0  # most negative = most extraction

        sp_totals.append({
            "sp": sp,
            "n_wells": len(rates),
            "extraction": extraction,
            "injection": injection,
            "net": net,
            "max_single": max_single,
        })

        lines.append(
            f"| {sp} | {len(rates)} | {extraction:.4f} | {injection:.4f} "
            f"| {net:.4f} | {max_single:.4f} |"
        )

    # Anomaly detection
    lines.append("\n### Anomaly Analysis")

    if len(sp_totals) > 1:
        # Check for >2x jumps between consecutive periods
        anomalies = []
        for i in range(1, len(sp_totals)):
            prev = abs(sp_totals[i-1]["extraction"])
            curr = abs(sp_totals[i]["extraction"])
            if prev > 0 and curr > 0:
                ratio = curr / prev if prev != 0 else 999
                if ratio > 2 or ratio < 0.5:
                    anomalies.append((sp_totals[i]["sp"], ratio, prev, curr))

        if anomalies:
            lines.append(f"\n**Anomalous rate changes detected ({len(anomalies)}):**")
            for sp, ratio, prev, curr in anomalies:
                direction = "increase" if ratio > 1 else "decrease"
                lines.append(
                    f"- SP {sp}: {ratio:.1f}x {direction} in total extraction "
                    f"({prev:.4f} → {curr:.4f})"
                )
            lines.append("\nLarge jumps may indicate:")
            lines.append("- Data entry errors in pumping rates")
            lines.append("- Seasonal changes that need smoothing")
            lines.append("- Missing stress periods in the pumping schedule")
        else:
            lines.append("No anomalous rate changes detected between consecutive stress periods.")

        # Cumulative volume analysis
        lines.append("\n### Cumulative Pumping")
        cumulative = 0.0
        for st in sp_totals:
            cumulative += abs(st["extraction"])
            # We don't have stress period lengths from WEL alone
        lines.append(f"- Total extraction rate (sum of all SP rates): {cumulative:.4f}")
        lines.append("  *(Note: multiply by stress period lengths for actual volume)*")
    else:
        lines.append("Only one stress period — no temporal comparison available.")

    return "\n".join(lines)


def _pumping_summary_via_flopy(ws_root: Path) -> str:
    """Try to analyze pumping via FloPy simulation loading."""
    lines = ["## Pumping Summary\n"]
    try:
        from gw.mf6.flopy_bridge import flopy_is_available
        if not flopy_is_available():
            return "## Pumping Summary\n\nNo WEL file found and FloPy is not available."

        import flopy

        # Try to load the simulation
        nam_files = list(ws_root.glob("mfsim.nam"))
        if not nam_files:
            return "## Pumping Summary\n\nNo WEL file or mfsim.nam found in workspace."

        sim = flopy.mf6.MFSimulation.load(
            sim_ws=str(ws_root),
            verbosity_level=0,
        )
        for model_name in sim.model_names:
            model = sim.get_model(model_name)
            if hasattr(model, "wel") and model.wel is not None:
                wel = model.wel
                lines.append(f"Found WEL package in model `{model_name}`")
                stress_data = wel.stress_period_data.get_data()
                if stress_data:
                    for sp, data in stress_data.items():
                        if data is not None and len(data) > 0:
                            rates = data["q"] if "q" in data.dtype.names else []
                            if len(rates) > 0:
                                lines.append(f"\n**Stress Period {sp}:** {len(rates)} wells")
                                extraction = float(np.sum(rates[rates < 0]))
                                injection = float(np.sum(rates[rates > 0]))
                                lines.append(f"- Extraction: {extraction:.4f}")
                                lines.append(f"- Injection: {injection:.4f}")
                                lines.append(f"- Net: {float(np.sum(rates)):.4f}")
                return "\n".join(lines)

        lines.append("No WEL package found in any model.")
    except Exception as e:
        lines.append(f"FloPy analysis failed: {type(e).__name__}: {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check: budget_timeseries
# ---------------------------------------------------------------------------

def check_budget_timeseries(ws_root: Path, **kwargs) -> str:
    """Extract IN/OUT per budget term across all timesteps from CBC file."""
    try:
        cbf, cbc_path, precision = _open_budget_file(ws_root)
    except (FileNotFoundError, RuntimeError) as e:
        return f"## Budget Time Series\n\n{e}"

    times = cbf.get_times()
    if not times:
        return "## Budget Time Series\n\nNo timesteps in CBC file."

    record_names_raw = cbf.get_unique_record_names()
    record_names = []
    for n in record_names_raw:
        if isinstance(n, bytes):
            record_names.append(n.decode().strip())
        else:
            record_names.append(str(n).strip())

    lines: List[str] = [
        "## Budget Time Series",
        f"**CBC file:** `{cbc_path.name}` (precision: {precision})",
        f"**Timesteps:** {len(times)}",
        f"**Budget terms:** {', '.join(record_names)}\n",
    ]

    # Sample timesteps for time series
    if len(times) <= 10:
        sample_indices = list(range(len(times)))
    else:
        step = max(1, len(times) // 10)
        sample_indices = list(range(0, len(times), step))
        if sample_indices[-1] != len(times) - 1:
            sample_indices.append(len(times) - 1)

    # Build time series data for each record
    ts_data: Dict[str, Dict[str, List[float]]] = {}  # {term: {"in": [...], "out": [...], "times": [...]}}

    for rname in record_names[:12]:  # limit to 12 budget terms
        ts_data[rname] = {"in": [], "out": [], "times": []}
        rname_b = rname.encode() if isinstance(rname, str) else rname

        for idx in sample_indices:
            t = times[idx]
            try:
                data = cbf.get_data(totim=t, text=rname_b)
                if not data:
                    continue

                total_in = 0.0
                total_out = 0.0
                for arr in data:
                    if hasattr(arr, 'dtype') and arr.dtype.names and 'q' in arr.dtype.names:
                        q = arr['q']
                        finite = q[np.isfinite(q)]
                        total_in += float(np.sum(finite[finite > 0]))
                        total_out += float(np.sum(finite[finite < 0]))
                    else:
                        arr_np = np.asarray(arr)
                        finite = arr_np[np.isfinite(arr_np) & (np.abs(arr_np) < 1e29)]
                        total_in += float(np.sum(finite[finite > 0]))
                        total_out += float(np.sum(finite[finite < 0]))

                ts_data[rname]["in"].append(total_in)
                ts_data[rname]["out"].append(total_out)
                ts_data[rname]["times"].append(t)
            except Exception:
                continue

    # Format as tables per term
    for rname, data in ts_data.items():
        if not data["times"]:
            continue

        lines.append(f"\n### {rname}")
        lines.append("| Time | IN | OUT | Net |")
        lines.append("|---:|---:|---:|---:|")

        for i, t in enumerate(data["times"]):
            flow_in = data["in"][i]
            flow_out = data["out"][i]
            net = flow_in + flow_out
            lines.append(f"| {t:.4g} | {flow_in:.4f} | {flow_out:.4f} | {net:.4f} |")

        # Summary stats
        if len(data["in"]) > 1:
            in_arr = np.array(data["in"])
            out_arr = np.array(data["out"])
            net_arr = in_arr + out_arr

            lines.append(f"\n**Summary:** IN range [{in_arr.min():.4f}, {in_arr.max():.4f}], "
                        f"OUT range [{out_arr.min():.4f}, {out_arr.max():.4f}], "
                        f"Net range [{net_arr.min():.4f}, {net_arr.max():.4f}]")

    # Overall budget check
    lines.append("\n### Overall Budget Balance (Final Timestep)")
    final_t = times[-1]
    total_in_all = 0.0
    total_out_all = 0.0
    for rname in record_names:
        rname_b = rname.encode() if isinstance(rname, str) else rname
        try:
            data = cbf.get_data(totim=final_t, text=rname_b)
            for arr in (data or []):
                if hasattr(arr, 'dtype') and arr.dtype.names and 'q' in arr.dtype.names:
                    q = arr['q']
                    finite = q[np.isfinite(q)]
                    total_in_all += float(np.sum(finite[finite > 0]))
                    total_out_all += float(np.sum(finite[finite < 0]))
                else:
                    arr_np = np.asarray(arr)
                    finite = arr_np[np.isfinite(arr_np) & (np.abs(arr_np) < 1e29)]
                    total_in_all += float(np.sum(finite[finite > 0]))
                    total_out_all += float(np.sum(finite[finite < 0]))
        except Exception:
            continue

    net_all = total_in_all + total_out_all
    pct = abs(net_all) / max(total_in_all, abs(total_out_all), 1e-30) * 100

    lines.append(f"- Total IN: {total_in_all:.4f}")
    lines.append(f"- Total OUT: {total_out_all:.4f}")
    lines.append(f"- Net: {net_all:.4f}")
    lines.append(f"- Percent discrepancy: {pct:.6f}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check: head_gradient
# ---------------------------------------------------------------------------

def check_head_gradient(ws_root: Path, **kwargs) -> str:
    """Compute cell-to-cell head gradients and flag extreme values."""
    try:
        hf, hds_path = _open_head_file(ws_root)
    except (FileNotFoundError, RuntimeError) as e:
        return f"## Head Gradient Check\n\n{e}"

    times = hf.get_times()
    if not times:
        return "## Head Gradient Check\n\nNo timesteps in HDS file."

    # Use the final timestep
    target_time = kwargs.get("time", times[-1])
    data = hf.get_data(totim=target_time)

    lines: List[str] = [
        "## Head Gradient Check",
        f"**HDS file:** `{hds_path.name}`",
        f"**Timestep:** {target_time} (final)\n",
    ]

    if len(data.shape) != 3:
        lines.append("HDS data is not 3D — cannot compute spatial gradients.")
        return "\n".join(lines)

    nlay, nrow, ncol = data.shape
    lines.append(f"**Grid:** {nlay} layers, {nrow} rows, {ncol} columns\n")

    lines.append("| Layer | Median Gradient | Max Gradient | Location (row,col) | Cells >10x Median | Status |")
    lines.append("|:---:|---:|---:|:---:|:---:|:---:|")

    flagged_layers = []

    for k in range(nlay):
        layer = data[k].copy()

        # Mask dry/inactive cells
        mask = (layer >= NODATA_THRESH) | (~np.isfinite(layer))
        layer[mask] = np.nan

        # Compute gradients in row and column directions
        if nrow < 2 or ncol < 2:
            lines.append(f"| {k+1} | N/A | N/A | N/A | N/A | Too small |")
            continue

        grad_row = np.abs(np.diff(layer, axis=0))  # (nrow-1, ncol)
        grad_col = np.abs(np.diff(layer, axis=1))  # (nrow, ncol-1)

        # Combine: max gradient at each cell
        # We need a common shape, so pad/combine carefully
        all_grads_row = grad_row[np.isfinite(grad_row)]
        all_grads_col = grad_col[np.isfinite(grad_col)]
        all_grads = np.concatenate([all_grads_row, all_grads_col])

        if len(all_grads) == 0:
            lines.append(f"| {k+1} | N/A | N/A | N/A | N/A | All dry/inactive |")
            continue

        median_grad = float(np.median(all_grads))
        max_grad = float(np.max(all_grads))

        # Find location of max gradient
        max_row_grad = float(np.nanmax(grad_row)) if grad_row.size > 0 else 0
        max_col_grad = float(np.nanmax(grad_col)) if grad_col.size > 0 else 0

        if max_row_grad >= max_col_grad and grad_row.size > 0:
            flat_idx = np.nanargmax(grad_row)
            r, c = np.unravel_index(flat_idx, grad_row.shape)
        elif grad_col.size > 0:
            flat_idx = np.nanargmax(grad_col)
            r, c = np.unravel_index(flat_idx, grad_col.shape)
        else:
            r, c = 0, 0

        # Count cells with extreme gradients
        threshold = 10 * median_grad if median_grad > 0 else max_grad * 0.9
        extreme_count = int(np.sum(all_grads > threshold)) if threshold > 0 else 0

        if extreme_count > 0 and median_grad > 0:
            ratio = max_grad / median_grad
            if ratio > 100:
                status = "**EXTREME**"
                flagged_layers.append(k + 1)
            elif ratio > 10:
                status = "HIGH"
                flagged_layers.append(k + 1)
            else:
                status = "OK"
        else:
            status = "OK"

        lines.append(
            f"| {k+1} | {median_grad:.6f} | {max_grad:.6f} | ({r},{c}) "
            f"| {extreme_count} | {status} |"
        )

    # Summary
    lines.append("")
    if flagged_layers:
        lines.append("### Issues Found")
        lines.append(f"Layers with extreme gradients: {', '.join(str(l) for l in flagged_layers)}")
        lines.append("\nExtreme head gradients may indicate:")
        lines.append("- Grid resolution too coarse near pumping wells or boundaries")
        lines.append("- Unrealistic hydraulic conductivity contrasts")
        lines.append("- Boundary condition issues (e.g., specified head adjacent to low-K zone)")
        lines.append("- Numerical artifacts from dry cell rewetting")
        lines.append("\n**Recommendation:** Consider local grid refinement in areas of high gradient")
    else:
        lines.append("### Summary")
        lines.append("No extreme head gradients detected. Gradient distribution appears reasonable.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check: property_check
# ---------------------------------------------------------------------------

def check_property_check(ws_root: Path, **kwargs) -> str:
    """Check hydraulic property values for unreasonable ranges."""
    lines: List[str] = ["## Property Check\n"]

    try:
        from gw.mf6.flopy_bridge import flopy_is_available
        if not flopy_is_available():
            return "## Property Check\n\nFloPy is not available for property analysis."

        import flopy

        nam_files = list(ws_root.glob("mfsim.nam"))
        if not nam_files:
            # Try to find NPF file directly
            return _property_check_from_files(ws_root)

        sim = flopy.mf6.MFSimulation.load(
            sim_ws=str(ws_root),
            verbosity_level=0,
        )

        for model_name in sim.model_names:
            model = sim.get_model(model_name)
            lines.append(f"### Model: `{model_name}`\n")

            # NPF package — hydraulic conductivity
            if hasattr(model, "npf") and model.npf is not None:
                npf = model.npf
                lines.append("#### Hydraulic Conductivity (NPF)")

                k_data = None
                try:
                    k_data = npf.k.array
                except Exception:
                    pass

                if k_data is not None:
                    k_arr = np.asarray(k_data, dtype=float)
                    valid = k_arr[np.isfinite(k_arr) & (k_arr > 0)]

                    if valid.size > 0:
                        lines.append(f"- Shape: {list(k_arr.shape)}")
                        lines.append(f"- Range: {float(np.min(valid)):.2e} to {float(np.max(valid)):.2e} m/d")
                        lines.append(f"- Mean: {float(np.mean(valid)):.2e} m/d")
                        lines.append(f"- Std: {float(np.std(valid)):.2e} m/d")

                        # Flag unreasonable values
                        too_low = int(np.sum(valid < 1e-12))
                        too_high = int(np.sum(valid > 1e4))
                        if too_low > 0:
                            lines.append(f"- **WARNING:** {too_low} cells with K < 1e-12 m/d (essentially impermeable)")
                        if too_high > 0:
                            lines.append(f"- **WARNING:** {too_high} cells with K > 10,000 m/d (unusually high)")

                        # Per-layer breakdown
                        if len(k_arr.shape) == 3:
                            nlay = k_arr.shape[0]
                            lines.append("\n**Per-layer K statistics:**")
                            lines.append("| Layer | Min K | Max K | Mean K | K Range Ratio |")
                            lines.append("|:---:|---:|---:|---:|---:|")

                            prev_mean = None
                            for k in range(nlay):
                                layer = k_arr[k]
                                lv = layer[np.isfinite(layer) & (layer > 0)]
                                if lv.size == 0:
                                    lines.append(f"| {k+1} | N/A | N/A | N/A | N/A |")
                                    continue
                                k_min = float(np.min(lv))
                                k_max = float(np.max(lv))
                                k_mean = float(np.mean(lv))
                                ratio = k_max / k_min if k_min > 0 else float("inf")
                                lines.append(f"| {k+1} | {k_min:.2e} | {k_max:.2e} | {k_mean:.2e} | {ratio:.1f} |")

                                # Check for layer inversions
                                if prev_mean is not None and k_mean > 100 * prev_mean:
                                    lines.append(f"  **Note:** Layer {k+1} K is >100x Layer {k} — potential layer inversion")
                                prev_mean = k_mean

                # K33 (vertical)
                k33_data = None
                try:
                    k33_data = npf.k33.array
                except Exception:
                    pass

                if k33_data is not None:
                    lines.append("\n#### Vertical Conductivity (K33)")
                    k33_arr = np.asarray(k33_data, dtype=float)
                    valid = k33_arr[np.isfinite(k33_arr) & (k33_arr > 0)]
                    if valid.size > 0:
                        lines.append(f"- Range: {float(np.min(valid)):.2e} to {float(np.max(valid)):.2e} m/d")
                        lines.append(f"- Mean: {float(np.mean(valid)):.2e} m/d")

                        # Check Kv/Kh ratio
                        if k_data is not None:
                            k_arr_flat = np.asarray(k_data, dtype=float).ravel()
                            k33_flat = k33_arr.ravel()
                            both_valid = np.isfinite(k_arr_flat) & np.isfinite(k33_flat) & (k_arr_flat > 0) & (k33_flat > 0)
                            if np.sum(both_valid) > 0:
                                ratios = k33_flat[both_valid] / k_arr_flat[both_valid]
                                lines.append(f"- Kv/Kh ratio range: {float(np.min(ratios)):.4f} to {float(np.max(ratios)):.4f}")
                                lines.append(f"- Kv/Kh median: {float(np.median(ratios)):.4f}")
                                if np.max(ratios) > 1.0:
                                    lines.append("  **Note:** Some cells have Kv > Kh, which is unusual for most geologic settings")

            # STO package — storage
            if hasattr(model, "sto") and model.sto is not None:
                sto = model.sto
                lines.append("\n#### Storage Properties (STO)")

                ss_data = None
                try:
                    ss_data = sto.ss.array
                except Exception:
                    pass

                if ss_data is not None:
                    ss_arr = np.asarray(ss_data, dtype=float)
                    valid = ss_arr[np.isfinite(ss_arr) & (ss_arr > 0)]
                    if valid.size > 0:
                        lines.append(f"- Specific Storage (Ss): {float(np.min(valid)):.2e} to {float(np.max(valid)):.2e} 1/m")
                        if np.min(valid) < 1e-7:
                            lines.append("  **Note:** Very low Ss values — ensure these are intentional")
                        if np.max(valid) > 1e-2:
                            lines.append("  **WARNING:** Ss > 0.01 is unusually high for most geologic materials")

                sy_data = None
                try:
                    sy_data = sto.sy.array
                except Exception:
                    pass

                if sy_data is not None:
                    sy_arr = np.asarray(sy_data, dtype=float)
                    valid = sy_arr[np.isfinite(sy_arr)]
                    if valid.size > 0:
                        lines.append(f"- Specific Yield (Sy): {float(np.min(valid)):.4f} to {float(np.max(valid)):.4f}")
                        if np.max(valid) > 0.45:
                            lines.append("  **WARNING:** Sy > 0.45 exceeds physical limits for most materials")
                        if np.min(valid) < 0.01:
                            lines.append("  **Note:** Very low Sy values — typical only for dense rock")

            # IC package — initial conditions
            if hasattr(model, "ic") and model.ic is not None:
                lines.append("\n#### Initial Conditions (IC)")
                try:
                    strt_data = model.ic.strt.array
                    strt_arr = np.asarray(strt_data, dtype=float)
                    valid = strt_arr[np.isfinite(strt_arr) & (np.abs(strt_arr) < 1e20)]
                    if valid.size > 0:
                        lines.append(f"- Starting head range: {float(np.min(valid)):.4f} to {float(np.max(valid)):.4f}")
                        head_range = float(np.max(valid) - np.min(valid))
                        lines.append(f"- Head range: {head_range:.4f}")
                        if head_range > 1000:
                            lines.append(f"  **WARNING:** Starting head range > 1000m — verify this is correct for your model")
                except Exception:
                    pass

        return "\n".join(lines)

    except Exception as e:
        return f"## Property Check\n\nFailed to load model via FloPy: {type(e).__name__}: {e}\n\nTrying file-based analysis...\n\n" + _property_check_from_files(ws_root)


def _property_check_from_files(ws_root: Path) -> str:
    """Fallback: check properties by parsing NPF text file directly."""
    lines: List[str] = ["### File-based Property Analysis\n"]

    npf_files = list(ws_root.glob("*.npf")) + list(ws_root.glob("*/*.npf"))
    if not npf_files:
        lines.append("No NPF package file found.")
        return "\n".join(lines)

    npf_path = npf_files[0]
    lines.append(f"**NPF file:** `{npf_path.name}`")

    try:
        text = npf_path.read_text(encoding="utf-8", errors="replace")
        # Check for CONSTANT values
        const_pattern = re.compile(r"CONSTANT\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", re.IGNORECASE)
        constants = const_pattern.findall(text)
        if constants:
            lines.append(f"\nConstant values found: {', '.join(constants)}")
            for c in constants:
                val = float(c)
                if val < 1e-12 or val > 1e4:
                    lines.append(f"  **WARNING:** Value {val:.2e} may be unreasonable for hydraulic conductivity")
    except Exception as e:
        lines.append(f"Error reading NPF file: {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

AVAILABLE_CHECKS = {
    "mass_balance": check_mass_balance,
    "dry_cells": check_dry_cells,
    "convergence": check_convergence,
    "pumping_summary": check_pumping_summary,
    "budget_timeseries": check_budget_timeseries,
    "head_gradient": check_head_gradient,
    "property_check": check_property_check,
}


def run_qa_check(ws_root: str, check_name: str, **kwargs) -> str:
    """Run a specific QA/QC diagnostic check.

    Parameters
    ----------
    ws_root : str
        Path to the workspace root directory.
    check_name : str
        Name of the check to run. One of: mass_balance, dry_cells,
        convergence, pumping_summary, budget_timeseries, head_gradient,
        property_check.
    **kwargs
        Additional parameters passed to the check function.

    Returns
    -------
    str
        Markdown-formatted diagnostic report.
    """
    ws = Path(ws_root)
    if not ws.exists():
        return f"Workspace not found: {ws_root}"

    check_fn = AVAILABLE_CHECKS.get(check_name)
    if check_fn is None:
        available = ", ".join(sorted(AVAILABLE_CHECKS.keys()))
        return f"Unknown check: `{check_name}`. Available checks: {available}"

    try:
        return check_fn(ws, **kwargs)
    except Exception as e:
        logger.exception("QA check %s failed", check_name)
        return f"## {check_name}\n\nCheck failed with error: {type(e).__name__}: {e}"
