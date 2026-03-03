from __future__ import annotations

"""QA/QC diagnostic checks for MODFLOW models (MF6, MF2005, NWT).

Each check function returns a markdown-formatted string that provides
senior-modeler-level analysis. These are designed to be called by the
LLM tool loop via the `run_qa_check` tool.

Available checks:
  - mass_balance            — Parse listing file for volumetric budget, compute % discrepancy
  - dry_cells               — Count cells with HDRY/1e30 per layer per time step
  - convergence             — Parse listing file for solver iterations, failures, and warnings
  - pumping_summary         — Analyze WEL package rates by stress period
  - budget_timeseries       — Extract IN/OUT per budget term across timesteps
  - head_gradient           — Compute cell-to-cell gradients, flag extremes
  - property_check          — Check K/SS/SY ranges for unreasonable values
  - observation_comparison  — Analyze observation outputs and calibration metrics
  - listing_budget_detail   — Per-package IN/OUT budget tables + solver warnings
  - property_zones          — Spatial K zone analysis with contrast detection
  - advanced_packages       — Summarize SFR, LAK, MAW, UZF, CSUB, EVT packages
  - save_snapshot           — Save lightweight post-run snapshot for comparison
  - compare_runs            — Compare two run snapshots (heads, budget, convergence)
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

    # --- Solver warnings (Phase 2B enhancement) ---
    solver_warning_patterns = {
        "UNDER_RELAXATION": re.compile(r"UNDER.?RELAX", re.IGNORECASE),
        "BACKTRACKING": re.compile(r"BACKTRACK", re.IGNORECASE),
        "DAMPING": re.compile(r"DAMP(?:ING|ED)", re.IGNORECASE),
        "TIMESTEP_REDUCED": re.compile(r"TIME\s*STEP\s*(?:REDUCED|CUT|DECREASE)", re.IGNORECASE),
        "MATRIX_SOLVER_FAILURE": re.compile(r"MATRIX\s*SOLVER.*?FAIL|LINEAR\s*SOLVER.*?FAIL", re.IGNORECASE),
    }

    solver_warning_counts: Dict[str, int] = {}
    for name, pat in solver_warning_patterns.items():
        count = len(pat.findall(text))
        if count > 0:
            solver_warning_counts[name] = count

    if solver_warning_counts:
        lines.append("\n### Solver Warnings Detected\n")
        lines.append("| Warning Type | Occurrences |")
        lines.append("|---|---|")
        for name in sorted(solver_warning_counts, key=lambda x: -solver_warning_counts[x]):
            lines.append(f"| {name} | {solver_warning_counts[name]} |")

        lines.append("")
        if "UNDER_RELAXATION" in solver_warning_counts:
            lines.append("- **Under-relaxation** active: solver is dampening head changes to improve convergence")
        if "BACKTRACKING" in solver_warning_counts:
            lines.append("- **Backtracking** detected: solver reduced step size after divergence")
        if "TIMESTEP_REDUCED" in solver_warning_counts:
            lines.append("- **Timestep cuts** detected: adaptive time stepping reduced Δt to converge")
        if "DAMPING" in solver_warning_counts:
            lines.append("- **Damping** applied: may indicate Newton solver difficulty")

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

        # Try MF6 first
        nam_files = list(ws_root.glob("mfsim.nam"))
        if nam_files:
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
            return "\n".join(lines)

        # Try MF2005/NWT
        mf2k_nams = [f for f in ws_root.glob("*.nam") if f.name.lower() != "mfsim.nam"]
        if mf2k_nams:
            m = flopy.modflow.Modflow.load(
                mf2k_nams[0].name,
                model_ws=str(ws_root),
                load_only=["wel"],
                check=False,
                verbose=False,
            )
            wel = m.get_package("WEL")
            if wel is not None and hasattr(wel, "stress_period_data"):
                spd = wel.stress_period_data
                lines.append(f"Found WEL package in MF2005 model `{mf2k_nams[0].stem}`")
                for iper in range(m.nper):
                    try:
                        data = spd[iper]
                        if data is not None and len(data) > 0:
                            rates = data["flux"] if "flux" in data.dtype.names else []
                            if len(rates) > 0:
                                lines.append(f"\n**Stress Period {iper}:** {len(rates)} wells")
                                extraction = float(np.sum(rates[rates < 0]))
                                injection = float(np.sum(rates[rates > 0]))
                                lines.append(f"- Extraction: {extraction:.4f}")
                                lines.append(f"- Injection: {injection:.4f}")
                                lines.append(f"- Net: {float(np.sum(rates)):.4f}")
                    except (KeyError, IndexError):
                        pass
                return "\n".join(lines)

        return "## Pumping Summary\n\nNo WEL file or name file found in workspace."

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
            # Try MF2005/NWT
            mf2k_nams = [f for f in ws_root.glob("*.nam") if f.name.lower() != "mfsim.nam"]
            if mf2k_nams:
                return _property_check_mf2005(ws_root, mf2k_nams[0])
            # Try to find NPF or LPF file directly
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
    """Fallback: check properties by parsing NPF/LPF/UPW text file directly."""
    lines: List[str] = ["### File-based Property Analysis\n"]

    # Look for MF6 NPF first, then MF2005 LPF/UPW/BCF
    prop_files = (
        list(ws_root.glob("*.npf")) + list(ws_root.glob("*/*.npf")) +
        list(ws_root.glob("*.lpf")) + list(ws_root.glob("*/*.lpf")) +
        list(ws_root.glob("*.upw")) + list(ws_root.glob("*/*.upw")) +
        list(ws_root.glob("*.bcf")) + list(ws_root.glob("*/*.bcf"))
    )
    if not prop_files:
        lines.append("No property package file found (NPF, LPF, UPW, or BCF).")
        return "\n".join(lines)

    prop_path = prop_files[0]
    lines.append(f"**Property file:** `{prop_path.name}`")

    try:
        text = prop_path.read_text(encoding="utf-8", errors="replace")
        # Check for CONSTANT values (MF6 pattern)
        const_pattern = re.compile(r"CONSTANT\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", re.IGNORECASE)
        constants = const_pattern.findall(text)
        if constants:
            lines.append(f"\nConstant values found: {', '.join(constants)}")
            for c in constants:
                val = float(c)
                if val < 1e-12 or val > 1e4:
                    lines.append(f"  **WARNING:** Value {val:.2e} may be unreasonable for hydraulic conductivity")
    except Exception as e:
        lines.append(f"Error reading property file: {e}")

    return "\n".join(lines)


def _property_check_mf2005(ws_root: Path, nam_path: Path) -> str:
    """Property check for MF2005/NWT models via FloPy."""
    lines: List[str] = ["## Property Check (MODFLOW-2005/NWT)\n"]

    try:
        import flopy

        m = flopy.modflow.Modflow.load(
            nam_path.name,
            model_ws=str(ws_root),
            check=False,
            verbose=False,
        )

        # Detect which flow package is active
        flow_pkg = None
        flow_pkg_name = None
        for pkg_name in ["LPF", "UPW", "BCF6"]:
            pkg = m.get_package(pkg_name)
            if pkg is not None:
                flow_pkg = pkg
                flow_pkg_name = pkg_name
                break

        if flow_pkg is None:
            lines.append("No flow package (LPF, UPW, BCF6) found in model.")
            return "\n".join(lines)

        lines.append(f"### Flow Package: {flow_pkg_name}\n")

        # Horizontal hydraulic conductivity
        hk_data = None
        hk_label = "HK"
        try:
            if hasattr(flow_pkg, "hk"):
                hk_data = flow_pkg.hk.array
            elif hasattr(flow_pkg, "hy"):
                hk_data = flow_pkg.hy.array
                hk_label = "HY"
            elif hasattr(flow_pkg, "tran"):
                hk_data = flow_pkg.tran.array
                hk_label = "TRAN (transmissivity)"
        except Exception:
            pass

        if hk_data is not None:
            k_arr = np.asarray(hk_data, dtype=float)
            valid = k_arr[np.isfinite(k_arr) & (k_arr > 0)]

            if valid.size > 0:
                lines.append(f"#### {hk_label}")
                lines.append(f"- Shape: {list(k_arr.shape)}")
                lines.append(f"- Range: {float(np.min(valid)):.2e} to {float(np.max(valid)):.2e}")
                lines.append(f"- Mean: {float(np.mean(valid)):.2e}")
                lines.append(f"- Std: {float(np.std(valid)):.2e}")

                too_low = int(np.sum(valid < 1e-12))
                too_high = int(np.sum(valid > 1e4))
                if too_low > 0:
                    lines.append(f"- **WARNING:** {too_low} cells with {hk_label} < 1e-12 (essentially impermeable)")
                if too_high > 0:
                    lines.append(f"- **WARNING:** {too_high} cells with {hk_label} > 10,000 (unusually high)")

                # Per-layer breakdown
                if len(k_arr.shape) == 3:
                    nlay = k_arr.shape[0]
                    lines.append(f"\n**Per-layer {hk_label} statistics:**")
                    lines.append("| Layer | Min | Max | Mean | Range Ratio |")
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
                        if prev_mean is not None and k_mean > 100 * prev_mean:
                            lines.append(f"  **Note:** Layer {k+1} is >100x Layer {k} — potential layer inversion")
                        prev_mean = k_mean

        # Vertical conductivity / anisotropy
        vka_data = None
        try:
            if hasattr(flow_pkg, "vka"):
                vka_data = flow_pkg.vka.array
        except Exception:
            pass

        if vka_data is not None and hk_data is not None:
            lines.append("\n#### Vertical Conductivity / Anisotropy (VKA)")
            vka_arr = np.asarray(vka_data, dtype=float)
            valid = vka_arr[np.isfinite(vka_arr) & (vka_arr > 0)]
            if valid.size > 0:
                lines.append(f"- Range: {float(np.min(valid)):.2e} to {float(np.max(valid)):.2e}")
                lines.append(f"- Mean: {float(np.mean(valid)):.2e}")

        # Storage properties
        ss_data = None
        sy_data = None
        try:
            if hasattr(flow_pkg, "ss"):
                ss_data = flow_pkg.ss.array
            if hasattr(flow_pkg, "sy"):
                sy_data = flow_pkg.sy.array
        except Exception:
            pass

        if ss_data is not None:
            lines.append("\n#### Storage Properties")
            ss_arr = np.asarray(ss_data, dtype=float)
            valid = ss_arr[np.isfinite(ss_arr) & (ss_arr > 0)]
            if valid.size > 0:
                lines.append(f"- Specific Storage (Ss): {float(np.min(valid)):.2e} to {float(np.max(valid)):.2e} 1/m")
                if np.min(valid) < 1e-7:
                    lines.append("  **Note:** Very low Ss values — ensure these are intentional")
                if np.max(valid) > 1e-2:
                    lines.append("  **WARNING:** Ss > 0.01 is unusually high")

        if sy_data is not None:
            sy_arr = np.asarray(sy_data, dtype=float)
            valid = sy_arr[np.isfinite(sy_arr)]
            if valid.size > 0:
                lines.append(f"- Specific Yield (Sy): {float(np.min(valid)):.4f} to {float(np.max(valid)):.4f}")
                if np.max(valid) > 0.45:
                    lines.append("  **WARNING:** Sy > 0.45 exceeds physical limits for most materials")
                if np.min(valid) < 0.01:
                    lines.append("  **Note:** Very low Sy values — typical only for dense rock")

        # Starting heads from BAS6
        bas = m.get_package("BAS6")
        if bas is not None:
            lines.append("\n#### Starting Heads (BAS6)")
            try:
                strt_data = bas.strt.array
                strt_arr = np.asarray(strt_data, dtype=float)
                valid = strt_arr[np.isfinite(strt_arr) & (np.abs(strt_arr) < 1e20)]
                if valid.size > 0:
                    lines.append(f"- Range: {float(np.min(valid)):.4f} to {float(np.max(valid)):.4f}")
                    head_range = float(np.max(valid) - np.min(valid))
                    lines.append(f"- Head range: {head_range:.4f}")
                    if head_range > 1000:
                        lines.append("  **WARNING:** Starting head range > 1000m — verify correctness")
            except Exception:
                pass

        return "\n".join(lines)

    except Exception as e:
        return f"## Property Check\n\nFailed to load MF2005 model via FloPy: {type(e).__name__}: {e}\n\nTrying file-based analysis...\n\n" + _property_check_from_files(ws_root)


# ---------------------------------------------------------------------------
# Check: observation_comparison  (Phase 1C)
# ---------------------------------------------------------------------------

def check_observation_comparison(ws_root: Path, **kwargs) -> str:
    """Analyze MF6 observation output files and compare with observed data if available.

    Looks for:
    - ``*.obs.csv`` — MF6 continuous observation output
    - ``*_observed*.csv`` or ``*_target*.csv`` — user-provided observed/target data

    Returns per-observation statistics and calibration metrics (RMSE, MAE, R², bias)
    if paired observed data is found.
    """
    lines: List[str] = ["## Observation Data Analysis"]

    # --- Discover observation output files ---
    obs_csvs = sorted(ws_root.glob("*.obs.csv"))
    if not obs_csvs:
        obs_csvs = sorted(ws_root.glob("*/*.obs.csv"))
    if not obs_csvs:
        # Also look for output_obs.csv naming convention
        obs_csvs = sorted(ws_root.glob("*obs*.csv"))
        if not obs_csvs:
            obs_csvs = sorted(ws_root.glob("*/*obs*.csv"))

    if not obs_csvs:
        lines.append("\nNo observation output CSV files (``*.obs.csv``) found in workspace.")
        lines.append("To enable observation analysis:")
        lines.append("- Add an OBS package block to your model")
        lines.append("- Re-run the simulation so MF6 writes ``*.obs.csv`` output")
        return "\n".join(lines)

    lines.append(f"\nFound **{len(obs_csvs)}** observation output file(s).\n")

    # --- Discover observed/target data ---
    target_csvs: List[Path] = []
    for pat in ("*observed*", "*target*", "*measured*", "*calibration*"):
        target_csvs.extend(ws_root.glob(f"{pat}.csv"))
        target_csvs.extend(ws_root.glob(f"*/{pat}.csv"))
    target_csvs = sorted(set(target_csvs))
    # Exclude obs output files themselves
    target_csvs = [t for t in target_csvs if t not in obs_csvs]

    if target_csvs:
        lines.append(f"Found **{len(target_csvs)}** observed/target data file(s): "
                      f"{', '.join(f'`{t.name}`' for t in target_csvs[:5])}\n")

    # --- Read and analyze each observation output ---
    all_sim_data: Dict[str, Any] = {}  # obs_name -> {times, values}

    for obs_csv in obs_csvs[:10]:  # cap at 10 files
        lines.append(f"### `{obs_csv.name}`\n")
        try:
            import csv as csv_mod

            with obs_csv.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv_mod.reader(f)
                header = next(reader, None)
                if not header:
                    lines.append("Empty file — no header row.\n")
                    continue

                # Clean header
                header = [h.strip().strip('"').strip() for h in header]
                time_col_idx = 0  # first col is typically time
                obs_col_names = header[1:]

                if not obs_col_names:
                    lines.append("No observation columns found.\n")
                    continue

                # Read data
                data_rows: List[List[float]] = []
                for row in reader:
                    try:
                        vals = [float(v.strip()) for v in row if v.strip()]
                        if len(vals) == len(header):
                            data_rows.append(vals)
                    except (ValueError, IndexError):
                        continue

                if not data_rows:
                    lines.append("No data rows parsed.\n")
                    continue

                arr = np.array(data_rows)
                times_arr = arr[:, 0]

                lines.append(f"- Time range: {times_arr[0]:.4g} → {times_arr[-1]:.4g} ({len(times_arr)} records)")
                lines.append(f"- Observation columns: {len(obs_col_names)}\n")

                # Per-obs statistics (cap at 30 columns to avoid bloat)
                n_show = min(len(obs_col_names), 30)
                lines.append("| Observation | Min | Max | Mean | Range | First | Last | Change | Trend |")
                lines.append("|---|---|---|---|---|---|---|---|---|")

                for j in range(n_show):
                    col = arr[:, j + 1]
                    valid = col[np.isfinite(col)]
                    if len(valid) == 0:
                        continue
                    obs_name = obs_col_names[j]
                    mn, mx, mean = float(np.min(valid)), float(np.max(valid)), float(np.mean(valid))
                    rng = mx - mn
                    first_v, last_v = float(valid[0]), float(valid[-1])
                    change = last_v - first_v

                    # Simple trend: rising/falling/stable
                    if len(valid) >= 3:
                        half = len(valid) // 2
                        first_half_mean = float(np.mean(valid[:half]))
                        second_half_mean = float(np.mean(valid[half:]))
                        diff_pct = abs(second_half_mean - first_half_mean) / (abs(first_half_mean) + 1e-30) * 100
                        if diff_pct < 1:
                            trend = "stable"
                        elif second_half_mean > first_half_mean:
                            trend = "rising"
                        else:
                            trend = "falling"
                    else:
                        trend = "—"

                    lines.append(
                        f"| {obs_name} | {mn:.4g} | {mx:.4g} | {mean:.4g} | "
                        f"{rng:.4g} | {first_v:.4g} | {last_v:.4g} | {change:+.4g} | {trend} |"
                    )

                    # Store for calibration comparison
                    all_sim_data[obs_name.lower()] = {
                        "times": times_arr.tolist(),
                        "values": col.tolist(),
                    }

                if len(obs_col_names) > n_show:
                    lines.append(f"\n*... and {len(obs_col_names) - n_show} more observation columns (truncated)*\n")

        except Exception as e:
            lines.append(f"Error reading: {type(e).__name__}: {e}\n")

    # --- Calibration comparison ---
    if target_csvs and all_sim_data:
        lines.append("\n---\n### Calibration Comparison\n")

        for tgt_csv in target_csvs[:5]:
            lines.append(f"#### Target file: `{tgt_csv.name}`\n")
            try:
                import csv as csv_mod

                with tgt_csv.open("r", encoding="utf-8", errors="replace") as f:
                    reader = csv_mod.reader(f)
                    header = next(reader, None)
                    if not header:
                        lines.append("Empty target file.\n")
                        continue

                    header = [h.strip().strip('"').strip() for h in header]
                    obs_names_tgt = header[1:]

                    data_rows_tgt: List[List[float]] = []
                    for row in reader:
                        try:
                            vals = [float(v.strip()) for v in row if v.strip()]
                            if len(vals) == len(header):
                                data_rows_tgt.append(vals)
                        except (ValueError, IndexError):
                            continue

                    if not data_rows_tgt:
                        lines.append("No data rows in target file.\n")
                        continue

                    arr_tgt = np.array(data_rows_tgt)
                    times_tgt = arr_tgt[:, 0]

                    matched = 0
                    lines.append("| Observation | N Pairs | RMSE | MAE | Bias | R² | Max Residual |")
                    lines.append("|---|---|---|---|---|---|---|")

                    for j, obs_name in enumerate(obs_names_tgt):
                        sim_key = obs_name.lower()
                        if sim_key not in all_sim_data:
                            # Try fuzzy match: strip common prefixes/suffixes
                            for k in all_sim_data:
                                if sim_key in k or k in sim_key:
                                    sim_key = k
                                    break
                            else:
                                continue

                        sim_info = all_sim_data[sim_key]
                        obs_vals = arr_tgt[:, j + 1]

                        # Interpolate simulated to observed times
                        try:
                            sim_interp = np.interp(times_tgt, sim_info["times"], sim_info["values"])
                        except Exception:
                            continue

                        # Only use valid pairs
                        mask = np.isfinite(obs_vals) & np.isfinite(sim_interp)
                        if np.sum(mask) < 2:
                            continue

                        obs_v = obs_vals[mask]
                        sim_v = sim_interp[mask]
                        residuals = sim_v - obs_v

                        n_pairs = int(np.sum(mask))
                        rmse = float(np.sqrt(np.mean(residuals ** 2)))
                        mae = float(np.mean(np.abs(residuals)))
                        bias = float(np.mean(residuals))
                        max_resid = float(np.max(np.abs(residuals)))

                        # R-squared
                        ss_res = float(np.sum(residuals ** 2))
                        ss_tot = float(np.sum((obs_v - np.mean(obs_v)) ** 2))
                        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                        lines.append(
                            f"| {obs_name} | {n_pairs} | {rmse:.4g} | {mae:.4g} | "
                            f"{bias:+.4g} | {r_sq:.4f} | {max_resid:.4g} |"
                        )
                        matched += 1

                    if matched == 0:
                        lines.append(
                            "\n*No observation names matched between simulated and target data. "
                            "Ensure column names match (case-insensitive).*\n"
                        )
                    else:
                        lines.append(f"\n**{matched}** observation(s) matched for calibration comparison.\n")
                        lines.append("**Interpretation guidance:**")
                        lines.append("- RMSE < 10% of observed range → excellent calibration")
                        lines.append("- RMSE 10–25% → acceptable for most applications")
                        lines.append("- R² > 0.9 → strong correlation; R² < 0.5 → poor fit")
                        lines.append("- Consistent positive/negative bias → systematic error in BCs or parameters")

            except Exception as e:
                lines.append(f"Error reading target file: {type(e).__name__}: {e}\n")

    elif not target_csvs and all_sim_data:
        lines.append("\n---\n### Calibration Comparison\n")
        lines.append("No observed/target data files found for calibration comparison.")
        lines.append("To enable automatic calibration metrics, place a CSV file named")
        lines.append("``*_observed.csv`` or ``*_target.csv`` in the workspace with matching")
        lines.append("column names and time values.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check: listing_budget_detail  (Phase 2A)
# ---------------------------------------------------------------------------

def check_listing_budget_detail(ws_root: Path, **kwargs) -> str:
    """Parse the listing file for per-package IN/OUT budget tables, solver warnings,
    dry-cell messages, and timestep reduction events.

    Goes deeper than ``check_mass_balance`` which only extracts overall discrepancy.
    """
    lst_path = _find_listing_file(ws_root)
    if lst_path is None:
        return "## Listing Budget Detail\n\nNo listing file (.lst) found in the workspace."

    try:
        text = lst_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"## Listing Budget Detail\n\nError reading listing file: {e}"

    lines: List[str] = ["## Listing Budget Detail", f"**Listing file:** `{lst_path.name}` ({lst_path.stat().st_size / 1024:.0f} KB)\n"]

    # --- 1. Per-package budget table ---
    # MF6 listing file budget block looks like:
    #          IN:                    OUT:
    #          ---                    ---
    #     STO-SS =       0.0000        STO-SS =       123.456
    #     STO-SY =       0.0000        STO-SY =        78.910
    #        WEL =       0.0000           WEL =     12345.678
    # We parse the LAST budget block (most representative — final timestep)

    # Find all budget blocks
    budget_block_re = re.compile(
        r"VOLUME\s+BUDGET\s+FOR\s+ENTIRE\s+MODEL.*?PERCENT\s+DISCREPANCY",
        re.IGNORECASE | re.DOTALL,
    )
    blocks = list(budget_block_re.finditer(text))

    if blocks:
        # Parse the LAST budget block
        last_block = blocks[-1].group()

        # Extract per-component IN and OUT lines
        #  Pattern:  COMPONENT-NAME =  <value>
        component_re = re.compile(
            r"^\s+([\w\-]+)\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
            re.MULTILINE,
        )

        # Split block into IN and OUT halves
        in_section = ""
        out_section = ""
        in_start = last_block.upper().find("IN:")
        out_start = last_block.upper().find("OUT:")
        total_in_pos = last_block.upper().find("TOTAL IN")
        total_out_pos = last_block.upper().find("TOTAL OUT")

        # Collect all component mentions and classify
        pkg_in: Dict[str, float] = {}
        pkg_out: Dict[str, float] = {}

        # Strategy: parse line by line between markers
        block_lines = last_block.split("\n")
        in_section_active = False
        for bl in block_lines:
            bl_upper = bl.upper().strip()
            if "IN:" in bl_upper:
                in_section_active = True
                continue
            if "OUT:" in bl_upper:
                in_section_active = False
                continue
            if "TOTAL IN" in bl_upper or "TOTAL OUT" in bl_upper:
                continue

            m = component_re.match(bl)
            if m:
                comp = m.group(1).strip()
                val = float(m.group(2))
                # MF6 prints IN and OUT side by side on same line
                # We'll collect them and deduplicate
                if in_section_active:
                    pkg_in[comp] = val
                else:
                    pkg_out[comp] = val

        # Sometimes MF6 prints IN and OUT on the same line:
        #    WEL =    0.0000       WEL =   12345.6
        # Re-parse with a dual-column regex
        dual_re = re.compile(
            r"^\s+([\w\-]+)\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
            r"\s+([\w\-]+)\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
            re.MULTILINE,
        )
        for m in dual_re.finditer(last_block):
            comp_left, val_left = m.group(1), float(m.group(2))
            comp_right, val_right = m.group(3), float(m.group(4))
            pkg_in[comp_left] = val_left
            pkg_out[comp_right] = val_right

        if pkg_in or pkg_out:
            all_comps = sorted(set(list(pkg_in.keys()) + list(pkg_out.keys())))
            lines.append("### Per-Package Budget (Final Timestep)\n")
            lines.append("| Package | IN | OUT | NET |")
            lines.append("|---|---|---|---|")

            total_in_val = 0.0
            total_out_val = 0.0
            for comp in all_comps:
                in_val = pkg_in.get(comp, 0.0)
                out_val = pkg_out.get(comp, 0.0)
                net = in_val - out_val
                total_in_val += in_val
                total_out_val += out_val
                lines.append(f"| {comp} | {in_val:.4g} | {out_val:.4g} | {net:+.4g} |")

            net_total = total_in_val - total_out_val
            lines.append(f"| **TOTAL** | **{total_in_val:.4g}** | **{total_out_val:.4g}** | **{net_total:+.4g}** |")

            if total_in_val > 0:
                pct_disc = abs(net_total) / ((total_in_val + total_out_val) / 2) * 100
                lines.append(f"\n**Percent discrepancy:** {pct_disc:.4f}%")
                if pct_disc < 0.5:
                    lines.append("Budget closure is **good** (<0.5%).")
                elif pct_disc < 1.0:
                    lines.append("Budget closure is **marginal** (0.5–1%). Review solver settings.")
                else:
                    lines.append("Budget closure is **poor** (>1%). Investigate boundary conditions and solver convergence.")

        lines.append(f"\n*Parsed from {len(blocks)} budget block(s); showing the last one.*\n")

    else:
        lines.append("No volumetric budget blocks found in the listing file.\n")

    # --- 2. Solver warnings ---
    lines.append("### Solver & Stability Warnings\n")

    warning_patterns = {
        "UNDER_RELAXATION": re.compile(r"UNDER.?RELAX", re.IGNORECASE),
        "BACKTRACKING": re.compile(r"BACKTRACK", re.IGNORECASE),
        "DAMPING": re.compile(r"DAMP(?:ING|ED)", re.IGNORECASE),
        "TIMESTEP_REDUCED": re.compile(r"TIME\s*STEP\s*(?:REDUCED|CUT|DECREASE)", re.IGNORECASE),
        "DRY_CELL": re.compile(r"DRY\s*CELL|CELL\s*(?:IS|WENT|BECAME)\s*DRY", re.IGNORECASE),
        "REWET": re.compile(r"RE.?WET|CELL\s*(?:RE)?WETTED", re.IGNORECASE),
        "WARNING": re.compile(r"^\s*WARNING", re.IGNORECASE | re.MULTILINE),
        "BUDGET_ERROR": re.compile(r"BUDGET\s*ERROR|PERCENT\s*ERROR", re.IGNORECASE),
    }

    warning_counts: Dict[str, int] = {}
    warning_examples: Dict[str, List[str]] = {}

    for name, pat in warning_patterns.items():
        matches = list(pat.finditer(text))
        if matches:
            warning_counts[name] = len(matches)
            # Capture a few example lines for context
            examples = []
            for m in matches[:3]:
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 80)
                snippet = text[start:end].replace("\n", " ").strip()[:120]
                examples.append(snippet)
            warning_examples[name] = examples

    if warning_counts:
        lines.append("| Warning Type | Count |")
        lines.append("|---|---|")
        for name in sorted(warning_counts, key=lambda x: -warning_counts[x]):
            lines.append(f"| {name} | {warning_counts[name]} |")

        lines.append("")
        for name in sorted(warning_counts, key=lambda x: -warning_counts[x])[:5]:
            if warning_examples.get(name):
                lines.append(f"\n**{name}** example(s):")
                for ex in warning_examples[name]:
                    lines.append(f"- `{ex}`")
    else:
        lines.append("No solver warnings, dry-cell messages, or timestep reductions detected.")
        lines.append("This is a positive sign for model stability.")

    # --- 3. Dry cell locations from listing ---
    dry_cell_re = re.compile(
        r"DRY\s+CELL.*?LAY(?:ER)?\s*=?\s*(\d+).*?ROW\s*=?\s*(\d+).*?COL\s*=?\s*(\d+)",
        re.IGNORECASE,
    )
    dry_matches = list(dry_cell_re.finditer(text))
    if dry_matches:
        lines.append(f"\n### Dry Cell Locations from Listing ({len(dry_matches)} messages)")
        # Aggregate by layer
        dry_by_layer: Dict[int, int] = {}
        for m in dry_matches:
            lay = int(m.group(1))
            dry_by_layer[lay] = dry_by_layer.get(lay, 0) + 1
        lines.append("\n| Layer | Dry Cell Messages |")
        lines.append("|---|---|")
        for lay in sorted(dry_by_layer):
            lines.append(f"| {lay} | {dry_by_layer[lay]} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check: property_zones  (Phase 3A)
# ---------------------------------------------------------------------------

def check_property_zones(ws_root: Path, **kwargs) -> str:
    """Analyze spatial distribution of hydraulic properties (K) per layer.

    Identifies distinct value clusters/zones, their extents, K ranges,
    and flags extreme contrasts between adjacent cells.
    """
    lines: List[str] = ["## Hydraulic Property Zone Analysis"]

    try:
        import flopy
    except ImportError:
        lines.append("\nFloPy is required for property zone analysis but is not installed.")
        return "\n".join(lines)

    # Try to load model via FloPy — MF6 first, then MF2005/NWT
    k_arr = None
    model_label = ""

    # Try MF6
    try:
        sim_path = ws_root / "mfsim.nam"
        if sim_path.exists():
            sim = flopy.mf6.MFSimulation.load(
                sim_ws=str(ws_root), verbosity_level=0,
            )
            model_names = list(sim.model_names)
            if model_names:
                gwf = sim.get_model(model_names[0])
                if hasattr(gwf, "npf") and gwf.npf is not None:
                    k_arr = gwf.npf.k.array
                    model_label = f"MF6 model `{model_names[0]}`"
    except Exception:
        pass

    # Try MF2005/NWT
    if k_arr is None:
        try:
            mf2k_nams = [f for f in ws_root.glob("*.nam") if f.name.lower() != "mfsim.nam"]
            if mf2k_nams:
                m = flopy.modflow.Modflow.load(
                    mf2k_nams[0].name,
                    model_ws=str(ws_root),
                    check=False,
                    verbose=False,
                )
                for pkg_name, attr_name in [("LPF", "hk"), ("UPW", "hk"), ("BCF6", "hy")]:
                    pkg = m.get_package(pkg_name)
                    if pkg is not None and hasattr(pkg, attr_name):
                        k_arr = getattr(pkg, attr_name).array
                        model_label = f"MF2005 model ({pkg_name})"
                        break
        except Exception:
            pass

    if k_arr is None:
        # Fallback: try to read property arrays from files directly
        lines.append("\nCould not load model via FloPy. Attempting direct file analysis.\n")
        return _property_zones_from_file(ws_root, lines)

    if k_arr is None or k_arr.size == 0:
        lines.append("\nK array is empty.")
        return "\n".join(lines)

    nlay = k_arr.shape[0]
    if model_label:
        lines.append(f"\n**Source:** {model_label}")
    lines.append(f"**Model layers:** {nlay}")
    lines.append(f"**K array shape:** {k_arr.shape}\n")

    for lay in range(min(nlay, 20)):  # cap at 20 layers
        k_layer = k_arr[lay]
        valid = k_layer[np.isfinite(k_layer) & (k_layer > 0)]
        if len(valid) == 0:
            lines.append(f"\n### Layer {lay + 1}: No valid K values")
            continue

        lines.append(f"\n### Layer {lay + 1}")
        lines.append(f"- K range: {np.min(valid):.4g} → {np.max(valid):.4g}")
        lines.append(f"- Mean: {np.mean(valid):.4g}, Median: {np.median(valid):.4g}")
        lines.append(f"- Std: {np.std(valid):.4g}")

        # Coefficient of variation
        cv = float(np.std(valid) / np.mean(valid)) if np.mean(valid) > 0 else 0
        lines.append(f"- Coefficient of variation: {cv:.3f}")
        if cv < 0.1:
            lines.append("  - Essentially uniform K distribution")
        elif cv < 0.5:
            lines.append("  - Moderate spatial variability")
        elif cv < 2.0:
            lines.append("  - High spatial variability — multiple zones likely")
        else:
            lines.append("  - **Extreme variability** — check for unrealistic values")

        # Log-binning to identify distinct K zones
        log_k = np.log10(valid)
        log_min, log_max = float(np.min(log_k)), float(np.max(log_k))
        log_range = log_max - log_min

        if log_range < 0.5:
            lines.append(f"  - K spans <0.5 orders of magnitude — effectively one zone")
        else:
            # Bin into half-decade groups
            n_bins = min(int(log_range / 0.5) + 1, 10)
            bin_edges = np.linspace(log_min, log_max, n_bins + 1)
            lines.append(f"\n**K zones (log-binned, {n_bins} bins across {log_range:.1f} orders of magnitude):**\n")
            lines.append("| Zone | K Range (m/d) | Cell Count | % of Layer |")
            lines.append("|---|---|---|---|")

            for b in range(n_bins):
                mask = (log_k >= bin_edges[b]) & (log_k < bin_edges[b + 1] + 1e-10)
                count = int(np.sum(mask))
                if count == 0:
                    continue
                lo = 10 ** bin_edges[b]
                hi = 10 ** bin_edges[b + 1]
                pct = count / len(valid) * 100
                lines.append(f"| {b + 1} | {lo:.4g} – {hi:.4g} | {count} | {pct:.1f}% |")

        # Adjacent cell contrast check (only for 2D+ arrays)
        if len(k_layer.shape) >= 2:
            nrow, ncol = k_layer.shape[-2], k_layer.shape[-1]
            max_contrast = 1.0
            contrast_count = 0

            # Row-direction contrasts
            if nrow > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_r = k_layer[1:, :] / k_layer[:-1, :]
                    ratio_r = np.where(np.isfinite(ratio_r) & (ratio_r > 0), ratio_r, 1.0)
                    contrast_r = np.maximum(ratio_r, 1.0 / ratio_r)
                    max_contrast = max(max_contrast, float(np.max(contrast_r)))
                    contrast_count += int(np.sum(contrast_r > 1000))

            # Column-direction contrasts
            if ncol > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_c = k_layer[:, 1:] / k_layer[:, :-1]
                    ratio_c = np.where(np.isfinite(ratio_c) & (ratio_c > 0), ratio_c, 1.0)
                    contrast_c = np.maximum(ratio_c, 1.0 / ratio_c)
                    max_contrast = max(max_contrast, float(np.max(contrast_c)))
                    contrast_count += int(np.sum(contrast_c > 1000))

            lines.append(f"\n- Max adjacent-cell K contrast: **{max_contrast:.1f}x**")
            if contrast_count > 0:
                lines.append(f"- Cells with >1000x contrast to neighbor: **{contrast_count}**")
                lines.append("  - **WARNING:** Extreme K contrasts can cause numerical instability")
            elif max_contrast > 100:
                lines.append("  - Notable contrast, but below critical threshold (1000x)")
            else:
                lines.append("  - Contrasts are moderate — good for solver stability")

    # Check for vertical K if available (K33 for MF6, VKA for MF2005)
    k33 = None
    try:
        # Try MF6: gwf.npf.k33
        sim_path = ws_root / "mfsim.nam"
        if sim_path.exists():
            _sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws_root), verbosity_level=0)
            _gwf = _sim.get_model(list(_sim.model_names)[0])
            k33 = _gwf.npf.k33.array
        else:
            # Try MF2005: LPF/UPW vka
            mf2k_nams = [f for f in ws_root.glob("*.nam") if f.name.lower() != "mfsim.nam"]
            if mf2k_nams:
                _m = flopy.modflow.Modflow.load(
                    mf2k_nams[0].name, model_ws=str(ws_root), check=False, verbose=False,
                )
                for _pkg_name in ["LPF", "UPW"]:
                    _pkg = _m.get_package(_pkg_name)
                    if _pkg is not None and hasattr(_pkg, "vka"):
                        k33 = _pkg.vka.array
                        break
    except Exception:
        pass

    if k33 is not None and k33.size > 0:
        lines.append("\n### Vertical K Summary\n")
        for lay in range(min(nlay, 10)):
            try:
                kv = k33[lay]
                kh = k_arr[lay]
                valid_kv = kv[np.isfinite(kv) & (kv > 0)]
                valid_kh = kh[np.isfinite(kh) & (kh > 0)]
                if len(valid_kv) > 0 and len(valid_kh) > 0:
                    ratio = np.mean(valid_kv) / np.mean(valid_kh)
                    lines.append(f"- Layer {lay + 1}: Kv/Kh = {ratio:.4g}")
                    if ratio > 1.0:
                        lines.append("  - **Unusual:** Kv > Kh — this is physically uncommon")
                    elif ratio < 0.001:
                        lines.append("  - Very low Kv/Kh ratio — may impede vertical flow significantly")
            except Exception:
                pass

    return "\n".join(lines)


def _property_zones_from_file(ws_root: Path, lines: List[str]) -> str:
    """Fallback: analyze property file text for K values when FloPy load fails."""
    prop_files = (
        list(ws_root.glob("*.npf")) + list(ws_root.glob("*/*.npf")) +
        list(ws_root.glob("*.lpf")) + list(ws_root.glob("*/*.lpf")) +
        list(ws_root.glob("*.upw")) + list(ws_root.glob("*/*.upw"))
    )
    if not prop_files:
        lines.append("No property package file found (NPF, LPF, or UPW).")
        return "\n".join(lines)

    npf_path = prop_files[0]
    lines.append(f"**Property file:** `{npf_path.name}`\n")

    try:
        text = npf_path.read_text(encoding="utf-8", errors="replace")

        # Look for CONSTANT values
        const_re = re.compile(r"CONSTANT\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", re.IGNORECASE)
        constants = const_re.findall(text)

        # Look for INTERNAL/OPEN-CLOSE array references
        internal_re = re.compile(r"INTERNAL\s+FACTOR\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", re.IGNORECASE)
        internals = internal_re.findall(text)

        if constants:
            lines.append("**Constant K values found:**")
            for c in constants:
                val = float(c)
                lines.append(f"- K = {val:.4g} m/d")
                if val < 1e-8:
                    lines.append("  - Extremely low — essentially impermeable")
                elif val > 1e4:
                    lines.append("  - **Very high** — unusual for natural materials")

        if internals:
            lines.append(f"\n**Internal array blocks:** {len(internals)} (with scale factors: {', '.join(internals[:5])})")

        # Look for OPEN/CLOSE references to external files
        openclose_re = re.compile(r"OPEN/CLOSE\s+(\S+)", re.IGNORECASE)
        ext_files = openclose_re.findall(text)
        if ext_files:
            lines.append(f"\n**External array files:** {', '.join(f'`{f}`' for f in ext_files[:10])}")
            lines.append("(Detailed zone analysis requires FloPy to read these files)")

    except Exception as e:
        lines.append(f"Error reading NPF file: {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check: advanced_packages  (Phase 3B)
# ---------------------------------------------------------------------------

def check_advanced_packages(ws_root: Path, **kwargs) -> str:
    """Parse and summarize complex MF6 packages: SFR, LAK, MAW, UZF.

    Provides structured summaries the LLM can reason about for each detected
    advanced package.
    """
    lines: List[str] = ["## Advanced Package Summary"]

    # Detect which advanced packages are present
    pkg_map: Dict[str, List[Path]] = {}
    for ext in ("sfr", "lak", "maw", "uzf", "csub", "evt"):
        files = list(ws_root.glob(f"*.{ext}")) + list(ws_root.glob(f"*/*.{ext}"))
        if files:
            pkg_map[ext.upper()] = files

    if not pkg_map:
        lines.append("\nNo advanced packages (SFR, LAK, MAW, UZF, CSUB, EVT) detected.")
        return "\n".join(lines)

    lines.append(f"\nDetected **{len(pkg_map)}** advanced package type(s): {', '.join(sorted(pkg_map.keys()))}\n")

    for pkg_type, files in sorted(pkg_map.items()):
        for pkg_file in files[:3]:  # cap per type
            lines.append(f"### {pkg_type}: `{pkg_file.name}`\n")
            try:
                text = pkg_file.read_text(encoding="utf-8", errors="replace")
                text_upper = text.upper()

                if pkg_type == "SFR":
                    _parse_sfr_summary(text, text_upper, lines)
                elif pkg_type == "LAK":
                    _parse_lak_summary(text, text_upper, lines)
                elif pkg_type == "MAW":
                    _parse_maw_summary(text, text_upper, lines)
                elif pkg_type == "UZF":
                    _parse_uzf_summary(text, text_upper, lines)
                elif pkg_type == "CSUB":
                    _parse_csub_summary(text, text_upper, lines)
                elif pkg_type == "EVT":
                    _parse_evt_summary(text, text_upper, lines)
                else:
                    lines.append(f"Basic file size: {pkg_file.stat().st_size} bytes")

            except Exception as e:
                lines.append(f"Error reading: {type(e).__name__}: {e}")

    return "\n".join(lines)


def _parse_sfr_summary(text: str, text_upper: str, lines: List[str]) -> None:
    """Parse SFR (Streamflow Routing) package summary."""
    # Count reaches in PACKAGEDATA
    pkg_data_start = text_upper.find("BEGIN PACKAGEDATA")
    pkg_data_end = text_upper.find("END PACKAGEDATA")
    reach_count = 0
    if pkg_data_start >= 0 and pkg_data_end >= 0:
        pkg_block = text[pkg_data_start:pkg_data_end]
        data_lines = [l.strip() for l in pkg_block.split("\n")
                      if l.strip() and not l.strip().startswith("#")
                      and not l.strip().upper().startswith("BEGIN")
                      and not l.strip().upper().startswith("END")]
        reach_count = len(data_lines)

    lines.append(f"- **Reaches:** {reach_count}")

    # Parse reach properties from PACKAGEDATA
    width_re = re.compile(r"^\s*\d+\s+.*?(?:\s+)([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", re.MULTILINE)
    widths = []
    gradients = []
    lengths = []
    if pkg_data_start >= 0 and pkg_data_end >= 0:
        pkg_block = text[pkg_data_start:pkg_data_end]
        for line in pkg_block.split("\n"):
            parts = line.strip().split()
            if len(parts) >= 7 and parts[0].isdigit():
                try:
                    # PACKAGEDATA: rno cellid rlen rwid rgrd rtp rbth rbhk
                    length = float(parts[-6]) if len(parts) >= 9 else 0
                    width = float(parts[-5]) if len(parts) >= 9 else 0
                    gradient = float(parts[-4]) if len(parts) >= 9 else 0
                    if length > 0:
                        lengths.append(length)
                    if width > 0:
                        widths.append(width)
                    if gradient > 0:
                        gradients.append(gradient)
                except (ValueError, IndexError):
                    pass

    if widths:
        lines.append(f"- **Width range:** {min(widths):.2f} – {max(widths):.2f}")
    if gradients:
        lines.append(f"- **Gradient range:** {min(gradients):.6f} – {max(gradients):.6f}")
    if lengths:
        lines.append(f"- **Reach length range:** {min(lengths):.1f} – {max(lengths):.1f}")
        lines.append(f"- **Total stream length:** {sum(lengths):.1f}")

    # Connection/diversion info
    conn_start = text_upper.find("BEGIN CONNECTIONDATA")
    if conn_start >= 0:
        conn_end = text_upper.find("END CONNECTIONDATA", conn_start)
        if conn_end >= 0:
            conn_block = text[conn_start:conn_end]
            conn_lines = [l for l in conn_block.split("\n")
                         if l.strip() and not l.strip().startswith("#")
                         and "BEGIN" not in l.upper() and "END" not in l.upper()]
            lines.append(f"- **Connections defined:** {len(conn_lines)}")

    div_start = text_upper.find("BEGIN DIVERSIONS")
    if div_start >= 0:
        lines.append("- **Diversions:** present")

    # Count stress period blocks
    period_count = text_upper.count("BEGIN PERIOD")
    lines.append(f"- **Stress period data blocks:** {period_count}")
    lines.append("")


def _parse_lak_summary(text: str, text_upper: str, lines: List[str]) -> None:
    """Parse LAK (Lake) package summary."""
    # Count lakes in PACKAGEDATA
    pkg_start = text_upper.find("BEGIN PACKAGEDATA")
    pkg_end = text_upper.find("END PACKAGEDATA")
    lake_count = 0
    stages = []
    if pkg_start >= 0 and pkg_end >= 0:
        pkg_block = text[pkg_start:pkg_end]
        for line in pkg_block.split("\n"):
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                lake_count += 1
                try:
                    stage = float(parts[1]) if len(parts) > 1 else None
                    if stage is not None:
                        stages.append(stage)
                except (ValueError, IndexError):
                    pass

    lines.append(f"- **Lakes:** {lake_count}")
    if stages:
        lines.append(f"- **Initial stage range:** {min(stages):.4g} – {max(stages):.4g}")

    # Outlets
    outlet_start = text_upper.find("BEGIN OUTLETS")
    outlet_count = 0
    if outlet_start >= 0:
        outlet_end = text_upper.find("END OUTLETS", outlet_start)
        if outlet_end >= 0:
            outlet_block = text[outlet_start:outlet_end]
            for line in outlet_block.split("\n"):
                if line.strip() and line.strip()[0].isdigit():
                    outlet_count += 1
    lines.append(f"- **Outlets:** {outlet_count}")

    # Connection data
    conn_start = text_upper.find("BEGIN CONNECTIONDATA")
    conn_count = 0
    if conn_start >= 0:
        conn_end = text_upper.find("END CONNECTIONDATA", conn_start)
        if conn_end >= 0:
            conn_block = text[conn_start:conn_end]
            for line in conn_block.split("\n"):
                if line.strip() and line.strip()[0].isdigit():
                    conn_count += 1
    lines.append(f"- **Lake-aquifer connections:** {conn_count}")

    period_count = text_upper.count("BEGIN PERIOD")
    lines.append(f"- **Stress period data blocks:** {period_count}")
    lines.append("")


def _parse_maw_summary(text: str, text_upper: str, lines: List[str]) -> None:
    """Parse MAW (Multi-Aquifer Well) package summary."""
    pkg_start = text_upper.find("BEGIN PACKAGEDATA")
    pkg_end = text_upper.find("END PACKAGEDATA")
    well_count = 0
    radii = []
    if pkg_start >= 0 and pkg_end >= 0:
        pkg_block = text[pkg_start:pkg_end]
        for line in pkg_block.split("\n"):
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                well_count += 1
                try:
                    radius = float(parts[1]) if len(parts) > 1 else None
                    if radius and radius > 0:
                        radii.append(radius)
                except (ValueError, IndexError):
                    pass

    lines.append(f"- **Multi-aquifer wells:** {well_count}")
    if radii:
        lines.append(f"- **Well radius range:** {min(radii):.4g} – {max(radii):.4g}")

    # Connection data (screens)
    conn_start = text_upper.find("BEGIN CONNECTIONDATA")
    screen_count = 0
    if conn_start >= 0:
        conn_end = text_upper.find("END CONNECTIONDATA", conn_start)
        if conn_end >= 0:
            conn_block = text[conn_start:conn_end]
            for line in conn_block.split("\n"):
                if line.strip() and line.strip()[0].isdigit():
                    screen_count += 1
    lines.append(f"- **Well-aquifer connections (screens):** {screen_count}")
    if well_count > 0 and screen_count > 0:
        lines.append(f"- **Avg screens per well:** {screen_count / well_count:.1f}")

    period_count = text_upper.count("BEGIN PERIOD")
    lines.append(f"- **Stress period data blocks:** {period_count}")
    lines.append("")


def _parse_uzf_summary(text: str, text_upper: str, lines: List[str]) -> None:
    """Parse UZF (Unsaturated Zone Flow) package summary."""
    pkg_start = text_upper.find("BEGIN PACKAGEDATA")
    pkg_end = text_upper.find("END PACKAGEDATA")
    cell_count = 0
    if pkg_start >= 0 and pkg_end >= 0:
        pkg_block = text[pkg_start:pkg_end]
        for line in pkg_block.split("\n"):
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                cell_count += 1

    lines.append(f"- **UZF cells:** {cell_count}")

    # Look for infiltration rates in PERIOD blocks
    period_count = text_upper.count("BEGIN PERIOD")
    lines.append(f"- **Stress period data blocks:** {period_count}")

    # Options
    if "SIMULATE_ET" in text_upper:
        lines.append("- **ET simulation:** enabled")
    if "SIMULATE_GWSEEP" in text_upper:
        lines.append("- **GW seepage:** enabled")
    if "UNSAT_ETWC" in text_upper:
        lines.append("- **Unsaturated ET:** enabled")
    if "LINEAR_GWET" in text_upper:
        lines.append("- **Linear GW ET:** enabled")

    lines.append("")


def _parse_csub_summary(text: str, text_upper: str, lines: List[str]) -> None:
    """Parse CSUB (Skeletal Storage/Compaction/Subsidence) package summary."""
    pkg_start = text_upper.find("BEGIN PACKAGEDATA")
    pkg_end = text_upper.find("END PACKAGEDATA")
    interbed_count = 0
    if pkg_start >= 0 and pkg_end >= 0:
        pkg_block = text[pkg_start:pkg_end]
        for line in pkg_block.split("\n"):
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                interbed_count += 1

    lines.append(f"- **Interbeds:** {interbed_count}")

    if "HEAD_BASED" in text_upper:
        lines.append("- **Formulation:** head-based")
    elif "EFFECTIVE_STRESS" in text_upper or "EFFECTIVE-STRESS" in text_upper:
        lines.append("- **Formulation:** effective-stress-based")

    if "DELAY_INTERBEDS" in text_upper or "NODELAY" not in text_upper:
        lines.append("- **Delay interbeds:** may be present")

    period_count = text_upper.count("BEGIN PERIOD")
    lines.append(f"- **Stress period data blocks:** {period_count}")
    lines.append("")


def _parse_evt_summary(text: str, text_upper: str, lines: List[str]) -> None:
    """Parse EVT (Evapotranspiration) package summary."""
    # Check for options
    if "READASARRAYS" in text_upper.replace(" ", ""):
        lines.append("- **Format:** array-based (READASARRAYS)")
    else:
        lines.append("- **Format:** list-based")

    # Count cells/entries in PERIOD blocks
    period_count = text_upper.count("BEGIN PERIOD")
    lines.append(f"- **Stress period data blocks:** {period_count}")

    if "SURF_RATE_SPECIFIED" in text_upper:
        lines.append("- **Surface rate:** specified")
    if "FIXED_CELL" in text_upper:
        lines.append("- **Fixed cells:** enabled")

    # Look for NSEG
    nseg_re = re.compile(r"NSEG\s+(\d+)", re.IGNORECASE)
    nseg_m = nseg_re.search(text)
    if nseg_m:
        lines.append(f"- **ET segments:** {nseg_m.group(1)}")

    lines.append("")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _check_save_snapshot(ws_root: Path, **kwargs) -> str:
    from gw.api.run_snapshots import check_save_snapshot
    return check_save_snapshot(ws_root, **kwargs)


def _check_compare_runs(ws_root: Path, **kwargs) -> str:
    from gw.api.run_snapshots import check_compare_runs
    return check_compare_runs(ws_root, **kwargs)


AVAILABLE_CHECKS = {
    "mass_balance": check_mass_balance,
    "dry_cells": check_dry_cells,
    "convergence": check_convergence,
    "pumping_summary": check_pumping_summary,
    "budget_timeseries": check_budget_timeseries,
    "head_gradient": check_head_gradient,
    "property_check": check_property_check,
    "observation_comparison": check_observation_comparison,
    "listing_budget_detail": check_listing_budget_detail,
    "property_zones": check_property_zones,
    "advanced_packages": check_advanced_packages,
    "save_snapshot": _check_save_snapshot,
    "compare_runs": _check_compare_runs,
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
        property_check, observation_comparison, listing_budget_detail,
        property_zones, advanced_packages, save_snapshot, compare_runs.
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
