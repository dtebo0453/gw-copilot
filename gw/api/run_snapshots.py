"""Run snapshot storage and comparison for MODFLOW 6 models.

Captures lightweight post-run snapshots (per-layer head stats, budget totals,
dry cell counts, mass balance) and supports before/after comparison.

Snapshots are stored in:
  <workspace>/.gw_copilot/runs/<timestamp>.json

QA checks ``save_snapshot`` and ``compare_runs`` delegate to this module.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot storage directory
# ---------------------------------------------------------------------------

def _snapshot_dir(ws_root: Path) -> Path:
    d = ws_root / ".gw_copilot" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Save a snapshot
# ---------------------------------------------------------------------------

def save_run_snapshot(ws_root: Path, label: Optional[str] = None) -> Dict[str, Any]:
    """Capture a lightweight snapshot of the current model outputs.

    Captures:
    - Per-layer head statistics (min, max, mean, dry cell count)
    - Total budget IN/OUT and % discrepancy
    - Number of dry cells per layer
    - Convergence failure count
    - Timestamp

    Returns the snapshot dict (also written to disk).
    """
    snapshot: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label or "auto",
        "workspace": str(ws_root),
    }

    # --- Head statistics ---
    head_stats = _capture_head_stats(ws_root)
    snapshot["head_stats"] = head_stats

    # --- Budget summary ---
    budget = _capture_budget_summary(ws_root)
    snapshot["budget"] = budget

    # --- Convergence info ---
    convergence = _capture_convergence_info(ws_root)
    snapshot["convergence"] = convergence

    # Write to disk
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = label.replace(" ", "_")[:30] if label else "snapshot"
    filename = f"{ts_str}_{slug}.json"
    out_path = _snapshot_dir(ws_root) / filename

    try:
        out_path.write_text(
            json.dumps(snapshot, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        snapshot["saved_to"] = str(out_path)
    except Exception as e:
        snapshot["save_error"] = str(e)

    return snapshot


def _capture_head_stats(ws_root: Path) -> Dict[str, Any]:
    """Read HDS file and compute per-layer statistics."""
    try:
        import flopy
    except ImportError:
        return {"error": "FloPy not available"}

    hds_files = list(ws_root.glob("*.hds")) + list(ws_root.glob("*/*.hds"))
    if not hds_files:
        return {"error": "No .hds file found"}

    try:
        hf = flopy.utils.HeadFile(str(hds_files[0]))
        times = hf.get_times()
        if not times:
            return {"error": "No timesteps in HDS file"}

        # Use final timestep
        data = hf.get_data(totim=times[-1])
        nlay = data.shape[0]

        layers: List[Dict[str, Any]] = []
        total_dry = 0
        for k in range(nlay):
            layer = data[k]
            dry_mask = (layer >= 1e20) | (~np.isfinite(layer))
            dry_count = int(np.sum(dry_mask))
            total_dry += dry_count

            valid = layer[~dry_mask]
            if len(valid) > 0:
                layers.append({
                    "layer": k + 1,
                    "min": float(np.min(valid)),
                    "max": float(np.max(valid)),
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                    "dry_cells": dry_count,
                    "valid_cells": int(len(valid)),
                })
            else:
                layers.append({
                    "layer": k + 1,
                    "min": None, "max": None, "mean": None, "std": None,
                    "dry_cells": dry_count,
                    "valid_cells": 0,
                })

        return {
            "file": hds_files[0].name,
            "final_time": float(times[-1]),
            "n_timesteps": len(times),
            "n_layers": nlay,
            "total_dry_cells": total_dry,
            "layers": layers,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _capture_budget_summary(ws_root: Path) -> Dict[str, Any]:
    """Parse listing file for final budget block totals."""
    import re

    lst_files = list(ws_root.glob("*.lst")) + list(ws_root.glob("*/*.lst"))
    if not lst_files:
        return {"error": "No .lst file found"}

    try:
        lst_path = max(lst_files, key=lambda p: p.stat().st_size)
        text = lst_path.read_text(encoding="utf-8", errors="replace")

        # Find total IN/OUT from the last budget block
        total_in_re = re.compile(
            r"TOTAL\s+IN\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        )
        total_out_re = re.compile(
            r"TOTAL\s+OUT\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        )
        pct_disc_re = re.compile(
            r"PERCENT\s+DISCREPANCY\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        )

        in_matches = total_in_re.findall(text)
        out_matches = total_out_re.findall(text)
        pct_matches = pct_disc_re.findall(text)

        result: Dict[str, Any] = {"file": lst_path.name}
        if in_matches:
            result["total_in"] = float(in_matches[-1])
        if out_matches:
            result["total_out"] = float(out_matches[-1])
        if pct_matches:
            result["percent_discrepancy"] = float(pct_matches[-1])

        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _capture_convergence_info(ws_root: Path) -> Dict[str, Any]:
    """Count convergence failures from listing file."""
    import re

    lst_files = list(ws_root.glob("*.lst")) + list(ws_root.glob("*/*.lst"))
    if not lst_files:
        return {"error": "No .lst file found"}

    try:
        lst_path = max(lst_files, key=lambda p: p.stat().st_size)
        text = lst_path.read_text(encoding="utf-8", errors="replace")

        failure_re = re.compile(
            r"FAILED?\s+TO\s+CONVERGE|CONVERGENCE\s+FAILURE|"
            r"DID\s+NOT\s+CONVERGE|NO\s+CONVERGENCE",
            re.IGNORECASE,
        )
        failures = failure_re.findall(text)

        return {
            "failure_count": len(failures),
            "converged": len(failures) == 0,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# List snapshots
# ---------------------------------------------------------------------------

def list_run_snapshots(ws_root: Path) -> List[Dict[str, Any]]:
    """List available snapshots for the workspace, newest first."""
    snap_dir = _snapshot_dir(ws_root)
    snapshots = []
    for f in sorted(snap_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            snapshots.append({
                "file": f.name,
                "timestamp": data.get("timestamp", "unknown"),
                "label": data.get("label", ""),
            })
        except Exception:
            snapshots.append({"file": f.name, "error": "could not parse"})
    return snapshots


# ---------------------------------------------------------------------------
# Compare two snapshots
# ---------------------------------------------------------------------------

def compare_run_snapshots(
    ws_root: Path,
    snapshot_a: Optional[str] = None,
    snapshot_b: Optional[str] = None,
) -> str:
    """Compare two run snapshots and return a markdown diff report.

    If snapshot_a/b are not specified, compares the two most recent snapshots.
    """
    snap_dir = _snapshot_dir(ws_root)
    files = sorted(snap_dir.glob("*.json"), reverse=True)

    if len(files) < 2 and (snapshot_a is None or snapshot_b is None):
        return ("## Run Comparison\n\n"
                "Need at least 2 snapshots for comparison. "
                f"Found {len(files)} snapshot(s).\n"
                "Use `save_snapshot` after each model run to enable comparison.")

    # Resolve file references
    def _load(ref: Optional[str], idx: int) -> Dict[str, Any]:
        if ref:
            path = snap_dir / ref
            if not path.exists():
                # Try matching by prefix
                matches = list(snap_dir.glob(f"{ref}*"))
                if matches:
                    path = matches[0]
                else:
                    return {"error": f"Snapshot not found: {ref}"}
        else:
            path = files[idx] if idx < len(files) else None
            if path is None:
                return {"error": "Not enough snapshots"}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"error": str(e)}

    a = _load(snapshot_a, 1)  # older (second most recent)
    b = _load(snapshot_b, 0)  # newer (most recent)

    if "error" in a or "error" in b:
        errors = []
        if "error" in a:
            errors.append(f"Snapshot A: {a['error']}")
        if "error" in b:
            errors.append(f"Snapshot B: {b['error']}")
        return "## Run Comparison\n\n" + "\n".join(errors)

    lines: List[str] = ["## Run Comparison"]
    lines.append(f"\n**Run A:** {a.get('label', '?')} ({a.get('timestamp', '?')})")
    lines.append(f"**Run B:** {b.get('label', '?')} ({b.get('timestamp', '?')})\n")

    # --- Head comparison ---
    hs_a = a.get("head_stats", {})
    hs_b = b.get("head_stats", {})

    if hs_a.get("layers") and hs_b.get("layers"):
        lines.append("### Head Statistics Comparison\n")
        lines.append("| Layer | Mean A | Mean B | Δ Mean | Dry A | Dry B | Δ Dry |")
        lines.append("|---|---|---|---|---|---|---|")

        layers_a = {l["layer"]: l for l in hs_a["layers"]}
        layers_b = {l["layer"]: l for l in hs_b["layers"]}
        all_layers = sorted(set(layers_a.keys()) | set(layers_b.keys()))

        for lay in all_layers[:20]:
            la = layers_a.get(lay, {})
            lb = layers_b.get(lay, {})
            mean_a = la.get("mean")
            mean_b = lb.get("mean")
            dry_a = la.get("dry_cells", 0)
            dry_b = lb.get("dry_cells", 0)

            mean_a_str = f"{mean_a:.4g}" if mean_a is not None else "—"
            mean_b_str = f"{mean_b:.4g}" if mean_b is not None else "—"
            if mean_a is not None and mean_b is not None:
                d_mean = f"{mean_b - mean_a:+.4g}"
            else:
                d_mean = "—"
            d_dry = f"{dry_b - dry_a:+d}" if dry_a is not None and dry_b is not None else "—"

            lines.append(
                f"| {lay} | {mean_a_str} | {mean_b_str} | {d_mean} | "
                f"{dry_a} | {dry_b} | {d_dry} |"
            )

        # Total dry cell comparison
        td_a = hs_a.get("total_dry_cells", 0)
        td_b = hs_b.get("total_dry_cells", 0)
        lines.append(f"\n**Total dry cells:** A={td_a}, B={td_b}, Δ={td_b - td_a:+d}")
        if td_b > td_a:
            lines.append("  - Dry cells **increased** — model may be less stable")
        elif td_b < td_a:
            lines.append("  - Dry cells **decreased** — improvement")

    # --- Budget comparison ---
    bgt_a = a.get("budget", {})
    bgt_b = b.get("budget", {})

    if bgt_a and bgt_b and "error" not in bgt_a and "error" not in bgt_b:
        lines.append("\n### Budget Comparison\n")
        lines.append("| Metric | Run A | Run B | Δ |")
        lines.append("|---|---|---|---|")

        for key, label in [
            ("total_in", "Total IN"),
            ("total_out", "Total OUT"),
            ("percent_discrepancy", "% Discrepancy"),
        ]:
            va = bgt_a.get(key)
            vb = bgt_b.get(key)
            if va is not None and vb is not None:
                d = vb - va
                lines.append(f"| {label} | {va:.4g} | {vb:.4g} | {d:+.4g} |")

    # --- Convergence comparison ---
    conv_a = a.get("convergence", {})
    conv_b = b.get("convergence", {})

    if conv_a and conv_b and "error" not in conv_a and "error" not in conv_b:
        lines.append("\n### Convergence Comparison\n")
        fa = conv_a.get("failure_count", 0)
        fb = conv_b.get("failure_count", 0)
        lines.append(f"- Convergence failures: A={fa}, B={fb}, Δ={fb - fa:+d}")
        if fb > fa:
            lines.append("  - **More failures** — solver performance degraded")
        elif fb < fa:
            lines.append("  - **Fewer failures** — improvement")
        elif fa == 0 and fb == 0:
            lines.append("  - Both runs converged fully")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# QA check wrappers (called via run_qa_check dispatcher)
# ---------------------------------------------------------------------------

def check_save_snapshot(ws_root: Path, **kwargs) -> str:
    """QA check wrapper: save a snapshot and return a summary."""
    label = kwargs.get("label", None)
    snap = save_run_snapshot(ws_root, label=label)

    lines: List[str] = ["## Run Snapshot Saved"]
    lines.append(f"\n**Timestamp:** {snap.get('timestamp', 'unknown')}")
    if snap.get("saved_to"):
        lines.append(f"**Saved to:** `{Path(snap['saved_to']).name}`")

    hs = snap.get("head_stats", {})
    if hs.get("layers"):
        lines.append(f"\n**Head data:** {hs.get('n_layers', '?')} layers, "
                      f"{hs.get('n_timesteps', '?')} timesteps, "
                      f"{hs.get('total_dry_cells', 0)} total dry cells")

    bgt = snap.get("budget", {})
    if bgt.get("percent_discrepancy") is not None:
        lines.append(f"**Budget discrepancy:** {bgt['percent_discrepancy']:.4f}%")

    conv = snap.get("convergence", {})
    if conv.get("failure_count") is not None:
        status = "converged" if conv["converged"] else f"{conv['failure_count']} failures"
        lines.append(f"**Convergence:** {status}")

    # List available snapshots
    snaps = list_run_snapshots(ws_root)
    lines.append(f"\n**Total snapshots available:** {len(snaps)}")
    if len(snaps) >= 2:
        lines.append("You can now run `compare_runs` to compare this snapshot with previous ones.")

    return "\n".join(lines)


def check_compare_runs(ws_root: Path, **kwargs) -> str:
    """QA check wrapper: compare two run snapshots."""
    snapshot_a = kwargs.get("snapshot_a", None)
    snapshot_b = kwargs.get("snapshot_b", None)
    return compare_run_snapshots(ws_root, snapshot_a=snapshot_a, snapshot_b=snapshot_b)
