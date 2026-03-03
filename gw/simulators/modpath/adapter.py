"""MODPATH 7 concrete simulator adapter.

Implements the ``SimulatorAdapter`` interface for MODPATH 7 particle
tracking models.  MODPATH is a post-processor that reads MODFLOW
head/budget output and traces particle paths through the flow field.

Key differences from flow-model adapters (MF6, MF2005):
- MODPATH does NOT produce head or budget files.
- It produces endpoint (.mpend), pathline (.mppth), and timeseries (.mpts) files.
- FloPy's ``Modpath7`` does NOT have a ``load()`` method.
- Grid information comes from the linked MODFLOW model, not from MODPATH.
- Stress packages do not exist in MODPATH.
- Time discretisation is inherited from MODFLOW.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from gw.simulators.base import (
    FileTypeInfo,
    GridInfo,
    OutputCapability,
    PackageArrayInfo,
    PackagePropertyInfo,
    QACheck,
    SimulatorAdapter,
    SimulatorInfo,
)

logger = logging.getLogger(__name__)


class ModpathAdapter(SimulatorAdapter):
    """Adapter for MODPATH 7 particle tracking models."""

    # ── Identity ──────────────────────────────────────────────────────────

    def info(self) -> SimulatorInfo:
        return SimulatorInfo(
            name="modpath",
            display_name="MODPATH 7",
        )

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(self, ws_root: Path) -> float:
        """Detect MODPATH models.

        Detection strategy:
        - .mpsim or .mpnam file present   -> 0.95  (definitive input files)
        - Only .mpend or .mppth present   -> 0.70  (output files only)
        - No MODPATH files                -> 0.0
        """
        has_mpsim = bool(list(ws_root.glob("*.mpsim")))
        has_mpnam = bool(list(ws_root.glob("*.mpnam")))

        if has_mpsim or has_mpnam:
            return 0.95

        # Check for MODPATH output files
        has_mpend = bool(list(ws_root.glob("*.mpend")))
        has_mppth = bool(list(ws_root.glob("*.mppth")))
        has_mpts = bool(list(ws_root.glob("*.mpts")))
        has_mpbas = bool(list(ws_root.glob("*.mpbas")))

        if has_mpend or has_mppth or has_mpts:
            return 0.70

        if has_mpbas:
            return 0.50

        return 0.0

    # ── Model loading ─────────────────────────────────────────────────────

    def build_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        """Build model snapshot by parsing MODPATH input and output files.

        Since MODPATH has no FloPy load(), we parse the .mpsim and .mpnam
        files directly and summarise any output files present.
        """
        from gw.simulators.modpath.io import (
            find_linked_flow_model,
            parse_mpnam,
            parse_mpsim,
            read_endpoints,
            read_pathlines,
        )

        snapshot: Dict[str, Any] = {
            "ok": True,
            "simulator": "modpath",
            "workspace_root": str(ws_root),
        }

        # ── Parse .mpsim ──────────────────────────────────────────────────
        mpsim_files = sorted(ws_root.glob("*.mpsim"))
        if mpsim_files:
            mpsim_data = parse_mpsim(mpsim_files[0])
            snapshot["mpsim_file"] = mpsim_files[0].name
            snapshot["simulation_type"] = mpsim_data.get("simulation_type", "unknown")
            snapshot["tracking_direction"] = mpsim_data.get("tracking_direction", "unknown")
            snapshot["weak_sink_option"] = mpsim_data.get("weak_sink_option")
            snapshot["weak_source_option"] = mpsim_data.get("weak_source_option")
            snapshot["reference_time_option"] = mpsim_data.get("reference_time_option")
            snapshot["stop_time_option"] = mpsim_data.get("stop_time_option")
            snapshot["option_flags"] = mpsim_data.get("option_flags", {})
        else:
            snapshot["mpsim_file"] = None
            snapshot["simulation_type"] = "unknown"
            snapshot["tracking_direction"] = "unknown"

        # ── Parse .mpnam ──────────────────────────────────────────────────
        mpnam_files = sorted(ws_root.glob("*.mpnam"))
        if mpnam_files:
            nam_data = parse_mpnam(mpnam_files[0])
            snapshot["mpnam_file"] = mpnam_files[0].name
            snapshot["linked_head_file"] = nam_data.get("head_file")
            snapshot["linked_budget_file"] = nam_data.get("budget_file")
            snapshot["linked_dis_file"] = nam_data.get("dis_file")
            snapshot["linked_dis_type"] = nam_data.get("dis_type")
            snapshot["linked_grb_file"] = nam_data.get("grb_file")
        else:
            snapshot["mpnam_file"] = None

        # ── Linked flow model ────────────────────────────────────────────
        linked = find_linked_flow_model(ws_root)
        snapshot["linked_flow_model"] = linked

        # ── MPBAS file ───────────────────────────────────────────────────
        mpbas_files = sorted(ws_root.glob("*.mpbas"))
        if mpbas_files:
            snapshot["mpbas_file"] = mpbas_files[0].name
            # Try to extract porosity info from MPBAS header
            try:
                text = mpbas_files[0].read_text(
                    encoding="utf-8", errors="replace"
                )[:3000]
                snapshot["mpbas_preview"] = text[:500]
            except Exception:
                pass

        # ── Output files present ─────────────────────────────────────────
        outputs_present: Dict[str, Any] = {}

        for ext, key in [
            (".mpend", "endpoints"),
            (".mppth", "pathlines"),
            (".mpts", "timeseries"),
            (".mplst", "listing"),
        ]:
            found = sorted(ws_root.glob(f"*{ext}"))
            if found:
                outputs_present[key] = {
                    "file": found[0].name,
                    "size_bytes": found[0].stat().st_size,
                }

        snapshot["outputs_present"] = outputs_present

        # ── Summarise endpoint/pathline data if present ──────────────────
        if "endpoints" in outputs_present:
            try:
                ep_path = ws_root / outputs_present["endpoints"]["file"]
                ep_summary = read_endpoints(ep_path)
                if ep_summary.get("ok"):
                    snapshot["endpoint_summary"] = {
                        "particle_count": ep_summary.get("particle_count", 0),
                        "termination_status": ep_summary.get("termination_status", {}),
                        "travel_time_stats": ep_summary.get("travel_time_stats"),
                    }
            except Exception as exc:
                logger.debug("Could not summarise endpoints: %s", exc)

        if "pathlines" in outputs_present:
            try:
                pl_path = ws_root / outputs_present["pathlines"]["file"]
                pl_summary = read_pathlines(pl_path)
                if pl_summary.get("ok"):
                    snapshot["pathline_summary"] = {
                        "particle_count": pl_summary.get("particle_count", 0),
                        "total_points": pl_summary.get("total_pathline_points", 0),
                        "travel_time_stats": pl_summary.get("travel_time_stats"),
                        "path_length_stats": pl_summary.get("path_length_stats"),
                    }
            except Exception as exc:
                logger.debug("Could not summarise pathlines: %s", exc)

        return snapshot

    def build_model_brief(self, snapshot: Dict[str, Any]) -> str:
        """Build a compact text summary for LLM prompt injection."""
        parts: List[str] = []

        parts.append("MODPATH 7")

        # Tracking info
        direction = snapshot.get("tracking_direction", "unknown")
        sim_type = snapshot.get("simulation_type", "unknown")
        parts.append(f"{direction} {sim_type} simulation")

        # Linked flow model
        linked = snapshot.get("linked_flow_model")
        if linked:
            parts.append(f"linked to {linked}")

        # Output files
        outputs = snapshot.get("outputs_present", {})
        if outputs:
            out_types = list(outputs.keys())
            parts.append(f"outputs: {', '.join(out_types)}")

        # Particle count
        ep_sum = snapshot.get("endpoint_summary", {})
        if ep_sum:
            n = ep_sum.get("particle_count", 0)
            parts.append(f"{n:,} particles")

        pl_sum = snapshot.get("pathline_summary", {})
        if pl_sum and not ep_sum:
            n = pl_sum.get("particle_count", 0)
            parts.append(f"{n:,} particles")

        return " | ".join(parts)

    # ── Stress I/O ────────────────────────────────────────────────────────
    # MODPATH has no stress packages.

    def read_stress_package(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        value_cols: int = 1,
        keep_aux: bool = True,
    ) -> Any:
        """MODPATH has no stress packages — return empty DataFrame."""
        import pandas as pd
        return pd.DataFrame()

    def read_time_discretization(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        """MODPATH uses MODFLOW's time discretisation — return empty DataFrame.

        To read time discretisation, use the linked MODFLOW model's adapter.
        """
        import pandas as pd
        return pd.DataFrame(
            columns=["per", "perlen", "nstp", "tsmult", "t_start", "t_end", "t_mid"]
        )

    def read_namefile(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        """Read the MODPATH name file (.mpnam) as a DataFrame."""
        from gw.simulators.modpath.io import parse_mpnam

        if rel_path:
            path = ws_root / rel_path
        else:
            mpnam_files = sorted(ws_root.glob("*.mpnam"))
            path = mpnam_files[0] if mpnam_files else ws_root / "model.mpnam"

        import pandas as pd

        nam_data = parse_mpnam(path)
        if not nam_data:
            return pd.DataFrame(columns=["ftype", "fname"])

        rows = [{"ftype": k.replace("_file", "").upper(), "fname": v}
                for k, v in nam_data.items()]
        return pd.DataFrame(rows, columns=["ftype", "fname"])

    # ── Binary output ─────────────────────────────────────────────────────
    # MODPATH does NOT produce head or budget files.

    def probe_head_file(
        self, ws_root: Path, rel_path: str
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": (
                "MODPATH does not produce head files. "
                "Head output comes from the linked MODFLOW model. "
                "Check the .mpnam file for the head file reference."
            ),
        }

    def probe_budget_file(
        self, ws_root: Path, rel_path: str
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": (
                "MODPATH does not produce budget files. "
                "Budget output comes from the linked MODFLOW model. "
                "Check the .mpnam file for the budget file reference."
            ),
        }

    def extract_head_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_times: int = 5,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": "MODPATH does not produce head data. Use the linked MODFLOW model.",
        }

    def extract_budget_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_records: int = 8,
        max_chars: int = 30_000,
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": "MODPATH does not produce budget data. Use the linked MODFLOW model.",
        }

    def extract_head_timeseries(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        cells: list,
        max_chars: int = 50_000,
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": "MODPATH does not produce head data. Use the linked MODFLOW model.",
        }

    def extract_budget_cells(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        component: str,
        max_cells: int = 50,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": "MODPATH does not produce budget data. Use the linked MODFLOW model.",
        }

    # ── Particle tracking output (MODPATH-specific) ──────────────────────

    def probe_pathline_file(
        self, ws_root: Path, rel_path: str
    ) -> Dict[str, Any]:
        """Probe a MODPATH pathline or endpoint file for metadata."""
        full_path = ws_root / rel_path
        ext = full_path.suffix.lower()

        if ext == ".mpend":
            from gw.simulators.modpath.io import read_endpoints
            return read_endpoints(full_path)
        elif ext == ".mppth":
            from gw.simulators.modpath.io import read_pathlines
            return read_pathlines(full_path)
        elif ext == ".mpts":
            from gw.simulators.modpath.io import read_timeseries
            return read_timeseries(full_path)
        else:
            return {
                "ok": False,
                "error": (
                    f"Unrecognised particle output extension: {ext}. "
                    f"Expected .mpend, .mppth, or .mpts."
                ),
            }

    def extract_endpoint_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        """Extract endpoint data with termination analysis."""
        from gw.simulators.modpath.io import read_endpoints

        full_path = ws_root / rel_path
        result = read_endpoints(full_path)

        if not result.get("ok"):
            return result

        # Build a summary text within character budget
        lines: List[str] = []
        lines.append(f"## Endpoint Summary: {full_path.name}")
        lines.append(f"Particles: {result.get('particle_count', 0):,}")

        term = result.get("termination_status", {})
        if term:
            lines.append("\n### Termination Status")
            for status, count in sorted(term.items(), key=lambda x: -x[1]):
                pct = 100.0 * count / result["particle_count"] if result["particle_count"] else 0
                lines.append(f"- {status}: {count:,} ({pct:.1f}%)")

        tt = result.get("travel_time_stats")
        if tt:
            lines.append("\n### Travel Time Statistics")
            lines.append(f"- Min: {tt['min']:.4g}")
            lines.append(f"- Max: {tt['max']:.4g}")
            lines.append(f"- Mean: {tt['mean']:.4g}")
            lines.append(f"- Median: {tt['median']:.4g}")
            lines.append(f"- Std: {tt['std']:.4g}")
            lines.append(f"- P10: {tt['p10']:.4g}")
            lines.append(f"- P90: {tt['p90']:.4g}")

        result["summary_text"] = "\n".join(lines)[:max_chars]
        return result

    def extract_pathline_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_chars: int = 50_000,
    ) -> Dict[str, Any]:
        """Extract pathline data with path statistics."""
        from gw.simulators.modpath.io import read_pathlines

        full_path = ws_root / rel_path
        result = read_pathlines(full_path)

        if not result.get("ok"):
            return result

        # Build a summary text
        lines: List[str] = []
        lines.append(f"## Pathline Summary: {full_path.name}")
        lines.append(f"Particles: {result.get('particle_count', 0):,}")
        lines.append(f"Total pathline points: {result.get('total_pathline_points', 0):,}")

        tr = result.get("time_range")
        if tr:
            lines.append(f"\n### Time Range")
            lines.append(f"- Min: {tr['min']:.4g}")
            lines.append(f"- Max: {tr['max']:.4g}")

        tt = result.get("travel_time_stats")
        if tt:
            lines.append("\n### Travel Time Statistics")
            lines.append(f"- Min: {tt['min']:.4g}")
            lines.append(f"- Max: {tt['max']:.4g}")
            lines.append(f"- Mean: {tt['mean']:.4g}")
            lines.append(f"- Median: {tt['median']:.4g}")

        pl = result.get("path_length_stats")
        if pl:
            lines.append("\n### Path Length Statistics")
            lines.append(f"- Min: {pl['min']:.4g}")
            lines.append(f"- Max: {pl['max']:.4g}")
            lines.append(f"- Mean: {pl['mean']:.4g}")
            lines.append(f"- Median: {pl['median']:.4g}")

        result["summary_text"] = "\n".join(lines)[:max_chars]
        return result

    # ── Model execution ───────────────────────────────────────────────────

    def run_model(
        self,
        ws_root: Path,
        *,
        exe_path: Optional[str] = None,
        timeout_sec: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        from gw.run.modpath_runner import run_modpath
        return run_modpath(
            workspace=str(ws_root),
            mp_path=exe_path,
            timeout_sec=timeout_sec,
        )

    def find_executable(self, exe_path: Optional[str] = None) -> str:
        from gw.run.modpath_runner import find_modpath
        return find_modpath(exe_path)

    # ── QA diagnostics ────────────────────────────────────────────────────

    def available_qa_checks(self) -> List[QACheck]:
        """Return QA checks available for MODPATH models.

        These are particle-tracking-specific checks, NOT flow-model checks.
        """
        return [
            QACheck(
                "particle_termination",
                "Analyze particle termination status: where and why particles "
                "stop. Flags high stranded or inactive-cell fractions.",
            ),
            QACheck(
                "travel_time_distribution",
                "Statistical analysis of particle travel times. Flags "
                "unreasonably short (<1 day) or long (>1000 yr) travel times.",
            ),
            QACheck(
                "capture_zones",
                "Zone-based capture analysis: which termination zones receive "
                "the most particles, zone distribution by particle group.",
            ),
            QACheck(
                "dry_cell_encounters",
                "Check if particles encounter dry or inactive cells during "
                "tracking. High counts indicate MODFLOW model boundary issues.",
            ),
            QACheck(
                "mass_balance_prereq",
                "Verify the underlying MODFLOW model has acceptable mass "
                "balance (<1% discrepancy). Poor mass balance invalidates "
                "particle tracking results.",
            ),
        ]

    def run_qa_check(
        self, ws_root: Path, check_name: str, **kwargs: Any
    ) -> str:
        """Run a MODPATH-specific QA check.

        Implements checks directly since they are particle-tracking-specific
        and not covered by the generic ``qa_diagnostics`` module.
        """
        handlers = {
            "particle_termination": self._qa_particle_termination,
            "travel_time_distribution": self._qa_travel_time_distribution,
            "capture_zones": self._qa_capture_zones,
            "dry_cell_encounters": self._qa_dry_cell_encounters,
            "mass_balance_prereq": self._qa_mass_balance_prereq,
        }

        handler = handlers.get(check_name)
        if handler is None:
            available = ", ".join(handlers.keys())
            return (
                f"## Unknown QA Check: `{check_name}`\n\n"
                f"Available MODPATH checks: {available}"
            )

        try:
            return handler(ws_root, **kwargs)
        except Exception as exc:
            return (
                f"## QA Check Failed: `{check_name}`\n\n"
                f"Error: {type(exc).__name__}: {exc}"
            )

    def _qa_particle_termination(self, ws_root: Path, **kwargs: Any) -> str:
        """Analyze particle termination status."""
        from gw.simulators.modpath.io import read_endpoints

        mpend_files = sorted(ws_root.glob("*.mpend"))
        if not mpend_files:
            return (
                "## Particle Termination Check\n\n"
                "No endpoint file (.mpend) found in workspace. "
                "Run MODPATH first or check simulation type."
            )

        result = read_endpoints(mpend_files[0])
        if not result.get("ok"):
            return (
                f"## Particle Termination Check\n\n"
                f"Could not read endpoint file: {result.get('error', 'unknown error')}"
            )

        lines: List[str] = [
            "## Particle Termination Analysis",
            f"\n**File:** `{mpend_files[0].name}`",
            f"**Total particles:** {result['particle_count']:,}",
        ]

        term = result.get("termination_status", {})
        if term:
            lines.append("\n### Termination Breakdown\n")
            lines.append("| Status | Count | Percentage |")
            lines.append("|--------|------:|----------:|")
            total = result["particle_count"]
            for status, count in sorted(term.items(), key=lambda x: -x[1]):
                pct = 100.0 * count / total if total else 0
                lines.append(f"| {status} | {count:,} | {pct:.1f}% |")

            # Warnings
            stranded = term.get("stranded", 0)
            inactive = term.get("entered_inactive_cell", 0)
            still_active = term.get("still_active", 0)

            warnings: List[str] = []
            if total > 0:
                if stranded / total > 0.05:
                    warnings.append(
                        f"- **High stranded fraction** ({stranded}/{total} = "
                        f"{100*stranded/total:.1f}%): Particles stuck at stagnation "
                        f"points. Consider checking flow field near stagnation zones."
                    )
                if inactive / total > 0.02:
                    warnings.append(
                        f"- **Particles entering inactive cells** ({inactive}/{total} = "
                        f"{100*inactive/total:.1f}%): Check IBOUND/IDOMAIN in the "
                        f"MODFLOW model."
                    )
                if still_active / total > 0.01:
                    warnings.append(
                        f"- **Particles still active** ({still_active}/{total} = "
                        f"{100*still_active/total:.1f}%): Tracking may have been "
                        f"cut short. Consider increasing stop time."
                    )

            if warnings:
                lines.append("\n### Warnings\n")
                lines.extend(warnings)
            else:
                lines.append("\n### Assessment\n")
                lines.append("Particle termination looks reasonable. Most particles "
                             "terminated normally at model boundaries.")

        return "\n".join(lines)

    def _qa_travel_time_distribution(self, ws_root: Path, **kwargs: Any) -> str:
        """Analyze travel time distribution."""
        from gw.simulators.modpath.io import read_endpoints

        mpend_files = sorted(ws_root.glob("*.mpend"))
        if not mpend_files:
            return (
                "## Travel Time Distribution Check\n\n"
                "No endpoint file (.mpend) found. Cannot analyze travel times."
            )

        result = read_endpoints(mpend_files[0])
        if not result.get("ok"):
            return (
                f"## Travel Time Distribution Check\n\n"
                f"Could not read endpoint file: {result.get('error', 'unknown error')}"
            )

        tt = result.get("travel_time_stats")
        if not tt:
            return (
                "## Travel Time Distribution Check\n\n"
                "No travel time data found in endpoint file."
            )

        lines: List[str] = [
            "## Travel Time Distribution Analysis",
            f"\n**File:** `{mpend_files[0].name}`",
            f"**Particles:** {result['particle_count']:,}",
            "\n### Statistics\n",
            "| Statistic | Value |",
            "|-----------|------:|",
            f"| Minimum | {tt['min']:.4g} |",
            f"| Maximum | {tt['max']:.4g} |",
            f"| Mean | {tt['mean']:.4g} |",
            f"| Median | {tt['median']:.4g} |",
            f"| Std Dev | {tt['std']:.4g} |",
            f"| P10 | {tt['p10']:.4g} |",
            f"| P90 | {tt['p90']:.4g} |",
        ]

        # Warnings
        warnings: List[str] = []
        if tt["min"] < 1.0:
            warnings.append(
                f"- **Very short travel times** (min = {tt['min']:.4g}): "
                f"Some particles travel extremely fast. Check porosity values "
                f"and grid resolution near boundaries."
            )
        if tt["max"] > 365250:  # ~1000 years in days
            warnings.append(
                f"- **Very long travel times** (max = {tt['max']:.4g}): "
                f"Some particles take >1000 years. This may be expected for "
                f"deep flow paths or may indicate low-K zones trapping particles."
            )
        if tt["std"] / tt["mean"] > 2.0 and tt["mean"] > 0:
            warnings.append(
                f"- **High travel time variability** (CV = {tt['std']/tt['mean']:.2f}): "
                f"Wide spread in travel times suggests heterogeneous flow paths."
            )

        if warnings:
            lines.append("\n### Warnings\n")
            lines.extend(warnings)
        else:
            lines.append("\n### Assessment\n")
            lines.append("Travel time distribution appears reasonable.")

        return "\n".join(lines)

    def _qa_capture_zones(self, ws_root: Path, **kwargs: Any) -> str:
        """Analyze capture zones from endpoint data."""
        from gw.simulators.modpath.io import read_endpoints

        mpend_files = sorted(ws_root.glob("*.mpend"))
        if not mpend_files:
            return (
                "## Capture Zone Analysis\n\n"
                "No endpoint file (.mpend) found. Cannot analyze capture zones."
            )

        result = read_endpoints(mpend_files[0])
        if not result.get("ok"):
            return (
                f"## Capture Zone Analysis\n\n"
                f"Could not read endpoint file: {result.get('error', 'unknown error')}"
            )

        lines: List[str] = [
            "## Capture Zone Analysis",
            f"\n**File:** `{mpend_files[0].name}`",
            f"**Particles:** {result['particle_count']:,}",
        ]

        # Look for zone distribution data
        zone_data_found = False
        for key in sorted(result.keys()):
            if "zone" in key.lower() and "distribution" in key.lower():
                zone_data_found = True
                dist = result[key]
                lines.append(f"\n### {key.replace('_', ' ').title()}\n")
                lines.append("| Zone | Count | Percentage |")
                lines.append("|------|------:|----------:|")
                total = sum(dist.values())
                for zone, count in sorted(dist.items(),
                                          key=lambda x: -x[1]):
                    pct = 100.0 * count / total if total else 0
                    lines.append(f"| {zone} | {count:,} | {pct:.1f}% |")

        if not zone_data_found:
            # Fall back to termination status as a proxy
            term = result.get("termination_status", {})
            if term:
                lines.append("\n### Termination-Based Capture\n")
                lines.append("(Zone data not available; using termination status)\n")
                lines.append("| Status | Count | Percentage |")
                lines.append("|--------|------:|----------:|")
                total = result["particle_count"]
                for status, count in sorted(term.items(), key=lambda x: -x[1]):
                    pct = 100.0 * count / total if total else 0
                    lines.append(f"| {status} | {count:,} | {pct:.1f}% |")
            else:
                lines.append("\nNo zone or termination data available.")

        return "\n".join(lines)

    def _qa_dry_cell_encounters(self, ws_root: Path, **kwargs: Any) -> str:
        """Check for particles encountering dry/inactive cells."""
        from gw.simulators.modpath.io import read_endpoints

        mpend_files = sorted(ws_root.glob("*.mpend"))
        if not mpend_files:
            return (
                "## Dry Cell Encounter Check\n\n"
                "No endpoint file (.mpend) found."
            )

        result = read_endpoints(mpend_files[0])
        if not result.get("ok"):
            return (
                f"## Dry Cell Encounter Check\n\n"
                f"Could not read endpoint file: {result.get('error', 'unknown error')}"
            )

        term = result.get("termination_status", {})
        total = result.get("particle_count", 0)

        inactive_count = term.get("entered_inactive_cell", 0)
        stranded_count = term.get("stranded", 0)

        lines: List[str] = [
            "## Dry Cell / Inactive Cell Encounter Analysis",
            f"\n**File:** `{mpend_files[0].name}`",
            f"**Total particles:** {total:,}",
            f"\n### Results\n",
            f"- Particles entering inactive cells: {inactive_count:,}",
            f"- Stranded particles (no flow path): {stranded_count:,}",
        ]

        if total > 0:
            inactive_pct = 100.0 * inactive_count / total
            stranded_pct = 100.0 * stranded_count / total
            lines.append(f"- Inactive cell fraction: {inactive_pct:.2f}%")
            lines.append(f"- Stranded fraction: {stranded_pct:.2f}%")

            if inactive_pct > 2.0:
                lines.append(
                    f"\n### Warning\n"
                    f"**{inactive_pct:.1f}% of particles entered inactive cells.** "
                    f"This suggests issues with the MODFLOW model's IBOUND/IDOMAIN "
                    f"array or dry cells in the flow solution. Review the MODFLOW "
                    f"model for dry cell warnings and consider adjusting boundary "
                    f"conditions or rewetting options."
                )
            elif inactive_pct > 0:
                lines.append(
                    f"\n### Note\n"
                    f"A small fraction ({inactive_pct:.2f}%) of particles entered "
                    f"inactive cells. This is generally acceptable but worth monitoring."
                )
            else:
                lines.append(
                    "\n### Assessment\n"
                    "No particles entered inactive cells. Flow field appears "
                    "well-defined for particle tracking."
                )
        else:
            lines.append("\nNo particles to analyze.")

        return "\n".join(lines)

    def _qa_mass_balance_prereq(self, ws_root: Path, **kwargs: Any) -> str:
        """Check MODFLOW mass balance as a prerequisite for particle tracking."""
        lines: List[str] = [
            "## Mass Balance Prerequisite Check",
            "\nMODPATH particle tracking is only reliable if the underlying "
            "MODFLOW model has good mass balance (<1% discrepancy).\n",
        ]

        # Try to find and parse the MODFLOW listing file
        lst_files = sorted(ws_root.glob("*.lst"))
        if not lst_files:
            lst_files = sorted(ws_root.glob("*.list"))

        if not lst_files:
            lines.append(
                "**No MODFLOW listing file found** in the workspace. "
                "Cannot verify mass balance. Ensure the MODFLOW model has "
                "been run successfully before running MODPATH."
            )
            return "\n".join(lines)

        # Try to use the generic mass balance QA check
        try:
            from gw.api.qa_diagnostics import run_qa_check as _run
            mb_report = _run(str(ws_root), "mass_balance")
            lines.append("### MODFLOW Mass Balance Report\n")
            lines.append(mb_report)

            # Add interpretation
            if "poor" in mb_report.lower() or "> 1%" in mb_report.lower():
                lines.append(
                    "\n### MODPATH Impact\n"
                    "**Poor mass balance in the MODFLOW model.** Particle tracking "
                    "results may be unreliable. Fix the MODFLOW model first:\n"
                    "- Check solver convergence settings\n"
                    "- Review boundary condition specifications\n"
                    "- Check for dry cells or dewatered layers\n"
                    "- Re-run MODFLOW and verify mass balance before MODPATH"
                )
            else:
                lines.append(
                    "\n### MODPATH Impact\n"
                    "Mass balance is acceptable. Particle tracking results "
                    "should be reliable from a flow-field perspective."
                )
        except Exception as exc:
            lines.append(
                f"Could not run mass balance check: {type(exc).__name__}: {exc}\n\n"
                f"Listing file found: `{lst_files[0].name}` "
                f"({lst_files[0].stat().st_size:,} bytes)"
            )

        return "\n".join(lines)

    # ── Output capabilities ──────────────────────────────────────────────

    def output_capabilities(self) -> OutputCapability:
        return OutputCapability(
            heads=False,
            budget=False,
            concentration=False,
            pathlines=True,
            endpoints=True,
        )

    # ── File-type knowledge ───────────────────────────────────────────────

    def file_type_knowledge(self) -> Dict[str, FileTypeInfo]:
        from gw.simulators.modpath.knowledge import _EXT_KB
        return dict(_EXT_KB)

    def package_properties(self) -> Dict[str, PackagePropertyInfo]:
        from gw.simulators.modpath.knowledge import PACKAGE_PROPERTIES
        return dict(PACKAGE_PROPERTIES)

    def file_extensions(self) -> Set[str]:
        from gw.simulators.modpath.knowledge import _EXT_KB
        return set(_EXT_KB.keys()) | {
            ".sloc", ".mpzon", ".mp7",
        }

    # ── LLM knowledge ─────────────────────────────────────────────────────

    def system_prompt_fragment(self) -> str:
        from gw.simulators.modpath.knowledge import modpath_system_prompt_fragment
        return modpath_system_prompt_fragment()

    def tool_descriptions(self) -> Dict[str, str]:
        from gw.simulators.modpath.knowledge import modpath_tool_description_overrides
        return modpath_tool_description_overrides()

    def file_mention_pattern(self) -> str:
        from gw.simulators.modpath.knowledge import modpath_file_mention_regex
        return modpath_file_mention_regex()

    # ── Grid ──────────────────────────────────────────────────────────────

    def get_grid_info(self, ws_root: Path) -> Optional[GridInfo]:
        """MODPATH uses the MODFLOW model's grid — return None.

        Grid information should be obtained from the linked MODFLOW model.
        """
        return None

    # ── FloPy bridge ──────────────────────────────────────────────────────

    def get_simulation(
        self, ws_root: Path
    ) -> Tuple[Optional[Any], Optional[str]]:
        """MODPATH has no FloPy load() method.

        FloPy's Modpath7 class can CREATE new models but cannot LOAD
        existing ones from files.
        """
        return None, (
            "MODPATH 7 has no FloPy load() method. "
            "Existing MODPATH models cannot be loaded via FloPy. "
            "Use the MODPATH adapter's direct file parsers instead."
        )

    def clear_simulation_cache(self, ws_root: Optional[Path] = None) -> None:
        """No simulation cache for MODPATH — no-op."""
        pass
