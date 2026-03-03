"""MODFLOW 6 concrete simulator adapter.

Thin delegation wrapper — each method delegates to existing MF6 modules.
No logic is rewritten; the adapter just wires them behind the
``SimulatorAdapter`` interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from gw.simulators.base import (
    FileTypeInfo,
    GridInfo,
    PackageArrayInfo,
    PackagePropertyInfo,
    QACheck,
    SimulatorAdapter,
    SimulatorInfo,
)


class MF6Adapter(SimulatorAdapter):
    """Adapter for MODFLOW 6 models."""

    # ── Identity ──────────────────────────────────────────────────────────

    def info(self) -> SimulatorInfo:
        return SimulatorInfo(
            name="mf6",
            display_name="MODFLOW 6",
        )

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(self, ws_root: Path) -> float:
        # mfsim.nam is the definitive MF6 signature
        if (ws_root / "mfsim.nam").exists():
            return 1.0

        # Check for MF6-specific file extensions
        mf6_exts = {".dis", ".disv", ".disu", ".npf", ".ic", ".sto", ".ims", ".tdis"}
        found = 0
        try:
            for p in ws_root.iterdir():
                if p.is_file() and p.suffix.lower() in mf6_exts:
                    found += 1
        except OSError:
            pass
        if found >= 2:
            return 0.8

        # Check for .nam files with MF6-style BEGIN/END blocks
        for nam in ws_root.glob("*.nam"):
            try:
                txt = nam.read_text(errors="ignore")[:2000]
                if "BEGIN" in txt.upper() and "END" in txt.upper():
                    return 0.6
            except OSError:
                pass

        return 0.0

    # ── Model loading ─────────────────────────────────────────────────────

    def build_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        from gw.api.model_snapshot import build_model_snapshot
        return build_model_snapshot(ws_root)

    def build_model_brief(self, snapshot: Dict[str, Any]) -> str:
        from gw.api.model_snapshot import build_model_brief
        return build_model_brief(snapshot)

    # ── Stress I/O ────────────────────────────────────────────────────────

    def read_stress_package(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        value_cols: int = 1,
        keep_aux: bool = True,
    ) -> Any:
        from gw.mf6.io import parse_stress_package
        return parse_stress_package(
            ws_root / rel_path, value_cols=value_cols, keep_aux=keep_aux,
        )

    def read_time_discretization(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        from gw.mf6.io import read_tdis_times
        path = ws_root / (rel_path or "mfsim.tdis")
        return read_tdis_times(path)

    def read_namefile(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        from gw.mf6.io import read_nam
        path = ws_root / (rel_path or "mfsim.nam")
        return read_nam(path)

    # ── Binary output ─────────────────────────────────────────────────────

    def probe_head_file(self, ws_root: Path, rel_path: str) -> Dict[str, Any]:
        from gw.api.output_probes import probe_hds
        return probe_hds(ws_root, rel_path)

    def probe_budget_file(self, ws_root: Path, rel_path: str) -> Dict[str, Any]:
        from gw.api.output_probes import probe_cbc
        return probe_cbc(ws_root, rel_path)

    def extract_head_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_times: int = 5,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        from gw.api.output_probes import extract_hds_data
        return extract_hds_data(ws_root, rel_path, max_times=max_times, max_chars=max_chars)

    def extract_budget_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_records: int = 8,
        max_chars: int = 30_000,
    ) -> Dict[str, Any]:
        from gw.api.output_probes import extract_cbc_data
        return extract_cbc_data(ws_root, rel_path, max_records=max_records, max_chars=max_chars)

    def extract_head_timeseries(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        cells: list,
        max_chars: int = 50_000,
    ) -> Dict[str, Any]:
        from gw.api.output_probes import extract_hds_timeseries
        return extract_hds_timeseries(ws_root, rel_path, cells=cells, max_chars=max_chars)

    def extract_budget_cells(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        component: str,
        max_cells: int = 50,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        from gw.api.output_probes import extract_cbc_cell_flows
        return extract_cbc_cell_flows(
            ws_root, rel_path, component=component,
            max_cells=max_cells, max_chars=max_chars,
        )

    # ── Model execution ───────────────────────────────────────────────────

    def run_model(
        self,
        ws_root: Path,
        *,
        exe_path: Optional[str] = None,
        timeout_sec: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        from gw.run.mf6_runner import run_mf6
        return run_mf6(
            workspace=str(ws_root),
            mf6_path=exe_path,
            timeout_sec=timeout_sec,
        )

    def find_executable(self, exe_path: Optional[str] = None) -> str:
        from gw.run.mf6_runner import find_mf6
        return find_mf6(exe_path)

    # ── QA diagnostics ────────────────────────────────────────────────────

    def available_qa_checks(self) -> List[QACheck]:
        return [
            QACheck("mass_balance",
                    "Parse listing file for volumetric budget, compute % discrepancy per stress period"),
            QACheck("dry_cells",
                    "Count cells with dry/inactive heads per layer and timestep"),
            QACheck("convergence",
                    "Parse listing file for solver iteration counts, convergence failures, and solver warnings"),
            QACheck("pumping_summary",
                    "Analyze WEL package pumping rates by stress period, detect anomalous jumps"),
            QACheck("budget_timeseries",
                    "Extract IN/OUT per budget term across all timesteps"),
            QACheck("head_gradient",
                    "Compute cell-to-cell gradients per layer, flag extreme values"),
            QACheck("property_check",
                    "Check K/SS/SY ranges for unreasonable values and layer inversions"),
            QACheck("observation_comparison",
                    "Analyze MF6 observation output CSV files, compute per-obs statistics, and compare with observed/target data (RMSE, MAE, R², bias) if available"),
            QACheck("listing_budget_detail",
                    "Deep listing file parse: per-package IN/OUT budget tables, solver warnings, dry-cell messages, timestep reductions"),
            QACheck("property_zones",
                    "Spatial K zone analysis per layer: distinct value clusters, contrast ratios, coefficient of variation, adjacent-cell contrast flags"),
            QACheck("advanced_packages",
                    "Parse and summarize complex packages (SFR, LAK, MAW, UZF, CSUB, EVT): reach counts, widths, connections, screens, etc."),
            QACheck("save_snapshot",
                    "Save a lightweight snapshot of current model outputs (heads, budget, convergence) for later comparison"),
            QACheck("compare_runs",
                    "Compare two run snapshots side-by-side: head changes, budget differences, convergence changes"),
        ]

    def run_qa_check(self, ws_root: Path, check_name: str, **kwargs: Any) -> str:
        from gw.api.qa_diagnostics import run_qa_check as _run
        return _run(str(ws_root), check_name, **kwargs)

    # ── File-type knowledge ───────────────────────────────────────────────

    def file_type_knowledge(self) -> Dict[str, FileTypeInfo]:
        from gw.llm.mf6_filetype_knowledge import _EXT_KB
        # Convert from the existing FileTypeInfo to our base dataclass.
        # They have the same fields, but different class identity.  If the
        # existing class IS the same (re-exported), just return directly.
        return {
            ext: FileTypeInfo(kind=info.kind, purpose=info.purpose,
                              what_to_look_for=info.what_to_look_for)
            for ext, info in _EXT_KB.items()
        }

    def package_properties(self) -> Dict[str, PackagePropertyInfo]:
        from gw.llm.mf6_filetype_knowledge import PACKAGE_PROPERTIES
        return {
            pkg: PackagePropertyInfo(
                file_ext=info.file_ext,
                block=info.block,
                arrays={
                    name: PackageArrayInfo(label=arr.label, per_layer=arr.per_layer)
                    for name, arr in info.arrays.items()
                },
            )
            for pkg, info in PACKAGE_PROPERTIES.items()
        }

    def file_extensions(self) -> Set[str]:
        from gw.llm.mf6_filetype_knowledge import _EXT_KB
        return set(_EXT_KB.keys()) | {
            # Extensions used in file whitelisting but not in the KB
            ".tdis", ".ims", ".sfr", ".uzf", ".lak", ".evt",
            ".maw", ".csub", ".gwt", ".grb",
        }

    # ── LLM knowledge ─────────────────────────────────────────────────────

    def system_prompt_fragment(self) -> str:
        from gw.simulators.mf6.knowledge import mf6_system_prompt_fragment
        return mf6_system_prompt_fragment()

    def tool_descriptions(self) -> Dict[str, str]:
        from gw.simulators.mf6.knowledge import mf6_tool_description_overrides
        return mf6_tool_description_overrides()

    def file_mention_pattern(self) -> str:
        from gw.simulators.mf6.knowledge import mf6_file_mention_regex
        return mf6_file_mention_regex()

    # ── Grid ──────────────────────────────────────────────────────────────

    def get_grid_info(self, ws_root: Path) -> Optional[GridInfo]:
        try:
            from gw.api.viz import load_grid
            dis = load_grid(ws_root)
            if dis is None:
                return None
            return GridInfo(
                grid_type=getattr(dis, "grid_type", "dis"),
                nlay=getattr(dis, "nlay", 0),
                ncpl=getattr(dis, "ncpl", 0),
                total_cells=getattr(dis, "nlay", 0) * getattr(dis, "ncpl", 0),
                nrow=getattr(dis, "nrow", None),
                ncol=getattr(dis, "ncol", None),
                xorigin=getattr(dis, "xorigin", 0.0),
                yorigin=getattr(dis, "yorigin", 0.0),
                angrot=getattr(dis, "angrot", 0.0),
            )
        except Exception:
            return None

    # ── FloPy bridge ──────────────────────────────────────────────────────

    def get_simulation(self, ws_root: Path) -> Tuple[Optional[Any], Optional[str]]:
        from gw.mf6.flopy_bridge import get_simulation
        return get_simulation(ws_root)

    def clear_simulation_cache(self, ws_root: Optional[Path] = None) -> None:
        from gw.mf6.flopy_bridge import clear_cache
        clear_cache(ws_root)
