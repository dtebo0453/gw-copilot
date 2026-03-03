"""MODFLOW-USG concrete simulator adapter.

Implements the ``SimulatorAdapter`` interface for MODFLOW-USG models.
Supports both structured (DIS) and unstructured (DISU) grids.  Binary
output reading uses either standard FloPy HeadFile (structured) or
HeadUFile (unstructured), and CellBudgetFile for budgets (works for both).
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


class MFUSGAdapter(SimulatorAdapter):
    """Adapter for MODFLOW-USG models (structured DIS and unstructured DISU)."""

    # ── Identity ──────────────────────────────────────────────────────────

    def info(self) -> SimulatorInfo:
        return SimulatorInfo(
            name="mfusg",
            display_name="MODFLOW-USG",
        )

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(self, ws_root: Path) -> float:
        """Detect whether *ws_root* contains a MODFLOW-USG model.

        Strategy:
        - ``mfsim.nam`` exists -> 0.0 (that is MF6)
        - ``.nam`` references DISU, SMS, CLN, or GNC -> 0.95
        - ``.nam`` has MF2005-style format + ``.disu`` or ``.sms`` files -> 0.9
        - No USG indicators -> 0.0
        """
        # MF6 disqualifier
        if (ws_root / "mfsim.nam").exists():
            return 0.0

        # Look for a .nam file (not mfsim.nam)
        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return 0.0

        # USG-specific package types that distinguish from MF2005
        usg_pkg_types = {"DISU", "SMS", "CLN", "GNC", "BCT", "DDF"}

        # Check namefile contents for USG packages
        for nam in nam_files:
            try:
                txt = nam.read_text(errors="ignore")[:4000].upper()

                # MF6 block-based name files -> not USG
                if "BEGIN" in txt and "END" in txt:
                    return 0.0

                # Scan for USG-specific package types in the namefile
                found_usg_pkg = False
                for line in txt.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if parts and parts[0] in usg_pkg_types:
                        found_usg_pkg = True
                        break

                if found_usg_pkg:
                    return 0.95

            except OSError:
                pass

        # Check for USG-specific files in the workspace (.disu, .sms, .cln, .gnc)
        usg_exts = {".disu", ".sms", ".cln", ".gnc", ".bct", ".ddf"}
        found_usg_files = 0
        try:
            for p in ws_root.iterdir():
                if p.is_file() and p.suffix.lower() in usg_exts:
                    found_usg_files += 1
        except OSError:
            pass

        if found_usg_files >= 1:
            return 0.9

        # No USG indicators
        return 0.0

    # ── Model loading ─────────────────────────────────────────────────────

    def build_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        """Build model snapshot.

        Attempts FloPy-based extraction first via flopy.mfusg.MfUsg,
        falls back to text parsing.
        """
        snapshot = self._try_flopy_snapshot(ws_root)
        if snapshot.get("ok", False):
            return snapshot

        # Fallback: basic text-based snapshot
        return self._build_text_snapshot(ws_root)

    def _try_flopy_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        """Build snapshot using FloPy's MfUsg loader."""
        try:
            import flopy
        except ImportError:
            return {"ok": False, "error": "FloPy not available"}

        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return {"ok": False, "error": "No .nam file found"}

        try:
            m = flopy.mfusg.MfUsg.load(
                nam_files[0].name,
                model_ws=str(ws_root),
                check=False,
                verbose=False,
            )

            # Detect grid type
            from gw.simulators.mfusg.io import detect_grid_type
            grid_type = detect_grid_type(ws_root)

            # Grid info
            grid: Dict[str, Any] = {"type": grid_type.upper()}

            if grid_type == "disu":
                # Unstructured grid
                disu_pkg = m.get_package("DISU")
                if disu_pkg is not None:
                    grid["nodes"] = int(getattr(disu_pkg, "nodes", 0))
                    grid["nlay"] = int(getattr(disu_pkg, "nlay", 0))
                    grid["njag"] = int(getattr(disu_pkg, "njag", 0))
                else:
                    grid["nodes"] = 0
                    grid["nlay"] = getattr(m, "nlay", 0)
            else:
                # Structured grid
                grid["nlay"] = m.nlay
                grid["nrow"] = m.nrow
                grid["ncol"] = m.ncol
                if m.dis is not None:
                    grid["delr"] = {"min": float(m.dis.delr.array.min()),
                                    "max": float(m.dis.delr.array.max())}
                    grid["delc"] = {"min": float(m.dis.delc.array.min()),
                                    "max": float(m.dis.delc.array.max())}

            # Time discretisation
            tdis: Dict[str, Any] = {"nper": m.nper}
            if m.dis is not None:
                perlen = m.dis.perlen.array
                nstp_arr = m.dis.nstp.array
                tdis["total_time"] = float(perlen.sum())
                tdis["perlen_range"] = {"min": float(perlen.min()),
                                         "max": float(perlen.max())}
                tdis["nstp_range"] = {"min": int(nstp_arr.min()),
                                       "max": int(nstp_arr.max())}

            # Packages
            packages: Dict[str, Any] = {}
            for pkg_name in m.get_package_list():
                packages[pkg_name.upper()] = {"active": True}

            # Stress summaries
            stress_sums: Dict[str, Any] = {}
            for pkg_type in ["WEL", "CHD", "GHB", "RIV", "DRN"]:
                pkg = m.get_package(pkg_type)
                if pkg is not None and hasattr(pkg, "stress_period_data"):
                    spd = pkg.stress_period_data
                    total_records = 0
                    periods_with_data = 0
                    if spd is not None:
                        for iper in range(m.nper):
                            try:
                                data = spd[iper]
                                if data is not None and len(data) > 0:
                                    total_records += len(data)
                                    periods_with_data += 1
                            except (KeyError, IndexError):
                                pass
                    stress_sums[pkg_type] = {
                        "total_records": total_records,
                        "periods_with_data": periods_with_data,
                    }

            # Output files
            outputs_present: Dict[str, Any] = {}
            for ext, key in [(".hds", "hds"), (".cbc", "cbc"), (".lst", "lst")]:
                found = list(ws_root.glob(f"*{ext}"))
                if found:
                    outputs_present[key] = {
                        "file": found[0].name,
                        "size_bytes": found[0].stat().st_size,
                    }

            return {
                "ok": True,
                "simulator": "mfusg",
                "grid": grid,
                "tdis": tdis,
                "packages": packages,
                "stress_summaries": stress_sums,
                "outputs_present": outputs_present,
                "workspace_root": str(ws_root),
            }
        except Exception as e:
            return {"ok": False, "error": f"FloPy MfUsg load failed: {type(e).__name__}: {e}"}

    def _build_text_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        """Minimal text-based snapshot when FloPy fails."""
        from gw.simulators.mfusg.io import read_mfusg_nam, detect_grid_type

        snapshot: Dict[str, Any] = {
            "ok": True,
            "simulator": "mfusg",
            "workspace_root": str(ws_root),
        }

        grid_type = detect_grid_type(ws_root)
        snapshot["grid"] = {"type": grid_type.upper()}

        # Parse name file
        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if nam_files:
            nam_df = read_mfusg_nam(nam_files[0])
            packages = {}
            for _, row in nam_df.iterrows():
                packages[row["ftype"]] = {"fname": row["fname"], "active": True}
            snapshot["packages"] = packages

        # Try to parse DIS/DISU header for grid info
        if grid_type == "disu":
            disu_files = list(ws_root.glob("*.disu"))
            if disu_files:
                try:
                    text = disu_files[0].read_text(encoding="utf-8", errors="replace")
                    lines = [l.strip() for l in text.splitlines()
                             if l.strip() and not l.strip().startswith("#")]
                    if lines:
                        parts = lines[0].split()
                        # DISU header: NODES NLAY NJAG IVSD NPER ...
                        snapshot["grid"]["nodes"] = int(parts[0])
                        snapshot["grid"]["nlay"] = int(parts[1])
                        snapshot["grid"]["njag"] = int(parts[2])
                        snapshot["tdis"] = {"nper": int(parts[4])}
                except Exception:
                    pass
        else:
            dis_files = list(ws_root.glob("*.dis"))
            if dis_files:
                try:
                    text = dis_files[0].read_text(encoding="utf-8", errors="replace")
                    lines = [l.strip() for l in text.splitlines()
                             if l.strip() and not l.strip().startswith("#")]
                    if lines:
                        parts = lines[0].split()
                        # DIS header: NLAY NROW NCOL NPER ...
                        snapshot["grid"]["nlay"] = int(parts[0])
                        snapshot["grid"]["nrow"] = int(parts[1])
                        snapshot["grid"]["ncol"] = int(parts[2])
                        snapshot["tdis"] = {"nper": int(parts[3])}
                except Exception:
                    pass

        # Output files
        outputs: Dict[str, Any] = {}
        for ext, key in [(".hds", "hds"), (".cbc", "cbc"), (".lst", "lst")]:
            found = list(ws_root.glob(f"*{ext}"))
            if found:
                outputs[key] = {
                    "file": found[0].name,
                    "size_bytes": found[0].stat().st_size,
                }
        snapshot["outputs_present"] = outputs

        return snapshot

    def build_model_brief(self, snapshot: Dict[str, Any]) -> str:
        """Build a compact text summary for LLM prompt injection."""
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
        from gw.simulators.mfusg.io import parse_mfusg_stress_package
        ext = Path(rel_path).suffix.lower().lstrip(".")
        pkg_type = ext.upper() if ext else "WEL"
        return parse_mfusg_stress_package(
            ws_root / rel_path, pkg_type=pkg_type, value_cols=value_cols,
        )

    def read_time_discretization(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        from gw.simulators.mfusg.io import read_mfusg_dis_times
        if rel_path:
            path = ws_root / rel_path
        else:
            # Try DISU first, then DIS
            disu_files = list(ws_root.glob("*.disu"))
            dis_files = list(ws_root.glob("*.dis"))
            if disu_files:
                path = disu_files[0]
            elif dis_files:
                path = dis_files[0]
            else:
                path = ws_root / "model.dis"
        return read_mfusg_dis_times(path)

    def read_namefile(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        from gw.simulators.mfusg.io import read_mfusg_nam
        if rel_path:
            path = ws_root / rel_path
        else:
            nam_files = [f for f in ws_root.glob("*.nam")
                         if f.name.lower() != "mfsim.nam"]
            path = nam_files[0] if nam_files else ws_root / "model.nam"
        return read_mfusg_nam(path)

    # ── Binary output ─────────────────────────────────────────────────────
    # DUAL PATH: uses HeadUFile for DISU grids, HeadFile for DIS grids.
    # CellBudgetFile works for both grid types.

    def probe_head_file(self, ws_root: Path, rel_path: str) -> Dict[str, Any]:
        """Probe a head output file.

        Uses HeadUFile for unstructured (DISU) grids, standard HeadFile
        for structured (DIS) grids.
        """
        from gw.simulators.mfusg.io import is_unstructured
        if is_unstructured(ws_root):
            from gw.api.output_probes import probe_hds_unstructured
            return probe_hds_unstructured(ws_root, rel_path)
        else:
            from gw.api.output_probes import probe_hds
            return probe_hds(ws_root, rel_path)

    def probe_budget_file(self, ws_root: Path, rel_path: str) -> Dict[str, Any]:
        """Probe a budget output file. CellBudgetFile works for both grid types."""
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
        """Extract head data.

        Uses unstructured reader for DISU grids, standard reader for DIS grids.
        """
        from gw.simulators.mfusg.io import is_unstructured
        if is_unstructured(ws_root):
            from gw.api.output_probes import extract_hds_data_unstructured
            return extract_hds_data_unstructured(
                ws_root, rel_path, max_times=max_times, max_chars=max_chars,
            )
        else:
            from gw.api.output_probes import extract_hds_data
            return extract_hds_data(
                ws_root, rel_path, max_times=max_times, max_chars=max_chars,
            )

    def extract_budget_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_records: int = 8,
        max_chars: int = 30_000,
    ) -> Dict[str, Any]:
        """Extract budget data. CellBudgetFile works for both grid types."""
        from gw.api.output_probes import extract_cbc_data
        return extract_cbc_data(
            ws_root, rel_path, max_records=max_records, max_chars=max_chars,
        )

    def extract_head_timeseries(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        cells: list,
        max_chars: int = 50_000,
    ) -> Dict[str, Any]:
        """Extract head time-series at specific cells.

        For structured (DIS) grids, reuses standard FloPy HeadFile extraction.
        For unstructured (DISU) grids, returns not-supported (HeadUFile does
        not support structured cell indexing).
        """
        from gw.simulators.mfusg.io import is_unstructured
        if is_unstructured(ws_root):
            return {
                "ok": False,
                "error": (
                    "Cell-level head time-series extraction is not supported "
                    "for unstructured (DISU) grids. Use extract_head_data for "
                    "per-layer summaries instead."
                ),
            }
        from gw.api.output_probes import extract_hds_timeseries
        return extract_hds_timeseries(
            ws_root, rel_path, cells=cells, max_chars=max_chars,
        )

    def extract_budget_cells(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        component: str,
        max_cells: int = 50,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        """Extract per-cell budget flows. CellBudgetFile works for both grid types."""
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
        from gw.run.mfusg_runner import run_mfusg
        return run_mfusg(
            workspace=str(ws_root),
            mfusg_path=exe_path,
            timeout_sec=timeout_sec,
        )

    def find_executable(self, exe_path: Optional[str] = None) -> str:
        from gw.run.mfusg_runner import find_mfusg
        return find_mfusg(exe_path)

    # ── QA diagnostics ────────────────────────────────────────────────────

    def available_qa_checks(self) -> List[QACheck]:
        """Return QA checks available for MFUSG models.

        Most checks work unchanged (listing file + binary output format
        conventions are the same).
        """
        return [
            QACheck("mass_balance",
                    "Parse listing file for volumetric budget, compute % discrepancy per stress period"),
            QACheck("dry_cells",
                    "Count cells with dry/inactive heads per layer and timestep"),
            QACheck("convergence",
                    "Parse listing file for solver iteration counts, convergence failures"),
            QACheck("pumping_summary",
                    "Analyze WEL package pumping rates by stress period, detect anomalous jumps"),
            QACheck("budget_timeseries",
                    "Extract IN/OUT per budget term across all timesteps"),
            QACheck("head_gradient",
                    "Compute cell-to-cell gradients per layer, flag extreme values"),
            QACheck("property_check",
                    "Check K/SS/SY ranges from LPF or BCF6 packages"),
            QACheck("listing_budget_detail",
                    "Deep listing file parse: per-package IN/OUT budget tables, solver warnings"),
            QACheck("property_zones",
                    "Spatial K zone analysis per layer: distinct value clusters, contrast ratios"),
            QACheck("save_snapshot",
                    "Save a lightweight snapshot of current model outputs for later comparison"),
            QACheck("compare_runs",
                    "Compare two run snapshots side-by-side: head changes, budget differences"),
        ]

    def run_qa_check(self, ws_root: Path, check_name: str, **kwargs: Any) -> str:
        from gw.api.qa_diagnostics import run_qa_check as _run
        return _run(str(ws_root), check_name, **kwargs)

    # ── File-type knowledge ───────────────────────────────────────────────

    def file_type_knowledge(self) -> Dict[str, FileTypeInfo]:
        from gw.simulators.mfusg.knowledge import _EXT_KB
        return dict(_EXT_KB)

    def package_properties(self) -> Dict[str, PackagePropertyInfo]:
        from gw.simulators.mfusg.knowledge import PACKAGE_PROPERTIES
        return dict(PACKAGE_PROPERTIES)

    def file_extensions(self) -> Set[str]:
        from gw.simulators.mfusg.knowledge import _EXT_KB
        return set(_EXT_KB.keys()) | {
            ".sms", ".cln", ".gnc", ".bct", ".ddf",
            ".sfr", ".lak", ".uzf",
        }

    # ── LLM knowledge ─────────────────────────────────────────────────────

    def system_prompt_fragment(self) -> str:
        from gw.simulators.mfusg.knowledge import mfusg_system_prompt_fragment
        return mfusg_system_prompt_fragment()

    def tool_descriptions(self) -> Dict[str, str]:
        from gw.simulators.mfusg.knowledge import mfusg_tool_description_overrides
        return mfusg_tool_description_overrides()

    def file_mention_pattern(self) -> str:
        from gw.simulators.mfusg.knowledge import mfusg_file_mention_regex
        return mfusg_file_mention_regex()

    # ── Output capabilities ───────────────────────────────────────────────

    def output_capabilities(self) -> OutputCapability:
        return OutputCapability(heads=True, budget=True)

    # ── Grid ──────────────────────────────────────────────────────────────

    def get_grid_info(self, ws_root: Path) -> Optional[GridInfo]:
        """Load grid information, handling both DIS and DISU grids."""
        from gw.simulators.mfusg.io import detect_grid_type

        grid_type = detect_grid_type(ws_root)

        # Try FloPy first
        try:
            import flopy
        except ImportError:
            return self._grid_info_from_text(ws_root, grid_type)

        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return self._grid_info_from_text(ws_root, grid_type)

        try:
            m = flopy.mfusg.MfUsg.load(
                nam_files[0].name,
                model_ws=str(ws_root),
                load_only=["dis", "disu"],
                check=False,
                verbose=False,
            )

            if grid_type == "disu":
                disu_pkg = m.get_package("DISU")
                if disu_pkg is not None:
                    nodes = int(getattr(disu_pkg, "nodes", 0))
                    nlay = int(getattr(disu_pkg, "nlay", 0))
                    ncpl = nodes // nlay if nlay > 0 else nodes
                    return GridInfo(
                        grid_type="disu",
                        nlay=nlay,
                        ncpl=ncpl,
                        total_cells=nodes,
                    )
                return self._grid_info_from_text(ws_root, grid_type)
            else:
                return GridInfo(
                    grid_type="dis",
                    nlay=m.nlay,
                    ncpl=m.nrow * m.ncol,
                    total_cells=m.nlay * m.nrow * m.ncol,
                    nrow=m.nrow,
                    ncol=m.ncol,
                    xorigin=getattr(m.modelgrid, "xoffset", 0.0) or 0.0,
                    yorigin=getattr(m.modelgrid, "yoffset", 0.0) or 0.0,
                    angrot=getattr(m.modelgrid, "angrot", 0.0) or 0.0,
                )
        except Exception:
            return self._grid_info_from_text(ws_root, grid_type)

    def _grid_info_from_text(
        self, ws_root: Path, grid_type: str
    ) -> Optional[GridInfo]:
        """Parse DIS/DISU header for basic grid info when FloPy is unavailable."""
        if grid_type == "disu":
            disu_files = list(ws_root.glob("*.disu"))
            if not disu_files:
                return None
            try:
                text = disu_files[0].read_text(encoding="utf-8", errors="replace")
                lines = [l.strip() for l in text.splitlines()
                         if l.strip() and not l.strip().startswith("#")]
                if not lines:
                    return None
                parts = lines[0].split()
                # DISU header: NODES NLAY NJAG IVSD NPER ...
                nodes = int(parts[0])
                nlay = int(parts[1])
                ncpl = nodes // nlay if nlay > 0 else nodes
                return GridInfo(
                    grid_type="disu",
                    nlay=nlay,
                    ncpl=ncpl,
                    total_cells=nodes,
                )
            except Exception:
                return None
        else:
            dis_files = list(ws_root.glob("*.dis"))
            if not dis_files:
                return None
            try:
                text = dis_files[0].read_text(encoding="utf-8", errors="replace")
                lines = [l.strip() for l in text.splitlines()
                         if l.strip() and not l.strip().startswith("#")]
                if not lines:
                    return None
                parts = lines[0].split()
                # DIS header: NLAY NROW NCOL NPER ...
                nlay = int(parts[0])
                nrow = int(parts[1])
                ncol = int(parts[2])
                return GridInfo(
                    grid_type="dis",
                    nlay=nlay,
                    ncpl=nrow * ncol,
                    total_cells=nlay * nrow * ncol,
                    nrow=nrow,
                    ncol=ncol,
                )
            except Exception:
                return None

    # ── FloPy bridge ──────────────────────────────────────────────────────

    _sim_cache: Dict[str, Any] = {}

    def get_simulation(self, ws_root: Path) -> Tuple[Optional[Any], Optional[str]]:
        """Load model via FloPy's MfUsg loader with caching.

        Returns ``(model, error_msg)`` -- note this returns an MfUsg object
        not an MFSimulation, but the interface allows Any.
        """
        cache_key = str(ws_root)
        if cache_key in self._sim_cache:
            return self._sim_cache[cache_key], None

        try:
            import flopy
        except ImportError:
            return None, "FloPy not available"

        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return None, "No .nam file found"

        try:
            m = flopy.mfusg.MfUsg.load(
                nam_files[0].name,
                model_ws=str(ws_root),
                check=False,
                verbose=False,
            )
            self._sim_cache[cache_key] = m
            return m, None
        except Exception as e:
            return None, f"FloPy MfUsg load failed: {type(e).__name__}: {e}"

    def clear_simulation_cache(self, ws_root: Optional[Path] = None) -> None:
        """Clear cached simulation objects."""
        if ws_root is not None:
            self._sim_cache.pop(str(ws_root), None)
        else:
            self._sim_cache.clear()
