"""MODFLOW-2005 / MODFLOW-NWT concrete simulator adapter.

Implements the ``SimulatorAdapter`` interface for classic MODFLOW models.
Binary output reading reuses the same FloPy HeadFile / CellBudgetFile
readers as MF6 (identical binary format).
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


class MF2005Adapter(SimulatorAdapter):
    """Adapter for MODFLOW-2005 and MODFLOW-NWT models."""

    # ── Identity ──────────────────────────────────────────────────────────

    def info(self) -> SimulatorInfo:
        return SimulatorInfo(
            name="mf2005",
            display_name="MODFLOW-2005/NWT",
        )

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(self, ws_root: Path) -> float:
        # MF6's mfsim.nam is a disqualifier — if it exists, this is NOT MF2005
        if (ws_root / "mfsim.nam").exists():
            return 0.0

        # Look for a .nam file (that isn't mfsim.nam)
        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]

        if not nam_files:
            return 0.0

        # Check for MF2005-specific package extensions
        mf2005_exts = {".bas", ".bcf", ".lpf", ".upw", ".pcg", ".nwt",
                       ".sip", ".sor", ".gmg", ".de4"}
        found_mf2005 = 0
        try:
            for p in ws_root.iterdir():
                if p.is_file() and p.suffix.lower() in mf2005_exts:
                    found_mf2005 += 1
        except OSError:
            pass

        if found_mf2005 >= 2:
            return 0.95

        # Has a .nam file — check if it's free-format (not MF6 block-based)
        for nam in nam_files:
            try:
                txt = nam.read_text(errors="ignore")[:2000].upper()
                # MF6 name files use BEGIN/END blocks
                if "BEGIN" in txt and "END" in txt:
                    return 0.0  # Likely MF6
                # MF2005 name files have FTYPE UNIT FNAME format
                # Look for typical package types
                if any(pkg in txt for pkg in ["LIST", "BAS6", "LPF", "BCF6",
                                               "UPW", "PCG", "NWT", "DIS"]):
                    return 0.9
            except OSError:
                pass

        # Has .nam but couldn't confirm format
        return 0.4

    # ── Model loading ─────────────────────────────────────────────────────

    def build_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        """Build model snapshot.

        Attempts FloPy-based extraction first, falls back to text parsing.
        """
        snapshot = self._try_flopy_snapshot(ws_root)
        if snapshot.get("ok", False):
            return snapshot

        # Fallback: basic text-based snapshot
        return self._build_text_snapshot(ws_root)

    def _try_flopy_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        """Build snapshot using FloPy's Modflow loader."""
        try:
            import flopy
        except ImportError:
            return {"ok": False, "error": "FloPy not available"}

        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return {"ok": False, "error": "No .nam file found"}

        try:
            m = flopy.modflow.Modflow.load(
                nam_files[0].name,
                model_ws=str(ws_root),
                check=False,
                verbose=False,
            )

            # Grid info
            grid: Dict[str, Any] = {
                "type": "DIS",
                "nlay": m.nlay,
                "nrow": m.nrow,
                "ncol": m.ncol,
            }
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

            # Stress summaries (for WEL, CHD, etc.)
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

            # Determine variant
            from gw.simulators.mf2005.io import detect_mf2005_variant
            variant = detect_mf2005_variant(ws_root)

            return {
                "ok": True,
                "simulator": "mf2005",
                "variant": variant,
                "grid": grid,
                "tdis": tdis,
                "packages": packages,
                "stress_summaries": stress_sums,
                "outputs_present": outputs_present,
                "workspace_root": str(ws_root),
            }
        except Exception as e:
            return {"ok": False, "error": f"FloPy load failed: {type(e).__name__}: {e}"}

    def _build_text_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        """Minimal text-based snapshot when FloPy fails."""
        from gw.simulators.mf2005.io import read_mf2005_nam, detect_mf2005_variant

        snapshot: Dict[str, Any] = {
            "ok": True,
            "simulator": "mf2005",
            "variant": detect_mf2005_variant(ws_root),
            "workspace_root": str(ws_root),
        }

        # Parse name file
        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if nam_files:
            nam_df = read_mf2005_nam(nam_files[0])
            packages = {}
            for _, row in nam_df.iterrows():
                packages[row["ftype"]] = {"fname": row["fname"], "active": True}
            snapshot["packages"] = packages

        # Try to parse DIS header for grid info
        dis_files = list(ws_root.glob("*.dis"))
        if dis_files:
            try:
                text = dis_files[0].read_text(encoding="utf-8", errors="replace")
                lines = [l.strip() for l in text.splitlines()
                         if l.strip() and not l.strip().startswith("#")]
                if lines:
                    parts = lines[0].split()
                    snapshot["grid"] = {
                        "type": "DIS",
                        "nlay": int(parts[0]),
                        "nrow": int(parts[1]),
                        "ncol": int(parts[2]),
                    }
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
        # Reuse the generic brief builder — it works with MF2005 snapshots too
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
        from gw.simulators.mf2005.io import parse_mf2005_stress_package
        # Infer package type from extension
        ext = Path(rel_path).suffix.lower().lstrip(".")
        pkg_type = ext.upper() if ext else "WEL"
        return parse_mf2005_stress_package(
            ws_root / rel_path, pkg_type=pkg_type, value_cols=value_cols,
        )

    def read_time_discretization(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        from gw.simulators.mf2005.io import read_mf2005_dis_times
        # Find the DIS file
        if rel_path:
            path = ws_root / rel_path
        else:
            dis_files = list(ws_root.glob("*.dis"))
            path = dis_files[0] if dis_files else ws_root / "model.dis"
        return read_mf2005_dis_times(path)

    def read_namefile(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        from gw.simulators.mf2005.io import read_mf2005_nam
        if rel_path:
            path = ws_root / rel_path
        else:
            nam_files = [f for f in ws_root.glob("*.nam")
                         if f.name.lower() != "mfsim.nam"]
            path = nam_files[0] if nam_files else ws_root / "model.nam"
        return read_mf2005_nam(path)

    # ── Binary output ─────────────────────────────────────────────────────
    # Binary HDS/CBC format is identical across all MODFLOW versions.
    # We reuse the same output_probes functions as MF6.

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
        from gw.run.mf2005_runner import run_mf2005
        from gw.simulators.mf2005.io import detect_mf2005_variant
        prefer_nwt = detect_mf2005_variant(ws_root) == "mfnwt"
        return run_mf2005(
            workspace=str(ws_root),
            mf2005_path=exe_path,
            timeout_sec=timeout_sec,
            prefer_nwt=prefer_nwt,
        )

    def find_executable(self, exe_path: Optional[str] = None) -> str:
        from gw.run.mf2005_runner import find_mf2005
        return find_mf2005(exe_path)

    # ── QA diagnostics ────────────────────────────────────────────────────

    def available_qa_checks(self) -> List[QACheck]:
        """Return QA checks available for MF2005/NWT models.

        Most checks work unchanged (listing file + binary output format
        are the same).  Some are excluded (e.g. advanced_packages and
        observation_comparison need MF2005-specific parsers).
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
                    "Check K/SS/SY ranges from LPF, BCF6, or UPW packages"),
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
        from gw.simulators.mf2005.knowledge import _EXT_KB
        return dict(_EXT_KB)

    def package_properties(self) -> Dict[str, PackagePropertyInfo]:
        from gw.simulators.mf2005.knowledge import PACKAGE_PROPERTIES
        return dict(PACKAGE_PROPERTIES)

    def file_extensions(self) -> Set[str]:
        from gw.simulators.mf2005.knowledge import _EXT_KB
        return set(_EXT_KB.keys()) | {
            ".sip", ".sor", ".gmg", ".de4",
            ".sfr", ".lak", ".mnw2", ".uzf", ".sub", ".hob",
        }

    # ── LLM knowledge ─────────────────────────────────────────────────────

    def system_prompt_fragment(self) -> str:
        from gw.simulators.mf2005.knowledge import mf2005_system_prompt_fragment
        return mf2005_system_prompt_fragment()

    def tool_descriptions(self) -> Dict[str, str]:
        from gw.simulators.mf2005.knowledge import mf2005_tool_description_overrides
        return mf2005_tool_description_overrides()

    def file_mention_pattern(self) -> str:
        from gw.simulators.mf2005.knowledge import mf2005_file_mention_regex
        return mf2005_file_mention_regex()

    # ── Grid ──────────────────────────────────────────────────────────────

    def get_grid_info(self, ws_root: Path) -> Optional[GridInfo]:
        try:
            import flopy
        except ImportError:
            return self._grid_info_from_text(ws_root)

        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return self._grid_info_from_text(ws_root)

        try:
            m = flopy.modflow.Modflow.load(
                nam_files[0].name,
                model_ws=str(ws_root),
                load_only=["dis"],
                check=False,
                verbose=False,
            )
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
            return self._grid_info_from_text(ws_root)

    def _grid_info_from_text(self, ws_root: Path) -> Optional[GridInfo]:
        """Parse DIS header for basic grid info when FloPy is unavailable."""
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

    def get_simulation(self, ws_root: Path) -> Tuple[Optional[Any], Optional[str]]:
        """Load model via FloPy's Modflow loader.

        Returns ``(model, error_msg)`` — note this returns a Modflow object
        not an MFSimulation, but the interface allows Any.
        """
        try:
            import flopy
        except ImportError:
            return None, "FloPy not available"

        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return None, "No .nam file found"

        try:
            m = flopy.modflow.Modflow.load(
                nam_files[0].name,
                model_ws=str(ws_root),
                check=False,
                verbose=False,
            )
            return m, None
        except Exception as e:
            return None, f"FloPy load failed: {type(e).__name__}: {e}"

    def clear_simulation_cache(self, ws_root: Optional[Path] = None) -> None:
        """No simulation cache for MF2005 — no-op."""
        pass
