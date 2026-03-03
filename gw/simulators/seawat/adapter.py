"""SEAWAT v4 concrete simulator adapter.

Implements the ``SimulatorAdapter`` interface for SEAWAT models — coupled
variable-density groundwater flow and solute transport.  SEAWAT combines
MODFLOW-2000 + MT3DMS + VDF (Variable-Density Flow) + VSC (Viscosity).

Binary output reading reuses the same FloPy readers as MF2005 (identical
HDS/CBC format) plus FloPy's UcnFile for concentration output.
Stress-package I/O and name-file parsing delegate to the MF2005 io module
(same free-format file layout).
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

# Cache for loaded SEAWAT models (keyed by resolved workspace path)
_MODEL_CACHE: Dict[str, Any] = {}


class SeawatAdapter(SimulatorAdapter):
    """Adapter for SEAWAT v4 variable-density flow and transport models."""

    # ── Identity ──────────────────────────────────────────────────────────

    def info(self) -> SimulatorInfo:
        return SimulatorInfo(
            name="seawat",
            display_name="SEAWAT v4",
        )

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(self, ws_root: Path) -> float:
        """Detect whether *ws_root* contains a SEAWAT model.

        Strategy:
        - mfsim.nam exists -> 0.0 (this is MF6, not SEAWAT)
        - .nam file contains VDF or VSC package references -> 0.95 (unique to SEAWAT)
        - .nam file contains BOTH MODFLOW (DIS, BAS, LPF) AND MT3D (BTN, ADV) -> 0.85
        - Otherwise -> 0.0
        """
        # MF6 disqualifier
        if (ws_root / "mfsim.nam").exists():
            return 0.0

        # Find .nam files (not mfsim.nam)
        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return 0.0

        best_score = 0.0

        for nam in nam_files:
            try:
                txt = nam.read_text(encoding="utf-8", errors="ignore")[:8000].upper()
            except OSError:
                continue

            # MF6 block-based name files use BEGIN/END — skip
            if "BEGIN" in txt and "END" in txt:
                continue

            # Check for SEAWAT-unique packages (VDF, VSC)
            has_vdf = False
            has_vsc = False
            has_flow = False
            has_transport = False

            for line in txt.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ftype = line.split()[0] if line.split() else ""

                if ftype == "VDF":
                    has_vdf = True
                elif ftype == "VSC":
                    has_vsc = True
                elif ftype in ("DIS", "BAS6", "BAS", "LPF", "BCF6", "BCF", "UPW"):
                    has_flow = True
                elif ftype in ("BTN", "ADV", "DSP", "SSM", "GCG"):
                    has_transport = True

            # VDF or VSC is unique to SEAWAT
            if has_vdf or has_vsc:
                best_score = max(best_score, 0.95)
            # Both flow and transport packages without VDF/VSC — likely SEAWAT
            # but could be standalone MT3DMS pointing to a separate flow model
            elif has_flow and has_transport:
                best_score = max(best_score, 0.85)

        # Also check for standalone VDF/VSC files in workspace
        if best_score == 0.0:
            vdf_files = list(ws_root.glob("*.vdf"))
            vsc_files = list(ws_root.glob("*.vsc"))
            if vdf_files or vsc_files:
                best_score = 0.7

        return best_score

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
        """Build snapshot using FloPy's Seawat loader."""
        try:
            import flopy
        except ImportError:
            return {"ok": False, "error": "FloPy not available"}

        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return {"ok": False, "error": "No .nam file found"}

        try:
            m = flopy.seawat.Seawat.load(
                nam_files[0].name,
                model_ws=str(ws_root),
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

            # Packages — collect all flow + transport + VDF/VSC
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

            # VDF summary
            vdf_info: Dict[str, Any] = {}
            vdf_pkg = m.get_package("VDF")
            if vdf_pkg is not None:
                for attr in ("mtdnconc", "denseref", "drhodc", "densemin",
                             "densemax", "nswtcpl", "dnscrit", "firstdt",
                             "iwtable"):
                    val = getattr(vdf_pkg, attr, None)
                    if val is not None:
                        try:
                            vdf_info[attr.upper()] = float(val)
                        except (TypeError, ValueError):
                            vdf_info[attr.upper()] = str(val)

            # VSC summary
            vsc_info: Dict[str, Any] = {}
            vsc_pkg = m.get_package("VSC")
            if vsc_pkg is not None:
                for attr in ("viscref", "dmudc", "viscmin", "viscmax",
                             "nsmueos", "mtmuspec"):
                    val = getattr(vsc_pkg, attr, None)
                    if val is not None:
                        try:
                            vsc_info[attr.upper()] = float(val)
                        except (TypeError, ValueError):
                            vsc_info[attr.upper()] = str(val)

            # Transport summary (BTN)
            transport_info: Dict[str, Any] = {}
            btn_pkg = m.get_package("BTN")
            if btn_pkg is not None:
                for attr in ("ncomp", "mcomp"):
                    val = getattr(btn_pkg, attr, None)
                    if val is not None:
                        transport_info[attr.upper()] = int(val)
                prsity = getattr(btn_pkg, "prsity", None)
                if prsity is not None:
                    try:
                        arr = prsity.array if hasattr(prsity, "array") else prsity
                        import numpy as np
                        transport_info["porosity_range"] = {
                            "min": float(np.min(arr)),
                            "max": float(np.max(arr)),
                        }
                    except Exception:
                        pass

            # Output files
            outputs_present: Dict[str, Any] = {}
            for ext, key in [(".hds", "hds"), (".cbc", "cbc"),
                             (".ucn", "ucn"), (".lst", "lst"),
                             (".mas", "mas")]:
                found = list(ws_root.glob(f"*{ext}"))
                if found:
                    outputs_present[key] = {
                        "file": found[0].name,
                        "size_bytes": found[0].stat().st_size,
                    }

            result: Dict[str, Any] = {
                "ok": True,
                "simulator": "seawat",
                "grid": grid,
                "tdis": tdis,
                "packages": packages,
                "stress_summaries": stress_sums,
                "outputs_present": outputs_present,
                "workspace_root": str(ws_root),
            }
            if vdf_info:
                result["vdf"] = vdf_info
            if vsc_info:
                result["vsc"] = vsc_info
            if transport_info:
                result["transport"] = transport_info

            return result

        except Exception as e:
            return {"ok": False, "error": f"FloPy SEAWAT load failed: {type(e).__name__}: {e}"}

    def _build_text_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        """Minimal text-based snapshot when FloPy fails."""
        from gw.simulators.mf2005.io import read_mf2005_nam

        snapshot: Dict[str, Any] = {
            "ok": True,
            "simulator": "seawat",
            "workspace_root": str(ws_root),
        }

        # Parse name file
        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if nam_files:
            nam_df = read_mf2005_nam(nam_files[0])
            packages: Dict[str, Any] = {}
            has_vdf = False
            has_vsc = False
            for _, row in nam_df.iterrows():
                packages[row["ftype"]] = {"fname": row["fname"], "active": True}
                if row["ftype"] == "VDF":
                    has_vdf = True
                if row["ftype"] == "VSC":
                    has_vsc = True
            snapshot["packages"] = packages
            snapshot["has_vdf"] = has_vdf
            snapshot["has_vsc"] = has_vsc

        # Try to parse DIS header for grid info
        dis_files = list(ws_root.glob("*.dis"))
        if dis_files:
            try:
                text = dis_files[0].read_text(encoding="utf-8", errors="replace")
                lines = [ln.strip() for ln in text.splitlines()
                         if ln.strip() and not ln.strip().startswith("#")]
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
        for ext, key in [(".hds", "hds"), (".cbc", "cbc"),
                         (".ucn", "ucn"), (".lst", "lst"),
                         (".mas", "mas")]:
            found = list(ws_root.glob(f"*{ext}"))
            if found:
                outputs[key] = {
                    "file": found[0].name,
                    "size_bytes": found[0].stat().st_size,
                }
        snapshot["outputs_present"] = outputs

        return snapshot

    def build_model_brief(self, snapshot: Dict[str, Any]) -> str:
        """Build a compact text summary for LLM prompt injection.

        Extends the generic brief with density coupling info.
        """
        from gw.api.model_snapshot import build_model_brief
        brief = build_model_brief(snapshot)

        # Append SEAWAT-specific density info
        extras: List[str] = []
        vdf = snapshot.get("vdf", {})
        if vdf:
            dref = vdf.get("DENSEREF", "?")
            drho = vdf.get("DRHODC", "?")
            ncpl = vdf.get("NSWTCPL", "?")
            extras.append(
                f"VDF: rho_ref={dref}, drhodc={drho}, coupling_iters={ncpl}"
            )
        vsc = snapshot.get("vsc", {})
        if vsc:
            vref = vsc.get("VISCREF", "?")
            extras.append(f"VSC: viscref={vref}")
        transport = snapshot.get("transport", {})
        if transport:
            ncomp = transport.get("NCOMP", "?")
            por = transport.get("porosity_range", {})
            por_str = ""
            if por:
                por_str = f", porosity={por.get('min','?')}-{por.get('max','?')}"
            extras.append(f"Transport: {ncomp} species{por_str}")

        if extras:
            brief += "\n" + "; ".join(extras)

        return brief

    # ── Stress I/O ────────────────────────────────────────────────────────
    # SEAWAT uses the same free-format stress-package layout as MF2005.

    def read_stress_package(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        value_cols: int = 1,
        keep_aux: bool = True,
    ) -> Any:
        from gw.simulators.mf2005.io import parse_mf2005_stress_package
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
    # HDS/CBC format is identical to MF2005.  UCN format is MT3DMS standard.

    def probe_head_file(self, ws_root: Path, rel_path: str) -> Dict[str, Any]:
        from gw.api.output_probes import probe_hds
        return probe_hds(ws_root, rel_path)

    def probe_budget_file(self, ws_root: Path, rel_path: str) -> Dict[str, Any]:
        from gw.api.output_probes import probe_cbc
        return probe_cbc(ws_root, rel_path)

    def probe_concentration_file(
        self, ws_root: Path, rel_path: str
    ) -> Dict[str, Any]:
        from gw.api.output_probes import probe_ucn
        return probe_ucn(ws_root, rel_path)

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

    def extract_concentration_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_times: int = 5,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        from gw.api.output_probes import extract_ucn_data
        return extract_ucn_data(ws_root, rel_path, max_times=max_times, max_chars=max_chars)

    def extract_concentration_timeseries(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        cells: List[Dict[str, int]],
        max_chars: int = 50_000,
    ) -> Dict[str, Any]:
        from gw.api.output_probes import extract_ucn_timeseries
        return extract_ucn_timeseries(ws_root, rel_path, cells=cells, max_chars=max_chars)

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
        from gw.run.seawat_runner import run_seawat
        return run_seawat(
            workspace=str(ws_root),
            seawat_path=exe_path,
            timeout_sec=timeout_sec,
        )

    def find_executable(self, exe_path: Optional[str] = None) -> str:
        from gw.run.seawat_runner import find_seawat
        return find_seawat(exe_path)

    # ── QA diagnostics ────────────────────────────────────────────────────

    def available_qa_checks(self) -> List[QACheck]:
        """Return QA checks available for SEAWAT models.

        Includes all MF2005 flow checks, transport checks, and
        SEAWAT-specific density/coupling checks.
        """
        return [
            # ── Flow checks (same as MF2005) ──
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
            # ── Transport checks ──
            QACheck("transport_mass_balance",
                    "Check solute mass balance from transport listing output (.mas file)"),
            QACheck("negative_concentrations",
                    "Detect negative concentrations in UCN output (numerical artifact indicator)"),
            QACheck("courant_number",
                    "Estimate grid Courant numbers for advection stability assessment"),
            # ── SEAWAT-specific density/coupling checks ──
            QACheck("density_range",
                    "Check computed densities are within physical bounds (990-1300 kg/m3)"),
            QACheck("density_concentration_consistency",
                    "Verify rho = rho_ref + drhodc * C relationship holds across the domain"),
            QACheck("coupling_convergence",
                    "Check flow-transport coupling iteration counts per timestep; flag poor convergence"),
            # ── Comparison ──
            QACheck("save_snapshot",
                    "Save a lightweight snapshot of current model outputs for later comparison"),
            QACheck("compare_runs",
                    "Compare two run snapshots side-by-side: head changes, budget differences"),
        ]

    def run_qa_check(self, ws_root: Path, check_name: str, **kwargs: Any) -> str:
        """Run a QA check.

        Standard flow checks delegate to qa_diagnostics.  SEAWAT-specific
        checks (density_range, negative_concentrations, density_concentration_consistency,
        coupling_convergence, transport_mass_balance, courant_number) are
        implemented here.
        """
        # SEAWAT-specific checks
        seawat_checks = {
            "density_range": self._qa_density_range,
            "negative_concentrations": self._qa_negative_concentrations,
            "density_concentration_consistency": self._qa_density_consistency,
            "coupling_convergence": self._qa_coupling_convergence,
            "transport_mass_balance": self._qa_transport_mass_balance,
            "courant_number": self._qa_courant_number,
        }

        fn = seawat_checks.get(check_name)
        if fn is not None:
            try:
                return fn(ws_root, **kwargs)
            except Exception as e:
                return f"## {check_name}\n\nError: {type(e).__name__}: {e}"

        # Delegate standard checks to the shared qa_diagnostics module
        from gw.api.qa_diagnostics import run_qa_check as _run
        return _run(str(ws_root), check_name, **kwargs)

    # ---- SEAWAT-specific QA check implementations ----

    def _qa_density_range(self, ws_root: Path, **kwargs: Any) -> str:
        """Check that computed densities are within physical bounds."""
        import numpy as np

        lines: List[str] = ["## Density Range Check", ""]

        # Get VDF parameters from snapshot or VDF file
        snapshot = self.build_snapshot(ws_root)
        vdf = snapshot.get("vdf", {})
        denseref = vdf.get("DENSEREF", 1000.0)
        drhodc = vdf.get("DRHODC", 0.7143)

        lines.append(f"VDF parameters: DENSEREF={denseref}, DRHODC={drhodc}")
        lines.append("")

        # Read concentration data to compute density
        ucn_files = list(ws_root.glob("*.ucn"))
        if not ucn_files:
            lines.append("No .ucn concentration files found. Cannot compute densities.")
            return "\n".join(lines)

        try:
            import flopy
            uf = flopy.utils.UcnFile(str(ucn_files[0]))
            times = uf.get_times()
            if not times:
                lines.append("No timesteps in UCN file.")
                return "\n".join(lines)

            issues = 0
            for t in [times[0], times[-1]]:
                conc = uf.get_data(totim=t)
                valid = conc[np.isfinite(conc)]
                if valid.size == 0:
                    continue

                rho = float(denseref) + float(drhodc) * valid
                rho_min = float(rho.min())
                rho_max = float(rho.max())

                status = "OK"
                if rho_min < 990.0 or rho_max > 1300.0:
                    status = "WARNING"
                    issues += 1

                lines.append(f"### Time = {t}")
                lines.append(f"  Concentration: min={float(valid.min()):.4g}, "
                             f"max={float(valid.max()):.4g}")
                lines.append(f"  Computed density: min={rho_min:.2f}, max={rho_max:.2f} kg/m3")
                lines.append(f"  Status: **{status}**")
                if rho_min < 990.0:
                    lines.append(f"  - Density below 990 kg/m3 (unrealistic for groundwater)")
                if rho_max > 1300.0:
                    lines.append(f"  - Density above 1300 kg/m3 (exceeds typical brine)")
                lines.append("")

            if issues == 0:
                lines.append("**Result: All computed densities are within physical bounds.**")
            else:
                lines.append(f"**Result: {issues} timestep(s) with out-of-range densities.**")
                lines.append("Consider checking concentration values and DRHODC setting.")

        except Exception as e:
            lines.append(f"Error reading UCN file: {type(e).__name__}: {e}")

        return "\n".join(lines)

    def _qa_negative_concentrations(self, ws_root: Path, **kwargs: Any) -> str:
        """Detect negative concentrations in UCN output."""
        import numpy as np

        lines: List[str] = ["## Negative Concentration Check", ""]

        ucn_files = list(ws_root.glob("*.ucn"))
        if not ucn_files:
            lines.append("No .ucn concentration files found.")
            return "\n".join(lines)

        try:
            import flopy
            uf = flopy.utils.UcnFile(str(ucn_files[0]))
            times = uf.get_times()
            if not times:
                lines.append("No timesteps in UCN file.")
                return "\n".join(lines)

            total_neg = 0
            for t in times:
                conc = uf.get_data(totim=t)
                valid = conc[np.isfinite(conc)]
                neg_count = int(np.sum(valid < 0))
                if neg_count > 0:
                    total_neg += neg_count
                    neg_vals = valid[valid < 0]
                    lines.append(f"Time={t}: {neg_count} negative cells, "
                                 f"min={float(neg_vals.min()):.4g}")

            lines.append("")
            if total_neg == 0:
                lines.append("**Result: No negative concentrations detected.**")
            else:
                lines.append(f"**Result: {total_neg} total negative concentration occurrences.**")
                lines.append("")
                lines.append("Negative concentrations are numerical artifacts. Consider:")
                lines.append("- Switch advection scheme (TVD or HMOC in ADV package)")
                lines.append("- Reduce timestep length")
                lines.append("- Increase grid resolution in areas of sharp concentration gradients")

        except Exception as e:
            lines.append(f"Error reading UCN file: {type(e).__name__}: {e}")

        return "\n".join(lines)

    def _qa_density_consistency(self, ws_root: Path, **kwargs: Any) -> str:
        """Verify that density = rho_ref + drhodc * C across the domain."""
        import numpy as np

        lines: List[str] = ["## Density-Concentration Consistency Check", ""]

        snapshot = self.build_snapshot(ws_root)
        vdf = snapshot.get("vdf", {})
        denseref = float(vdf.get("DENSEREF", 1000.0))
        drhodc = float(vdf.get("DRHODC", 0.7143))

        lines.append(f"Equation of state: rho = {denseref} + {drhodc} * C")
        lines.append("")

        ucn_files = list(ws_root.glob("*.ucn"))
        if not ucn_files:
            lines.append("No .ucn file found. Cannot verify density-concentration relationship.")
            return "\n".join(lines)

        try:
            import flopy
            uf = flopy.utils.UcnFile(str(ucn_files[0]))
            times = uf.get_times()
            if not times:
                lines.append("No timesteps in UCN file.")
                return "\n".join(lines)

            conc = uf.get_data(totim=times[-1])
            valid = conc[np.isfinite(conc)]
            if valid.size == 0:
                lines.append("No valid concentration data at final timestep.")
                return "\n".join(lines)

            c_min, c_max = float(valid.min()), float(valid.max())
            rho_min = denseref + drhodc * c_min
            rho_max = denseref + drhodc * c_max

            lines.append(f"Final timestep (t={times[-1]}):")
            lines.append(f"  Concentration range: {c_min:.4g} to {c_max:.4g}")
            lines.append(f"  Implied density range: {rho_min:.2f} to {rho_max:.2f} kg/m3")
            lines.append("")

            # Sanity checks
            issues: List[str] = []
            if c_min < 0:
                issues.append(f"Negative concentrations present (min={c_min:.4g})")
            if rho_min < 990:
                issues.append(f"Implied minimum density ({rho_min:.1f}) below 990 kg/m3")
            if rho_max > 1300:
                issues.append(f"Implied maximum density ({rho_max:.1f}) above 1300 kg/m3")

            # Check expected seawater density
            if drhodc > 0 and c_max > 0:
                rho_sw = denseref + drhodc * 35.0
                lines.append(f"Expected seawater density (C=35): {rho_sw:.2f} kg/m3")
                if abs(rho_sw - 1025.0) > 5.0:
                    issues.append(
                        f"Seawater density ({rho_sw:.1f}) differs significantly from "
                        f"expected ~1025 kg/m3. Check DRHODC and concentration units."
                    )

            if issues:
                lines.append("### Issues Found:")
                for iss in issues:
                    lines.append(f"  - {iss}")
            else:
                lines.append("**Result: Density-concentration relationship is consistent.**")

        except Exception as e:
            lines.append(f"Error: {type(e).__name__}: {e}")

        return "\n".join(lines)

    def _qa_coupling_convergence(self, ws_root: Path, **kwargs: Any) -> str:
        """Check flow-transport coupling iteration counts from listing file."""
        import re

        lines: List[str] = ["## Flow-Transport Coupling Convergence", ""]

        # Get NSWTCPL from snapshot
        snapshot = self.build_snapshot(ws_root)
        vdf = snapshot.get("vdf", {})
        nswtcpl = vdf.get("NSWTCPL", "unknown")
        lines.append(f"NSWTCPL (max coupling iterations): {nswtcpl}")
        lines.append("")

        # Parse listing file for coupling iteration info
        lst_files = list(ws_root.glob("*.lst")) + list(ws_root.glob("*.list"))
        if not lst_files:
            lines.append("No listing file found. Cannot check coupling convergence.")
            return "\n".join(lines)

        lst_path = max(lst_files, key=lambda p: p.stat().st_size)
        try:
            text = lst_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            lines.append(f"Error reading listing file: {e}")
            return "\n".join(lines)

        # Look for coupling iteration messages
        coupling_pattern = re.compile(
            r"COUPLING\s+ITERATION\s+(\d+)", re.IGNORECASE
        )
        density_conv_pattern = re.compile(
            r"DENSITY\s+CONVERGENCE\s+CRITERION\s*=\s*([\d.Ee+-]+)", re.IGNORECASE
        )
        max_cpl_reached_pattern = re.compile(
            r"MAX(?:IMUM)?\s+COUPLING\s+ITERATIONS?\s+REACHED", re.IGNORECASE
        )

        coupling_iters: List[int] = []
        max_reached_count = 0

        for line in text.splitlines():
            m = coupling_pattern.search(line)
            if m:
                coupling_iters.append(int(m.group(1)))
            if max_cpl_reached_pattern.search(line):
                max_reached_count += 1

        if coupling_iters:
            import numpy as np
            arr = np.array(coupling_iters)
            lines.append(f"Coupling iterations found: {len(arr)} records")
            lines.append(f"  Min: {int(arr.min())}, Mean: {float(arr.mean()):.1f}, "
                         f"Max: {int(arr.max())}")
            if max_reached_count > 0:
                lines.append(f"  Max coupling iterations reached: {max_reached_count} time(s)")
                lines.append("  **WARNING**: Flow-transport coupling did not converge at some timesteps.")
                lines.append("  Consider reducing timestep length or increasing NSWTCPL.")
            else:
                lines.append("  All timesteps converged within coupling iteration limit.")
            lines.append("")
            if float(arr.mean()) > 5:
                lines.append("**Result: Average coupling iterations > 5 suggests large density contrasts.**")
                lines.append("Consider smaller timesteps for better coupling convergence.")
            else:
                lines.append("**Result: Coupling convergence is satisfactory.**")
        else:
            lines.append("No coupling iteration records found in listing file.")
            lines.append("This may indicate NSWTCPL=1 (explicit coupling with no iteration).")
            if nswtcpl != "unknown":
                try:
                    if int(float(nswtcpl)) <= 1:
                        lines.append("")
                        lines.append("**Note**: NSWTCPL <= 1 means explicit coupling (no iteration).")
                        lines.append("This is fast but may be less accurate for problems with")
                        lines.append("strong density contrasts. Consider NSWTCPL >= 2 for better accuracy.")
                except (ValueError, TypeError):
                    pass

        return "\n".join(lines)

    def _qa_transport_mass_balance(self, ws_root: Path, **kwargs: Any) -> str:
        """Check solute mass balance from .mas file or listing output."""
        lines: List[str] = ["## Transport Mass Balance", ""]

        # Try .mas file first (MT3DMS mass balance output)
        mas_files = list(ws_root.glob("*.mas"))
        if mas_files:
            try:
                text = mas_files[0].read_text(encoding="utf-8", errors="replace")
                mas_lines = text.strip().splitlines()
                if len(mas_lines) > 1:
                    lines.append(f"Mass balance file: {mas_files[0].name}")
                    lines.append(f"Total records: {len(mas_lines) - 1}")
                    lines.append("")
                    # Show header + last few lines
                    lines.append("Header: " + mas_lines[0].strip())
                    lines.append("")
                    lines.append("Last 5 records:")
                    for ml in mas_lines[-5:]:
                        lines.append(f"  {ml.strip()}")
                    lines.append("")

                    # Try to parse percent discrepancy from last line
                    import re
                    last_line = mas_lines[-1]
                    parts = last_line.split()
                    if len(parts) >= 5:
                        try:
                            disc = float(parts[-1])
                            if abs(disc) < 0.5:
                                lines.append(f"**Result: Final mass balance discrepancy = {disc:.4f}% (GOOD)**")
                            elif abs(disc) < 1.0:
                                lines.append(f"**Result: Final mass balance discrepancy = {disc:.4f}% (MARGINAL)**")
                            else:
                                lines.append(f"**Result: Final mass balance discrepancy = {disc:.4f}% (POOR)**")
                                lines.append("Consider reducing timestep or using a more stable advection scheme.")
                        except ValueError:
                            lines.append("Could not parse mass balance discrepancy from last record.")
                    return "\n".join(lines)
            except Exception as e:
                lines.append(f"Error reading .mas file: {e}")
                lines.append("")

        # Fallback: parse listing file for transport budget
        lst_files = list(ws_root.glob("*.lst")) + list(ws_root.glob("*.list"))
        if lst_files:
            lst_path = max(lst_files, key=lambda p: p.stat().st_size)
            try:
                text = lst_path.read_text(encoding="utf-8", errors="replace")
                # Look for MT3DMS mass balance sections
                import re
                disc_pattern = re.compile(
                    r"PERCENT\s+DISCREPANCY\s*=?\s*([\d.Ee+-]+)", re.IGNORECASE
                )
                discrepancies: List[float] = []
                for line in text.splitlines():
                    m = disc_pattern.search(line)
                    if m:
                        try:
                            discrepancies.append(float(m.group(1)))
                        except ValueError:
                            pass

                if discrepancies:
                    import numpy as np
                    arr = np.array(discrepancies)
                    lines.append(f"Transport mass balance discrepancies found: {len(arr)} records")
                    lines.append(f"  Min: {float(arr.min()):.4f}%")
                    lines.append(f"  Mean: {float(arr.mean()):.4f}%")
                    lines.append(f"  Max: {float(abs(arr).max()):.4f}%")
                    lines.append("")
                    max_disc = float(abs(arr).max())
                    if max_disc < 0.5:
                        lines.append(f"**Result: Transport mass balance is GOOD (max {max_disc:.4f}%)**")
                    elif max_disc < 1.0:
                        lines.append(f"**Result: Transport mass balance is MARGINAL (max {max_disc:.4f}%)**")
                    else:
                        lines.append(f"**Result: Transport mass balance is POOR (max {max_disc:.4f}%)**")
                        lines.append("Consider smaller timesteps or a more stable advection scheme.")
                else:
                    lines.append("No transport mass balance records found in listing file.")
            except Exception as e:
                lines.append(f"Error reading listing file: {e}")
        else:
            lines.append("No .mas or listing file found.")

        return "\n".join(lines)

    def _qa_courant_number(self, ws_root: Path, **kwargs: Any) -> str:
        """Estimate grid Courant numbers for advection stability."""
        import numpy as np

        lines: List[str] = ["## Courant Number Estimation", ""]

        # We need: velocity (from budget), porosity (from BTN), cell sizes (from DIS)
        snapshot = self.build_snapshot(ws_root)
        grid = snapshot.get("grid", {})
        nlay = grid.get("nlay", 0)
        nrow = grid.get("nrow", 0)
        ncol = grid.get("ncol", 0)

        if not (nlay and nrow and ncol):
            lines.append("Grid dimensions not available. Cannot estimate Courant numbers.")
            return "\n".join(lines)

        try:
            import flopy

            # Load model for cell sizes
            nam_files = [f for f in ws_root.glob("*.nam")
                         if f.name.lower() != "mfsim.nam"]
            if not nam_files:
                lines.append("No .nam file found.")
                return "\n".join(lines)

            m = flopy.seawat.Seawat.load(
                nam_files[0].name,
                model_ws=str(ws_root),
                verbose=False,
            )

            # Get cell sizes
            delr = m.dis.delr.array if m.dis is not None else None
            delc = m.dis.delc.array if m.dis is not None else None
            if delr is None or delc is None:
                lines.append("Cell sizes not available from DIS package.")
                return "\n".join(lines)

            dx_min = float(min(delr.min(), delc.min()))
            dx_max = float(max(delr.max(), delc.max()))
            lines.append(f"Grid cell sizes: min={dx_min:.2f}, max={dx_max:.2f}")

            # Get porosity from BTN
            btn_pkg = m.get_package("BTN")
            porosity = 0.3  # default
            if btn_pkg is not None:
                prsity = getattr(btn_pkg, "prsity", None)
                if prsity is not None:
                    try:
                        arr = prsity.array if hasattr(prsity, "array") else prsity
                        porosity = float(np.mean(arr))
                    except Exception:
                        pass
            lines.append(f"Mean porosity: {porosity:.3f}")

            # Get timestep info
            tdis = snapshot.get("tdis", {})
            perlen_range = tdis.get("perlen_range", {})
            nstp_range = tdis.get("nstp_range", {})

            if perlen_range and nstp_range:
                dt_min = perlen_range.get("min", 1.0) / max(nstp_range.get("max", 1), 1)
                dt_max = perlen_range.get("max", 1.0) / max(nstp_range.get("min", 1), 1)
                lines.append(f"Estimated timestep range: {dt_min:.4g} to {dt_max:.4g}")
            else:
                dt_max = 1.0
                lines.append("Timestep info not available; using dt=1.0 for estimation")

            lines.append("")

            # Try to estimate velocity from CBC file
            cbc_files = list(ws_root.glob("*.cbc"))
            v_est = None
            if cbc_files:
                try:
                    cbf = flopy.utils.CellBudgetFile(str(cbc_files[0]))
                    records = cbf.get_unique_record_names()
                    # Look for flow-right-face or flow-front-face
                    flow_records = [r for r in records
                                    if b"FLOW RIGHT" in r or b"FLOW FRONT" in r]
                    if flow_records:
                        data = cbf.get_data(text=flow_records[0])[-1]
                        # Specific discharge = Q / (area * porosity)
                        # Area ~ delc * thickness; approximate with cell area
                        area = dx_min * dx_min  # rough approximation
                        v_data = np.abs(data) / (area * porosity)
                        valid = v_data[np.isfinite(v_data) & (v_data > 0)]
                        if valid.size > 0:
                            v_est = float(np.percentile(valid, 95))
                            lines.append(f"Estimated 95th percentile velocity: {v_est:.4g}")
                except Exception:
                    pass

            if v_est is not None:
                courant = v_est * dt_max / dx_min
                lines.append(f"Estimated maximum Courant number: {courant:.4g}")
                lines.append("")
                if courant <= 1.0:
                    lines.append(f"**Result: Courant number ({courant:.4g}) <= 1.0 (STABLE)**")
                else:
                    lines.append(f"**Result: Courant number ({courant:.4g}) > 1.0 (POTENTIALLY UNSTABLE)**")
                    lines.append("Consider reducing timestep or using MOC/HMOC advection scheme.")
            else:
                lines.append("Could not estimate velocity from budget file.")
                lines.append("Courant number estimation requires flow-right-face or flow-front-face data.")

        except ImportError:
            lines.append("FloPy not available. Cannot estimate Courant numbers.")
        except Exception as e:
            lines.append(f"Error: {type(e).__name__}: {e}")

        return "\n".join(lines)

    # ── File-type knowledge ───────────────────────────────────────────────

    def file_type_knowledge(self) -> Dict[str, FileTypeInfo]:
        from gw.simulators.seawat.knowledge import _EXT_KB
        return dict(_EXT_KB)

    def package_properties(self) -> Dict[str, PackagePropertyInfo]:
        from gw.simulators.seawat.knowledge import PACKAGE_PROPERTIES
        return dict(PACKAGE_PROPERTIES)

    def file_extensions(self) -> Set[str]:
        from gw.simulators.seawat.knowledge import _EXT_KB
        return set(_EXT_KB.keys()) | {
            # Extra extensions not in _EXT_KB
            ".sip", ".sor", ".gmg", ".de4",
            ".sfr", ".lak", ".mnw2", ".uzf", ".sub", ".hob",
        }

    # ── LLM knowledge ─────────────────────────────────────────────────────

    def system_prompt_fragment(self) -> str:
        from gw.simulators.seawat.knowledge import seawat_system_prompt_fragment
        return seawat_system_prompt_fragment()

    def tool_descriptions(self) -> Dict[str, str]:
        from gw.simulators.seawat.knowledge import seawat_tool_description_overrides
        return seawat_tool_description_overrides()

    def file_mention_pattern(self) -> str:
        from gw.simulators.seawat.knowledge import seawat_file_mention_regex
        return seawat_file_mention_regex()

    # ── Output capabilities ───────────────────────────────────────────────

    def output_capabilities(self) -> OutputCapability:
        return OutputCapability(
            heads=True,
            budget=True,
            concentration=True,
            pathlines=False,
            endpoints=False,
        )

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
            m = flopy.seawat.Seawat.load(
                nam_files[0].name,
                model_ws=str(ws_root),
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
            lines = [ln.strip() for ln in text.splitlines()
                     if ln.strip() and not ln.strip().startswith("#")]
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
        """Load model via FloPy's Seawat loader with caching.

        Returns ``(model, error_msg)`` — note this returns a Seawat object
        not an MFSimulation, but the interface allows Any.
        """
        cache_key = str(ws_root.resolve())

        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key], None

        try:
            import flopy
        except ImportError:
            return None, "FloPy not available"

        nam_files = [f for f in ws_root.glob("*.nam")
                     if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return None, "No .nam file found"

        try:
            m = flopy.seawat.Seawat.load(
                nam_files[0].name,
                model_ws=str(ws_root),
                verbose=False,
            )
            _MODEL_CACHE[cache_key] = m
            return m, None
        except Exception as e:
            return None, f"FloPy SEAWAT load failed: {type(e).__name__}: {e}"

    def clear_simulation_cache(self, ws_root: Optional[Path] = None) -> None:
        """Clear cached SEAWAT model objects."""
        if ws_root is not None:
            key = str(ws_root.resolve())
            _MODEL_CACHE.pop(key, None)
        else:
            _MODEL_CACHE.clear()
