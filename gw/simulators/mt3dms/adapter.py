"""MT3DMS / MT3D-USGS concrete simulator adapter.

Implements the ``SimulatorAdapter`` interface for solute-transport models.
MT3DMS is a solute transport model that uses the flow field from a MODFLOW
model.  Binary concentration output (.ucn) is read via FloPy's UcnFile.
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


class MT3DMSAdapter(SimulatorAdapter):
    """Adapter for MT3DMS and MT3D-USGS transport models."""

    # ── Identity ──────────────────────────────────────────────────────────

    def info(self) -> SimulatorInfo:
        return SimulatorInfo(
            name="mt3dms",
            display_name="MT3DMS/MT3D-USGS",
        )

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(self, ws_root: Path) -> float:
        """Detect MT3DMS/MT3D-USGS model in the workspace.

        Detection strategy:
        - .btn file exists (or BTN in .nam) -> 0.95
        - .mtnam extension -> 0.95 (MT3D-USGS convention)
        - .nam contains BTN/ADV/DSP/GCG -> 0.9
        - .nam also has VDF/VSC -> 0.0 (that's SEAWAT, not plain MT3D)
        - mfsim.nam exists -> 0.0 (MF6 has its own GWT)
        """
        # MF6 disqualifier: if mfsim.nam exists, this is MF6 (with its own GWT)
        if (ws_root / "mfsim.nam").exists():
            return 0.0

        # Check for SEAWAT disqualifier first (VDF/VSC packages)
        seawat_exts = {".vdf", ".vsc"}
        has_seawat = False
        try:
            for p in ws_root.iterdir():
                if p.is_file() and p.suffix.lower() in seawat_exts:
                    has_seawat = True
                    break
        except OSError:
            pass
        if has_seawat:
            return 0.0

        # Also check name files for VDF/VSC entries
        all_nam = list(ws_root.glob("*.nam")) + list(ws_root.glob("*.mtnam"))
        for nf in all_nam:
            try:
                txt = nf.read_text(encoding="utf-8", errors="replace")[:4000].upper()
                if "VDF" in txt or "VSC" in txt:
                    return 0.0
            except OSError:
                pass

        # .mtnam extension is a strong MT3D-USGS indicator
        if list(ws_root.glob("*.mtnam")):
            return 0.95

        # .btn file is the definitive MT3D indicator
        btn_files = list(ws_root.glob("*.btn"))
        if btn_files:
            return 0.95

        # Check name files for MT3D package types
        mt3d_types = {"BTN", "ADV", "DSP", "GCG", "SSM", "RCT", "TOB"}
        for nf in all_nam:
            try:
                txt = nf.read_text(encoding="utf-8", errors="replace")[:4000].upper()
                found_mt3d = sum(1 for pkg in mt3d_types if pkg in txt)
                if found_mt3d >= 2:
                    return 0.9
            except OSError:
                pass

        # UCN files present without other strong indicators
        ucn_files = list(ws_root.glob("*.ucn")) + list(ws_root.glob("*.UCN"))
        if ucn_files:
            return 0.5

        return 0.0

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
        """Build snapshot using FloPy's Mt3dms loader."""
        try:
            import flopy
        except ImportError:
            return {"ok": False, "error": "FloPy not available"}

        # Find MT3D name file (.nam or .mtnam)
        nam_file = self._find_mt3d_nam(ws_root)
        if nam_file is None:
            return {"ok": False, "error": "No MT3D name file found"}

        try:
            mt = flopy.mt3d.Mt3dms.load(
                nam_file.name,
                model_ws=str(ws_root),
                verbose=False,
            )

            # Grid info (from BTN)
            grid: Dict[str, Any] = {
                "type": "DIS",
                "nlay": mt.nlay,
                "nrow": mt.nrow,
                "ncol": mt.ncol,
            }

            # Transport time discretisation
            tdis: Dict[str, Any] = {"nper": mt.nper}
            if mt.btn is not None:
                # Species count
                ncomp = getattr(mt.btn, "ncomp", 1) or 1
                mcomp = getattr(mt.btn, "mcomp", ncomp) or ncomp
                tdis["ncomp"] = ncomp
                tdis["mcomp"] = mcomp

            # Packages
            packages: Dict[str, Any] = {}
            for pkg_name in mt.get_package_list():
                packages[pkg_name.upper()] = {"active": True}

            # Porosity summary from BTN
            transport_props: Dict[str, Any] = {}
            if mt.btn is not None:
                try:
                    prsity = mt.btn.prsity.array
                    import numpy as np
                    transport_props["porosity"] = {
                        "min": float(np.min(prsity)),
                        "max": float(np.max(prsity)),
                        "mean": float(np.mean(prsity)),
                    }
                except Exception:
                    pass

                # Initial concentration summary
                try:
                    sconc = mt.btn.sconc.array
                    import numpy as np
                    transport_props["initial_concentration"] = {
                        "min": float(np.min(sconc)),
                        "max": float(np.max(sconc)),
                        "mean": float(np.mean(sconc)),
                    }
                except Exception:
                    pass

            # Dispersivity from DSP
            if mt.dsp is not None:
                try:
                    al = mt.dsp.al.array
                    import numpy as np
                    transport_props["longitudinal_dispersivity"] = {
                        "min": float(np.min(al)),
                        "max": float(np.max(al)),
                    }
                except Exception:
                    pass

            # Advection method from ADV
            if mt.adv is not None:
                try:
                    mixelm = int(getattr(mt.adv, "mixelm", 0))
                    method_names = {
                        0: "Upstream Finite Difference",
                        1: "Method of Characteristics (MOC)",
                        2: "Modified MOC (MMOC)",
                        3: "Hybrid MOC (HMOC)",
                        -1: "TVD (Third-Order TVD)",
                    }
                    transport_props["advection_method"] = method_names.get(
                        mixelm, f"MIXELM={mixelm}"
                    )
                except Exception:
                    pass

            # Output files
            from gw.simulators.mt3dms.io import find_ucn_files
            outputs_present: Dict[str, Any] = {}
            ucn_files = find_ucn_files(ws_root)
            if ucn_files:
                outputs_present["ucn"] = [
                    {"file": f.name, "size_bytes": f.stat().st_size}
                    for f in ucn_files[:10]
                ]

            for ext, key in [(".mas", "mas"), (".lst", "lst"), (".obs", "obs")]:
                found = list(ws_root.glob(f"*{ext}"))
                if found:
                    outputs_present[key] = {
                        "file": found[0].name,
                        "size_bytes": found[0].stat().st_size,
                    }

            # Determine variant
            from gw.simulators.mt3dms.io import detect_mt3d_variant
            variant = detect_mt3d_variant(ws_root)

            return {
                "ok": True,
                "simulator": "mt3dms",
                "variant": variant,
                "grid": grid,
                "tdis": tdis,
                "packages": packages,
                "transport_properties": transport_props,
                "outputs_present": outputs_present,
                "workspace_root": str(ws_root),
            }
        except Exception as e:
            return {"ok": False, "error": f"FloPy load failed: {type(e).__name__}: {e}"}

    def _build_text_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        """Minimal text-based snapshot when FloPy fails."""
        from gw.simulators.mt3dms.io import (
            read_mt3d_nam,
            detect_mt3d_variant,
            find_ucn_files,
        )

        snapshot: Dict[str, Any] = {
            "ok": True,
            "simulator": "mt3dms",
            "variant": detect_mt3d_variant(ws_root),
            "workspace_root": str(ws_root),
        }

        # Parse name file
        nam_file = self._find_mt3d_nam(ws_root)
        if nam_file is not None:
            nam_df = read_mt3d_nam(nam_file)
            packages = {}
            for _, row in nam_df.iterrows():
                packages[row["ftype"]] = {"fname": row["fname"], "active": True}
            snapshot["packages"] = packages

        # Try to parse BTN header for grid info
        btn_files = list(ws_root.glob("*.btn"))
        if btn_files:
            try:
                text = btn_files[0].read_text(encoding="utf-8", errors="replace")
                lines = [l.strip() for l in text.splitlines()
                         if l.strip() and not l.strip().startswith("#")]
                # BTN first data line: NLAY NROW NCOL NPER NCOMP MCOMP
                if lines:
                    parts = lines[0].split()
                    if len(parts) >= 4:
                        snapshot["grid"] = {
                            "type": "DIS",
                            "nlay": int(parts[0]),
                            "nrow": int(parts[1]),
                            "ncol": int(parts[2]),
                        }
                        snapshot["tdis"] = {"nper": int(parts[3])}
                        if len(parts) >= 5:
                            snapshot["tdis"]["ncomp"] = int(parts[4])
                        if len(parts) >= 6:
                            snapshot["tdis"]["mcomp"] = int(parts[5])
            except Exception:
                pass

        # Output files
        outputs: Dict[str, Any] = {}
        ucn_files = find_ucn_files(ws_root)
        if ucn_files:
            outputs["ucn"] = [
                {"file": f.name, "size_bytes": f.stat().st_size}
                for f in ucn_files[:10]
            ]
        for ext, key in [(".mas", "mas"), (".lst", "lst"), (".obs", "obs")]:
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
        parts: List[str] = []

        variant = snapshot.get("variant", "mt3dms")
        parts.append(f"Simulator: {variant.upper()}")

        grid = snapshot.get("grid", {})
        if grid:
            parts.append(
                f"Grid: {grid.get('nlay', '?')}L x {grid.get('nrow', '?')}R x {grid.get('ncol', '?')}C"
            )

        tdis = snapshot.get("tdis", {})
        if tdis:
            nper = tdis.get("nper", "?")
            ncomp = tdis.get("ncomp", 1)
            parts.append(f"Periods: {nper}, Species: {ncomp}")

        pkgs = snapshot.get("packages", {})
        if pkgs:
            parts.append(f"Packages: {', '.join(sorted(pkgs.keys()))}")

        tp = snapshot.get("transport_properties", {})
        if tp.get("advection_method"):
            parts.append(f"Advection: {tp['advection_method']}")
        if tp.get("porosity"):
            por = tp["porosity"]
            parts.append(f"Porosity: {por.get('min', '?')}-{por.get('max', '?')}")

        outputs = snapshot.get("outputs_present", {})
        out_keys = [k for k in ["ucn", "mas", "lst", "obs"] if k in outputs]
        if out_keys:
            parts.append(f"Outputs: {', '.join(out_keys)}")

        return " | ".join(parts)

    # ── Stress I/O ────────────────────────────────────────────────────────

    def read_stress_package(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        value_cols: int = 1,
        keep_aux: bool = True,
    ) -> Any:
        """MT3D stress data lives in SSM (linked to MODFLOW).

        Return an empty DataFrame since MT3D does not have its own
        stress packages in the MODFLOW sense.  The SSM package references
        MODFLOW stress packages for source concentrations.
        """
        import pandas as pd
        return pd.DataFrame(
            columns=["per", "layer", "row", "col", "css", "itype"]
        )

    def read_time_discretization(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        """Read time discretisation from BTN or linked MODFLOW DIS."""
        # Try to read from linked MODFLOW model first
        from gw.simulators.mt3dms.io import find_linked_modflow_model
        linked = find_linked_modflow_model(ws_root)
        if linked is not None:
            try:
                from gw.simulators.mf2005.io import read_mf2005_dis_times
                dis_files = list(ws_root.glob("*.dis"))
                if dis_files:
                    return read_mf2005_dis_times(dis_files[0])
            except Exception:
                pass

        # Fallback: return empty DataFrame
        import pandas as pd
        return pd.DataFrame(
            columns=["per", "perlen", "nstp", "tsmult", "t_start", "t_end", "t_mid"]
        )

    def read_namefile(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        from gw.simulators.mt3dms.io import read_mt3d_nam
        if rel_path:
            path = ws_root / rel_path
        else:
            nam_file = self._find_mt3d_nam(ws_root)
            path = nam_file if nam_file else ws_root / "mt3d.nam"
        return read_mt3d_nam(path)

    # ── Binary output — Heads (not supported) ────────────────────────────

    def probe_head_file(self, ws_root: Path, rel_path: str) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": "MT3DMS does not produce head files. "
                     "Head output comes from the linked MODFLOW model.",
        }

    def probe_budget_file(self, ws_root: Path, rel_path: str) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": "MT3DMS does not produce cell-by-cell budget files. "
                     "Flow budgets come from the linked MODFLOW model. "
                     "Use the .mas file or listing file for transport mass balance.",
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
            "error": "MT3DMS does not produce head output. "
                     "Use the linked MODFLOW model for head data.",
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
            "error": "MT3DMS does not produce cell-by-cell budget files. "
                     "Use the linked MODFLOW model for flow budget data.",
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
            "error": "MT3DMS does not produce head output. "
                     "Use the linked MODFLOW model for head time-series.",
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
            "error": "MT3DMS does not produce cell-by-cell budget files. "
                     "Use the linked MODFLOW model for per-cell flow data.",
        }

    # ── Binary output — Concentration (UCN) ──────────────────────────────

    def probe_concentration_file(
        self, ws_root: Path, rel_path: str
    ) -> Dict[str, Any]:
        from gw.api.output_probes import probe_ucn
        return probe_ucn(ws_root, rel_path)

    def extract_concentration_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_times: int = 5,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        from gw.api.output_probes import extract_ucn_data
        return extract_ucn_data(
            ws_root, rel_path, max_times=max_times, max_chars=max_chars
        )

    def extract_concentration_timeseries(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        cells: List[Dict[str, int]],
        max_chars: int = 50_000,
    ) -> Dict[str, Any]:
        from gw.api.output_probes import extract_ucn_timeseries
        return extract_ucn_timeseries(
            ws_root, rel_path, cells=cells, max_chars=max_chars
        )

    # ── Output capabilities ──────────────────────────────────────────────

    def output_capabilities(self) -> OutputCapability:
        return OutputCapability(
            heads=False,
            budget=False,
            concentration=True,
            pathlines=False,
            endpoints=False,
        )

    # ── Model execution ───────────────────────────────────────────────────

    def run_model(
        self,
        ws_root: Path,
        *,
        exe_path: Optional[str] = None,
        timeout_sec: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        from gw.run.mt3dms_runner import run_mt3dms
        from gw.simulators.mt3dms.io import detect_mt3d_variant
        prefer_usgs = detect_mt3d_variant(ws_root) == "mt3d-usgs"
        return run_mt3dms(
            workspace=str(ws_root),
            mt3d_path=exe_path,
            timeout_sec=timeout_sec,
            prefer_usgs=prefer_usgs,
        )

    def find_executable(self, exe_path: Optional[str] = None) -> str:
        from gw.run.mt3dms_runner import find_mt3dms
        return find_mt3dms(exe_path)

    # ── QA diagnostics ────────────────────────────────────────────────────

    def available_qa_checks(self) -> List[QACheck]:
        return [
            QACheck(
                "transport_mass_balance",
                "Parse .mas file or listing file for transport mass balance, "
                "compute % discrepancy per transport step",
            ),
            QACheck(
                "negative_concentrations",
                "Scan UCN output for negative concentrations indicating "
                "numerical instability (common with advection-dominated transport)",
            ),
            QACheck(
                "courant_number",
                "Estimate Courant number (Cr = v*dt/dx) from flow field and "
                "transport time stepping. Target: Cr < 1",
            ),
            QACheck(
                "peclet_number",
                "Estimate grid Peclet number (Pe = v*dx/D) from velocity "
                "field and dispersivity. Target: Pe < 2",
            ),
            QACheck(
                "concentration_bounds",
                "Check concentrations against physical bounds (non-negative, "
                "below source concentration). Flags unrealistic values",
            ),
            QACheck(
                "solver_convergence",
                "Parse listing file for GCG solver iterations, convergence "
                "failures, and transport solver warnings",
            ),
        ]

    def run_qa_check(self, ws_root: Path, check_name: str, **kwargs: Any) -> str:
        """Run a QA check on the MT3D model.

        Some checks delegate to the generic ``qa_diagnostics`` module;
        transport-specific checks are implemented here.
        """
        if check_name == "transport_mass_balance":
            return self._qa_transport_mass_balance(ws_root)
        if check_name == "negative_concentrations":
            return self._qa_negative_concentrations(ws_root)
        if check_name == "concentration_bounds":
            return self._qa_concentration_bounds(ws_root)
        if check_name == "solver_convergence":
            return self._qa_solver_convergence(ws_root)
        if check_name in ("courant_number", "peclet_number"):
            return self._qa_dimensionless_number(ws_root, check_name)

        return f"Unknown QA check: {check_name}"

    def _qa_transport_mass_balance(self, ws_root: Path) -> str:
        """Parse .mas file for transport mass balance."""
        from gw.simulators.mt3dms.io import parse_mas_file

        mas_files = list(ws_root.glob("*.mas"))
        if not mas_files:
            return (
                "## Transport Mass Balance\n\n"
                "No .mas (mass balance) file found in workspace. "
                "Ensure MT3DMS is configured to write the mass balance summary."
            )

        result = parse_mas_file(mas_files[0])
        if not result["ok"]:
            return f"## Transport Mass Balance\n\nError: {result['error']}"

        records = result.get("records", [])
        if not records:
            return "## Transport Mass Balance\n\nNo mass balance records found."

        lines = [
            f"## Transport Mass Balance — {result['file']}",
            f"Total records: {len(records)}",
            "",
        ]

        # Show first and last few records
        show = records[:5] + (records[-3:] if len(records) > 8 else [])
        if show:
            cols = list(show[0].keys())
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for rec in show:
                vals = []
                for c in cols:
                    v = rec.get(c, "")
                    if isinstance(v, float):
                        vals.append(f"{v:.4g}")
                    else:
                        vals.append(str(v))
                lines.append("| " + " | ".join(vals) + " |")

        return "\n".join(lines)

    def _qa_negative_concentrations(self, ws_root: Path) -> str:
        """Scan UCN files for negative concentrations."""
        from gw.simulators.mt3dms.io import find_ucn_files

        ucn_files = find_ucn_files(ws_root)
        if not ucn_files:
            return (
                "## Negative Concentration Check\n\n"
                "No UCN files found in workspace."
            )

        try:
            import flopy
            import numpy as np
        except ImportError:
            return (
                "## Negative Concentration Check\n\n"
                "FloPy not available — cannot read UCN files."
            )

        lines = ["## Negative Concentration Check", ""]
        total_neg = 0

        for ucn_path in ucn_files:
            try:
                uf = flopy.utils.UcnFile(str(ucn_path))
                times = uf.get_times()
                if not times:
                    lines.append(f"**{ucn_path.name}**: no timesteps")
                    continue

                # Check last timestep
                data = uf.get_data(totim=times[-1])
                valid = data[np.isfinite(data)]
                neg_count = int(np.sum(valid < 0))
                neg_min = float(valid.min()) if valid.size else 0.0
                total_cells = int(valid.size)
                neg_pct = (neg_count / total_cells * 100) if total_cells else 0.0

                total_neg += neg_count

                if neg_count == 0:
                    lines.append(f"**{ucn_path.name}** (t={times[-1]:.4g}): "
                                 f"No negative concentrations. All values >= 0.")
                else:
                    lines.append(
                        f"**{ucn_path.name}** (t={times[-1]:.4g}): "
                        f"{neg_count} negative cells ({neg_pct:.1f}% of {total_cells}), "
                        f"most negative = {neg_min:.4g}"
                    )

                    # Per-layer breakdown
                    if data.ndim >= 3:
                        for lay in range(data.shape[0]):
                            layer_data = data[lay]
                            lvalid = layer_data[np.isfinite(layer_data)]
                            lneg = int(np.sum(lvalid < 0))
                            if lneg > 0:
                                lines.append(
                                    f"  Layer {lay + 1}: {lneg} negative cells, "
                                    f"min = {float(lvalid.min()):.4g}"
                                )
            except Exception as e:
                lines.append(f"**{ucn_path.name}**: Error reading — {e}")

        lines.append("")
        if total_neg == 0:
            lines.append("**Result: PASS** — No negative concentrations detected.")
        else:
            lines.append(
                f"**Result: WARNING** — {total_neg} total negative cells found. "
                "Consider: refining grid, increasing dispersivity, "
                "switching to TVD or MOC advection method."
            )

        return "\n".join(lines)

    def _qa_concentration_bounds(self, ws_root: Path) -> str:
        """Check concentrations against physical bounds."""
        from gw.simulators.mt3dms.io import find_ucn_files

        ucn_files = find_ucn_files(ws_root)
        if not ucn_files:
            return "## Concentration Bounds Check\n\nNo UCN files found."

        try:
            import flopy
            import numpy as np
        except ImportError:
            return "## Concentration Bounds Check\n\nFloPy not available."

        lines = ["## Concentration Bounds Check", ""]

        for ucn_path in ucn_files:
            try:
                uf = flopy.utils.UcnFile(str(ucn_path))
                times = uf.get_times()
                if not times:
                    continue

                data = uf.get_data(totim=times[-1])
                valid = data[np.isfinite(data)]
                if valid.size == 0:
                    lines.append(f"**{ucn_path.name}**: no valid data")
                    continue

                cmin = float(valid.min())
                cmax = float(valid.max())
                cmean = float(valid.mean())
                neg_count = int(np.sum(valid < 0))

                lines.append(f"**{ucn_path.name}** (t={times[-1]:.4g}):")
                lines.append(f"  Range: [{cmin:.4g}, {cmax:.4g}], Mean: {cmean:.4g}")

                issues = []
                if neg_count > 0:
                    issues.append(f"{neg_count} negative cells")
                if cmax > 1e6:
                    issues.append(f"very high max ({cmax:.4g}) — check source terms")

                if issues:
                    lines.append(f"  Issues: {'; '.join(issues)}")
                else:
                    lines.append("  No bounds issues detected.")
                lines.append("")

            except Exception as e:
                lines.append(f"**{ucn_path.name}**: Error — {e}")

        return "\n".join(lines)

    def _qa_solver_convergence(self, ws_root: Path) -> str:
        """Parse listing file for GCG solver convergence info."""
        lst_files = list(ws_root.glob("*.lst"))
        if not lst_files:
            return "## Solver Convergence\n\nNo listing file (.lst) found."

        # Use the largest .lst file (likely the transport listing)
        lst_file = max(lst_files, key=lambda f: f.stat().st_size)

        try:
            text = lst_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"## Solver Convergence\n\nCannot read listing file: {e}"

        lines = [f"## Solver Convergence — {lst_file.name}", ""]

        import re

        # Look for GCG convergence messages
        failures = []
        max_iters = []
        for line in text.splitlines():
            upper = line.upper().strip()
            if "FAILED TO CONVERGE" in upper or "DID NOT CONVERGE" in upper:
                failures.append(line.strip())
            # Look for iteration counts
            m = re.search(r"TOTAL\s+ITER.*?=\s*(\d+)", upper)
            if m:
                max_iters.append(int(m.group(1)))

        if failures:
            lines.append(f"**Convergence failures: {len(failures)}**")
            for f in failures[:10]:
                lines.append(f"  - {f}")
            if len(failures) > 10:
                lines.append(f"  ... and {len(failures) - 10} more")
        else:
            lines.append("No convergence failures detected.")

        if max_iters:
            lines.append("")
            lines.append(f"Iteration counts: min={min(max_iters)}, "
                         f"max={max(max_iters)}, mean={sum(max_iters)/len(max_iters):.1f}")

        return "\n".join(lines)

    def _qa_dimensionless_number(self, ws_root: Path, check_name: str) -> str:
        """Estimate Courant or Peclet number (requires flow + transport data)."""
        title = "Courant Number" if check_name == "courant_number" else "Peclet Number"
        lines = [f"## {title} Estimate", ""]

        # This requires velocity data from the MODFLOW model + transport params
        # Provide guidance rather than full computation when data is unavailable
        try:
            import flopy
            import numpy as np
        except ImportError:
            lines.append("FloPy not available — cannot compute velocity field.")
            return "\n".join(lines)

        # Try to load the MT3D model for grid/transport info
        nam_file = self._find_mt3d_nam(ws_root)
        if nam_file is None:
            lines.append("No MT3D name file found.")
            return "\n".join(lines)

        try:
            mt = flopy.mt3d.Mt3dms.load(
                nam_file.name,
                model_ws=str(ws_root),
                verbose=False,
            )
        except Exception as e:
            lines.append(f"Cannot load MT3D model: {e}")
            lines.append("")
            lines.append("**Manual estimation guidance:**")
            if check_name == "courant_number":
                lines.append("- Cr = v * dt / dx")
                lines.append("- Target: Cr < 1.0")
                lines.append("- If Cr > 1, reduce transport time step (DT0) or refine grid")
            else:
                lines.append("- Pe = v * dx / D")
                lines.append("- Target: Pe < 2 for upstream FD, < 4 for TVD")
                lines.append("- If Pe > threshold, refine grid or increase dispersivity")
            return "\n".join(lines)

        # Get grid spacing
        dx = dy = None
        try:
            from gw.simulators.mt3dms.io import find_linked_modflow_model
            linked = find_linked_modflow_model(ws_root)
            if linked:
                mf = flopy.modflow.Modflow.load(
                    linked.name,
                    model_ws=str(ws_root),
                    load_only=["dis"],
                    check=False,
                    verbose=False,
                )
                dx = float(np.mean(mf.dis.delr.array))
                dy = float(np.mean(mf.dis.delc.array))
        except Exception:
            pass

        if dx is None:
            lines.append("Could not determine grid spacing from linked MODFLOW model.")
            lines.append("Provide MODFLOW DIS file in the same workspace for estimation.")
            return "\n".join(lines)

        # Get dispersivity if checking Peclet
        if check_name == "peclet_number" and mt.dsp is not None:
            try:
                al = float(np.mean(mt.dsp.al.array))
                pe_est = dx / al if al > 0 else float("inf")
                lines.append(f"Grid spacing (mean): dx = {dx:.2f}")
                lines.append(f"Longitudinal dispersivity (mean): AL = {al:.4g}")
                lines.append(f"Grid Peclet number estimate (dx/AL): **{pe_est:.2f}**")
                lines.append("")
                if pe_est <= 2:
                    lines.append("**Result: PASS** — Pe <= 2, suitable for upstream FD.")
                elif pe_est <= 4:
                    lines.append("**Result: MARGINAL** — Pe 2-4, consider TVD method.")
                else:
                    lines.append(
                        "**Result: WARNING** — Pe > 4, high risk of numerical dispersion. "
                        "Refine grid, increase dispersivity, or use MOC/TVD method."
                    )
            except Exception as e:
                lines.append(f"Error computing Peclet number: {e}")
        elif check_name == "courant_number":
            lines.append(f"Grid spacing (mean): dx = {dx:.2f}")
            lines.append("")
            lines.append(
                "Courant number requires velocity magnitude (v = K*i/n) which needs "
                "the completed MODFLOW head solution. Check the transport listing file "
                "for any Courant number warnings."
            )
            # Check listing for Courant warnings
            lst_files = list(ws_root.glob("*.lst"))
            if lst_files:
                lst = max(lst_files, key=lambda f: f.stat().st_size)
                try:
                    txt = lst.read_text(encoding="utf-8", errors="replace")
                    courant_lines = [l.strip() for l in txt.splitlines()
                                     if "COURANT" in l.upper()]
                    if courant_lines:
                        lines.append("")
                        lines.append("Courant number references in listing file:")
                        for cl in courant_lines[:10]:
                            lines.append(f"  {cl}")
                except Exception:
                    pass
        else:
            lines.append("Dispersion package (DSP) not found — cannot compute Peclet.")

        return "\n".join(lines)

    # ── File-type knowledge ───────────────────────────────────────────────

    def file_type_knowledge(self) -> Dict[str, FileTypeInfo]:
        from gw.simulators.mt3dms.knowledge import _EXT_KB
        return dict(_EXT_KB)

    def package_properties(self) -> Dict[str, PackagePropertyInfo]:
        from gw.simulators.mt3dms.knowledge import PACKAGE_PROPERTIES
        return dict(PACKAGE_PROPERTIES)

    def file_extensions(self) -> Set[str]:
        from gw.simulators.mt3dms.knowledge import _EXT_KB
        return set(_EXT_KB.keys()) | {
            ".sft", ".lkt", ".uzt", ".cts", ".obs",
        }

    # ── LLM knowledge ─────────────────────────────────────────────────────

    def system_prompt_fragment(self) -> str:
        from gw.simulators.mt3dms.knowledge import mt3dms_system_prompt_fragment
        return mt3dms_system_prompt_fragment()

    def tool_descriptions(self) -> Dict[str, str]:
        from gw.simulators.mt3dms.knowledge import mt3dms_tool_description_overrides
        return mt3dms_tool_description_overrides()

    def file_mention_pattern(self) -> str:
        from gw.simulators.mt3dms.knowledge import mt3dms_file_mention_regex
        return mt3dms_file_mention_regex()

    # ── Grid ──────────────────────────────────────────────────────────────

    def get_grid_info(self, ws_root: Path) -> Optional[GridInfo]:
        """Get grid info from the MT3D model.

        MT3D shares the same grid as its linked MODFLOW model.
        Tries FloPy Mt3dms loader first, then linked MODFLOW, then text parse.
        """
        try:
            import flopy
        except ImportError:
            return self._grid_info_from_text(ws_root)

        # Try loading MT3D model directly
        nam_file = self._find_mt3d_nam(ws_root)
        if nam_file is not None:
            try:
                mt = flopy.mt3d.Mt3dms.load(
                    nam_file.name,
                    model_ws=str(ws_root),
                    verbose=False,
                )
                return GridInfo(
                    grid_type="dis",
                    nlay=mt.nlay,
                    ncpl=mt.nrow * mt.ncol,
                    total_cells=mt.nlay * mt.nrow * mt.ncol,
                    nrow=mt.nrow,
                    ncol=mt.ncol,
                )
            except Exception:
                pass

        # Try linked MODFLOW model
        from gw.simulators.mt3dms.io import find_linked_modflow_model
        linked = find_linked_modflow_model(ws_root)
        if linked is not None:
            try:
                m = flopy.modflow.Modflow.load(
                    linked.name,
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
                pass

        return self._grid_info_from_text(ws_root)

    def _grid_info_from_text(self, ws_root: Path) -> Optional[GridInfo]:
        """Parse BTN header for basic grid info when FloPy is unavailable."""
        btn_files = list(ws_root.glob("*.btn"))
        if not btn_files:
            return None
        try:
            text = btn_files[0].read_text(encoding="utf-8", errors="replace")
            lines = [l.strip() for l in text.splitlines()
                     if l.strip() and not l.strip().startswith("#")]
            if not lines:
                return None
            parts = lines[0].split()
            if len(parts) < 3:
                return None
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
        """Load model via FloPy's Mt3dms loader.

        Returns ``(model, error_msg)`` — note this returns an Mt3dms object
        not an MFSimulation, but the interface allows Any.
        """
        try:
            import flopy
        except ImportError:
            return None, "FloPy not available"

        nam_file = self._find_mt3d_nam(ws_root)
        if nam_file is None:
            return None, "No MT3D name file found"

        try:
            mt = flopy.mt3d.Mt3dms.load(
                nam_file.name,
                model_ws=str(ws_root),
                verbose=False,
            )
            return mt, None
        except Exception as e:
            return None, f"FloPy load failed: {type(e).__name__}: {e}"

    def clear_simulation_cache(self, ws_root: Optional[Path] = None) -> None:
        """No simulation cache for MT3DMS — no-op."""
        pass

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _find_mt3d_nam(ws_root: Path) -> Optional[Path]:
        """Find the MT3D name file (.mtnam first, then .nam with MT3D content)."""
        # Prefer .mtnam (MT3D-USGS convention)
        mtnam_files = list(ws_root.glob("*.mtnam"))
        if mtnam_files:
            return mtnam_files[0]

        # Look for .nam files that contain MT3D packages
        mt3d_indicators = {"BTN", "ADV", "DSP", "SSM", "RCT", "GCG", "TOB"}
        modflow_indicators = {"DIS", "BAS6", "BAS", "LPF", "BCF6", "UPW",
                              "PCG", "NWT", "WEL", "CHD", "RIV"}

        best_nam = None
        best_mt3d_count = 0

        for nf in ws_root.glob("*.nam"):
            if nf.name.lower() == "mfsim.nam":
                continue
            try:
                txt = nf.read_text(encoding="utf-8", errors="replace")[:4000].upper()
                mt3d_count = sum(1 for pkg in mt3d_indicators if pkg in txt)
                modflow_count = sum(1 for pkg in modflow_indicators if pkg in txt)

                # It's an MT3D name file if it has MT3D packages and few/no MODFLOW packages
                if mt3d_count >= 2 and mt3d_count > modflow_count:
                    if mt3d_count > best_mt3d_count:
                        best_mt3d_count = mt3d_count
                        best_nam = nf
            except OSError:
                pass

        return best_nam
