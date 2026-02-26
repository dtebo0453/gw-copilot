from __future__ import annotations

"""Model profile: combined analysis cache.

Orchestrates the full workspace analysis pipeline (snapshot, brief, output
probes, cross-package checks) and caches the result under:

  <workspace>/.gw_copilot/model_profile.json

Cache invalidation uses the same (file_count, newest_mtime) fingerprint as
the workspace scan.  Callers should prefer ``get_model_profile()`` which
returns the cached profile when fresh, rebuilding only when stale.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gw.api.model_snapshot import build_model_snapshot, build_model_brief
from gw.api.output_probes import probe_workspace_outputs
from gw.api.workspace_files import resolve_workspace_root
from gw.api.workspace_scan import (
    _compute_fingerprint,
    build_file_index,
    _cache_dir,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Cross-package analysis
# ---------------------------------------------------------------------------

def _analyze_cross_package(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Run basic cross-package consistency checks.

    These heuristics are *not* full model validation but catch the most
    common configuration issues that confuse plotting and chat.  Each check
    produces an ``{ok, code, message}`` entry.
    """
    checks: List[Dict[str, Any]] = []
    grid = snapshot.get("grid", {})
    nlay = int(grid.get("nlay") or 0)
    nrow = int(grid.get("nrow") or 0)
    ncol = int(grid.get("ncol") or 0)
    grid_type = (grid.get("type") or "").lower()
    packages = snapshot.get("packages", {})
    outputs = snapshot.get("outputs_present", {})
    tdis = snapshot.get("tdis", {})
    stress_sums = snapshot.get("stress_summaries", {})
    out_meta = snapshot.get("output_metadata", {})

    # 1. Wells in potentially inactive cells  --------------------------------
    # Heuristic: if WEL total_records > total active cells (approx), warn.
    wel_info = stress_sums.get("WEL", {})
    total_wel = int(wel_info.get("total_records") or 0)
    if total_wel and nlay and nrow and ncol:
        total_cells = nlay * nrow * ncol
        if total_wel > total_cells:
            checks.append({
                "ok": False,
                "code": "wel_exceeds_cells",
                "message": (
                    f"WEL has {total_wel} total stress-period records vs "
                    f"{total_cells} grid cells.  Some wells may reference "
                    "inactive or duplicate cells."
                ),
            })
        else:
            checks.append({
                "ok": True,
                "code": "wel_count_ok",
                "message": f"WEL record count ({total_wel}) is within grid cell count ({total_cells}).",
            })

    # 2. CHD / WEL adjacency warning  ----------------------------------------
    has_chd = "CHD" in packages or "CHD" in stress_sums
    has_wel = "WEL" in packages or "WEL" in stress_sums
    if has_chd and has_wel:
        checks.append({
            "ok": True,  # informational, not an error
            "code": "chd_wel_coexist",
            "message": (
                "Both CHD and WEL are present.  Ensure wells are not placed "
                "in CHD cells (CHD overrides computed heads)."
            ),
        })

    # 3. Recharge layer check  ------------------------------------------------
    rch_info = stress_sums.get("RCH", {})
    if rch_info and nlay and nlay > 1:
        checks.append({
            "ok": True,
            "code": "rch_multilayer",
            "message": (
                f"RCH is present in a {nlay}-layer model.  Verify that "
                "recharge is applied to the intended layer (top by default)."
            ),
        })

    # 4. TDIS period consistency  ---------------------------------------------
    nper = int(tdis.get("nper") or 0)
    for pkg_type in ("WEL", "CHD", "GHB", "RIV", "DRN"):
        pkg_info = stress_sums.get(pkg_type, {})
        periods_with_data = int(pkg_info.get("periods_with_data") or 0)
        if periods_with_data and nper and periods_with_data > nper:
            checks.append({
                "ok": False,
                "code": f"{pkg_type.lower()}_excess_periods",
                "message": (
                    f"{pkg_type} has data in {periods_with_data} periods but "
                    f"TDIS defines only {nper} stress periods."
                ),
            })

    # 5. Output availability vs OC expectation  --------------------------------
    has_oc = "OC" in packages or "OC6" in packages
    if has_oc:
        if not outputs.get("hds"):
            checks.append({
                "ok": False,
                "code": "oc_no_hds",
                "message": "OC package is present but no .hds file was found.  The model may not have been run or SAVE HEAD is missing.",
            })
        if not outputs.get("cbc"):
            checks.append({
                "ok": False,
                "code": "oc_no_cbc",
                "message": "OC package is present but no .cbc file was found.  SAVE BUDGET may be missing from OC.",
            })

    # 6. Binary output probe consistency  -------------------------------------
    hds_probe = out_meta.get("hds", {})
    if hds_probe.get("ok"):
        hds_ntimes = int(hds_probe.get("ntimes") or 0)
        if nper and hds_ntimes and hds_ntimes < nper:
            checks.append({
                "ok": False,
                "code": "hds_fewer_times",
                "message": (
                    f"HDS file has {hds_ntimes} time steps but TDIS defines "
                    f"{nper} stress periods.  OC may only save selected periods."
                ),
            })

    return {
        "checks": checks,
        "ok": all(c.get("ok", True) for c in checks),
        "count_passed": sum(1 for c in checks if c.get("ok", True)),
        "count_failed": sum(1 for c in checks if not c.get("ok", True)),
    }


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def build_model_profile(ws_root: Path, *, force: bool = False) -> Dict[str, Any]:
    """Build or return cached model profile.

    The profile bundles the enriched snapshot, brief, output probes, and
    cross-package analysis into a single JSON-serialisable dict.
    """
    cache_path = _cache_dir(ws_root) / "model_profile.json"
    cur_count, cur_newest = _compute_fingerprint(ws_root)

    # Try cache
    if not force and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            fp = data.get("fingerprint", {})
            if (
                int(fp.get("file_count", -1)) == int(cur_count)
                and float(fp.get("newest_mtime", -2.0)) == float(cur_newest)
            ):
                return data
        except Exception:
            pass

    # Rebuild
    snapshot = build_model_snapshot(ws_root)
    brief = build_model_brief(snapshot)
    file_index = build_file_index(ws_root)

    output_probes: Dict[str, Any] = {}
    try:
        output_probes = probe_workspace_outputs(ws_root, file_index)
    except Exception:
        pass

    cross_pkg = _analyze_cross_package(snapshot)

    from gw.mf6.flopy_bridge import flopy_is_available

    profile: Dict[str, Any] = {
        "version": 1,
        "fingerprint": {
            "file_count": int(cur_count),
            "newest_mtime": float(cur_newest),
        },
        "workspace_root": str(ws_root),
        "snapshot": snapshot,
        "brief": brief,
        "output_probes": output_probes,
        "cross_package_analysis": cross_pkg,
        "flopy_available": flopy_is_available(),
    }

    # Persist
    try:
        cache_path.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

    return profile


def get_model_profile(ws_root: Path) -> Dict[str, Any]:
    """Convenience wrapper — returns cached profile, rebuilding if stale."""
    return build_model_profile(ws_root, force=False)


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

class ProfileRequest(BaseModel):
    inputs_dir: str
    workspace: str | None = None
    force: bool = False


@router.post("/workspace/profile")
def workspace_profile(req: ProfileRequest):
    """Build / return the cached model profile."""
    try:
        ws_root = resolve_workspace_root(req.inputs_dir, req.workspace)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return build_model_profile(ws_root, force=req.force)
