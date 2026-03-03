"""Simulator registry — factory + auto-detection.

Mirrors the pattern used for LLM providers in ``gw.llm.registry``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from gw.simulators.base import SimulatorAdapter


def get_simulator(name: Optional[str] = None) -> SimulatorAdapter:
    """Return a simulator adapter by *name*.

    If *name* is ``None`` or empty, returns the default (MF6).
    """
    name_l = (name or "mf6").lower().strip()

    if name_l in ("mf6", "modflow6", "modflow-6", "modflow 6"):
        from gw.simulators.mf6.adapter import MF6Adapter
        return MF6Adapter()

    if name_l in ("mf2005", "modflow-2005", "modflow 2005", "modflow2005",
                   "mfnwt", "modflow-nwt", "nwt"):
        from gw.simulators.mf2005.adapter import MF2005Adapter
        return MF2005Adapter()

    if name_l in ("mfusg", "modflow-usg", "modflow usg", "modflowusg"):
        from gw.simulators.mfusg.adapter import MFUSGAdapter
        return MFUSGAdapter()

    if name_l in ("modpath", "modpath7", "modpath 7", "mp7"):
        from gw.simulators.modpath.adapter import ModpathAdapter
        return ModpathAdapter()

    if name_l in ("mt3dms", "mt3d", "mt3d-usgs", "mt3dusgs", "mt3d-ms"):
        from gw.simulators.mt3dms.adapter import MT3DMSAdapter
        return MT3DMSAdapter()

    if name_l in ("seawat", "swtv4", "swt", "swt_v4", "seawat v4"):
        from gw.simulators.seawat.adapter import SeawatAdapter
        return SeawatAdapter()

    raise ValueError(f"Unknown simulator: {name!r}")


def detect_simulator(ws_root: Path) -> SimulatorAdapter:
    """Auto-detect the simulator for a workspace.

    Calls ``detect()`` on every registered adapter and returns the one
    with the highest confidence score.  Falls back to MF6 if nothing
    matches.
    """
    best_adapter: Optional[SimulatorAdapter] = None
    best_score = 0.0

    for adapter in _all_adapters():
        try:
            score = adapter.detect(ws_root)
        except Exception:
            score = 0.0
        if score > best_score:
            best_score = score
            best_adapter = adapter

    if best_adapter is not None and best_score >= 0.1:
        return best_adapter

    # Default fallback
    from gw.simulators.mf6.adapter import MF6Adapter
    return MF6Adapter()


def _all_adapters() -> List[SimulatorAdapter]:
    """Return instances of all registered adapters.

    ORDER MATTERS for detection.  More specific adapters (SEAWAT, MFUSG)
    must come before the general MF2005 adapter, because SEAWAT and MFUSG
    are supersets of MF2005 (they include MF2005 packages but add VDF/VSC
    or DISU/SMS).  The detection loop picks the highest-scoring adapter,
    but when two adapters score similarly, listing order breaks ties.

    MF6 always wins when ``mfsim.nam`` exists (score 1.0), so its
    position doesn't matter much, but we list it first for clarity.
    """
    from gw.simulators.mf6.adapter import MF6Adapter
    from gw.simulators.seawat.adapter import SeawatAdapter
    from gw.simulators.mfusg.adapter import MFUSGAdapter
    from gw.simulators.mt3dms.adapter import MT3DMSAdapter
    from gw.simulators.modpath.adapter import ModpathAdapter
    from gw.simulators.mf2005.adapter import MF2005Adapter
    return [
        MF6Adapter(),
        SeawatAdapter(),      # before MF2005 — superset (VDF/VSC)
        MFUSGAdapter(),       # before MF2005 — superset (DISU/SMS)
        MT3DMSAdapter(),      # transport — BTN detection distinct from flow
        ModpathAdapter(),     # unique file types (.mpsim/.mpnam)
        MF2005Adapter(),      # most general fallback for classic MODFLOW
    ]
