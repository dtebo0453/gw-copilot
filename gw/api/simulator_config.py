"""Per-workspace simulator type storage.

Stores detected or user-selected simulator type in:
  ``<inputs_dir>/GW_Copilot/simulator.json``

Schema::

    {"simulator": "mf6", "detected": true, "confidence": 0.95}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from gw.simulators.base import SimulatorAdapter


def _cfg_path(ws_root: Path) -> Path:
    return ws_root / "GW_Copilot" / "simulator.json"


def get_simulator_type(ws_root: Path) -> Optional[str]:
    """Get the saved simulator type for a workspace, or ``None``."""
    p = _cfg_path(ws_root)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("simulator")
        except Exception:
            pass
    return None


def set_simulator_type(
    ws_root: Path,
    simulator: str,
    *,
    detected: bool = False,
    confidence: float = 0.0,
) -> None:
    """Persist the simulator type for a workspace."""
    p = _cfg_path(ws_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "simulator": simulator,
        "detected": detected,
        "confidence": confidence,
    }, indent=2), encoding="utf-8")


def get_adapter_for_workspace(ws_root: Path) -> SimulatorAdapter:
    """Return the ``SimulatorAdapter`` for *ws_root*.

    Resolution order:
      1. Saved config (``simulator.json``)
      2. Auto-detection via registry
      3. Default (MF6)
    """
    from gw.simulators.registry import get_simulator, detect_simulator

    saved = get_simulator_type(ws_root)
    if saved:
        try:
            return get_simulator(saved)
        except ValueError:
            pass  # unknown name — fall through to detection

    # Auto-detect and cache
    adapter = detect_simulator(ws_root)
    confidence = adapter.detect(ws_root)
    set_simulator_type(
        ws_root, adapter.info().name,
        detected=True, confidence=confidence,
    )
    return adapter
