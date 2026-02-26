from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# NOTE: This module intentionally does NOT import FastAPI.
# It provides a small in-memory cache for parsed model grids and derived products
# (mesh + scalar arrays) keyed by workspace + DIS path.

@dataclass
class ModelSession:
    ws_root: Path
    dis_mtime_ns: int
    dep_sig: Tuple[Tuple[str, int], ...]  # (relative_path, mtime_ns)
    dis: Any  # DisInfo from viz.py (kept loose to avoid circular typing)

    grid_type: str = "dis"           # "dis", "disv", or "disu"
    flopy_sim: Any = None            # cached MFSimulation for DISV/DISU
    flopy_model_name: str = ""       # primary model name for FloPy

    mesh_cache: Dict[Tuple[str, int], Dict[str, Any]] = field(default_factory=dict)  # (mode, layer) -> mesh payload
    scalars_cache: Dict[tuple, np.ndarray] = field(default_factory=dict)  # (key, layer[, mode]) -> array
    pkg_arrays_cache: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)  # pkg_type -> {array_name: flat_array}


class ModelSessionCache:
    def __init__(self, max_sessions: int = 4):
        self.max_sessions = max_sessions
        self._lru: "OrderedDict[Tuple[str, str], ModelSession]" = OrderedDict()

    def _evict_if_needed(self) -> None:
        while len(self._lru) > self.max_sessions:
            self._lru.popitem(last=False)

    def get(self, ws_root: Path, dis_path: Path, *, dis_mtime_ns: int, dep_sig: Tuple[Tuple[str, int], ...], dis_obj: Any) -> ModelSession:
        key = (str(ws_root.resolve()), str(dis_path.resolve()))
        existing = self._lru.get(key)
        if existing and existing.dis_mtime_ns == dis_mtime_ns and existing.dep_sig == dep_sig:
            # mark as most recently used
            self._lru.move_to_end(key)
            return existing

        sess = ModelSession(
            ws_root=ws_root,
            dis_mtime_ns=dis_mtime_ns,
            dep_sig=dep_sig,
            dis=dis_obj,
        )
        self._lru[key] = sess
        self._lru.move_to_end(key)
        self._evict_if_needed()
        return sess
