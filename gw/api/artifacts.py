from __future__ import annotations

from pathlib import Path
from typing import Optional, List

# workspace_files contains the hardened resolver for turning (inputs_dir, workspace)
# into an absolute root. We use it here to keep artifacts browsing consistent with
# workspace file browsing, and to avoid returning drive-qualified absolute paths
# (e.g. C:\\...) back to the UI where they may be rejected.
try:
    from gw.api.workspace_files import resolve_workspace_root
except Exception:  # pragma: no cover
    resolve_workspace_root = None  # type: ignore

def _has_gw_copilot(inputs_dir: str) -> bool:
    """Check if inputs_dir has a GW_Copilot/ subfolder with config.json."""
    return (Path(inputs_dir) / "GW_Copilot" / "config.json").exists()


def infer_workspace(inputs_dir: str) -> Optional[str]:
    """Return the *relative* workspace folder name when present.

    GW_Copilot layout: model files are directly in inputs_dir → return None
    (no workspace subfolder to navigate into).

    Legacy layout: return "workspace" if that subfolder exists.

    IMPORTANT: The frontend passes this value back to API endpoints. Returning a
    drive-qualified absolute path (Windows) can be rejected by our path guards.
    """
    # GW_Copilot: model files are at the root of inputs_dir
    if _has_gw_copilot(inputs_dir):
        return None

    p = Path(inputs_dir)
    ws = p / "workspace"
    if ws.exists() and ws.is_dir():
        # Always return the subfolder name, not a full path.
        return "workspace"
    return None

def artifacts_dir(inputs_dir: str, workspace: Optional[str]) -> Optional[str]:
    """Return the absolute artifacts directory path.

    GW_Copilot layout: ``<inputs_dir>/GW_Copilot/``
    Legacy layout: ``<workspace>/run_artifacts/``
    """
    # GW_Copilot layout
    if _has_gw_copilot(inputs_dir):
        gw_dir = Path(inputs_dir) / "GW_Copilot"
        gw_dir.mkdir(parents=True, exist_ok=True)
        return str(gw_dir)

    # Legacy layout
    ws = workspace or infer_workspace(inputs_dir)
    # Prefer hardened resolver if available so artifacts_dir is an absolute path
    # on the server regardless of how inputs_dir was provided.
    if resolve_workspace_root is not None:
        try:
            root = resolve_workspace_root(inputs_dir, ws)
            return str(Path(root) / "run_artifacts")
        except Exception:
            # Fall back to old behavior below
            pass

    if ws:
        return str(Path(inputs_dir) / ws / "run_artifacts")
    return str(Path(inputs_dir) / "run_artifacts")

def list_recent_artifacts(adir: Optional[str], limit: int = 30) -> List[str]:
    """Return recently-modified *file* paths inside *adir*.

    For the GW_Copilot layout, files live in subdirectories (validation/,
    plots/, qa/), so we recurse one level deep.  Returned paths are relative
    to *adir* so the UI can display them and pass back to ``/artifacts/read``.

    Only regular files are returned — directories are skipped.
    """
    if not adir:
        return []
    p = Path(adir)
    if not p.exists():
        return []

    items: List[tuple] = []
    # Recurse into subdirectories (max 2 levels for GW_Copilot layout)
    for item in p.rglob("*"):
        if item.is_file():
            # Skip config.json in GW_Copilot root — not a useful artifact to show
            if item.name == "config.json" and item.parent == p:
                continue
            try:
                rel = str(item.relative_to(p)).replace("\\", "/")
                items.append((rel, item.stat().st_mtime))
            except (OSError, ValueError):
                continue

    items.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _mtime in items[:limit]]
