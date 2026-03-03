"""SEAWAT v4 model runner.

Locates the SEAWAT executable and runs it in the model workspace directory.
SEAWAT uses the same invocation pattern as MODFLOW-2005: ``exe namefile``.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Common executable names (searched in order of preference)
_SEAWAT_NAMES = [
    "swtv4", "swtv4.exe",
    "swt_v4", "swt_v4.exe",
    "swtv4_64", "swtv4_64.exe",
    "swt_v4_64", "swt_v4_64.exe",
    "seawat", "seawat.exe",
    "SEAWAT.exe",
]

# Common installation directories on Windows
_COMMON_DIRS = [
    r"C:\WRDAPP",
    r"C:\Program Files\USGS",
    r"C:\Program Files (x86)\USGS",
    r"C:\MODFLOW",
    r"C:\SEAWAT",
    r"C:\GMS",
]


def find_seawat(exe_path: Optional[str] = None) -> str:
    """Locate the SEAWAT executable.

    Parameters
    ----------
    exe_path : str, optional
        Explicit path to the executable.

    Returns
    -------
    str
        Path to the executable.

    Raises
    ------
    FileNotFoundError
        If the executable cannot be found.
    """
    if exe_path:
        p = Path(exe_path)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"Specified executable not found: {exe_path}")

    # 1. Check PATH via shutil.which
    for name in _SEAWAT_NAMES:
        found = shutil.which(name)
        if found:
            return found

    # 2. Check common installation directories (Windows)
    if platform.system() == "Windows":
        for base_dir in _COMMON_DIRS:
            base = Path(base_dir)
            if not base.exists():
                continue
            for name in _SEAWAT_NAMES:
                # Check directly
                candidate = base / name
                if candidate.is_file():
                    return str(candidate)
                # Check one level deep
                try:
                    for sub in base.iterdir():
                        if sub.is_dir():
                            candidate = sub / name
                            if candidate.is_file():
                                return str(candidate)
                except OSError:
                    pass

    # 3. Check for FloPy-installed executables
    try:
        import flopy
        flopy_bin = Path(flopy.__file__).parent / "bin"
        for name in _SEAWAT_NAMES:
            candidate = flopy_bin / name
            if candidate.is_file():
                return str(candidate)
    except ImportError:
        pass

    searched = ", ".join(_SEAWAT_NAMES[:6])
    raise FileNotFoundError(
        f"SEAWAT executable not found. "
        f"Searched PATH for: {searched}. "
        f"Install SEAWAT v4 and add to PATH, "
        f"or provide exe_path explicitly."
    )


def run_seawat(
    workspace: str,
    *,
    seawat_path: Optional[str] = None,
    nam_file: Optional[str] = None,
    timeout_sec: Optional[int] = None,
) -> Tuple[int, str, str]:
    """Run SEAWAT in the given workspace.

    Parameters
    ----------
    workspace : str
        Path to the model workspace directory.
    seawat_path : str, optional
        Explicit path to the SEAWAT executable.
    nam_file : str, optional
        Name file to use. If None, auto-detects the .nam file.
    timeout_sec : int, optional
        Timeout in seconds. Defaults to 600 (10 minutes).

    Returns
    -------
    tuple
        (returncode, stdout, stderr)
    """
    ws = Path(workspace)
    if not ws.is_dir():
        return (1, "", f"Workspace directory not found: {workspace}")

    exe = find_seawat(seawat_path)

    # Find the name file
    if nam_file is None:
        nam_files = sorted(ws.glob("*.nam"))
        # Exclude mfsim.nam (MF6)
        nam_files = [f for f in nam_files if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return (1, "", "No .nam file found in workspace (excluding mfsim.nam)")
        nam_file = nam_files[0].name

    logger.info("Running SEAWAT: exe=%s  ws=%s  nam=%s", exe, workspace, nam_file)

    try:
        result = subprocess.run(
            [exe, nam_file],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=timeout_sec or 600,
        )
        return (result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (1, "", f"SEAWAT run timed out after {timeout_sec or 600} seconds")
    except FileNotFoundError:
        return (1, "", f"Executable not found: {exe}")
    except Exception as e:
        return (1, "", f"Error running SEAWAT: {type(e).__name__}: {e}")
