"""MODFLOW-USG model runner.

Locates the MODFLOW-USG executable and runs it in the model workspace
directory.  MODFLOW-USG uses the same invocation pattern as MF2005:
``mfusg <namefile>``.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Common executable names (searched in order)
_MFUSG_NAMES = [
    "mfusg", "mfusg.exe",
    "MODFLOW-USG.exe", "MODFLOW-USG_64.exe",
    "mfusg_64.exe",
    "mfusgdbl", "mfusgdbl.exe",
]

# Common installation directories on Windows
_COMMON_DIRS = [
    r"C:\WRDAPP",
    r"C:\Program Files\USGS",
    r"C:\Program Files (x86)\USGS",
    r"C:\MODFLOW",
    r"C:\MODFLOW-USG",
    r"C:\GMS",
]


def find_mfusg(exe_path: Optional[str] = None) -> str:
    """Locate the MODFLOW-USG executable.

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
    for name in _MFUSG_NAMES:
        found = shutil.which(name)
        if found:
            return found

    # 2. Check common installation directories (Windows)
    if platform.system() == "Windows":
        for base_dir in _COMMON_DIRS:
            base = Path(base_dir)
            if not base.exists():
                continue
            for name in _MFUSG_NAMES:
                # Check directly in the directory
                candidate = base / name
                if candidate.is_file():
                    return str(candidate)
                # Check one level deep (subdirectories)
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
        for name in _MFUSG_NAMES:
            candidate = flopy_bin / name
            if candidate.is_file():
                return str(candidate)
    except ImportError:
        pass

    searched = ", ".join(_MFUSG_NAMES[:4])
    raise FileNotFoundError(
        f"MODFLOW-USG executable not found. "
        f"Searched PATH for: {searched}. "
        f"Install MODFLOW-USG and add to PATH, "
        f"or provide exe_path explicitly."
    )


def run_mfusg(
    workspace: str,
    *,
    mfusg_path: Optional[str] = None,
    nam_file: Optional[str] = None,
    timeout_sec: Optional[int] = None,
) -> Tuple[int, str, str]:
    """Run MODFLOW-USG in the given workspace.

    Parameters
    ----------
    workspace : str
        Path to the model workspace directory.
    mfusg_path : str, optional
        Explicit path to the executable.
    nam_file : str, optional
        Name file to use. If None, auto-detects the .nam file.
    timeout_sec : int, optional
        Timeout in seconds.

    Returns
    -------
    tuple
        (returncode, stdout, stderr)
    """
    ws = Path(workspace)
    if not ws.is_dir():
        return (1, "", f"Workspace directory not found: {workspace}")

    exe = find_mfusg(mfusg_path)

    # Find the name file
    if nam_file is None:
        nam_files = sorted(ws.glob("*.nam"))
        # Exclude mfsim.nam (MF6)
        nam_files = [f for f in nam_files if f.name.lower() != "mfsim.nam"]
        if not nam_files:
            return (1, "", "No .nam file found in workspace (excluding mfsim.nam)")
        nam_file = nam_files[0].name

    logger.info("Running MODFLOW-USG: exe=%s  ws=%s  nam=%s", exe, workspace, nam_file)

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
        return (1, "", f"MODFLOW-USG run timed out after {timeout_sec or 600} seconds")
    except FileNotFoundError:
        return (1, "", f"Executable not found: {exe}")
    except Exception as e:
        return (1, "", f"Error running MODFLOW-USG: {type(e).__name__}: {e}")
