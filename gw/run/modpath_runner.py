"""MODPATH 7 model runner.

Locates the MODPATH executable and runs it in the model workspace
directory.  MODPATH reads the .mpsim simulation file which references
all other input files.
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
_MP7_NAMES = [
    "mp7",
    "mp7.exe",
    "mpath7",
    "mpath7.exe",
    "modpath7",
    "modpath7.exe",
    "mp7_64",
    "mp7_64.exe",
    "MODPATH_7",
    "MODPATH_7.exe",
]

# Common installation directories on Windows
_COMMON_DIRS = [
    r"C:\WRDAPP",
    r"C:\Program Files\USGS",
    r"C:\Program Files (x86)\USGS",
    r"C:\MODPATH",
    r"C:\MODFLOW",
    r"C:\GMS",
]


def find_modpath(exe_path: Optional[str] = None) -> str:
    """Locate the MODPATH 7 executable.

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
    for name in _MP7_NAMES:
        found = shutil.which(name)
        if found:
            return found

    # 2. Check common installation directories (Windows)
    if platform.system() == "Windows":
        for base_dir in _COMMON_DIRS:
            base = Path(base_dir)
            if not base.exists():
                continue
            for name in _MP7_NAMES:
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
                    continue

    # 3. Check for FloPy-installed executables
    try:
        import flopy
        flopy_bin = Path(flopy.__file__).parent / "bin"
        for name in _MP7_NAMES:
            candidate = flopy_bin / name
            if candidate.is_file():
                return str(candidate)
    except ImportError:
        pass

    searched = ", ".join(_MP7_NAMES[:4])
    raise FileNotFoundError(
        f"MODPATH 7 executable not found. "
        f"Searched PATH for: {searched}. "
        f"Install MODPATH 7 and add to PATH, "
        f"or provide exe_path explicitly."
    )


def run_modpath(
    workspace: str,
    *,
    mp_path: Optional[str] = None,
    mpsim_file: Optional[str] = None,
    timeout_sec: Optional[int] = None,
) -> Tuple[int, str, str]:
    """Run MODPATH 7 in the given workspace.

    Parameters
    ----------
    workspace : str
        Path to the model workspace directory.
    mp_path : str, optional
        Explicit path to the MODPATH executable.
    mpsim_file : str, optional
        MODPATH simulation file name.  If None, auto-detects the .mpsim
        file in the workspace.
    timeout_sec : int, optional
        Timeout in seconds.  Defaults to 300 (5 minutes).

    Returns
    -------
    tuple
        (returncode, stdout, stderr)
    """
    ws = Path(workspace)
    if not ws.is_dir():
        return (1, "", f"Workspace directory not found: {workspace}")

    try:
        exe = find_modpath(mp_path)
    except FileNotFoundError as exc:
        return (1, "", str(exc))

    # Find the simulation file
    if mpsim_file is None:
        mpsim_files = sorted(ws.glob("*.mpsim"))
        if not mpsim_files:
            return (1, "", "No .mpsim file found in workspace")
        mpsim_file = mpsim_files[0].name

    logger.info(
        "Running MODPATH: exe=%s  ws=%s  mpsim=%s",
        exe, workspace, mpsim_file,
    )

    try:
        result = subprocess.run(
            [exe, mpsim_file],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=timeout_sec or 300,
        )
        return (result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (
            1, "",
            f"MODPATH run timed out after {timeout_sec or 300} seconds",
        )
    except FileNotFoundError:
        return (1, "", f"Executable not found: {exe}")
    except Exception as e:
        return (1, "", f"Error running MODPATH: {type(e).__name__}: {e}")
