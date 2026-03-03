"""MT3DMS / MT3D-USGS model runner.

Locates the executable and runs it in the model workspace directory.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Common executable names (searched in order)
_MT3DMS_NAMES = ["mt3dms", "mt3dms.exe", "mt3dms5b", "mt3dms5b.exe"]
_MT3DUSGS_NAMES = [
    "mt3d-usgs", "mt3d-usgs.exe",
    "mt3dusgs", "mt3dusgs.exe",
    "mt3d-usgs_1.0.00.exe",
    "mt3d-usgs_1.1.0.exe",
]

# Common installation directories on Windows
_COMMON_DIRS = [
    r"C:\WRDAPP",
    r"C:\Program Files\USGS",
    r"C:\Program Files (x86)\USGS",
    r"C:\MT3DMS",
    r"C:\MT3D-USGS",
    r"C:\GMS",
]


def find_mt3dms(
    exe_path: Optional[str] = None,
    *,
    prefer_usgs: bool = False,
) -> str:
    """Locate the MT3DMS or MT3D-USGS executable.

    Parameters
    ----------
    exe_path : str, optional
        Explicit path to the executable.
    prefer_usgs : bool
        If True, search for MT3D-USGS executables first.

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

    # Determine search order
    if prefer_usgs:
        names = _MT3DUSGS_NAMES + _MT3DMS_NAMES
    else:
        names = _MT3DMS_NAMES + _MT3DUSGS_NAMES

    # 1. Check PATH via shutil.which
    for name in names:
        found = shutil.which(name)
        if found:
            return found

    # 2. Check common installation directories (Windows)
    if platform.system() == "Windows":
        for base_dir in _COMMON_DIRS:
            base = Path(base_dir)
            if not base.exists():
                continue
            for name in names:
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
        for name in names:
            candidate = flopy_bin / name
            if candidate.is_file():
                return str(candidate)
    except ImportError:
        pass

    searched = ", ".join(names[:4])
    raise FileNotFoundError(
        f"MT3DMS/MT3D-USGS executable not found. "
        f"Searched PATH for: {searched}. "
        f"Install MT3DMS or MT3D-USGS and add to PATH, "
        f"or provide exe_path explicitly."
    )


def run_mt3dms(
    workspace: str,
    *,
    mt3d_path: Optional[str] = None,
    nam_file: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    prefer_usgs: bool = False,
) -> Tuple[int, str, str]:
    """Run MT3DMS or MT3D-USGS in the given workspace.

    Parameters
    ----------
    workspace : str
        Path to the model workspace directory.
    mt3d_path : str, optional
        Explicit path to the executable.
    nam_file : str, optional
        Name file to use. If None, auto-detects .mtnam or .nam.
    timeout_sec : int, optional
        Timeout in seconds.
    prefer_usgs : bool
        Prefer MT3D-USGS executable over MT3DMS.

    Returns
    -------
    tuple
        (returncode, stdout, stderr)
    """
    ws = Path(workspace)
    if not ws.is_dir():
        return (1, "", f"Workspace directory not found: {workspace}")

    exe = find_mt3dms(mt3d_path, prefer_usgs=prefer_usgs)

    # Find the name file
    if nam_file is None:
        # Prefer .mtnam (MT3D-USGS convention)
        mtnam_files = sorted(ws.glob("*.mtnam"))
        if mtnam_files:
            nam_file = mtnam_files[0].name
        else:
            # Fall back to .nam files that look like MT3D name files
            all_nam = sorted(ws.glob("*.nam"))
            # Exclude mfsim.nam (MF6)
            all_nam = [f for f in all_nam if f.name.lower() != "mfsim.nam"]

            # Try to find the MT3D-specific .nam (has BTN, ADV, etc.)
            mt3d_indicators = {"BTN", "ADV", "DSP", "SSM", "GCG"}
            for nf in all_nam:
                try:
                    txt = nf.read_text(encoding="utf-8", errors="replace")[:4000].upper()
                    if sum(1 for pkg in mt3d_indicators if pkg in txt) >= 2:
                        nam_file = nf.name
                        break
                except OSError:
                    pass

            if nam_file is None:
                return (
                    1, "",
                    "No MT3D name file found (.mtnam or .nam with BTN/ADV/DSP packages)"
                )

    logger.info("Running MT3D: exe=%s  ws=%s  nam=%s", exe, workspace, nam_file)

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
        return (1, "", f"MT3D run timed out after {timeout_sec or 600} seconds")
    except FileNotFoundError:
        return (1, "", f"Executable not found: {exe}")
    except Exception as e:
        return (1, "", f"Error running MT3D: {type(e).__name__}: {e}")
