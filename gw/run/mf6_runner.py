import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

def _which(exe: str) -> Optional[str]:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if os.name == "nt":
        exts = os.environ.get("PATHEXT", ".EXE;.BAT;.CMD").split(";")
    else:
        exts = [""]
    for d in paths:
        d = d.strip('"')
        if not d:
            continue
        for ext in exts:
            cand = Path(d) / (exe + ext if ext and not exe.lower().endswith(ext.lower()) else exe)
            if cand.exists() and cand.is_file():
                return str(cand)
    return None

def find_mf6(mf6_path: Optional[str] = None) -> str:
    if mf6_path:
        p = Path(mf6_path)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"mf6 executable not found at: {mf6_path}")
    found = _which("mf6") or _which("mf6.exe")
    if found:
        return found
    raise FileNotFoundError(
        "Could not find 'mf6' on PATH. Install MODFLOW 6 or pass --mf6-path to the command."
    )

def run_mf6(*, workspace: str, mf6_path: Optional[str] = None, timeout_sec: Optional[int] = None) -> Tuple[int, str, str]:
    exe = find_mf6(mf6_path)
    wdir = Path(workspace)
    if not wdir.exists():
        raise FileNotFoundError(f"Workspace directory not found: {workspace}")
    if not (wdir / "mfsim.nam").exists():
        raise FileNotFoundError(f"mfsim.nam not found in workspace: {workspace}")

    proc = subprocess.run(
        [exe],
        cwd=str(wdir),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""

def write_run_artifacts(outdir: str, rc: int, stdout: str, stderr: str) -> None:
    from pathlib import Path
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "mf6_stdout.txt").write_text(stdout, encoding="utf-8", errors="ignore")
    (outp / "mf6_stderr.txt").write_text(stderr, encoding="utf-8", errors="ignore")
    (outp / "run_return_code.txt").write_text(str(rc) + "\n", encoding="utf-8")
