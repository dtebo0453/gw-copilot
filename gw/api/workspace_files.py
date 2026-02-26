from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# Optional: semicolon-separated list of allowed roots for browsing.
# Example (Windows):
#   set GW_FS_ROOTS=C:\dev\gw_projects;D:\data\cases
GW_FS_ROOTS_ENV = "GW_FS_ROOTS"


@dataclass(frozen=True)
class WorkspaceFileEntry:
    path_rel: str
    size: int
    mtime: str
    kind: str
    sha256: Optional[str] = None


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _is_path_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _deny_traversal(path_rel: str) -> None:
    if not path_rel or path_rel.strip() == "":
        raise ValueError("path_rel is required")
    p = path_rel.replace("\\", "/")
    if p.startswith("/") or p.startswith("~"):
        raise ValueError("absolute paths are not allowed")
    # Block drive-qualified paths like C:\... (drive appears in first segment)
    if ":" in p.split("/")[0]:
        raise ValueError("drive-qualified paths are not allowed")
    parts = [x for x in p.split("/") if x]
    if any(part == ".." for part in parts):
        raise ValueError("path traversal is not allowed")


def resolve_workspace_root(inputs_dir: str, workspace: Optional[str] = None) -> Path:
    """Resolve the workspace root directory safely.

    inputs_dir may be:
      - an absolute path to a project folder (often containing a 'workspace/' subfolder)
      - a relative identifier like 'runs/aoi_demo' that should be resolved under configured roots

    workspace may be:
      - None (we will auto-detect 'workspace/' if present)
      - a subfolder name (e.g., 'workspace')
      - in some UI flows it may redundantly include the inputs_dir prefix; we normalize that out.
    """
    raw_inputs = (inputs_dir or "").strip()
    if not raw_inputs:
        raise ValueError("inputs_dir is required")

    # Normalize separators early
    raw_inputs_norm = raw_inputs.replace("\\", "/").strip()
    raw_workspace_norm = (workspace or "").replace("\\", "/").strip() if workspace else None

    # Candidate bases to try, in order.
    candidates: list[Path] = []

    p = Path(raw_inputs).expanduser()
    if p.is_absolute():
        candidates.append(p)
    else:
        # 1) repo root (historical behavior)
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / p)
        # 2) current working directory
        candidates.append(Path.cwd() / p)

        # 3) configured roots (semicolon separated)
        roots_raw = os.environ.get(GW_FS_ROOTS_ENV, "").strip()
        if roots_raw:
            for r in roots_raw.split(";"):
                r = r.strip()
                if not r:
                    continue
                candidates.append(Path(r).expanduser() / p)

        # 4) optional GW_PROJECTS_ROOT (single root)
        proj_root = os.environ.get("GW_PROJECTS_ROOT", "").strip()
        if proj_root:
            candidates.append(Path(proj_root).expanduser() / p)

    base: Optional[Path] = None
    tried: list[str] = []
    for c in candidates:
        try:
            cr = c.resolve()
        except Exception:
            cr = c
        tried.append(str(cr))
        if cr.exists() and cr.is_dir():
            base = cr
            break

    if base is None:
        raise ValueError(f"workspace root does not exist or is not a directory (tried: {tried})")

    # Normalize workspace parameter: sometimes UI passes workspace that includes inputs_dir prefix.
    if raw_workspace_norm:
        inputs_dir_norm = raw_inputs_norm.strip("/")
        ws_norm = raw_workspace_norm.strip("/")
        if inputs_dir_norm and ws_norm.startswith(inputs_dir_norm + "/"):
            ws_norm = ws_norm[len(inputs_dir_norm) + 1 :]
        if ws_norm == inputs_dir_norm:
            ws_norm = ""
        if ws_norm:
            base = (base / ws_norm).resolve()

    # If workspace folder isn't explicitly provided, try conventional subfolder.
    # GW_Copilot layout: model files are at the root — skip the workspace/ guess.
    if raw_workspace_norm is None:
        gw_copilot_cfg = base / "GW_Copilot" / "config.json"
        if not gw_copilot_cfg.exists():
            ws_guess = (base / "workspace")
            if ws_guess.exists() and ws_guess.is_dir():
                base = ws_guess.resolve()

    # Enforce allowed roots if configured: base must be within at least one allowed root.
    roots_raw = os.environ.get(GW_FS_ROOTS_ENV, "").strip()
    if roots_raw:
        allowed_roots = []
        for r in roots_raw.split(";"):
            r = r.strip()
            if not r:
                continue
            try:
                allowed_roots.append(Path(r).expanduser().resolve())
            except Exception:
                continue
        if allowed_roots:
            ok = any(str(base).lower().startswith(str(ar).lower()) for ar in allowed_roots)
            if not ok:
                raise ValueError(f"workspace root is not under allowed roots ({GW_FS_ROOTS_ENV})")

    if not base.exists() or not base.is_dir():
        raise ValueError("workspace root does not exist or is not a directory")

    return base


def resolve_file(root: Path, path_rel: str) -> Path:
    _deny_traversal(path_rel)
    rel = Path(path_rel.replace("/", os.sep))
    abs_path = (root / rel).resolve()
    if not (_is_path_under(abs_path, root) or abs_path == root):
        raise ValueError("requested file is outside the workspace root")
    if not abs_path.exists() or not abs_path.is_file():
        raise FileNotFoundError("file not found")
    return abs_path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_kind_and_text(data: bytes) -> Tuple[str, Optional[str]]:
    # Fast path: null bytes almost always mean binary — skip expensive decode.
    if b"\x00" in data[:512]:
        return "binary", None
    try:
        return "text", data.decode("utf-8")
    except UnicodeDecodeError:
        return "binary", None


def _is_probably_text(path: Path, sniff_bytes: int = 4096) -> bool:
    try:
        with path.open("rb") as fh:
            data = fh.read(sniff_bytes)
    except Exception:
        return False
    kind, _ = _detect_kind_and_text(data)
    return kind == "text"


def list_workspace_files(
    root: Path,
    glob: Optional[str] = None,
    max_files: int = 2000,
    include_hash: bool = False,
) -> Tuple[list[WorkspaceFileEntry], bool]:
    """
    Returns (files, truncated)
    """
    pattern = glob or "**/*"
    files: list[WorkspaceFileEntry] = []
    truncated = False

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        rel = p.relative_to(root).as_posix()

        if pattern and pattern != "**/*":
            if not fnmatch.fnmatch(rel, pattern) and not fnmatch.fnmatch(p.name, pattern):
                continue

        st = p.stat()
        sha = _sha256_file(p) if include_hash else None
        kind = "text" if _is_probably_text(p) else "binary"

        files.append(
            WorkspaceFileEntry(
                path_rel=rel,
                size=int(st.st_size),
                mtime=_utc_iso(st.st_mtime),
                kind=kind,
                sha256=sha,
            )
        )

        if len(files) >= max_files:
            truncated = True
            break

    files.sort(key=lambda x: x.path_rel.lower())
    return files, truncated


def read_file_text(path: Path, max_bytes: int = 2_000_000) -> Tuple[Optional[str], bool, str, int, str]:
    """
    Returns: (content_or_none, truncated, sha256, size, kind)
    sha256 is for the full file.
    """
    size = int(path.stat().st_size)
    sha = _sha256_file(path)

    with path.open("rb") as f:
        data = f.read(max_bytes + 1)

    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]

    kind, text = _detect_kind_and_text(data)
    return text, truncated, sha, size, kind
