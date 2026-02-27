from __future__ import annotations

import hashlib
import inspect
import json
import mimetypes
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from gw.api.artifacts import artifacts_dir, infer_workspace, list_recent_artifacts
from gw.api.fs import find_paths
from gw.api.jobs import create_job, get_job, run_subprocess_job
from gw.api import workspace_files as wf
from gw.api.schemas import (
    ChatPlotOutput,
    ChatRequest,
    ChatResponse,
    ImportPlanModel,
    JobStatus,
    ProjectImportRequest,
    ProjectImportResponse,
    ProjectInspectRequest,
    ProjectInspectResponse,
    RunRequest,
    RunResponse,
    WorkspaceInfo,
    FsFindRequest,
    FsFindResponse,
    WorkspaceFilesResponse,
    WorkspaceFileReadResponse,
)

router = APIRouter()

# -----------------------------
# Helpers (robust adapter layer)
# -----------------------------

def _call(fn: Callable[..., Any], /, *pos: Any, **kw: Any) -> Any:
    """
    Call a function robustly even if keyword names drift:
    - If fn accepts **kwargs, pass everything.
    - Else, pass only kwargs that exist in signature.
    """
    sig = inspect.signature(fn)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*pos, **kw)

    filtered = {k: v for k, v in kw.items() if k in params}
    return fn(*pos, **filtered)


def _to_plain_dict(x: Any) -> dict:
    """
    Convert an object to a plain dict suitable for pydantic schemas.
    IMPORTANT: Never try dict(<string>) etc.
    """
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):  # pydantic v2
        return x.model_dump()
    if hasattr(x, "dict"):  # pydantic v1
        return x.dict()
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    # If we get here, it's likely not dict-like; don't do dict(x) blindly.
    raise TypeError(f"Cannot convert {type(x).__name__} to dict safely")


def _resolve_root(inputs_dir: str, workspace: str | None) -> Path:
    # Use helper but call positionally to avoid signature drift.
    try:
        return _call(wf.resolve_workspace_root, inputs_dir, workspace)
    except TypeError:
        return _call(wf.resolve_workspace_root, inputs_dir)


def _list_files(root: Path, glob: str, max_files: int, include_hash: bool) -> Tuple[list[Any], bool]:
    out = _call(wf.list_workspace_files, root, glob=glob, max_files=max_files, include_hash=include_hash)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0] or [], bool(out[1])
    if isinstance(out, list):
        return out, False
    raise RuntimeError(f"Unexpected return from list_workspace_files: {type(out).__name__}")


def _resolve_file(root: Path, path_rel: str) -> Path:
    # Prefer helper if present and stable
    try:
        p = _call(wf.resolve_file, root, path_rel)  # positional
        return Path(p)
    except Exception:
        # Fallback: safe join; root must already be validated by resolve_workspace_root.
        p = (Path(root) / path_rel).resolve()
        # Ensure it's still under root
        root_res = Path(root).resolve()
        if root_res not in p.parents and p != root_res:
            raise ValueError("path escapes workspace root")
        return p


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _guess_mime(p: Path) -> str:
    mt, _enc = mimetypes.guess_type(str(p))
    return mt or "application/octet-stream"


def _read_preview_text(p: Path, max_bytes: int) -> tuple[str | None, bool, str]:
    """
    Return (content, truncated, kind) where kind is 'text' or 'binary'.
    content is None for binary.
    """
    size = p.stat().st_size
    if max_bytes <= 0:
        max_bytes = 1

    # Read slightly over the limit so we can mark truncated
    to_read = min(size, max_bytes + 1)

    raw = b""
    with p.open("rb") as f:
        raw = f.read(to_read)

    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]

    # Heuristic: null bytes => binary
    if b"\x00" in raw:
        return None, truncated, "binary"

    # Try utf-8 first; if it fails, still show something Notepad-like (replace errors)
    try:
        txt = raw.decode("utf-8")
        return txt, truncated, "text"
    except UnicodeDecodeError:
        txt = raw.decode("utf-8", errors="replace")
        return txt, truncated, "text"


# -----------------------------
# CLI action runner + jobs
# -----------------------------

_ACTION_TO_CLI = {
    "validate": "revalidate",          # both use the robust revalidate command
    "revalidate": "revalidate",        # which handles missing stress_validate gracefully
    "suggest-fix": "llm-suggest-fixes", # CLI name differs from UI action name
    "apply-fixes": "apply-fixes",
}


def _resolve_config(req: RunRequest) -> Optional[str]:
    """Auto-find config.json when not explicitly provided."""
    if req.config:
        return req.config
    candidates = [
        Path(req.inputs_dir) / "GW_Copilot" / "config.json",
        Path(req.inputs_dir) / "config.json",
    ]
    if req.workspace:
        candidates.append(Path(req.workspace) / "config.json")
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _build_cmd(action: str, req: RunRequest) -> list[str]:
    cli_cmd = _ACTION_TO_CLI.get(action, action)
    import sys
    cmd = [sys.executable, "-m", "gw.cli_entry", cli_cmd, "--inputs-dir", req.inputs_dir]
    if req.workspace:
        # Resolve workspace to absolute path — the CLI subprocess may have a
        # different cwd than expected, so relative paths like "workspace" would
        # resolve incorrectly.
        ws_abs = str(Path(req.inputs_dir) / req.workspace)
        cmd += ["--workspace", ws_abs]

    # Auto-resolve config for commands that need it
    config = _resolve_config(req)
    if config:
        cmd += ["--config", config]

    # For LLM-backed commands, pass the active provider and model so the
    # subprocess uses the same settings as the server.
    if action == "suggest-fix":
        from gw.api.llm_config import get_active_provider, get_model
        cmd += ["--provider", get_active_provider()]
        model = get_model()
        if model:
            cmd += ["--model", model]

    if req.out_dir:
        cmd += ["--out-dir", req.out_dir]
    if action == "apply-fixes":
        if req.fix_plan:
            cmd += ["--fix-plan", req.fix_plan]
        if req.apply:
            cmd += ["--apply"]
        for c in req.confirm:
            cmd += ["--confirm", c]
    return cmd


# -----------------------------
# Workspace / project inspection
# -----------------------------

@router.get("/workspaces/inspect", response_model=WorkspaceInfo)
def inspect_workspace(inputs_dir: str):
    ws = infer_workspace(inputs_dir)
    ad = artifacts_dir(inputs_dir, ws)
    return WorkspaceInfo(
        inputs_dir=inputs_dir,
        workspace=ws,
        artifacts_dir=ad,
        recent_artifacts=list_recent_artifacts(ad),
    )


@router.post("/projects/inspect", response_model=ProjectInspectResponse)
def inspect_project(req: ProjectInspectRequest):
    from gw.projects.importer import detect_project

    p = req.path
    detected, _ctx = detect_project(p)

    if detected == "copilot_project":
        ws = infer_workspace(p)
        ad = artifacts_dir(p, ws)
        return ProjectInspectResponse(
            status="ok",
            workspace_info=WorkspaceInfo(
                inputs_dir=p,
                workspace=ws,
                artifacts_dir=ad,
                recent_artifacts=list_recent_artifacts(ad),
            ),
        )

    if detected == "mf6_workspace":
        # New flow: offer in-place GW_Copilot setup (no file copying)
        return ProjectInspectResponse(
            status="needs_setup",
            message=(
                "Detected a MODFLOW 6 workspace. "
                "A GW_Copilot/ folder will be created inside this directory "
                "to store validation reports, plots, and other tool outputs. "
                "Your original model files will not be modified."
            ),
            import_plan=ImportPlanModel(
                detected_type="mf6_workspace",
                source_path=p,
                proposed_project_path=p,
                actions=[],
                warnings=[],
                context=_ctx,
            ),
        )

    return ProjectInspectResponse(
        status="unknown",
        message=(
            "No Copilot project (config.json) or MF6 workspace signature was detected in that folder. "
            "Point Inspect at either an existing Copilot project root, or an MF6 example folder that contains .nam files."
        ),
    )


@router.post("/projects/setup")
def setup_project(req: ProjectInspectRequest):
    """Create GW_Copilot/ in-place inside an MF6 model directory."""
    from gw.projects.importer import setup_copilot_folder

    try:
        model_dir, info = setup_copilot_folder(req.path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ws = infer_workspace(model_dir)
    ad = artifacts_dir(model_dir, ws)
    return ProjectImportResponse(
        status="ok",
        project_path=model_dir,
        workspace_info=WorkspaceInfo(
            inputs_dir=model_dir,
            workspace=ws,
            artifacts_dir=ad,
            recent_artifacts=list_recent_artifacts(ad),
        ),
    )


@router.post("/projects/import", response_model=ProjectImportResponse)
def import_project(req: ProjectImportRequest):
    from gw.projects.importer import ImportAction, ImportPlan, execute_import_plan

    p = req.plan
    plan = ImportPlan(
        detected_type=p.detected_type,  # type: ignore[arg-type]
        source_path=p.source_path,
        proposed_project_path=p.proposed_project_path,
        actions=[
            ImportAction(
                op=a.op,
                path=a.path,
                src=a.src,
                bytes=a.bytes,
                count_estimate=a.count_estimate,
            )
            for a in (p.actions or [])
        ],
        warnings=p.warnings or [],
        context=p.context or {},
    )

    try:
        project_path = execute_import_plan(plan)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ws = infer_workspace(project_path)
    ad = artifacts_dir(project_path, ws)
    info = WorkspaceInfo(
        inputs_dir=project_path,
        workspace=ws,
        artifacts_dir=ad,
        recent_artifacts=list_recent_artifacts(ad),
    )
    return ProjectImportResponse(status="ok", project_path=project_path, workspace_info=info)


# -----------------------------
# Workspace file browsing (Model Files tab)
# -----------------------------

def _filter_workspace_files_for_view(files: list[Any], view: str) -> list[Any]:
    """Filter files for Model Files tab.
    view='modflow' hides tool/generated directories (artifacts/plots/ui/etc.) and shows the model workspace inputs.
    view='all' shows everything.
    """
    if not files:
        return []
    v = (view or "modflow").strip().lower()
    if v in ("all", "everything"):
        return files

    # Exclude common tool/build/generated folders that are not part of the MODFLOW workspace deliverable.
    EXCLUDE_PREFIXES = (
        "GW_Copilot/",
        "run_artifacts/",
        "artifacts/",
        "plots/",
        "ui/",
        "node_modules/",
        ".git/",
        "__pycache__/",
        ".venv/",
        "venv/",
        "dist/",
        "build/",
    )

    out = []
    for f in files:
        rel = getattr(f, "path_rel", None) or getattr(f, "path", None) or ""
        rel = str(rel).replace("\\", "/")
        # drop hidden junk
        name = rel.split("/")[-1]
        if name in (".DS_Store", "Thumbs.db"):
            continue
        if rel.startswith(EXCLUDE_PREFIXES):
            continue
        out.append(f)
    return out

def _validate_glob(pattern: str) -> str:
    """Sanitise a user-supplied glob pattern to prevent DoS / injection."""
    if not pattern or not pattern.strip():
        return "**/*"
    if len(pattern) > 200:
        raise HTTPException(status_code=400, detail="Glob pattern too long")
    if pattern.count("*") > 8:
        raise HTTPException(status_code=400, detail="Too many wildcards in glob pattern")
    import re as _re_glob
    if not _re_glob.match(r"^[a-zA-Z0-9_\-.*/?\\:\[\]]+$", pattern):
        raise HTTPException(status_code=400, detail="Glob pattern contains invalid characters")
    return pattern


@router.get("/workspace/files", response_model=WorkspaceFilesResponse)
def workspace_files(
    inputs_dir: str,
    workspace: str | None = None,
    glob: str = "**/*",
    max: int = 2000,
    include_hash: bool = False,
    view: str = "modflow",
):
    glob = _validate_glob(glob)
    max = min(max, 5000)  # Hard cap on file count
    try:
        root = _resolve_root(inputs_dir, workspace)
        files, truncated = _list_files(root=root, glob=glob, max_files=max, include_hash=include_hash)
        files = _filter_workspace_files_for_view(files, view)
        files_out = []
        for f in (files or []):
            try:
                files_out.append(_to_plain_dict(f))
            except Exception:
                # last resort: assume it has attributes
                files_out.append({
                    "path_rel": getattr(f, "path_rel", str(f)),
                    "size": getattr(f, "size", 0),
                    "mtime": getattr(f, "mtime", ""),
                    "kind": getattr(f, "kind", "text"),
                    "sha256": getattr(f, "sha256", None),
                })
        return WorkspaceFilesResponse(root=str(root), files=files_out, truncated=truncated)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workspace/file", response_model=WorkspaceFileReadResponse)
def workspace_file(
    inputs_dir: str,
    path_rel: str,
    workspace: str | None = None,
    max_bytes: int = 2_000_000,
):
    """
    Robust, Notepad-like file reader:
    - Resolves path safely under workspace root
    - Detects binary vs text
    - Returns text preview up to max_bytes + truncation flag
    - Includes sha256 of full file for chain-of-custody
    """
    try:
        root = _resolve_root(inputs_dir, workspace)
        p = _resolve_file(root, path_rel)

        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"file not found: {path_rel}")

        size = p.stat().st_size
        mime = _guess_mime(p)

        # Compute sha256 for provenance (full file)
        sha256 = _sha256_file(p)

        content, truncated, kind = _read_preview_text(p, max_bytes=max_bytes)

        obj = {
            "path_rel": path_rel,
            "content": content,
            "truncated": bool(truncated),
            "size": int(size),
            "sha256": sha256,
            "kind": kind,
            "mime": mime,
        }
        return WorkspaceFileReadResponse(**obj)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workspace/file/download")
def workspace_file_download(
    inputs_dir: str,
    path_rel: str,
    workspace: str | None = None,
):
    try:
        root = _resolve_root(inputs_dir, workspace)
        p = _resolve_file(root, path_rel)

        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"file not found: {path_rel}")

        return FileResponse(
            path=str(p),
            filename=p.name,
            media_type="application/octet-stream",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------
# Deterministic actions + jobs
# -----------------------------

@router.post("/run/{action}", response_model=RunResponse)
def run_action(action: str, req: RunRequest):
    allowed = {"validate", "suggest-fix", "apply-fixes", "revalidate"}
    if action not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: '{action}'. Valid actions: {', '.join(sorted(allowed))}",
        )

    # Validate inputs_dir exists
    if not Path(req.inputs_dir).exists():
        raise HTTPException(
            status_code=400,
            detail=f"inputs_dir does not exist: {req.inputs_dir}",
        )

    if req.workspace is None:
        req.workspace = infer_workspace(req.inputs_dir)

    # For suggest-fix, verify we can find config.json (required by the CLI)
    if action == "suggest-fix":
        config = _resolve_config(req)
        if not config:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not find config.json needed for suggest-fix. "
                    "Ensure config.json exists in your inputs directory or workspace."
                ),
            )
        from gw.api.llm_config import get_active_provider, get_api_key as _get_api_key
        active_provider = get_active_provider()
        api_key = _get_api_key(active_provider)
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No API key configured for the active LLM provider ('{active_provider}'). "
                    "Set your API key in the LLM Settings panel or via environment variables. "
                    "suggest-fix requires an LLM provider to analyze errors and propose fixes."
                ),
            )

    try:
        job = create_job()
        cmd = _build_cmd(action, req)
        run_subprocess_job(job, cmd=cmd, cwd=None)
        return RunResponse(job_id=job.id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start {action} job: {e}",
        )


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(
        job_id=job.id,
        state=job.state,
        exit_code=job.exit_code,
        last_lines=job.last_lines[-50:],
        result=job.result,
        error=job.error,
    )


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    def gen():
        import time
        last_len = 0
        while True:
            time.sleep(0.25)
            j = get_job(job_id)
            if not j:
                break
            cur = j.last_lines
            if len(cur) > last_len:
                for line in cur[last_len:]:
                    payload = json.dumps({"line": line})
                    yield f"data: {payload}\n\n"
                last_len = len(cur)
            if j.state in {"done", "error"}:
                payload = json.dumps({"final": True, "state": j.state, "exit_code": j.exit_code})
                yield f"data: {payload}\n\n"
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


# -----------------------------
# Artifacts
# -----------------------------

@router.get("/artifacts/read")
def read_artifact(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return PlainTextResponse(p.read_text(encoding="utf-8", errors="replace"))


# -----------------------------
# Filesystem search
# -----------------------------

@router.post("/fs/find", response_model=FsFindResponse)
def fs_find(req: FsFindRequest):
    matches, roots_used = find_paths(
        query=req.query,
        kind=req.kind,
        max_results=req.max_results,
        roots=req.roots,
    )
    return FsFindResponse(matches=matches, roots_used=roots_used)


@router.get("/fs/browse")
def fs_browse():
    """Open a native directory picker dialog and return the selected path."""
    import threading

    result: dict = {"path": None, "error": None}

    def _pick():
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = filedialog.askdirectory(title="Select model inputs directory")
            root.destroy()
            result["path"] = folder or None
        except Exception as e:
            result["error"] = str(e)

    # tkinter must run on a dedicated thread (not the asyncio event loop thread)
    t = threading.Thread(target=_pick, daemon=True)
    t.start()
    t.join(timeout=120)

    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return {"path": result["path"]}


# -----------------------------
# Chat
# -----------------------------

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        from gw.llm.chat_agent import chat_reply
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"chat module import failed: {e}")

    ws = req.workspace or (infer_workspace(req.inputs_dir) if req.inputs_dir else None)
    try:
        reply, suggestions, plot_outputs = chat_reply(
            message=req.message,
            inputs_dir=req.inputs_dir,
            workspace=ws,
            history=req.history,
        )
        return ChatResponse(
            reply=reply,
            suggestions=suggestions,
            plot_outputs=[ChatPlotOutput(**p) for p in plot_outputs] if plot_outputs else [],
        )
    except Exception as e:
        return ChatResponse(
            reply=(
                "I hit an error while generating a chat reply.\n\n"
                f"Error: {type(e).__name__}: {e}\n\n"
                "Tips:\n"
                "• If you want LLM replies, set OPENAI_API_KEY in your environment.\n"
                "• For deterministic actions, try: validate | revalidate | suggest-fix | apply-fixes\n"
                "• Check the server console for a traceback."
            ),
            suggestions=["validate", "revalidate", "suggest-fix"],
        )


# -----------------------------
# Debug: location context (temporary — remove after confirming)
# -----------------------------

@router.get("/debug/location-context")
def debug_location_context(inputs_dir: str):
    """Debug endpoint to test location context computation.

    Call: GET /debug/location-context?inputs_dir=<path>
    Returns the full diagnostic chain for location context loading.
    """
    import traceback
    result: dict = {"inputs_dir": inputs_dir, "steps": []}

    # Step 1: Check config path
    from pathlib import Path
    for candidate in [
        Path(inputs_dir) / "GW_Copilot" / "config.json",
        Path(inputs_dir).parent / "GW_Copilot" / "config.json",
    ]:
        if candidate.exists():
            result["config_path"] = str(candidate)
            result["steps"].append(f"Found config at {candidate}")
            break
    else:
        result["steps"].append("No GW_Copilot/config.json found")
        return result

    # Step 2: Read config
    import json as _json
    try:
        cfg = _json.loads(candidate.read_text(encoding="utf-8"))
        result["config_keys"] = list(cfg.keys())
        result["spatial_ref"] = cfg.get("spatial_ref")
        result["has_location_context"] = "location_context" in cfg
        result["location_context"] = cfg.get("location_context")
        result["steps"].append(f"Config loaded. Keys: {list(cfg.keys())}")
    except Exception as e:
        result["steps"].append(f"Failed to read config: {e}")
        return result

    # Step 3: Try load_location_context from viz
    try:
        from gw.api.viz import load_location_context
        loc = load_location_context(inputs_dir)
        result["viz_load_location_context"] = loc
        result["steps"].append(f"viz.load_location_context returned: {loc}")
    except Exception as e:
        result["viz_load_location_context_error"] = f"{type(e).__name__}: {e}"
        result["viz_load_location_context_traceback"] = traceback.format_exc()
        result["steps"].append(f"viz.load_location_context FAILED: {e}")

    # Step 4: Try the chat_agent fallback
    try:
        from gw.llm.chat_agent import _try_compute_location_from_config
        fallback = _try_compute_location_from_config(inputs_dir)
        result["chat_agent_fallback"] = fallback
        result["steps"].append(f"chat_agent fallback returned: {fallback}")
    except Exception as e:
        result["chat_agent_fallback_error"] = f"{type(e).__name__}: {e}"
        result["chat_agent_fallback_traceback"] = traceback.format_exc()
        result["steps"].append(f"chat_agent fallback FAILED: {e}")

    # Step 5: Re-read config to see if it was updated
    try:
        cfg2 = _json.loads(candidate.read_text(encoding="utf-8"))
        result["config_after"] = cfg2.get("location_context")
        result["steps"].append(f"Config after computation: location_context={cfg2.get('location_context')}")
    except Exception:
        pass

    return result
