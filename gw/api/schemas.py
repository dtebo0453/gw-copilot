from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# -----------------------------
# Deterministic run + jobs
# -----------------------------


class RunRequest(BaseModel):
    inputs_dir: str
    workspace: Optional[str] = None
    config: Optional[str] = None
    fix_plan: Optional[str] = None
    out_dir: Optional[str] = None
    apply: bool = False
    confirm: List[str] = []


class RunResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    state: str  # queued|running|done|error
    exit_code: Optional[int] = None
    last_lines: List[str] = []
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# -----------------------------
# Workspace info
# -----------------------------


class WorkspaceInfo(BaseModel):
    inputs_dir: str
    workspace: Optional[str]
    artifacts_dir: Optional[str]
    recent_artifacts: List[str] = []


# -----------------------------
# Chat
# -----------------------------


class ChatRequest(BaseModel):
    """Freeform chat request for the UI."""
    message: str
    inputs_dir: Optional[str] = None
    workspace: Optional[str] = None
    history: Optional[List[dict]] = None


class ChatPlotOutput(BaseModel):
    """A plot image generated during chat via the generate_plot tool."""
    name: str
    url: str


class ChatResponse(BaseModel):
    """Assistant reply returned to the UI."""
    reply: str
    suggestions: List[str] = []
    plot_outputs: List[ChatPlotOutput] = []


# -----------------------------
# Project inspect/import
# -----------------------------


class ProjectInspectRequest(BaseModel):
    path: str


class ImportActionModel(BaseModel):
    op: str
    path: str
    src: Optional[str] = None
    bytes: Optional[int] = None
    count_estimate: Optional[int] = None


class ImportPlanModel(BaseModel):
    detected_type: str
    source_path: str
    proposed_project_path: str
    actions: List[ImportActionModel] = []
    warnings: List[str] = []
    context: Dict[str, Any] = {}


class ProjectInspectResponse(BaseModel):
    status: str  # ok | needs_import | unknown
    workspace_info: Optional[WorkspaceInfo] = None
    import_plan: Optional[ImportPlanModel] = None
    message: Optional[str] = None


class ProjectImportRequest(BaseModel):
    plan: ImportPlanModel


class ProjectImportResponse(BaseModel):
    status: str  # ok
    project_path: str
    workspace_info: WorkspaceInfo


# -----------------------------
# Local filesystem search (robust option B)
# -----------------------------


class FsFindRequest(BaseModel):
    """Search for directories (and optionally files) under allowed roots."""
    query: str
    kind: str = "dir"  # dir | file
    max_results: int = 25
    roots: Optional[List[str]] = None


class FsFindResponse(BaseModel):
    matches: List[str] = []
    roots_used: List[str] = []


# -----------------------------
# Workspace file browsing (Model Files tab)
# -----------------------------


class WorkspaceFileEntry(BaseModel):
    path_rel: str
    size: int
    mtime: str  # ISO8601 UTC (string is fine; UI treats it as display text)
    kind: str  # text | binary | other
    sha256: Optional[str] = None


class WorkspaceFilesResponse(BaseModel):
    root: str
    files: List[WorkspaceFileEntry] = []
    truncated: bool = False


class WorkspaceFileReadResponse(BaseModel):
    path_rel: str
    content: Optional[str] = None
    truncated: bool = False
    size: int
    sha256: str
    kind: str  # text | binary
    mime: str = "text/plain"


# -----------------------------
# Plot planning / execution (Plots tab)
# -----------------------------


class PlotPlanRequest(BaseModel):
    prompt: str
    inputs_dir: str
    workspace: Optional[str] = None
    selected_files: Optional[List[str]] = None


class PlotPlanResponse(BaseModel):
    notes: str
    required_files: List[str] = []
    outputs: List[str] = []
    script: str
    context_hash: str


class PlotRunRequest(BaseModel):
    inputs_dir: str
    workspace: Optional[str] = None
    prompt: Optional[str] = None
    script: str
    confirm: str
    context_hash: Optional[str] = None


class PlotRunOutput(BaseModel):
    name: str
    path: str


class PlotRunResponse(BaseModel):
    run_id: str
    run_dir: str
    outputs: List[PlotRunOutput] = []
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class PlotRunSummary(BaseModel):
    run_id: str
    ts_utc: Optional[str] = None
    prompt: str = ""
    exit_code: Optional[int] = None


class PlotRunsResponse(BaseModel):
    runs: List[PlotRunSummary] = []
