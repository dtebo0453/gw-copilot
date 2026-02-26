from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


DetectedType = Literal["copilot_project", "mf6_workspace", "unknown"]


@dataclass(frozen=True)
class ImportAction:
    op: str  # mkdir | copytree | write_text
    path: str
    src: Optional[str] = None
    bytes: Optional[int] = None
    count_estimate: Optional[int] = None


@dataclass(frozen=True)
class ImportPlan:
    detected_type: DetectedType
    source_path: str
    proposed_project_path: str
    actions: List[ImportAction]
    warnings: List[str]
    context: Dict[str, Any]


def _safe_slug(name: str) -> str:
    s = "".join(c if c.isalnum() or c in {"-", "_"} else "-" for c in (name or "project"))
    s = "-".join([p for p in s.split("-") if p]).strip("-")
    return s or "project"


def _looks_like_copilot_project(p: Path) -> Tuple[bool, Optional[Path]]:
    """Return (is_project, workspace_path_if_any).

    Recognizes two project layouts:
    1. Legacy: ``<project>/config.json`` with workspace subdir
    2. GW_Copilot: ``<model_dir>/GW_Copilot/config.json`` — model files
       live directly in ``<model_dir>`` (the parent).
    """
    # --- New layout: GW_Copilot/ subfolder with config.json ---
    gw_cfg = p / "GW_Copilot" / "config.json"
    if gw_cfg.exists():
        try:
            obj = json.loads(gw_cfg.read_text(encoding="utf-8"))
            # workspace "." means model files are in p itself
            ws_rel = obj.get("workspace", ".")
            ws_path = (p / ws_rel).resolve() if ws_rel else p.resolve()
            if ws_path.exists():
                return (True, ws_path)
            return (True, p)
        except Exception:
            # GW_Copilot/config.json exists but is broken — still a project
            return (True, p)

    # --- Legacy layout: config.json at project root ---
    cfg = p / "config.json"
    if not cfg.exists():
        return (False, None)
    try:
        obj = json.loads(cfg.read_text(encoding="utf-8"))
        ws = obj.get("workspace")
        if ws:
            ws_path = Path(ws)
            if not ws_path.is_absolute():
                ws_path = (p / ws).resolve()
            if ws_path.exists():
                return (True, ws_path)
    except Exception:
        # If config.json exists but isn't parseable, treat it as not-a-project.
        return (False, None)
    # fallback: project-style workspace dir
    ws_dir = p / "workspace"
    if ws_dir.exists():
        return (True, ws_dir)
    return (True, None)


def _detect_mf6_signatures(p: Path) -> Dict[str, Any]:
    """Shallow-ish MF6 signature scan (fast, deterministic)."""
    ctx: Dict[str, Any] = {"namefiles": [], "mfsim": False, "package_files": []}

    # Shallow scan: top-level + one level deep. Keeps IO small.
    candidates: List[Path] = []
    candidates.extend(list(p.glob("*.nam")))
    candidates.extend(list(p.glob("*/*.nam")))
    # mfsim.nam is common
    for c in candidates:
        if c.name.lower() == "mfsim.nam":
            ctx["mfsim"] = True
        ctx["namefiles"].append(str(c))

    # Some other common MF6 files
    pkg_exts = {".dis", ".disu", ".npf", ".ic", ".sto", ".oc", ".rcha", ".wel", ".chd", ".drn", ".ghb", ".riv", ".evt", ".sfr", ".uzf", ".gwe", ".gwf", ".ims"}
    pkg_files: List[str] = []
    for f in list(p.glob("*")) + list(p.glob("*/*")):
        if f.is_file() and f.suffix.lower() in pkg_exts:
            pkg_files.append(str(f))
    ctx["package_files"] = sorted(pkg_files)[:50]
    return ctx


def detect_project(path: str) -> Tuple[DetectedType, Dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return ("unknown", {"reason": "path_not_found"})

    is_proj, ws = _looks_like_copilot_project(p)
    if is_proj:
        return (
            "copilot_project",
            {
                "project_root": str(p),
                "workspace": str(ws) if ws else None,
            },
        )

    mf6_ctx = _detect_mf6_signatures(p)
    if mf6_ctx.get("namefiles") or mf6_ctx.get("mfsim") or mf6_ctx.get("package_files"):
        return ("mf6_workspace", mf6_ctx)

    return ("unknown", {"reason": "no_project_markers"})


def build_import_plan(source_path: str, *, project_root: Optional[str] = None) -> ImportPlan:
    src = Path(source_path).expanduser().resolve()
    detected, ctx = detect_project(str(src))
    warnings: List[str] = []

    if detected == "copilot_project":
        raise ValueError("Source already looks like a Copilot project; import is not needed.")

    if detected == "unknown":
        raise ValueError("Could not detect a Copilot project or MF6 workspace in the provided path.")

    # Determine destination under <repo_root>/runs/imported
    repo_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    # If called from API, cwd should be repo root; but be defensive:
    if not (repo_root / "gw").exists() and (repo_root.parent / "gw").exists():
        repo_root = repo_root.parent

    imported_root = repo_root / "runs" / "imported"
    imported_root.mkdir(parents=True, exist_ok=True)

    slug = _safe_slug(src.name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = imported_root / f"{slug}_{ts}"

    # Estimate file count (cap to avoid heavy IO)
    count_est = 0
    try:
        for i, f in enumerate(src.rglob("*")):
            if i > 5000:
                count_est = 5000
                warnings.append("Large source folder: showing a capped file count estimate (>=5000).")
                break
            if f.is_file():
                count_est += 1
    except Exception:
        warnings.append("Could not estimate file count; import may still work.")

    # Minimal config.json for imported MF6 workspace
    rel_ws = (dest / "workspace")
    # store workspace relative to repo root for portability
    rel_ws_str = str(Path("runs") / "imported" / dest.name / "workspace").replace("\\", "/")
    cfg = {
        "model_name": slug,
        "workspace": rel_ws_str,
        "project_type": "mf6_workspace",
        "inputs": {
            "note": "Imported MF6 workspace. Stress CSV validation is not mapped yet."
        },
        "import": {
            "source_path": str(src),
            "detected": ctx,
            "imported_at": ts,
        },
    }
    cfg_text = json.dumps(cfg, indent=2)

    actions: List[ImportAction] = [
        ImportAction(op="mkdir", path=str(dest)),
        ImportAction(op="mkdir", path=str(rel_ws)),
        ImportAction(op="mkdir", path=str(dest / "run_artifacts")),
        ImportAction(op="mkdir", path=str(dest / "inputs")),
        ImportAction(op="copytree", path=str(rel_ws), src=str(src), count_estimate=count_est),
        ImportAction(op="write_text", path=str(dest / "config.json"), bytes=len(cfg_text.encode("utf-8"))),
    ]

    if detected == "mf6_workspace":
        warnings.append(
            "MF6 workspaces typically encode stresses in package files; Copilot stress CSV validation will be limited until you add a mapping/import step."
        )

    return ImportPlan(
        detected_type=detected,
        source_path=str(src),
        proposed_project_path=str(dest),
        actions=actions,
        warnings=warnings,
        context=ctx,
    )


def execute_import_plan(plan: ImportPlan) -> str:
    """Execute the import plan. Writes only under runs/imported/* by construction."""
    dest = Path(plan.proposed_project_path).resolve()
    src = Path(plan.source_path).resolve()
    ws = dest / "workspace"

    if dest.exists():
        raise ValueError(f"Destination already exists: {dest}")

    # Create dirs
    (dest / "workspace").mkdir(parents=True, exist_ok=False)
    (dest / "run_artifacts").mkdir(parents=True, exist_ok=False)
    (dest / "inputs").mkdir(parents=True, exist_ok=False)

    # Copy source contents into workspace
    # We copy file-by-file into the existing workspace directory.
    for item in src.iterdir():
        target = ws / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    # Write config.json (recompute to ensure it matches actual dest name)
    cfg_path = dest / "config.json"
    # If caller provided cfg via actions, just write a minimal one.
    rel_ws_str = str(Path("runs") / "imported" / dest.name / "workspace").replace("\\", "/")
    cfg = {
        "model_name": dest.name,
        "workspace": rel_ws_str,
        "project_type": plan.detected_type,
        "inputs": {
            "note": "Imported MF6 workspace. Stress CSV validation is not mapped yet."
        },
        "import": {
            "source_path": str(src),
            "detected": plan.context,
            "imported_at": datetime.now().isoformat(),
        },
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Also write an import log in run_artifacts for provenance
    log_path = dest / "run_artifacts" / "import_log.json"
    log_path.write_text(
        json.dumps(
            {
                "plan": {
                    "detected_type": plan.detected_type,
                    "source_path": plan.source_path,
                    "proposed_project_path": plan.proposed_project_path,
                    "warnings": plan.warnings,
                    "context": plan.context,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return str(dest)


def setup_copilot_folder(model_dir: str) -> Tuple[str, Dict[str, Any]]:
    """Create a ``GW_Copilot/`` subfolder inside an existing MF6 model directory.

    This is the **new** project-creation flow: no file copying.  The original
    model files stay where they are, and all tool outputs go into ``GW_Copilot/``.

    Returns ``(model_dir, workspace_info_dict)`` where *model_dir* is the
    absolute path to the model directory (also serves as inputs_dir).
    """
    md = Path(model_dir).expanduser().resolve()
    if not md.exists() or not md.is_dir():
        raise ValueError(f"Model directory does not exist: {md}")

    gw_dir = md / "GW_Copilot"
    gw_dir.mkdir(exist_ok=True)

    # Create organised subfolders
    (gw_dir / "validation").mkdir(exist_ok=True)
    (gw_dir / "plots").mkdir(exist_ok=True)
    (gw_dir / "qa").mkdir(exist_ok=True)

    # Write/update config.json
    cfg_path = gw_dir / "config.json"
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    slug = _safe_slug(md.name)
    cfg.update({
        "model_name": cfg.get("model_name", slug),
        "workspace": ".",  # model files are in the parent (model_dir)
        "project_type": "mf6_workspace",
        "copilot_dir": "GW_Copilot",
        "created_at": cfg.get("created_at", datetime.now().isoformat()),
        "updated_at": datetime.now().isoformat(),
    })
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    return str(md), {
        "inputs_dir": str(md),
        "workspace": None,       # model files are at the root of inputs_dir
        "artifacts_dir": str(gw_dir),
        "copilot_dir": str(gw_dir),
    }
