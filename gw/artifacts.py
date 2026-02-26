from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ArtifactPaths:
    artifacts_dir: str
    stress_md: str
    stress_json: str
    fix_plan_json: str
    applied_audit_json: str
    applied_audit_md: str
    config_updated_json: str
    revalidate_diff_md: str
    dashboard_json: str
    latest_json: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_gw_copilot(inputs_dir: str) -> bool:
    """Check if inputs_dir has a GW_Copilot/ subfolder with config.json."""
    return (Path(inputs_dir) / "GW_Copilot" / "config.json").exists()


def resolve_workspace(inputs_dir: str, workspace: Optional[str]) -> str:
    """Resolve workspace path (where MF6 model files live).

    With GW_Copilot layout: model files are directly in inputs_dir.
    Legacy layout: <inputs_dir>/workspace.
    Creates the directory if needed.
    """
    inputs_path = Path(inputs_dir)

    # GW_Copilot layout: model files are at the root of inputs_dir
    if _has_gw_copilot(inputs_dir):
        inputs_path.mkdir(parents=True, exist_ok=True)
        return str(inputs_path)

    ws = Path(workspace) if workspace else (inputs_path / "workspace")
    ws.mkdir(parents=True, exist_ok=True)
    return str(ws)


def resolve_artifacts_dir(inputs_dir: str, workspace: Optional[str], out_dir: Optional[str]) -> str:
    """Artifact directory (created if needed).

    With GW_Copilot layout: <inputs_dir>/GW_Copilot/
    Legacy: <workspace>/run_artifacts
    """
    if out_dir:
        p = Path(out_dir)
    elif _has_gw_copilot(inputs_dir):
        p = Path(inputs_dir) / "GW_Copilot"
    else:
        ws = Path(resolve_workspace(inputs_dir, workspace))
        p = ws / "run_artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def default_artifact_paths(inputs_dir: str, workspace: Optional[str], out_dir: Optional[str]) -> ArtifactPaths:
    adir = Path(resolve_artifacts_dir(inputs_dir, workspace, out_dir))

    # GW_Copilot layout: validation artifacts go into validation/ subfolder
    if _has_gw_copilot(inputs_dir) and not out_dir:
        vdir = adir / "validation"
        vdir.mkdir(parents=True, exist_ok=True)
        return ArtifactPaths(
            artifacts_dir=str(adir),
            stress_md=str(vdir / "stress_validation.md"),
            stress_json=str(vdir / "stress_validation.json"),
            fix_plan_json=str(vdir / "fix_plan.json"),
            applied_audit_json=str(vdir / "applied_fixes.audit.json"),
            applied_audit_md=str(vdir / "applied_fixes.audit.md"),
            config_updated_json=str(vdir / "config.updated.json"),
            revalidate_diff_md=str(vdir / "revalidate.diff.md"),
            dashboard_json=str(adir / "dashboard.json"),
            latest_json=str(adir / "latest.json"),
        )

    # Legacy layout: everything flat under run_artifacts/
    return ArtifactPaths(
        artifacts_dir=str(adir),
        stress_md=str(adir / "stress_validation.md"),
        stress_json=str(adir / "stress_validation.json"),
        fix_plan_json=str(adir / "fix_plan.json"),
        applied_audit_json=str(adir / "applied_fixes.audit.json"),
        applied_audit_md=str(adir / "applied_fixes.audit.md"),
        config_updated_json=str(adir / "config.updated.json"),
        revalidate_diff_md=str(adir / "revalidate.diff.md"),
        dashboard_json=str(adir / "dashboard.json"),
        latest_json=str(adir / "latest.json"),
    )


def write_latest(paths: ArtifactPaths, payload: Dict[str, Any]) -> None:
    """Writes a lightweight pointer file that GUIs can watch."""
    out = {
        "ts_utc": _utc_now_iso(),
        "artifacts_dir": paths.artifacts_dir,
        "artifacts": payload,
    }
    Path(paths.latest_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
