from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from gw.artifacts import default_artifact_paths, write_latest
from gw.dashboard import write_dashboard
from gw.revalidate import run_revalidate


@dataclass
class PipelineResult:
    ok: bool
    step: str
    message: str
    artifacts_dir: str
    dashboard_json: str
    exit_code: int = 0


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_pipeline(
    *,
    inputs_dir: str,
    base_config: Dict[str, Any],
    workspace: Optional[str] = None,
    out_dir: Optional[str] = None,
    apply: bool = False,
    provider: str = "openai",
    model: Optional[str] = None,
    max_actions: int = 8,
) -> PipelineResult:
    """validate -> (if errors) suggest-fix-plan -> apply-fixes -> revalidate"""
    paths = default_artifact_paths(inputs_dir, workspace, out_dir)

    r0 = run_revalidate(
        inputs_dir=inputs_dir,
        base_config=base_config,
        workspace=workspace,
        out_dir=out_dir,
        previous_json_path=None,
    )
    if r0.ok:
        msg = "Pipeline: validation already OK."
        write_dashboard(
            dashboard_path=paths.dashboard_json,
            step="pipeline",
            ok=True,
            counts={"errors": r0.errors, "warnings": r0.warnings, "info": r0.info},
            artifacts={"stress_md": r0.stress_report_md_path, "stress_json": r0.stress_report_json_path},
            message=msg,
            extra={"phase": "validate"},
        )
        write_latest(paths, {"dashboard": paths.dashboard_json})
        return PipelineResult(ok=True, step="validate", message=msg, artifacts_dir=paths.artifacts_dir, dashboard_json=paths.dashboard_json)

    # Suggest FixPlan (LLM)
    try:
        from gw.llm.suggest_fix_plan import suggest_fix_plan
    except Exception:
        try:
            from gw.llm.fix_plan_suggest import suggest_fix_plan  # fallback
        except Exception as e:
            msg = f"Pipeline failed: could not import suggest_fix_plan ({e})"
            write_dashboard(
                dashboard_path=paths.dashboard_json,
                step="pipeline",
                ok=False,
                counts={"errors": r0.errors, "warnings": r0.warnings, "info": r0.info},
                artifacts={"stress_md": r0.stress_report_md_path, "stress_json": r0.stress_report_json_path},
                message=msg,
                extra={"phase": "suggest"},
            )
            write_latest(paths, {"dashboard": paths.dashboard_json})
            return PipelineResult(ok=False, step="suggest", message=msg, artifacts_dir=paths.artifacts_dir, dashboard_json=paths.dashboard_json, exit_code=2)

    plan = suggest_fix_plan(
        provider=provider,
        model=model,
        config_path=str(Path(inputs_dir) / "config.json"),
        inputs_dir=inputs_dir,
        stress_validation_report_path=r0.stress_report_md_path,
        max_actions=max_actions,
    )

    Path(paths.fix_plan_json).write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    write_dashboard(
        dashboard_path=paths.dashboard_json,
        step="pipeline",
        ok=False,
        counts={"errors": r0.errors, "warnings": r0.warnings, "info": r0.info},
        artifacts={
            "stress_md": r0.stress_report_md_path,
            "stress_json": r0.stress_report_json_path,
            "fix_plan": paths.fix_plan_json,
        },
        message="Pipeline: FixPlan proposed.",
        extra={"phase": "suggest"},
    )
    write_latest(paths, {"fix_plan": paths.fix_plan_json, "dashboard": paths.dashboard_json})

    # Apply fixes
    try:
        from gw.fix_apply import apply_fix_plan
    except Exception as e:
        msg = f"Pipeline failed: could not import apply_fix_plan ({e})"
        write_dashboard(
            dashboard_path=paths.dashboard_json,
            step="pipeline",
            ok=False,
            counts={"errors": r0.errors, "warnings": r0.warnings, "info": r0.info},
            artifacts={"fix_plan": paths.fix_plan_json},
            message=msg,
            extra={"phase": "apply"},
        )
        return PipelineResult(ok=False, step="apply", message=msg, artifacts_dir=paths.artifacts_dir, dashboard_json=paths.dashboard_json, exit_code=2)

    try:
        from gw.llm.fix_plan import FixPlan
        plan_obj = FixPlan.model_validate(_load_json(paths.fix_plan_json))
    except Exception:
        plan_obj = _load_json(paths.fix_plan_json)

    updated_cfg, res = apply_fix_plan(
        fix_plan=plan_obj,
        base_config=base_config,
        inputs_dir=inputs_dir,
        workspace=workspace,
        out_dir=out_dir,
        dry_run=(not apply),
        user_confirmations=[],
        max_actions=max_actions,
    )

    artifacts = {
        "fix_plan": paths.fix_plan_json,
        "audit_json": getattr(res, "audit_json_path", None),
        "audit_md": getattr(res, "audit_md_path", None),
    }

    if apply and getattr(res, "changed_config", False):
        Path(paths.config_updated_json).write_text(json.dumps(updated_cfg, indent=2), encoding="utf-8")
        artifacts["config_updated"] = paths.config_updated_json

    write_dashboard(
        dashboard_path=paths.dashboard_json,
        step="pipeline",
        ok=False,
        counts={"errors": r0.errors, "warnings": r0.warnings, "info": r0.info},
        artifacts=artifacts,
        message="Pipeline: FixPlan applied (dry-run)" if not apply else "Pipeline: FixPlan applied.",
        extra={"phase": "apply", "dry_run": (not apply)},
    )
    write_latest(paths, {"dashboard": paths.dashboard_json, **artifacts})

    r1 = run_revalidate(
        inputs_dir=inputs_dir,
        base_config=updated_cfg if apply else base_config,
        workspace=workspace,
        out_dir=out_dir,
        previous_json_path=paths.stress_json,
    )

    ok = r1.ok
    msg = "Pipeline OK." if ok else "Pipeline still failing after fixes."
    write_dashboard(
        dashboard_path=paths.dashboard_json,
        step="pipeline",
        ok=ok,
        counts={"errors": r1.errors, "warnings": r1.warnings, "info": r1.info},
        artifacts={
            **artifacts,
            "revalidate_stress_md": r1.stress_report_md_path,
            "revalidate_stress_json": r1.stress_report_json_path,
            "revalidate_diff_md": r1.diff_md_path,
        },
        message=msg,
        extra={"phase": "revalidate"},
    )
    write_latest(paths, {"dashboard": paths.dashboard_json})
    return PipelineResult(ok=ok, step="revalidate", message=msg, artifacts_dir=paths.artifacts_dir, dashboard_json=paths.dashboard_json, exit_code=(0 if ok else 2))
