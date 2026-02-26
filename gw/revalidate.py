from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gw.artifacts import default_artifact_paths, write_latest
from gw.dashboard import write_dashboard


@dataclass
class RevalidateResult:
    """Artifacts + summary of revalidation run."""
    stress_report_md_path: str
    stress_report_json_path: Optional[str]
    diff_md_path: Optional[str]
    ok: bool
    errors: int
    warnings: int
    info: int
    # convenience for CLI/CI
    summary_line: str = ""
    exit_code: int = 0
    failed: bool = False


def _render_findings_md(findings: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# Stress validation report")
    lines.append("")
    if not findings:
        lines.append("✅ No findings.")
        return "\n".join(lines) + "\n"

    sev_counts = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        sev = str(f.get("severity") or f.get("level") or "info").lower()
        if sev.startswith("err"):
            sev_counts["error"] += 1
        elif sev.startswith("warn"):
            sev_counts["warning"] += 1
        else:
            sev_counts["info"] += 1

    lines.append(f"- errors: {sev_counts['error']}")
    lines.append(f"- warnings: {sev_counts['warning']}")
    lines.append(f"- info: {sev_counts['info']}")
    lines.append("")
    lines.append("| severity | code | message | file | row |")
    lines.append("|---|---|---|---|---|")
    for f in findings[:500]:
        sev = str(f.get("severity") or f.get("level") or "").strip()
        code = str(f.get("code") or "").strip()
        msg = str(f.get("message") or f.get("detail") or "").replace("\n", " ").strip()
        file = str(f.get("file") or f.get("filename") or "").strip()
        row = f.get("row") or f.get("row_index") or ""
        lines.append(f"| {sev} | {code} | {msg} | {file} | {row} |")
    if len(findings) > 500:
        lines.append("")
        lines.append(f"_Truncated: showing first 500 of {len(findings)} findings._")
    return "\n".join(lines) + "\n"


def _coerce_findings_to_dict_list(out: Any) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    findings: List[Dict[str, Any]] = []

    if isinstance(out, tuple) and len(out) == 2:
        _, maybe_json = out
        out = maybe_json

    if isinstance(out, dict):
        if "findings" in out and isinstance(out["findings"], list):
            for f in out["findings"]:
                findings.append(f if isinstance(f, dict) else {"message": str(f), "severity": "info"})
        elif isinstance(out.get("issues"), list):
            for f in out["issues"]:
                findings.append(f if isinstance(f, dict) else {"message": str(f), "severity": "info"})
        else:
            findings.append(out)
    elif isinstance(out, list):
        for f in out:
            if isinstance(f, dict):
                findings.append(f)
            else:
                d = {}
                for k in ("severity", "level", "code", "message", "detail", "file", "filename", "row", "row_index"):
                    if hasattr(f, k):
                        d[k] = getattr(f, k)
                if not d:
                    d = {"message": str(f)}
                if "severity" not in d and "level" not in d:
                    d["severity"] = "info"
                findings.append(d)
    else:
        findings = [{"message": str(out), "severity": "info"}]

    counts = {"errors": 0, "warnings": 0, "info": 0}
    for f in findings:
        sev = str(f.get("severity") or f.get("level") or "info").lower()
        if sev.startswith("err"):
            counts["errors"] += 1
        elif sev.startswith("warn"):
            counts["warnings"] += 1
        else:
            counts["info"] += 1
    return findings, counts


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(**filtered)


def _find_validator():
    candidates = [
        ("gw.build.stress_validate", "validate_stress_inputs"),
        ("gw.stress_validate", "validate_stress_inputs"),
    ]
    last_err = None
    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            return fn
        except Exception as e:
            last_err = e
    raise ModuleNotFoundError(
        f"Could not import validate_stress_inputs from candidates: {candidates}. Last error: {last_err}"
    )


def _diff_simple(prev: Dict[str, Any], curr: Dict[str, Any]) -> str:
    prev_findings = prev.get("findings") if isinstance(prev, dict) else None
    curr_findings = curr.get("findings") if isinstance(curr, dict) else None

    def _count(x):
        return len(x) if isinstance(x, list) else 0

    lines = ["# Revalidate diff", ""]
    lines.append(f"- previous findings: {_count(prev_findings)}")
    lines.append(f"- current findings:  {_count(curr_findings)}")
    return "\n".join(lines) + "\n"


def run_revalidate(
    *,
    inputs_dir: str,
    base_config: Dict[str, Any],
    workspace: Optional[str] = None,
    out_dir: Optional[str] = None,
    previous_json_path: Optional[str] = None,
) -> RevalidateResult:
    paths = default_artifact_paths(inputs_dir, workspace, out_dir)

    validator = _find_validator()

    out = _call_with_supported_kwargs(
        validator,
        inputs_dir=str(inputs_dir),
        base_config=base_config,
        cfg=base_config,
        workspace=workspace,
        out_dir=paths.artifacts_dir,
    )

    stress_md: str = ""
    stress_json: Optional[Dict[str, Any]] = None

    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], str):
        stress_md = out[0]
        stress_json = out[1] if isinstance(out[1], dict) else None
    elif isinstance(out, str):
        stress_md = out
        stress_json = None
    else:
        findings, counts = _coerce_findings_to_dict_list(out)
        stress_json = {"findings": findings, **counts}
        stress_md = _render_findings_md(findings)

    if stress_json is None:
        stress_json = {"findings": [], "errors": 0, "warnings": 0, "info": 0}

    findings = stress_json.get("findings", [])
    _, counts = _coerce_findings_to_dict_list({"findings": findings})

    Path(paths.stress_md).write_text(stress_md, encoding="utf-8")
    Path(paths.stress_json).write_text(json.dumps(stress_json, indent=2), encoding="utf-8")

    diff_md_path: Optional[str] = None
    if previous_json_path:
        prev_path = Path(previous_json_path)
        if prev_path.exists():
            try:
                prev = json.loads(prev_path.read_text(encoding="utf-8"))
                diff_md = _diff_simple(prev, stress_json)
                Path(paths.revalidate_diff_md).write_text(diff_md, encoding="utf-8")
                diff_md_path = paths.revalidate_diff_md
            except Exception:
                diff_md_path = None

    ok = counts["errors"] == 0
    summary = f"Revalidate ok={ok} (errors={counts['errors']}, warnings={counts['warnings']}, info={counts['info']})"
    exit_code = 0 if ok else 2

    write_dashboard(
        dashboard_path=paths.dashboard_json,
        step="revalidate",
        ok=ok,
        counts=counts,
        artifacts={
            "stress_md": paths.stress_md,
            "stress_json": paths.stress_json,
            "diff_md": diff_md_path,
        },
        message=summary,
    )
    write_latest(
        paths,
        {
            "stress_md": paths.stress_md,
            "stress_json": paths.stress_json,
            "diff_md": diff_md_path,
            "dashboard": paths.dashboard_json,
        },
    )

    return RevalidateResult(
        stress_report_md_path=paths.stress_md,
        stress_report_json_path=paths.stress_json,
        diff_md_path=diff_md_path,
        ok=ok,
        errors=counts["errors"],
        warnings=counts["warnings"],
        info=counts["info"],
        summary_line=summary,
        exit_code=exit_code,
        failed=not ok,
    )
