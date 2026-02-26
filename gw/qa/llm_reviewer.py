import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal

ReviewStyle = Literal["peer-review", "regulatory", "calibration", "numerics", "executive"]

def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def _format_findings(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return "No deterministic QA report was provided."
    findings: List[Dict[str, Any]] = report.get("findings", []) or []
    if not findings:
        return "Deterministic QA found no issues."
    lines: List[str] = []
    for f in findings:
        sev = f.get("severity", "INFO")
        code = f.get("code", "UNKNOWN")
        msg = f.get("message", "")
        sug = f.get("suggestion")
        lines.append(f"- [{sev}] {code}: {msg}")
        if sug:
            lines.append(f"  - Suggestion: {sug}")
    return "\n".join(lines)

def _instructions_for_style(style: ReviewStyle) -> str:
    hard_rules = (
        "Hard rules:\n"
        "- Only use information provided in CONFIG SUMMARY + DETERMINISTIC QA FINDINGS.\n"
        "- If something is not specified, write: 'Not specified in config.' Do not assume.\n"
        "- Do NOT claim you ran MODFLOW or saw simulation results.\n"
        "- Avoid vague advice; every recommendation must be actionable.\n"
        "- Keep it concise; prefer bullets over long paragraphs.\n"
    )

    if style == "peer-review":
        return (
            "You are a senior groundwater model reviewer writing an evidence-based peer review memo.\n\n"
            + hard_rules +
            "\nWrite Markdown with these sections:\n"
            "# MODFLOW 6 Review Memo\n"
            "## Executive summary (3–5 sentences)\n"
            "## Evidence-based observations (bullets; cite config fields like grid.nlay, periods.nper, npf.k)\n"
            "## Key risks / red flags (ranked; risk → why → fix)\n"
            "## Data gaps / questions (group: Boundaries, Stresses, Properties, Observations/Calibration)\n"
            "## Recommendations (ordered checklist; include minimum viable calibration inputs if missing)\n"
            "## Suggested next tests (bullets; include water budget, sensitivity, convergence diagnostics)\n"
        )

    if style == "regulatory":
        return (
            "You are a conservative regulatory/SGMA-facing reviewer. Focus on defensibility, documentation, and uncertainty.\n\n"
            + hard_rules +
            "\nWrite Markdown with these sections:\n"
            "# Regulatory / SGMA Readiness Memo\n"
            "## Summary (pass/conditional/fail + why)\n"
            "## Intended use & decision context (state what is and isn't supported by current config)\n"
            "## Boundary condition defensibility (what's specified vs missing)\n"
            "## Water budget / mass balance readiness (what inputs are needed; no results claimed)\n"
            "## Calibration & validation readiness (observations required; acceptance criteria suggestions)\n"
            "## Uncertainty and sensitivity plan (minimum plan)\n"
            "## Documentation checklist (bullets)\n"
        )

    if style == "calibration":
        return (
            "You are a calibration engineer preparing the model for parameter estimation and calibration.\n\n"
            + hard_rules +
            "\nWrite Markdown with these sections:\n"
            "# Calibration Readiness Memo\n"
            "## Executive summary (2–4 sentences)\n"
            "## Observation strategy (what to include; what is missing)\n"
            "## Parameterization critique (K, storage, zones/pilots; identifiability risks)\n"
            "## Stress representation (time discretization; stresses; gaps)\n"
            "## Initial conditions & spin-up (what is specified/missing)\n"
            "## Recommended next actions (ordered)\n"
            "## Suggested calibration tests (bullets; sensitivity, residual checks, water budget, split-sample)\n"
        )

    if style == "numerics":
        return (
            "You are a MODFLOW 6 numerics and stability debugger. Focus on solver/time discretization risks and likely failure modes.\n\n"
            + hard_rules +
            "\nWrite Markdown with these sections:\n"
            "# Numerics & Stability Memo\n"
            "## Executive summary (2–4 sentences)\n"
            "## Time discretization risks (periods.nper, perlen, nstp, tsmult)\n"
            "## Solver considerations (what is specified; what to tune; failure symptoms to watch)\n"
            "## Hydraulic property stability concerns (K contrasts, convertible layers, vertical K)\n"
            "## Common MF6 pitfalls checklist (bullets)\n"
            "## Recommended stability experiments (bullets; step size, solver options, dry cell handling)\n"
        )

    # executive
    return (
        "You are writing for a non-technical stakeholder. Keep jargon minimal and focus on decisions, risks, and next steps.\n\n"
        + hard_rules +
        "\nWrite Markdown with these sections:\n"
        "# Groundwater Model Readiness Brief\n"
        "## What we have now (3 bullets)\n"
        "## Biggest risks (top 3 bullets)\n"
        "## What data we still need (bullets)\n"
        "## Recommended next steps (ordered; include time/cost drivers qualitatively)\n"
    )

def write_llm_review_memo(
    *,
    config_path: str,
    qa_report_path: Optional[str],
    out_md_path: str,
    provider: str = "openai",
    model: Optional[str] = None,
    style: ReviewStyle = "peer-review",
) -> str:
    """Generate a human-friendly model review memo (Markdown) from config + QA report."""
    cfg = _load_json(config_path) or {}
    qa = _load_json(qa_report_path)

    grid = (cfg.get("grid") or {})
    periods = (cfg.get("periods") or {})
    npf = (cfg.get("npf") or {})
    sto = (cfg.get("sto") or {})

    cfg_summary = {
        "model_name": cfg.get("model_name"),
        "workspace": cfg.get("workspace"),
        "units": {"time_units": cfg.get("time_units"), "length_units": cfg.get("length_units")},
        "grid": {
            "nlay": grid.get("nlay"),
            "nrow": grid.get("nrow"),
            "ncol": grid.get("ncol"),
            "delr": grid.get("delr"),
            "delc": grid.get("delc"),
            "top": grid.get("top"),
            "botm_preview": (grid.get("botm") or [])[:10],
        },
        "periods": {
            "nper": periods.get("nper"),
            "perlen_preview": (periods.get("perlen") or [])[:10],
            "nstp_preview": (periods.get("nstp") or [])[:10],
            "steady_preview": (periods.get("steady") or [])[:10],
        },
        "npf": {"k": npf.get("k"), "k33": npf.get("k33"), "icelltype": npf.get("icelltype")},
        "sto": {"sy": sto.get("sy"), "ss": sto.get("ss")},
        "packages_present": sorted([k for k in cfg.keys() if isinstance(cfg.get(k), (dict, list))]),
    }

    findings_text = _format_findings(qa)

    instructions = _instructions_for_style(style)
    user_input = (
        "CONFIG SUMMARY (JSON):\n"
        f"{json.dumps(cfg_summary, indent=2)}\n\n"
        "DETERMINISTIC QA FINDINGS:\n"
        f"{findings_text}\n"
    )

    from gw.llm.registry import get_provider
    prov = get_provider(provider, model=model)
    memo = prov.generate_markdown(instructions=instructions, user_input=user_input)

    outp = Path(out_md_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(memo.strip() + "\n", encoding="utf-8")
    return memo
