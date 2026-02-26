import json
from pathlib import Path
from datetime import datetime

def write_reports(report: dict, outdir: str):
    if not outdir:
        return
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)

    # JSON output (machine readable)
    (outp / "qa_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    errors = report["summary"]["errors"]
    warns = report["summary"]["warnings"]
    info = report["summary"]["info"]
    findings = report.get("findings", [])

    status = "PASS" if errors == 0 else "FAIL"

    # Group findings
    by_sev = {"ERROR": [], "WARN": [], "INFO": []}
    for f in findings:
        by_sev.get(f.get("severity", "INFO"), by_sev["INFO"]).append(f)

    # Recommendations = anything that has a suggestion
    recs = []
    for f in findings:
        sug = f.get("suggestion")
        if sug:
            recs.append(f"{f['code']}: {sug}")

    checked = [
        "Grid/config integrity (nlay vs botm, top vs botm)",
        "Hydraulic parameter sanity (K, Sy)",
        "AOI/idomain consistency (dimensions, active fraction) when idomain.csv is present",
    ]

    lines = [
        "# Model QA Report",
        "",
        f"**Status:** {status}",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Errors: **{errors}**",
        f"- Warnings: **{warns}**",
        f"- Info: **{info}**",
        "",
        "## What this QA checked",
        *[f"- {c}" for c in checked],
        "",
    ]

    if errors == 0 and warns == 0 and info == 0:
        lines += [
            "## Summary",
            "No issues were detected by the automated checks in this QA pass.",
            "",
            "## Next suggested steps",
            "- Add boundary/stress packages appropriate for your conceptual model (e.g., recharge, pumping, GHB/CHD).",
            "- Add observation targets (heads/flows) and calibration metrics if this is a predictive model.",
            "",
        ]

    def add_section(title: str, items: list):
        if not items:
            return
        lines.append(f"## {title}")
        lines.append("")
        for f in items:
            lines.append(f"### {f['code']}")
            lines.append(f"- Severity: **{f['severity']}**")
            lines.append(f"- Message: {f['message']}")
            if f.get("details"):
                lines.append(f"- Details: `{json.dumps(f['details'])}`")
            if f.get("suggestion"):
                lines.append(f"- Suggestion: {f['suggestion']}")
            lines.append("")

    add_section("Errors", by_sev["ERROR"])
    add_section("Warnings", by_sev["WARN"])
    add_section("Info", by_sev["INFO"])

    if recs:
        lines += [
            "## Recommendations",
            "",
            *[f"- {r}" for r in recs],
            "",
        ]

    (outp / "qa_report.md").write_text("\n".join(lines), encoding="utf-8")
