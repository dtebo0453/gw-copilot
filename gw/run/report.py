import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def write_run_report(report: Dict[str, Any], outdir: str) -> None:
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# MODFLOW 6 Run Report",
        "",
        f"**Status:** {report.get('status')}",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        f"- Return code: {report.get('return_code')}",
        f"- Workspace: `{report.get('workspace')}`",
        f"- Listing: `{(report.get('listing') or {}).get('path')}`",
        f"- Normal termination: **{(report.get('listing') or {}).get('normal_termination')}**",
        f"- Convergence failure detected: **{(report.get('listing') or {}).get('convergence_failure')}**",
        f"- Percent discrepancy: `{(report.get('listing') or {}).get('percent_discrepancy')}`",
        "",
        "## Notes",
    ]

    notes = report.get("notes") or []
    listing_notes = (report.get("listing") or {}).get("notes") or []
    if not notes and not listing_notes:
        lines.append("- (none)")
    else:
        for n in notes:
            lines.append(f"- {n}")
        for n in listing_notes:
            lines.append(f"- {n}")

    (outp / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
