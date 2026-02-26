import re
from pathlib import Path
from typing import Dict, Any, Optional

_NORMAL_TERMINATION_RE = re.compile(r"normal termination", re.IGNORECASE)
_FAILED_RE = re.compile(r"error|failed|abnormal termination", re.IGNORECASE)
_PCT_DISC_RE = re.compile(r"percent\s+discrepancy\s*=\s*([+-]?[0-9]*\.?[0-9]+)", re.IGNORECASE)
_CONV_FAIL_RE = re.compile(r"failed\s+to\s+converge|did\s+not\s+converge|convergence\s+failure", re.IGNORECASE)

def parse_listing(listing_path: str) -> Dict[str, Any]:
    p = Path(listing_path)
    if not p.exists():
        return {
            "exists": False,
            "path": listing_path,
            "normal_termination": False,
            "convergence_failure": None,
            "percent_discrepancy": None,
            "failed_language": None,
            "notes": [f"Listing file not found: {listing_path}"],
        }

    text = p.read_text(encoding="utf-8", errors="ignore")
    normal = bool(_NORMAL_TERMINATION_RE.search(text))
    conv_fail = bool(_CONV_FAIL_RE.search(text))
    pct = None
    matches = _PCT_DISC_RE.findall(text)
    if matches:
        try:
            pct = float(matches[-1])
        except Exception:
            pct = None
    failed = bool(_FAILED_RE.search(text)) and not normal

    notes = []
    if failed:
        notes.append("Listing contains error/failure language and did not show normal termination.")
    if conv_fail:
        notes.append("Listing indicates a convergence failure.")
    if pct is not None:
        notes.append(f"Detected percent discrepancy value: {pct}")
    if normal:
        notes.append("Detected normal termination in listing.")

    return {
        "exists": True,
        "path": str(p),
        "normal_termination": normal,
        "convergence_failure": conv_fail,
        "percent_discrepancy": pct,
        "failed_language": failed,
        "notes": notes,
    }

def default_listing_path(workspace: str) -> Optional[str]:
    w = Path(workspace)
    cand = w / "mfsim.lst"
    if cand.exists():
        return str(cand)
    lsts = sorted(w.glob("*.lst"), key=lambda x: x.stat().st_mtime, reverse=True)
    if lsts:
        return str(lsts[0])
    return None
