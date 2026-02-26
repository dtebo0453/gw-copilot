from typing import Dict, Any, List
from pathlib import Path
import csv

def check_grid_and_idomain(cfg: Dict[str, Any], inputs_dir: str) -> List[Dict[str, Any]]:
    findings = []
    if not inputs_dir:
        return findings

    idomain_path = Path(inputs_dir) / "idomain.csv"
    if not idomain_path.exists():
        return findings

    with idomain_path.open() as f:
        reader = list(csv.reader(f))

    nrow = len(reader)
    ncol = len(reader[0]) if reader else 0

    grid = cfg.get("grid", {})
    if nrow != grid.get("nrow") or ncol != grid.get("ncol"):
        findings.append({
            "severity": "ERROR",
            "code": "IDOMAIN_DIM_MISMATCH",
            "message": "idomain dimensions do not match grid.",
            "details": {"idomain": (nrow, ncol), "grid": (grid.get("nrow"), grid.get("ncol"))},
            "suggestion": "Rebuild idomain from AOI or adjust grid."
        })

    active = sum(1 for row in reader for v in row if int(v) != 0)
    frac = active / (nrow * ncol) if nrow and ncol else 0
    if frac < 0.01:
        findings.append({
            "severity": "WARN",
            "code": "LOW_ACTIVE_FRACTION",
            "message": "Very low active cell fraction.",
            "details": {"active_fraction": frac},
            "suggestion": "Check AOI extent and grid resolution."
        })
    return findings