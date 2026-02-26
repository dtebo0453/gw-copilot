import json
from pathlib import Path
from typing import Dict, Any, List

from .checks_config import check_config_structure
from .checks_grid import check_grid_and_idomain
from .checks_hydro import check_hydro_params
from .report import write_reports

def run_qa(config_path: str, inputs_dir: str = None, outdir: str = None) -> Dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    findings: List[Dict[str, Any]] = []

    findings.extend(check_config_structure(config))
    findings.extend(check_hydro_params(config))
    findings.extend(check_grid_and_idomain(config, inputs_dir))

    report = {
        "summary": {
            "errors": sum(1 for f in findings if f["severity"] == "ERROR"),
            "warnings": sum(1 for f in findings if f["severity"] == "WARN"),
            "info": sum(1 for f in findings if f["severity"] == "INFO"),
        },
        "findings": findings,
    }

    write_reports(report, outdir)
    return report