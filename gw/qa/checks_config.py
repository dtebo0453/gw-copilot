from typing import Dict, Any, List

def check_config_structure(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    grid = cfg.get("grid", {})
    nlay = grid.get("nlay")
    botm = grid.get("botm")

    if nlay is None or botm is None:
        findings.append({
            "severity": "ERROR",
            "code": "GRID_MISSING",
            "message": "Grid definition missing nlay or botm.",
            "details": {},
            "suggestion": "Define grid.nlay and grid.botm."
        })
    elif len(botm) != nlay:
        findings.append({
            "severity": "ERROR",
            "code": "BOTM_LENGTH_MISMATCH",
            "message": "Length of botm does not match nlay.",
            "details": {"nlay": nlay, "len(botm)": len(botm)},
            "suggestion": "Ensure one bottom elevation per layer."
        })

    top = grid.get("top")
    if top is not None and botm:
        if top <= botm[0]:
            findings.append({
                "severity": "WARN",
                "code": "TOP_BELOW_BOTM",
                "message": "Model top is not above first layer bottom.",
                "details": {"top": top, "botm0": botm[0]},
                "suggestion": "Check layer elevations."
            })
    return findings