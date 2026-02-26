from typing import Dict, Any, List

def check_hydro_params(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    npf = cfg.get("npf", {})
    k = npf.get("k")

    if k is not None and k <= 0:
        findings.append({
            "severity": "ERROR",
            "code": "K_NONPOSITIVE",
            "message": "Hydraulic conductivity must be positive.",
            "details": {"k": k},
            "suggestion": "Set k > 0."
        })
    elif k is not None and k > 1000:
        findings.append({
            "severity": "WARN",
            "code": "K_HIGH",
            "message": "Hydraulic conductivity is unusually high.",
            "details": {"k": k},
            "suggestion": "Verify units and magnitude."
        })

    sto = cfg.get("sto", {})
    sy = sto.get("sy")
    if sy is not None and not (0 <= sy <= 0.35):
        findings.append({
            "severity": "WARN",
            "code": "SY_OUT_OF_RANGE",
            "message": "Specific yield outside typical range.",
            "details": {"sy": sy},
            "suggestion": "Typical sy range is 0–0.35."
        })
    return findings