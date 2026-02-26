from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import re

MF6_EXPECTED_FOR_TRANSIENT = ["dis", "ic", "npf", "sto", "oc", "ims", "tdis"]

@dataclass
class CritiqueResult:
    modeldir: str
    findings: List[str]
    recommendations: List[str]
    detected: Dict[str, List[str]]
    notes: List[str]

def _scan_files(modeldir: str) -> List[str]:
    p = Path(modeldir)
    if not p.exists():
        raise FileNotFoundError(modeldir)
    return [f.name.lower() for f in p.glob("**/*") if f.is_file()]

def _detect_packages(files: List[str]) -> Dict[str, List[str]]:
    detected: Dict[str, List[str]] = {}
    patterns = {
        "dis": re.compile(r"\.dis\b|\bdis\b"),
        "ic": re.compile(r"\.ic\b|\bic\b"),
        "npf": re.compile(r"\.npf\b|\bnpf\b"),
        "sto": re.compile(r"\.sto\b|\bsto\b"),
        "oc": re.compile(r"\.oc\b|\boc\b"),
        "tdis": re.compile(r"\.tdis\b|tdis"),
        "ims": re.compile(r"\.ims\b|ims"),
        "rch": re.compile(r"\.rch\b|\.rcha\b|rcha|\brch\b"),
        "wel": re.compile(r"\.wel\b|\bwel\b"),
        "ghb": re.compile(r"\.ghb\b|\bghb\b"),
        "drn": re.compile(r"\.drn\b|\bdrn\b"),
        "riv": re.compile(r"\.riv\b|\briv\b"),
        "obs": re.compile(r"\.obs\b|obs"),
    }
    for f in files:
        for pkg, pat in patterns.items():
            if pat.search(f):
                detected.setdefault(pkg, []).append(f)
    return detected

def critique_model(modeldir: str) -> CritiqueResult:
    files = _scan_files(modeldir)
    detected = _detect_packages(files)
    findings: List[str] = []; recs: List[str] = []; notes: List[str] = []

    if not any("mfsim.nam" in f for f in files):
        findings.append("No mfsim.nam found; folder may not be a complete MF6 simulation workspace.")
        recs.append("Point to the MF6 simulation workspace directory containing mfsim.nam.")

    missing_core = [p for p in MF6_EXPECTED_FOR_TRANSIENT if p not in detected]
    if missing_core:
        findings.append(f"Missing core components (heuristic): {', '.join(missing_core)}")
        recs.append("Confirm DIS/IC/NPF/STO/TDIS/IMS/OC are present and referenced in name files.")

    if ("tdis" in detected) and ("sto" not in detected):
        findings.append("TDIS detected but STO not detected; transient model without storage is unusual.")
        recs.append("If transient, add STO (Ss/Sy) or document why storage is not represented.")

    if all(p not in detected for p in ["wel","rch","riv","ghb","drn"]):
        findings.append("No common stress/boundary packages detected (WEL/RCH/RIV/GHB/DRN). Model may be overly simplified.")
        recs.append("Include stresses/boundaries appropriate to model purpose and document assumptions.")

    if "obs" not in detected:
        findings.append("No observations detected. Calibration/validation may be limited.")
        recs.append("Add head/flow observations (UTL-OBS) and maintain a clear list of calibration targets.")

    if not any(f.endswith(".lst") for f in files):
        notes.append("No .lst listing file detected; keep .lst for convergence diagnostics when running MF6.")

    return CritiqueResult(modeldir=modeldir, findings=findings, recommendations=recs, detected=detected, notes=notes)

def write_report(result: CritiqueResult, out_path: str) -> None:
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# MF6 Model Critique\n", f"**Model directory:** `{result.modeldir}`\n",
             "## Detected components (heuristic)\n"]
    if result.detected:
        for k in sorted(result.detected.keys()):
            lines.append(f"- **{k.upper()}**: {len(result.detected[k])} file(s)")
    else:
        lines.append("- (none detected)")
    lines.append("\n## Findings\n")
    lines += [f"- {f}" for f in result.findings] if result.findings else ["- No major issues detected by heuristic checks."]
    lines.append("\n## Recommendations\n")
    lines += [f"- {r}" for r in result.recommendations] if result.recommendations else ["- No recommendations."]
    if result.notes:
        lines.append("\n## Notes\n")
        lines += [f"- {n}" for n in result.notes]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
