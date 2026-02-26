"""Lightweight, offline MODFLOW 6 file/type knowledge.

This module provides a small, dependency-free knowledge base used as a fallback
when the user hasn't added official MF6 documentation into a docs folder for
local retrieval.

It is deliberately conservative and avoids overly specific claims. The chat
agent should prefer:
  1) the actual file contents,
  2) retrieved documentation snippets (if available),
and only then use this as a scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FileTypeInfo:
    kind: str
    purpose: str
    what_to_look_for: str


_EXT_KB: Dict[str, FileTypeInfo] = {
    ".nam": FileTypeInfo(
        kind="Name file",
        purpose=(
            "Lists and labels the input files used by a MODFLOW 6 simulation or model. "
            "Name files tell MF6 which packages to load and where to find their input."
        ),
        what_to_look_for=(
            "File records that map package types (e.g., DIS, NPF, CHD, OC) to filenames; "
            "comments about how the model was generated; paths to package files."
        ),
    ),
    ".dis": FileTypeInfo(
        kind="Structured discretization (DIS)",
        purpose=(
            "Defines the structured grid geometry (rows/cols/layers) and cell elevations. "
            "Provides TOP/BOTM and optionally IDOMAIN and other grid data." 
        ),
        what_to_look_for=(
            "DIMENSIONS, GRIDDATA, and how arrays are specified (CONSTANT/INTERNAL/OPEN/CLOSE); "
            "TOP/BOTM values; IDOMAIN active/inactive cells."
        ),
    ),
    ".disv": FileTypeInfo(
        kind="Vertex discretization (DISV)",
        purpose=(
            "Defines an unstructured 2D vertex-based grid with layers. Used for irregular polygons "
            "or refined meshes while retaining layered structure."
        ),
        what_to_look_for=(
            "VERTICES, CELL2D, TOP/BOTM, and IDOMAIN; how cells reference vertices."
        ),
    ),
    ".disu": FileTypeInfo(
        kind="Unstructured discretization (DISU)",
        purpose=(
            "Defines a fully unstructured 3D grid with connectivity. Used for general unstructured meshes."
        ),
        what_to_look_for=(
            "NODES, NJA, CONNECTIONDATA/IA/JA style connectivity, TOP/BOTM, IDOMAIN."
        ),
    ),
    ".chd": FileTypeInfo(
        kind="Constant-Head (CHD) package",
        purpose=(
            "Applies specified head boundary conditions at selected cells. "
            "CHD cells act like fixed-head boundaries for the stress periods where they are active." 
        ),
        what_to_look_for=(
            "OPTIONS (e.g., auxiliary variables), DIMENSIONS (like MAXBOUND), and PERIOD blocks "
            "listing (layer,row,col) or cellid plus head values (and optional auxiliary data)."
        ),
    ),
    ".wel": FileTypeInfo(
        kind="Well (WEL) package",
        purpose="Applies pumping/injection rates at selected cells per stress period.",
        what_to_look_for="PERIOD blocks with cellid plus Q (and optional auxiliaries).",
    ),
    ".drn": FileTypeInfo(
        kind="Drain (DRN) package",
        purpose="Represents drains that remove water when head is above a specified elevation.",
        what_to_look_for="Cell locations plus drain elevation and conductance in each stress period.",
    ),
    ".riv": FileTypeInfo(
        kind="River (RIV) package",
        purpose="Represents river boundary with stage, conductance, and bottom elevation.",
        what_to_look_for="Stage/cond/bottom fields per boundary cell per stress period.",
    ),
    ".ghb": FileTypeInfo(
        kind="General-Head Boundary (GHB) package",
        purpose="Represents a head-dependent flux boundary with specified boundary head and conductance.",
        what_to_look_for="Boundary head and conductance fields per cell per stress period.",
    ),
    ".rcha": FileTypeInfo(
        kind="Recharge (RCHA) package",
        purpose="Applies areal recharge to cells, commonly on the top layer.",
        what_to_look_for="Recharge array(s) per stress period; how recharge is mapped to cells.",
    ),
    ".npf": FileTypeInfo(
        kind="Node Property Flow (NPF) package",
        purpose="Defines hydraulic properties and flow options (K, anisotropy, cell types, etc.).",
        what_to_look_for="HK/HANI/VK or K33 arrays, icelltype, and any options controlling flow formulation.",
    ),
    ".ic": FileTypeInfo(
        kind="Initial Conditions (IC) package",
        purpose="Defines starting heads (and sometimes other initial state variables) for the simulation.",
        what_to_look_for="STRT array definition and how it is specified.",
    ),
    ".sto": FileTypeInfo(
        kind="Storage (STO) package",
        purpose="Defines storage properties (Ss/Sy) and transient/steady settings by stress period.",
        what_to_look_for="SS/SY arrays, steady-state/transient flags, and period settings.",
    ),
    ".oc": FileTypeInfo(
        kind="Output Control (OC) package",
        purpose="Controls which outputs are written (heads, budgets) and when.",
        what_to_look_for="SAVE/PRINT instructions by stress period; output file names and formats.",
    ),
    ".lst": FileTypeInfo(
        kind="Listing file",
        purpose="Text log output from MODFLOW 6 containing run messages, warnings, and summaries.",
        what_to_look_for="Warnings/errors, package summaries, convergence info, and timing details.",
    ),
    ".hds": FileTypeInfo(
        kind="Binary heads output",
        purpose="Binary output containing heads by time step/stress period.",
        what_to_look_for="Use a reader (FloPy or MF6 utilities) to interpret times and arrays.",
    ),
    ".cbc": FileTypeInfo(
        kind="Cell-by-cell budget output",
        purpose="Binary output containing flow terms by cell and time step.",
        what_to_look_for="Use a reader to extract budget terms (e.g., CHD, WEL, STORAGE, etc.).",
    ),
}


def guess_filetype(path: str) -> Optional[FileTypeInfo]:
    """Return FileTypeInfo based on file extension, if known."""
    p = path.lower().strip()
    for ext, info in _EXT_KB.items():
        if p.endswith(ext):
            return info
    return None


# ---------------------------------------------------------------------------
# Package property knowledge: maps package types to their GRIDDATA arrays
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PackageArrayInfo:
    label: str
    per_layer: bool = True


@dataclass(frozen=True)
class PackagePropertyInfo:
    file_ext: str
    block: str
    arrays: Dict[str, PackageArrayInfo]


PACKAGE_PROPERTIES: Dict[str, PackagePropertyInfo] = {
    "NPF": PackagePropertyInfo(
        file_ext=".npf",
        block="GRIDDATA",
        arrays={
            "k": PackageArrayInfo(label="Hydraulic Conductivity (K)"),
            "k22": PackageArrayInfo(label="Horizontal Anisotropy (K22)"),
            "k33": PackageArrayInfo(label="Vertical Conductivity (K33)"),
            "icelltype": PackageArrayInfo(label="Cell Type"),
        },
    ),
    "STO": PackagePropertyInfo(
        file_ext=".sto",
        block="GRIDDATA",
        arrays={
            "ss": PackageArrayInfo(label="Specific Storage (Ss)"),
            "sy": PackageArrayInfo(label="Specific Yield (Sy)"),
            "iconvert": PackageArrayInfo(label="Convertibility Flag"),
        },
    ),
    "IC": PackagePropertyInfo(
        file_ext=".ic",
        block="GRIDDATA",
        arrays={
            "strt": PackageArrayInfo(label="Starting Head (STRT)"),
        },
    ),
    "RCH": PackagePropertyInfo(
        file_ext=".rch",
        block="PERIOD",
        arrays={
            "recharge": PackageArrayInfo(label="Recharge Rate", per_layer=False),
        },
    ),
}

# Built-in DIS properties (not from package files)
DIS_PROPERTIES: Dict[str, str] = {
    "top": "Top Elevation",
    "botm": "Bottom Elevation",
    "idomain": "IDOMAIN",
}


def property_to_package(prop_key: str) -> Optional[Tuple[str, str, str]]:
    """Resolve a property key to its package.

    Returns (package_type, array_name, label) or None if the key is a DIS
    property or unknown.
    """
    k = (prop_key or "").strip().lower()
    for pkg_type, pkg_info in PACKAGE_PROPERTIES.items():
        for arr_name, arr_info in pkg_info.arrays.items():
            if arr_name == k:
                return (pkg_type, arr_name, arr_info.label)
    return None


def get_all_property_keys() -> List[Dict[str, str]]:
    """Return a flat list of all known property descriptors (DIS + packages).

    Each entry: {"key": ..., "label": ..., "source": ...}
    """
    out: List[Dict[str, str]] = []
    for key, label in DIS_PROPERTIES.items():
        out.append({"key": key, "label": label, "source": "DIS"})
    for pkg_type, pkg_info in PACKAGE_PROPERTIES.items():
        for arr_name, arr_info in pkg_info.arrays.items():
            out.append({"key": arr_name, "label": arr_info.label, "source": pkg_type})
    return out


def package_property_summary() -> str:
    """Return a concise human-readable summary of package→property mappings.

    Used to inject into LLM system prompts so the model knows which files
    contain which properties without needing to ask the user.
    """
    lines: List[str] = []
    for pkg_type, pkg_info in PACKAGE_PROPERTIES.items():
        arr_labels = [f"{name} ({info.label})" for name, info in pkg_info.arrays.items()]
        lines.append(f"- {pkg_type} ({pkg_info.file_ext}): {', '.join(arr_labels)}")
    return "\n".join(lines)
