"""MODFLOW-2005/NWT simulator-specific knowledge for LLM prompts and tools.

This module provides the MF2005/NWT-specific text fragments that are injected
into the system prompt and tool definitions.
"""

from __future__ import annotations

from gw.simulators.base import FileTypeInfo, PackagePropertyInfo, PackageArrayInfo


# ---------------------------------------------------------------------------
# System prompt fragment
# ---------------------------------------------------------------------------

def mf2005_system_prompt_fragment() -> str:
    """Return the MF2005/NWT-specific knowledge block for the LLM system prompt."""
    return (
        "SIMULATOR: MODFLOW-2005 / MODFLOW-NWT\n"
        "You understand MODFLOW-2005 and MODFLOW-NWT file formats deeply:\n"
        "NAM (free-format namefile), DIS (discretisation), BAS6 (basic package),\n"
        "LPF/BCF6/UPW (layer-property / block-centered flow / upstream weighting),\n"
        "PCG/NWT/SIP/SOR/GMG/DE4 (solvers), OC (output control),\n"
        "WEL, CHD, GHB, RIV, DRN, RCH, EVT, SFR, LAK, MNW2, UZF, SUB, etc.\n\n"

        "KEY DIFFERENCES FROM MODFLOW 6:\n"
        "- Free-format text files (no BEGIN/END blocks)\n"
        "- Name file lists FTYPE UNIT FILENAME (not block-structured)\n"
        "- Grid always DIS (structured), no DISV/DISU unstructured grids\n"
        "- Layer properties in LPF (most common), BCF6 (older), or UPW (NWT)\n"
        "- MODFLOW-NWT uses UPW package (upstream weighting) + NWT solver\n"
        "- Multi-node wells use MNW2 (not MAW as in MF6)\n"
        "- Cell indices are 1-based (layer, row, col) in all input files\n"
        "- Binary output files (.hds, .cbc) use the same format as MF6\n"
        "- Listing file (.lst) has same budget/convergence reporting as MF6\n\n"

        "QA/QC DIAGNOSTICS:\n"
        "When the user asks about model quality, mass balance, dry cells, convergence,\n"
        "pumping data issues, calibration, observations, or any QA/QC analysis, use\n"
        "the run_qa_check tool.\n\n"
        "Available checks:\n"
        "- mass_balance: Parse listing file for volumetric budget, compute % discrepancy\n"
        "  per stress period. Good balance: <0.5%, marginal: 0.5-1%, poor: >1%.\n"
        "- dry_cells: Count cells with dry/inactive heads per layer and timestep.\n"
        "  Reports spatial clusters and trends over time.\n"
        "- convergence: Parse listing file for solver iteration counts, failures,\n"
        "  and solver warnings.\n"
        "- pumping_summary: Analyze WEL package pumping rates by stress period.\n"
        "  Detects anomalous rate jumps (>2x between consecutive periods).\n"
        "- budget_timeseries: Extract IN/OUT per budget term across all timesteps.\n"
        "  Shows temporal trends in the water budget.\n"
        "- head_gradient: Compute cell-to-cell gradients per layer (final timestep).\n"
        "  Flags extreme gradients that may indicate grid resolution issues.\n"
        "- property_check: Check K, Kv/Kh, Ss, Sy ranges for unreasonable values.\n"
        "  Reads from LPF, BCF6, or UPW packages.\n"
        "- listing_budget_detail: Deep listing file parse — per-package IN/OUT budget\n"
        "  tables, solver warnings, dry-cell messages, timestep reductions.\n"
        "- property_zones: Spatial K zone analysis per layer — log-bins K into distinct\n"
        "  value clusters, computes coefficient of variation.\n"
        "- save_snapshot: Capture a lightweight snapshot of current model outputs.\n"
        "- compare_runs: Compare two run snapshots side-by-side.\n\n"

        "QA/QC DOMAIN KNOWLEDGE:\n"
        "- Hydraulic conductivity (K) typical ranges:\n"
        "  * Gravel: 10-1000 m/d, Sand: 0.1-100 m/d, Silt: 1e-4 to 1 m/d\n"
        "  * Clay: 1e-8 to 1e-3 m/d, Sandstone: 1e-4 to 10 m/d\n"
        "- Kv/Kh ratio: typically 0.01-1.0 (vertical always less than horizontal)\n"
        "- Specific storage (Ss): typically 1e-6 to 1e-4 per meter\n"
        "- Specific yield (Sy): sand 0.15-0.30, gravel 0.20-0.35, clay 0.01-0.05\n"
        "- Mass balance: <0.5% discrepancy is good, 0.5-1% needs attention, >1% is problematic\n"
        "- Convergence: PCG/NWT outer iterations >50 suggest difficulty\n"
        "- Dry cells: >10% of a layer dry warrants investigation\n\n"

        "MODFLOW-2005/NWT DOMAIN KNOWLEDGE:\n"
        "- In WEL packages, negative Q = extraction (pumping), positive Q = injection.\n"
        "- LPF uses HK (horizontal K) and VKA (vertical K anisotropy or Kv).\n"
        "- BCF6 uses TRAN (transmissivity), HY (hydraulic conductivity).\n"
        "- UPW is like LPF but with upstream weighting for the NWT solver.\n"
        "- The LAYTYP/LAYCON parameter controls confined (0) vs convertible (>0) layers.\n"
        "- OC (Output Control) uses unit numbers to specify .hds/.cbc output files.\n"
        "- The .hds file contains computed heads per timestep; .cbc contains cell budgets.\n"
        "- The .lst file contains the solver convergence history and water budget.\n"
        "- HOB package provides head observations at specific cells and times.\n\n"

        "TARGETED DATA EXTRACTION:\n"
        "The read_binary_output tool supports targeted extraction:\n"
        "- To get heads at specific cells: use 'cells' parameter with a list of\n"
        "  {layer, row, col} objects (1-based). Returns time-series + per-cell stats.\n"
        "- To get budget flows for a specific component: use 'component' parameter\n"
        "  (e.g. 'WELLS', 'RIVER LEAKAGE', 'STORAGE'). Returns top cells by flow magnitude.\n"
        "Use these for well analysis, boundary condition review, and targeted diagnostics.\n\n"

        "CROSS-ANALYSIS GUIDANCE:\n"
        "When investigating model behavior, combine tools for deeper insight:\n\n"
        "1. PUMPING vs HEAD DECLINE: Run pumping_summary to identify high-rate wells,\n"
        "   then use read_binary_output with cells=[well locations] to see head response.\n\n"
        "2. DRY CELLS vs STRESS CHANGES: Run dry_cells to find problematic areas,\n"
        "   then read the WEL/RCH package files to see if stresses changed.\n\n"
        "3. MASS BALANCE vs BUDGET COMPONENTS: If mass_balance shows poor closure,\n"
        "   run listing_budget_detail to identify which packages contribute most.\n\n"
        "4. CONVERGENCE vs PROPERTY CONTRASTS: If convergence shows high iterations,\n"
        "   run property_zones to check for extreme K contrasts.\n"
    )


# ---------------------------------------------------------------------------
# File-mention regex pattern
# ---------------------------------------------------------------------------

def mf2005_file_mention_regex() -> str:
    """Return the regex alternation for MF2005/NWT-specific file extensions."""
    return (
        "bas|bcf|lpf|upw|pcg|nwt|sip|sor|gmg|de4|"
        "dis|nam|oc|"
        "wel|chd|ghb|riv|drn|rch|evt|sfr|lak|mnw2|uzf|sub|hob|"
        "hds|cbc|lst"
    )


# ---------------------------------------------------------------------------
# Tool description overrides
# ---------------------------------------------------------------------------

def mf2005_tool_description_overrides() -> dict[str, str]:
    """Return MF2005-specific description text for tool definitions."""
    return {
        "read_file": (
            "Read a text file from the model workspace. Returns the file content "
            "(truncated to max_bytes if the file is large). Use this for MF2005/NWT "
            "package files (.dis, .bas, .lpf, .upw, .wel, .pcg, .nwt, etc.), "
            "CSV files, config files, listing files, or any other text-based file."
        ),
        "read_binary_output": (
            "Extract numerical data from a MODFLOW-2005/NWT binary output file "
            "(.hds or .cbc) via FloPy. Returns per-layer statistics (min, max, mean, "
            "median, std) for sampled timesteps, drawdown analysis (for HDS), or "
            "per-component budget breakdowns (for CBC). Binary format is identical "
            "to MODFLOW 6."
        ),
        "run_qa_check": (
            "Run a specialized QA/QC diagnostic check on the MODFLOW-2005/NWT model. "
            "Returns a detailed markdown report with analysis, tables, and recommendations.\n\n"
            "Available checks:\n"
            "- mass_balance: Parse listing file for volumetric budget\n"
            "- dry_cells: Count cells with HDRY per layer per timestep\n"
            "- convergence: Parse listing file for solver iterations and failures\n"
            "- pumping_summary: Analyze WEL package rates by stress period\n"
            "- budget_timeseries: Extract IN/OUT per budget term across timesteps\n"
            "- head_gradient: Compute cell-to-cell gradients per layer\n"
            "- property_check: Check K/SS/SY ranges (reads LPF, BCF6, or UPW)\n"
            "- listing_budget_detail: Per-package IN/OUT tables, solver warnings\n"
            "- property_zones: Spatial K zone analysis per layer\n"
            "- save_snapshot: Save model output snapshot for comparison\n"
            "- compare_runs: Compare two run snapshots\n\n"
            "Use this when the user asks about model quality, mass balance, dry cells, "
            "convergence, pumping data, property distribution, or any QA/QC analysis."
        ),
        "generate_plot": (
            "Generate a plot from the model workspace data. You MUST provide a full Python "
            "script that reads real data from workspace files. NEVER use random/dummy data.\n\n"
            "The script runs in a sandbox with access to matplotlib, numpy, pandas, flopy, "
            "and helper modules (gw_plot_io). Output images are saved to the "
            "plot output directory and returned as URLs.\n\n"
            "REQUIRED helper APIs in scripts:\n"
            "- import gw_plot_io — workspace file access\n"
            "- gw_plot_io.ws_path(rel) — get absolute path to a workspace file\n"
            "- gw_plot_io.out_path('name.png') — get output path for saving plots\n"
            "- gw_plot_io.find_one('.hds') — find a file by extension\n"
            "- Path(os.environ['GW_PLOT_OUTDIR']) — output directory\n\n"
            "COMMON PATTERNS (MODFLOW-2005/NWT):\n"
            "  # Load model:\n"
            "  import flopy\n"
            "  ws = str(gw_plot_io.ws_path('.'))\n"
            "  nam_file = gw_plot_io.find_one('.nam')  # finds the .nam file\n"
            "  m = flopy.modflow.Modflow.load(nam_file, model_ws=ws, check=False)\n\n"
            "  # Read heads from HDS:\n"
            "  hds = flopy.utils.HeadFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.hds'))))\n"
            "  data = hds.get_data(totim=hds.get_times()[-1])  # shape: (nlay, nrow, ncol)\n\n"
            "  # Save plot:\n"
            "  plt.savefig(gw_plot_io.out_path('my_plot.png'), dpi=150, bbox_inches='tight')\n\n"
            "  # K distribution from LPF:\n"
            "  hk = m.lpf.hk.array  # shape: (nlay, nrow, ncol)\n"
            "  # or for UPW (NWT): hk = m.upw.hk.array\n\n"
            "  # Head contour map:\n"
            "  fig, ax = plt.subplots(figsize=(10, 8))\n"
            "  pmv = flopy.plot.PlotMapView(model=m, layer=0, ax=ax)\n"
            "  cb = pmv.plot_array(hds_data[0], cmap='coolwarm')\n"
            "  cs = pmv.contour_array(hds_data[0], levels=10, colors='black', linewidths=0.5)\n"
            "  plt.clabel(cs, inline=True, fontsize=8)\n\n"
            "The tool returns image URLs in an 'images' array. Embed them in your response "
            "as markdown: ![Description](url)"
        ),
    }


# ---------------------------------------------------------------------------
# File-type knowledge base
# ---------------------------------------------------------------------------

_EXT_KB: dict[str, FileTypeInfo] = {
    ".nam": FileTypeInfo(
        kind="namefile",
        purpose="Master name file listing all packages, unit numbers, and filenames",
        what_to_look_for="FTYPE UNIT FNAME entries; which packages are active",
    ),
    ".dis": FileTypeInfo(
        kind="discretisation",
        purpose="Grid discretisation: NLAY, NROW, NCOL, NPER, cell sizes, top/bottom",
        what_to_look_for="Grid dimensions in header; DELR, DELC, TOP, BOTM arrays; stress period info at end",
    ),
    ".bas": FileTypeInfo(
        kind="basic",
        purpose="Basic package: IBOUND array, starting heads (STRT), dry cell head value (HDRY)",
        what_to_look_for="IBOUND values (-1=constant head, 0=inactive, 1=active); STRT (initial head)",
    ),
    ".lpf": FileTypeInfo(
        kind="flow",
        purpose="Layer-Property Flow: HK, VKA, Ss, Sy, LAYTYP (confined/convertible)",
        what_to_look_for="HK arrays per layer; VKA (vertical K or anisotropy); Ss/Sy for transient; LAYTYP flags",
    ),
    ".bcf": FileTypeInfo(
        kind="flow",
        purpose="Block-Centered Flow: TRAN or HY, Vcont, Sf1, Sf2, LAYCON",
        what_to_look_for="Transmissivity or K arrays; LAYCON (layer type); specific storage/yield",
    ),
    ".upw": FileTypeInfo(
        kind="flow",
        purpose="Upstream Weighting: HK, VKA, Ss, Sy (NWT-specific, like LPF with upstream weighting)",
        what_to_look_for="HK arrays per layer; VKA; Ss/Sy; same layout as LPF",
    ),
    ".pcg": FileTypeInfo(
        kind="solver",
        purpose="Preconditioned Conjugate-Gradient solver settings",
        what_to_look_for="MXITER, ITER1 (max outer/inner iterations); HCLOSE, RCLOSE (closure criteria)",
    ),
    ".nwt": FileTypeInfo(
        kind="solver",
        purpose="Newton solver settings (MODFLOW-NWT)",
        what_to_look_for="HEADTOL, FLUXTOL (convergence criteria); MAXITEROUT; solver options",
    ),
    ".oc": FileTypeInfo(
        kind="output_control",
        purpose="Output Control: specifies when to save heads and budgets",
        what_to_look_for="SAVE HEAD / SAVE BUDGET directives; output unit numbers",
    ),
    ".wel": FileTypeInfo(
        kind="stress",
        purpose="Well package: pumping/injection rates per stress period",
        what_to_look_for="Number of wells per period; Layer Row Col Q per entry",
    ),
    ".chd": FileTypeInfo(
        kind="stress",
        purpose="Constant-Head package: fixed-head boundaries",
        what_to_look_for="Layer Row Col Shead Ehead entries per stress period",
    ),
    ".ghb": FileTypeInfo(
        kind="stress",
        purpose="General Head Boundary: head-dependent flux boundaries",
        what_to_look_for="Layer Row Col Bhead Conductance entries per stress period",
    ),
    ".riv": FileTypeInfo(
        kind="stress",
        purpose="River package: river-aquifer interaction",
        what_to_look_for="Layer Row Col Stage Cond Rbot entries per stress period",
    ),
    ".drn": FileTypeInfo(
        kind="stress",
        purpose="Drain package: groundwater drains",
        what_to_look_for="Layer Row Col Elev Cond entries per stress period",
    ),
    ".rch": FileTypeInfo(
        kind="stress",
        purpose="Recharge package: areal recharge rates",
        what_to_look_for="NRCHOP (option); recharge array per stress period",
    ),
    ".evt": FileTypeInfo(
        kind="stress",
        purpose="Evapotranspiration package: ET rates and extinction depth",
        what_to_look_for="NEVTOP; SURF, EVTR, EXDP, IEVT arrays per period",
    ),
    ".sfr": FileTypeInfo(
        kind="advanced",
        purpose="Streamflow-Routing: surface water network",
        what_to_look_for="NSTRM, NSS (reaches, segments); reach/segment data; segment flow",
    ),
    ".lak": FileTypeInfo(
        kind="advanced",
        purpose="Lake package: lake-aquifer interaction",
        what_to_look_for="NLAKES, NSSITR; lake bed conductance; stage constraints",
    ),
    ".mnw2": FileTypeInfo(
        kind="advanced",
        purpose="Multi-Node Well 2: multi-screened wells spanning layers",
        what_to_look_for="MNWMAX; well IDs; screen intervals; pumping rates",
    ),
    ".uzf": FileTypeInfo(
        kind="advanced",
        purpose="Unsaturated Zone Flow: vadose zone simulation",
        what_to_look_for="NUZTOP, IUZFOPT; infiltration, ET, VKS arrays",
    ),
    ".sub": FileTypeInfo(
        kind="advanced",
        purpose="Subsidence package: land subsidence from groundwater withdrawal",
        what_to_look_for="Interbed properties; preconsolidation head; compressibility",
    ),
    ".hob": FileTypeInfo(
        kind="observation",
        purpose="Head Observation: comparison targets at specific cells/times",
        what_to_look_for="Observation names; Layer Row Col; time offsets; observed head values",
    ),
    ".hds": FileTypeInfo(
        kind="binary_output",
        purpose="Binary head output file — computed heads for each saved timestep",
        what_to_look_for="Binary data — use read_binary_output tool to extract statistics",
    ),
    ".cbc": FileTypeInfo(
        kind="binary_output",
        purpose="Binary cell-by-cell budget file — flow terms per cell",
        what_to_look_for="Binary data — use read_binary_output tool to extract budget breakdowns",
    ),
    ".lst": FileTypeInfo(
        kind="listing",
        purpose="Listing file — solver convergence, water budget, warnings",
        what_to_look_for="Volumetric budget tables; convergence history; dry cell warnings",
    ),
}


# ---------------------------------------------------------------------------
# Package property definitions
# ---------------------------------------------------------------------------

PACKAGE_PROPERTIES: dict[str, PackagePropertyInfo] = {
    "LPF": PackagePropertyInfo(
        file_ext=".lpf",
        block="GRIDDATA",
        arrays={
            "HK": PackageArrayInfo(label="Horizontal hydraulic conductivity", per_layer=True),
            "VKA": PackageArrayInfo(label="Vertical K or anisotropy ratio", per_layer=True),
            "SS": PackageArrayInfo(label="Specific storage", per_layer=True),
            "SY": PackageArrayInfo(label="Specific yield", per_layer=True),
            "HANI": PackageArrayInfo(label="Horizontal anisotropy", per_layer=True),
            "VKCB": PackageArrayInfo(label="Vertical K of confining bed", per_layer=True),
        },
    ),
    "UPW": PackagePropertyInfo(
        file_ext=".upw",
        block="GRIDDATA",
        arrays={
            "HK": PackageArrayInfo(label="Horizontal hydraulic conductivity", per_layer=True),
            "VKA": PackageArrayInfo(label="Vertical K or anisotropy ratio", per_layer=True),
            "SS": PackageArrayInfo(label="Specific storage", per_layer=True),
            "SY": PackageArrayInfo(label="Specific yield", per_layer=True),
            "HANI": PackageArrayInfo(label="Horizontal anisotropy", per_layer=True),
        },
    ),
    "BCF6": PackagePropertyInfo(
        file_ext=".bcf",
        block="GRIDDATA",
        arrays={
            "TRAN": PackageArrayInfo(label="Transmissivity", per_layer=True),
            "HY": PackageArrayInfo(label="Hydraulic conductivity (HY)", per_layer=True),
            "VCONT": PackageArrayInfo(label="Vertical conductance", per_layer=True),
            "SF1": PackageArrayInfo(label="Primary storage coefficient", per_layer=True),
            "SF2": PackageArrayInfo(label="Secondary storage coefficient", per_layer=True),
        },
    ),
    "BAS6": PackagePropertyInfo(
        file_ext=".bas",
        block="DATA",
        arrays={
            "IBOUND": PackageArrayInfo(label="Boundary array (active/inactive/CHD)", per_layer=True),
            "STRT": PackageArrayInfo(label="Starting heads", per_layer=True),
        },
    ),
}
