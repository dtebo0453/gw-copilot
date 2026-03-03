"""MT3DMS / MT3D-USGS simulator-specific knowledge for LLM prompts and tools.

This module provides the MT3D-specific text fragments that are injected
into the system prompt and tool definitions.  It covers solute-transport
concepts, package types, QA thresholds, and file-type metadata.
"""

from __future__ import annotations

from gw.simulators.base import FileTypeInfo, PackagePropertyInfo, PackageArrayInfo


# ---------------------------------------------------------------------------
# System prompt fragment
# ---------------------------------------------------------------------------

def mt3dms_system_prompt_fragment() -> str:
    """Return the MT3DMS/MT3D-USGS-specific knowledge block for the LLM system prompt."""
    return (
        "SIMULATOR: MT3DMS / MT3D-USGS (Solute Transport)\n"
        "You understand MT3DMS and MT3D-USGS file formats deeply:\n"
        "BTN (Basic Transport), ADV (Advection), DSP (Dispersion),\n"
        "SSM (Sink/Source Mixing), RCT (Chemical Reaction),\n"
        "GCG (Generalized Conjugate Gradient solver), TOB (Transport Observation),\n"
        "FTL (Flow-Transport Link — connects to a MODFLOW flow model).\n\n"

        "TRANSPORT CONCEPTS:\n"
        "- MT3DMS simulates advection, dispersion, and chemical reactions of solutes\n"
        "  in groundwater using the flow field from a MODFLOW model.\n"
        "- The flow solution is passed via FTL (Flow-Transport Link) file.\n"
        "- Multi-species transport is supported: each species gets its own UCN output.\n"
        "- Advection methods: upstream FD (MIXELM=0), MOC (1), MMOC (2), HMOC (3),\n"
        "  TVD/ULTIMATE (>0 with specific flags).\n"
        "- Dispersion includes longitudinal, transverse horizontal, and transverse\n"
        "  vertical dispersivities plus molecular diffusion.\n"
        "- Reactions: linear sorption, Freundlich, Langmuir, first-order decay,\n"
        "  zero-order production.\n\n"

        "KEY PACKAGES:\n"
        "- BTN: Grid overlay (NLAY, NROW, NCOL, NPER from flow model), porosity\n"
        "  (PRSITY), ICBUND array, initial concentrations (SCONC), time stepping\n"
        "  (DT0, MXSTRN, TTSMULT, TTSMAX), output control.\n"
        "- ADV: Advection solution method (MIXELM), particle tracking parameters.\n"
        "- DSP: Dispersivity arrays (AL — longitudinal, TRPT — TH/TL ratio,\n"
        "  TRPV — TV/TL ratio), molecular diffusion (DMCOEF).\n"
        "- SSM: Links transport to MODFLOW stress packages (WEL, RIV, GHB, etc.).\n"
        "  Source/sink concentrations are specified here.\n"
        "- RCT: Chemical reaction parameters — sorption type (ISOTHM),\n"
        "  bulk density (RHOB), distribution coefficient (SP1), decay rates.\n"
        "- GCG: Iterative solver — MXITER, ITER1, ISOLVE, CCLOSE.\n"
        "- TOB: Transport observations — concentration targets at specific\n"
        "  cells/times for calibration.\n\n"

        "MT3D-USGS EXTENSIONS:\n"
        "- SFT: Streamflow Transport — solute transport in SFR streams.\n"
        "- LKT: Lake Transport — solute transport in LAK packages.\n"
        "- UZT: Unsaturated Zone Transport — solute in UZF packages.\n"
        "- CTS: Contaminant Treatment System — pump-and-treat simulation.\n"
        "- Enhanced handling of dry cells and rewetting in transport.\n\n"

        "OUTPUT FILES:\n"
        "- MT3D001.UCN, MT3D002.UCN, ... — concentration output per species.\n"
        "  Binary format readable by FloPy's UcnFile.\n"
        "- .mas — mass balance summary file with total mass in/out per timestep.\n"
        "- .lst — listing file with solver convergence and transport budget.\n"
        "- .obs — observation output (if TOB is active).\n\n"

        "QA/QC DIAGNOSTICS:\n"
        "When the user asks about transport model quality, mass balance,\n"
        "negative concentrations, Courant/Peclet numbers, solver convergence,\n"
        "or any QA/QC analysis, use the run_qa_check tool.\n\n"
        "Available checks:\n"
        "- transport_mass_balance: Parse .mas file or listing file for transport\n"
        "  mass balance. Good: <0.5%, marginal: 0.5-1%, poor: >1%.\n"
        "- negative_concentrations: Scan UCN output for negative concentrations\n"
        "  which indicate numerical instability (common with advection-dominated\n"
        "  transport or coarse grids).\n"
        "- courant_number: Estimate Courant number from velocity field and grid\n"
        "  spacing. Target: Cr < 1 for numerical stability.\n"
        "- peclet_number: Estimate grid Peclet number (Pe = v*dx/D). Target:\n"
        "  Pe < 2 for advection-dominated problems to avoid numerical dispersion.\n"
        "- concentration_bounds: Check concentrations against physical bounds\n"
        "  (non-negative, below source concentration). Flags unrealistic values.\n"
        "- solver_convergence: Parse listing file for GCG solver iterations,\n"
        "  failures, and convergence history.\n\n"

        "QA/QC DOMAIN KNOWLEDGE (Transport):\n"
        "- Peclet number (Pe = v*dx/D): <2 for upstream FD, <4 for TVD.\n"
        "  High Pe causes numerical dispersion (artificial smearing).\n"
        "- Courant number (Cr = v*dt/dx): <1 for stability.\n"
        "  High Cr causes numerical oscillations.\n"
        "- Transport mass balance: <0.5% discrepancy is good, 0.5-1% marginal,\n"
        "  >1% is problematic. Worse than flow mass balance is expected.\n"
        "- Negative concentrations: Indicate numerical problems. Common causes:\n"
        "  coarse grid, high Pe, upstream FD method with sharp fronts.\n"
        "  Fix: refine grid, increase dispersivity, use TVD or MOC method.\n"
        "- Dispersivity typical ranges:\n"
        "  * Field scale longitudinal: 1-100 m (scale-dependent)\n"
        "  * TRPT (TH/TL ratio): 0.1-0.3 typically\n"
        "  * TRPV (TV/TL ratio): 0.01-0.1 typically\n"
        "  * Molecular diffusion: 1e-10 to 1e-9 m2/s\n"
        "- Porosity typical ranges: sand 0.25-0.40, gravel 0.25-0.35,\n"
        "  silt 0.35-0.50, clay 0.40-0.60\n"
        "- Sorption: Kd (distribution coefficient) is site/contaminant specific.\n"
        "  Retardation R = 1 + (rho_b * Kd) / n\n"
        "- First-order decay half-life: t_1/2 = ln(2) / lambda\n\n"

        "TARGETED DATA EXTRACTION:\n"
        "The read_binary_output tool supports UCN concentration files:\n"
        "- Probe: returns ntimes, shape, value_range for quick overview.\n"
        "- Extract data: per-layer concentration statistics per sampled timestep.\n"
        "- Extract time-series: concentration at specific cells across all times.\n"
        "  Use 'cells' parameter with list of {layer, row, col} (1-based).\n\n"

        "CONNECTION TO MODFLOW:\n"
        "- MT3DMS requires a completed MODFLOW flow solution.\n"
        "- The FTL file passes cell-by-cell flows to the transport model.\n"
        "- MT3D grid must match the MODFLOW grid exactly.\n"
        "- Stress periods in BTN can differ from MODFLOW (sub-stepping).\n"
        "- ICBUND in BTN mirrors IBOUND in BAS: -1=constant conc, 0=inactive, 1=active.\n\n"

        "CROSS-ANALYSIS GUIDANCE:\n"
        "When investigating transport model behavior, combine tools:\n\n"
        "1. MASS BALANCE vs CONCENTRATION TRENDS: Run transport_mass_balance,\n"
        "   then extract concentration time-series at key monitoring points.\n\n"
        "2. NEGATIVE CONCENTRATIONS vs PECLET/COURANT: If negative_concentrations\n"
        "   shows problems, check courant_number and peclet_number to diagnose cause.\n\n"
        "3. SOLVER CONVERGENCE vs MASS BALANCE: If solver_convergence shows issues,\n"
        "   transport_mass_balance will likely show poor closure.\n\n"
        "4. SOURCE STRENGTH vs PLUME EXTENT: Read SSM package to find source\n"
        "   concentrations, then extract concentration data to see plume spread.\n"
    )


# ---------------------------------------------------------------------------
# File-mention regex pattern
# ---------------------------------------------------------------------------

def mt3dms_file_mention_regex() -> str:
    """Return the regex alternation for MT3D-specific file extensions."""
    return (
        "btn|adv|dsp|ssm|rct|gcg|tob|ftl|ucn|mas|"
        "nam|mtnam|sft|lkt|uzt|cts|lst"
    )


# ---------------------------------------------------------------------------
# Tool description overrides
# ---------------------------------------------------------------------------

def mt3dms_tool_description_overrides() -> dict[str, str]:
    """Return MT3D-specific description text for tool definitions."""
    return {
        "read_file": (
            "Read a text file from the model workspace. Returns the file content "
            "(truncated to max_bytes if the file is large). Use this for MT3DMS / "
            "MT3D-USGS package files (.btn, .adv, .dsp, .ssm, .rct, .gcg, .tob, "
            ".ftl, .sft, .lkt, .uzt, .cts, etc.), the listing file (.lst), "
            "mass balance file (.mas), CSV files, or any other text-based file."
        ),
        "read_concentration_output": (
            "Extract numerical data from a MT3DMS/MT3D-USGS binary concentration "
            "file (.ucn) via FloPy. Returns per-layer statistics (min, max, mean, "
            "median, std) for sampled timesteps. UCN files are named MT3D001.UCN "
            "(species 1), MT3D002.UCN (species 2), etc. Use the 'cells' parameter "
            "to extract concentration time-series at specific cells (1-based "
            "{layer, row, col})."
        ),
        "run_qa_check": (
            "Run a specialized QA/QC diagnostic check on the MT3DMS/MT3D-USGS "
            "transport model. Returns a detailed markdown report.\n\n"
            "Available checks:\n"
            "- transport_mass_balance: Parse mass balance from .mas or listing file\n"
            "- negative_concentrations: Scan UCN output for negative values\n"
            "- courant_number: Estimate Courant number from flow field\n"
            "- peclet_number: Estimate grid Peclet number\n"
            "- concentration_bounds: Check concentrations against physical limits\n"
            "- solver_convergence: Parse GCG solver iterations and failures\n\n"
            "Use this when the user asks about transport quality, mass balance, "
            "negative concentrations, numerical stability, or any QA/QC analysis."
        ),
        "generate_plot": (
            "Generate a plot from the transport model workspace data. You MUST "
            "provide a full Python script that reads real data. NEVER use random "
            "or dummy data.\n\n"
            "REQUIRED helper APIs:\n"
            "- import gw_plot_io\n"
            "- gw_plot_io.ws_path(rel) -- get absolute path\n"
            "- gw_plot_io.out_path('name.png') -- output path for saving\n"
            "- gw_plot_io.find_one('.ucn') -- find a UCN file\n\n"
            "COMMON PATTERNS (MT3DMS/MT3D-USGS):\n"
            "  # Load MT3D model:\n"
            "  import flopy\n"
            "  ws = str(gw_plot_io.ws_path('.'))\n"
            "  # Read concentration from UCN:\n"
            "  ucn_path = str(gw_plot_io.ws_path(gw_plot_io.find_one('.ucn')))\n"
            "  ucn = flopy.utils.UcnFile(ucn_path)\n"
            "  times = ucn.get_times()\n"
            "  conc = ucn.get_data(totim=times[-1])  # (nlay, nrow, ncol)\n\n"
            "  # Concentration contour map:\n"
            "  import matplotlib.pyplot as plt\n"
            "  fig, ax = plt.subplots(figsize=(10, 8))\n"
            "  im = ax.imshow(conc[0], cmap='YlOrRd', origin='upper')\n"
            "  plt.colorbar(im, ax=ax, label='Concentration')\n"
            "  ax.set_title('Layer 1 Concentration')\n\n"
            "  # Breakthrough curve:\n"
            "  ts = ucn.get_ts((0, row, col))  # (time, conc) array\n"
            "  plt.plot(ts[:, 0], ts[:, 1])\n\n"
            "  # Save:\n"
            "  plt.savefig(gw_plot_io.out_path('conc_map.png'), dpi=150, "
            "bbox_inches='tight')\n\n"
            "The tool returns image URLs in an 'images' array. Embed them as "
            "markdown: ![Description](url)"
        ),
    }


# ---------------------------------------------------------------------------
# File-type knowledge base
# ---------------------------------------------------------------------------

_EXT_KB: dict[str, FileTypeInfo] = {
    ".nam": FileTypeInfo(
        kind="namefile",
        purpose="MT3D name file listing all transport packages, unit numbers, and filenames",
        what_to_look_for="FTYPE UNIT FNAME entries; which transport packages are active",
    ),
    ".mtnam": FileTypeInfo(
        kind="namefile",
        purpose="MT3D-USGS name file (alternate extension to distinguish from MODFLOW .nam)",
        what_to_look_for="FTYPE UNIT FNAME entries; look for SFT/LKT/UZT/CTS packages",
    ),
    ".btn": FileTypeInfo(
        kind="basic_transport",
        purpose="Basic Transport: grid overlay, porosity, ICBUND, initial concentrations, time stepping",
        what_to_look_for="NCOMP (number of species); PRSITY arrays; ICBUND; SCONC per species; DT0, MXSTRN",
    ),
    ".adv": FileTypeInfo(
        kind="advection",
        purpose="Advection package: solution method selection and particle tracking parameters",
        what_to_look_for="MIXELM (0=upstream FD, 1=MOC, 2=MMOC, 3=HMOC); PERCEL; MXPART",
    ),
    ".dsp": FileTypeInfo(
        kind="dispersion",
        purpose="Dispersion package: dispersivity arrays and molecular diffusion",
        what_to_look_for="AL (longitudinal dispersivity per layer); TRPT (TH/TL ratio); TRPV (TV/TL ratio); DMCOEF",
    ),
    ".ssm": FileTypeInfo(
        kind="source_sink",
        purpose="Sink/Source Mixing: links transport to MODFLOW stress packages with source concentrations",
        what_to_look_for="MXSS; ITYPE codes linking to WEL/RIV/GHB/etc.; source concentrations per species",
    ),
    ".rct": FileTypeInfo(
        kind="reaction",
        purpose="Chemical Reaction: sorption, decay, and production parameters",
        what_to_look_for="ISOTHM (sorption type); IREACT (reaction type); RHOB (bulk density); SP1/SP2; RC1/RC2",
    ),
    ".gcg": FileTypeInfo(
        kind="solver",
        purpose="Generalized Conjugate Gradient solver for transport equation",
        what_to_look_for="MXITER (max outer iterations); ITER1 (max inner); ISOLVE (method); CCLOSE (closure)",
    ),
    ".tob": FileTypeInfo(
        kind="observation",
        purpose="Transport Observation: concentration targets for calibration",
        what_to_look_for="Observation well names; Layer Row Col; time offsets; observed concentrations",
    ),
    ".ftl": FileTypeInfo(
        kind="link",
        purpose="Flow-Transport Link: binary file passing MODFLOW flows to MT3D",
        what_to_look_for="Binary file — presence confirms MT3D is linked to a MODFLOW model",
    ),
    ".ucn": FileTypeInfo(
        kind="binary_output",
        purpose="Binary concentration output — computed concentrations per saved timestep per species",
        what_to_look_for="Binary data — use read_binary_output tool to extract statistics",
    ),
    ".mas": FileTypeInfo(
        kind="mass_balance",
        purpose="Mass balance summary — total mass in/out per timestep",
        what_to_look_for="Columnar data: TIME, TOTAL IN, TOTAL OUT; compute % discrepancy",
    ),
    ".lst": FileTypeInfo(
        kind="listing",
        purpose="Listing file — solver convergence, transport budget, warnings",
        what_to_look_for="GCG solver iterations; transport budget tables; convergence failures",
    ),
    ".obs": FileTypeInfo(
        kind="observation_output",
        purpose="Transport observation output — simulated vs observed concentrations",
        what_to_look_for="Observation name, time, simulated concentration, observed concentration, residual",
    ),
    ".sft": FileTypeInfo(
        kind="stream_transport",
        purpose="Streamflow Transport (MT3D-USGS): solute transport in SFR stream network",
        what_to_look_for="Stream reach concentrations; mass loading; decay in streams",
    ),
    ".lkt": FileTypeInfo(
        kind="lake_transport",
        purpose="Lake Transport (MT3D-USGS): solute transport in LAK package lakes",
        what_to_look_for="Lake concentrations; mass flux between lake and aquifer",
    ),
    ".uzt": FileTypeInfo(
        kind="unsaturated_transport",
        purpose="Unsaturated Zone Transport (MT3D-USGS): solute transport through vadose zone",
        what_to_look_for="Infiltration concentrations; unsaturated zone mass storage",
    ),
    ".cts": FileTypeInfo(
        kind="treatment",
        purpose="Contaminant Treatment System (MT3D-USGS): pump-and-treat simulation",
        what_to_look_for="Treatment well locations; injection/extraction pairs; treatment efficiency",
    ),
}


# ---------------------------------------------------------------------------
# Package property definitions
# ---------------------------------------------------------------------------

PACKAGE_PROPERTIES: dict[str, PackagePropertyInfo] = {
    "BTN": PackagePropertyInfo(
        file_ext=".btn",
        block="GRIDDATA",
        arrays={
            "PRSITY": PackageArrayInfo(label="Effective porosity", per_layer=True),
            "ICBUND": PackageArrayInfo(label="Transport boundary array (active/inactive/const-conc)", per_layer=True),
            "SCONC": PackageArrayInfo(label="Starting concentration (per species)", per_layer=True),
        },
    ),
    "DSP": PackagePropertyInfo(
        file_ext=".dsp",
        block="GRIDDATA",
        arrays={
            "AL": PackageArrayInfo(label="Longitudinal dispersivity", per_layer=True),
            "TRPT": PackageArrayInfo(label="Ratio of TH to TL dispersivity", per_layer=False),
            "TRPV": PackageArrayInfo(label="Ratio of TV to TL dispersivity", per_layer=False),
            "DMCOEF": PackageArrayInfo(label="Molecular diffusion coefficient", per_layer=True),
        },
    ),
    "RCT": PackagePropertyInfo(
        file_ext=".rct",
        block="GRIDDATA",
        arrays={
            "RHOB": PackageArrayInfo(label="Bulk density of aquifer medium", per_layer=True),
            "SP1": PackageArrayInfo(label="First sorption parameter (Kd for linear)", per_layer=True),
            "SP2": PackageArrayInfo(label="Second sorption parameter", per_layer=True),
            "RC1": PackageArrayInfo(label="First-order reaction rate — dissolved phase", per_layer=True),
            "RC2": PackageArrayInfo(label="First-order reaction rate — sorbed phase", per_layer=True),
        },
    ),
}
