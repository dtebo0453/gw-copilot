"""SEAWAT v4 simulator-specific knowledge for LLM prompts and tools.

SEAWAT is a coupled groundwater flow and solute transport simulator that
combines MODFLOW-2000 + MT3DMS + Variable-Density Flow (VDF) + Viscosity
(VSC).  It reads both flow and transport packages from a single .nam file.

This module provides the SEAWAT-specific text fragments that are injected
into the system prompt and tool definitions.
"""

from __future__ import annotations

from gw.simulators.base import FileTypeInfo, PackagePropertyInfo, PackageArrayInfo


# ---------------------------------------------------------------------------
# System prompt fragment
# ---------------------------------------------------------------------------

def seawat_system_prompt_fragment() -> str:
    """Return the SEAWAT-specific knowledge block for the LLM system prompt."""
    return (
        "SIMULATOR: SEAWAT v4 (Coupled Flow + Transport + Variable-Density)\n"
        "You understand SEAWAT file formats deeply.  SEAWAT couples MODFLOW-2000\n"
        "with MT3DMS and adds two density/viscosity packages:\n"
        "- VDF (Variable-Density Flow): density-dependent flow coupling\n"
        "- VSC (Viscosity): viscosity-dependent flow coupling\n\n"

        "A single .nam file references BOTH flow packages (DIS, BAS6, LPF/BCF6,\n"
        "WEL, CHD, GHB, RIV, DRN, RCH, EVT, PCG, etc.) AND transport packages\n"
        "(BTN, ADV, DSP, SSM, GCG, RCT, etc.) AND SEAWAT-specific packages\n"
        "(VDF, VSC).\n\n"

        "VARIABLE-DENSITY FLOW (VDF PACKAGE):\n"
        "The VDF package implements density-dependent groundwater flow.\n"
        "Equation of state: rho = rho_ref + drhodc * (C - C_ref)\n"
        "- rho_ref: reference fluid density (typically 1000 kg/m3 for freshwater)\n"
        "- drhodc: density slope (drho/dC), typically 0.7143 for seawater at\n"
        "  35 kg/m3 TDS (rho = 1000 + 0.7143 * 35 = 1025 kg/m3)\n"
        "- C_ref: reference concentration (typically 0 mg/L)\n"
        "Typical density ranges:\n"
        "  * Freshwater: ~1000 kg/m3\n"
        "  * Brackish: 1000-1010 kg/m3\n"
        "  * Seawater: ~1025 kg/m3\n"
        "  * Brine: 1025-1200+ kg/m3\n\n"

        "Key VDF parameters:\n"
        "- MTDNCONC: MT3DMS species number used for density coupling (usually 1)\n"
        "- DENSEMIN / DENSEMAX: density limits for stability (0 = no limit)\n"
        "- DENSEREF: reference density (1000 for freshwater ref)\n"
        "- DRHODC: slope of density vs concentration\n"
        "- FIRSTDT: first transport timestep length\n"
        "- NSWTCPL: max coupling iterations between flow and transport per timestep\n"
        "- IWTABLE: flag for variable-density water table correction\n"
        "- DNSCRIT: density convergence criterion for coupling iterations\n\n"

        "VISCOSITY (VSC PACKAGE):\n"
        "The VSC package makes viscosity a function of concentration and/or\n"
        "temperature.  Key parameters:\n"
        "- VISCMIN / VISCMAX: viscosity limits\n"
        "- VISCREF: reference dynamic viscosity (typically 0.001 Pa-s at 20C)\n"
        "- DMUDC: slope of viscosity vs concentration (or APTS/MUTEMPOPT for\n"
        "  temperature-dependent viscosity)\n"
        "- NSMUEOS: number of species affecting viscosity\n"
        "- MTMUSPEC: MT3DMS species number for viscosity coupling\n\n"

        "MT3DMS TRANSPORT PACKAGES IN SEAWAT:\n"
        "- BTN (Basic Transport): porosity, ICBUND array, initial concentrations\n"
        "- ADV (Advection): MIXELM scheme (MOC=1, MMOC=2, HMOC=3, TVD=0, finite-diff=-1)\n"
        "- DSP (Dispersion): AL (longitudinal), TRPT (TH/TL ratio), TRPV (TV/TL ratio),\n"
        "  DMCOEF (molecular diffusion)\n"
        "- SSM (Sink/Source Mixing): links transport to flow BCs (WEL, CHD, GHB, etc.)\n"
        "- GCG (Generalized Conjugate Gradient solver for transport)\n"
        "- RCT (Reaction): chemical reactions (sorption, decay, dual-domain)\n\n"

        "COUPLING AND ITERATION:\n"
        "SEAWAT iterates between flow and transport within each timestep:\n"
        "1. Solve flow equation with current density field\n"
        "2. Solve transport equation with current velocity field\n"
        "3. Update density from new concentrations\n"
        "4. Repeat until DNSCRIT met or NSWTCPL reached\n"
        "Poor coupling convergence indicates timestep is too large or density\n"
        "contrasts are too sharp.\n\n"

        "SALTWATER INTRUSION CONCEPTS:\n"
        "- Toe position: landward extent of saltwater wedge at aquifer base\n"
        "- Mixing zone: transition zone between fresh and salt water (diffuse interface)\n"
        "- Upconing: upward movement of saltwater below a pumping well\n"
        "- Sharp interface: assumes no mixing (Ghyben-Herzberg approximation:\n"
        "  freshwater lens depth = 40x freshwater head above sea level)\n"
        "- Diffuse interface: explicitly models solute transport and mixing\n"
        "- SEAWAT uses the diffuse interface approach by default\n"
        "- Henry problem: classic benchmark for variable-density flow codes\n\n"

        "KEY DIFFERENCES FROM MF2005 / MT3DMS STANDALONE:\n"
        "- Single .nam file references BOTH flow AND transport packages\n"
        "- VDF and VSC packages are unique to SEAWAT (not in MF2005 or MT3DMS)\n"
        "- Density coupling adds nonlinearity; needs smaller timesteps\n"
        "- Binary output includes .hds (heads), .cbc (budgets), AND .ucn (concentrations)\n"
        "- Listing file contains both flow and transport convergence info\n"
        "- MODFLOW-2000 style name file (same format as MF2005)\n\n"

        "QA/QC DIAGNOSTICS:\n"
        "When the user asks about model quality, mass balance, dry cells, convergence,\n"
        "transport, density, saltwater intrusion, or any QA/QC analysis, use the\n"
        "run_qa_check tool.\n\n"
        "Available checks:\n"
        "- mass_balance: Parse listing file for volumetric budget, compute % discrepancy\n"
        "  per stress period. Good balance: <0.5%, marginal: 0.5-1%, poor: >1%.\n"
        "- dry_cells: Count cells with dry/inactive heads per layer and timestep.\n"
        "- convergence: Parse listing file for solver iteration counts and failures.\n"
        "- pumping_summary: Analyze WEL package pumping rates by stress period.\n"
        "- budget_timeseries: Extract IN/OUT per budget term across all timesteps.\n"
        "- head_gradient: Compute cell-to-cell gradients per layer.\n"
        "- property_check: Check K/SS/SY ranges from LPF, BCF6, or UPW packages.\n"
        "- listing_budget_detail: Deep listing file parse with solver warnings.\n"
        "- property_zones: Spatial K zone analysis per layer.\n"
        "- density_range: Check computed densities are within physical bounds.\n"
        "- negative_concentrations: Detect negative concentrations from numerical artifacts.\n"
        "- density_concentration_consistency: Verify rho = rho_ref + drhodc * C.\n"
        "- coupling_convergence: Check flow-transport coupling iteration counts.\n"
        "- transport_mass_balance: Check solute mass balance from transport listing.\n"
        "- courant_number: Estimate Courant numbers for advection stability.\n"
        "- save_snapshot: Save model output snapshot for comparison.\n"
        "- compare_runs: Compare two run snapshots side-by-side.\n\n"

        "QA/QC DOMAIN KNOWLEDGE (DENSITY-DEPENDENT MODELS):\n"
        "- Density should be between 990 and 1300 kg/m3 in most applications.\n"
        "- Freshwater density: 998-1002 kg/m3 (temperature dependent).\n"
        "- Seawater density: 1020-1030 kg/m3 (salinity dependent).\n"
        "- Concentrations must be non-negative (negative = numerical artifact).\n"
        "- Courant number should be < 1.0 for stability (ADV TVD scheme tolerates\n"
        "  slightly higher; MOC is less sensitive to Courant number).\n"
        "- Coupling iterations > 5 per timestep suggest the timestep is too large.\n"
        "- NSWTCPL = 1 means no coupling iteration (explicit coupling — fast but\n"
        "  less accurate); NSWTCPL >= 2 is implicit coupling (more accurate).\n"
        "- Transport mass balance errors > 1% indicate numerical problems.\n"
        "- Peclet number > 2 indicates grid is too coarse for given dispersivity.\n\n"

        "CROSS-ANALYSIS GUIDANCE:\n"
        "When investigating SEAWAT model behavior, combine tools for deeper insight:\n\n"
        "1. SALTWATER INTRUSION: Read UCN concentration files to map the mixing zone,\n"
        "   then check head gradients near the coast for inland flow (reversal).\n\n"
        "2. DENSITY vs CONCENTRATION: Run density_range and negative_concentrations\n"
        "   together; inconsistency often points to numerical instability.\n\n"
        "3. COUPLING STABILITY: If coupling_convergence shows many iterations,\n"
        "   check whether the density contrast (DRHODC * C_max) is large and\n"
        "   consider smaller timesteps or more coupling iterations.\n\n"
        "4. PUMPING vs UPCONING: Use pumping_summary + read concentration UCN\n"
        "   at well locations to detect saline upconing.\n"
    )


# ---------------------------------------------------------------------------
# File-mention regex pattern
# ---------------------------------------------------------------------------

def seawat_file_mention_regex() -> str:
    """Return the regex alternation for SEAWAT-specific file extensions.

    Includes all MF2005 extensions + MT3DMS extensions + VDF/VSC.
    """
    return (
        # Flow packages (MF2005)
        "bas|bcf|lpf|upw|pcg|nwt|sip|sor|gmg|de4|"
        "dis|nam|oc|"
        "wel|chd|ghb|riv|drn|rch|evt|sfr|lak|mnw2|uzf|sub|hob|"
        # Transport packages (MT3DMS)
        "btn|adv|dsp|ssm|gcg|rct|tob|"
        # SEAWAT-specific
        "vdf|vsc|"
        # Output files
        "hds|cbc|lst|ucn|mas|obs|cnf"
    )


# ---------------------------------------------------------------------------
# Tool description overrides
# ---------------------------------------------------------------------------

def seawat_tool_description_overrides() -> dict[str, str]:
    """Return SEAWAT-specific description text for tool definitions."""
    return {
        "read_file": (
            "Read a text file from the model workspace. Returns the file content "
            "(truncated to max_bytes if the file is large). Use this for SEAWAT "
            "package files (.dis, .bas, .lpf, .vdf, .vsc, .btn, .adv, .dsp, .ssm, "
            ".gcg, .rct, .wel, .pcg, etc.), CSV files, config files, listing files, "
            "or any other text-based file."
        ),
        "read_binary_output": (
            "Extract numerical data from a SEAWAT binary output file (.hds, .cbc, "
            "or .ucn) via FloPy. For .hds and .cbc files, returns per-layer statistics "
            "(min, max, mean, median, std) for sampled timesteps, drawdown analysis "
            "(for HDS), or per-component budget breakdowns (for CBC). For .ucn files, "
            "returns concentration data with per-layer statistics and negative "
            "concentration detection. Binary formats are identical to MODFLOW-2005 "
            "(.hds, .cbc) and MT3DMS (.ucn)."
        ),
        "run_qa_check": (
            "Run a specialized QA/QC diagnostic check on the SEAWAT model. "
            "Returns a detailed markdown report with analysis, tables, and "
            "recommendations.\n\n"
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
            "- density_range: Check computed densities within physical bounds\n"
            "- negative_concentrations: Detect negative concentrations\n"
            "- density_concentration_consistency: Verify density-concentration relation\n"
            "- coupling_convergence: Check flow-transport coupling iterations\n"
            "- transport_mass_balance: Check solute mass balance\n"
            "- courant_number: Estimate Courant numbers for advection stability\n"
            "- save_snapshot: Save model output snapshot for comparison\n"
            "- compare_runs: Compare two run snapshots\n\n"
            "Use this when the user asks about model quality, mass balance, density, "
            "saltwater intrusion, coupling convergence, transport stability, or any "
            "QA/QC analysis."
        ),
        "generate_plot": (
            "Generate a plot from the model workspace data. You MUST provide a full Python "
            "script that reads real data from workspace files. NEVER use random/dummy data.\n\n"
            "The script runs in a sandbox with access to matplotlib, numpy, pandas, flopy, "
            "and helper modules (gw_plot_io). Output images are saved to the "
            "plot output directory and returned as URLs.\n\n"
            "REQUIRED helper APIs in scripts:\n"
            "- import gw_plot_io -- workspace file access\n"
            "- gw_plot_io.ws_path(rel) -- get absolute path to a workspace file\n"
            "- gw_plot_io.out_path('name.png') -- get output path for saving plots\n"
            "- gw_plot_io.find_one('.hds') -- find a file by extension\n"
            "- Path(os.environ['GW_PLOT_OUTDIR']) -- output directory\n\n"
            "COMMON PATTERNS (SEAWAT):\n"
            "  # Load SEAWAT model:\n"
            "  import flopy\n"
            "  ws = str(gw_plot_io.ws_path('.'))\n"
            "  nam_file = gw_plot_io.find_one('.nam')\n"
            "  m = flopy.seawat.Seawat.load(nam_file, model_ws=ws, verbose=False)\n\n"
            "  # Read heads from HDS:\n"
            "  hds = flopy.utils.HeadFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.hds'))))\n"
            "  data = hds.get_data(totim=hds.get_times()[-1])\n\n"
            "  # Read concentrations from UCN:\n"
            "  ucn = flopy.utils.UcnFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.ucn'))))\n"
            "  conc = ucn.get_data(totim=ucn.get_times()[-1])\n\n"
            "  # Density field from concentration:\n"
            "  rho_ref = 1000.0; drhodc = 0.7143\n"
            "  rho = rho_ref + drhodc * conc  # kg/m3\n\n"
            "  # Cross-section with concentration contours:\n"
            "  fig, ax = plt.subplots(figsize=(12, 4))\n"
            "  xsect = flopy.plot.PlotCrossSection(model=m, line={'row': 0}, ax=ax)\n"
            "  cb = xsect.plot_array(conc, cmap='RdYlBu_r')\n"
            "  cs = xsect.contour_array(conc, levels=[0.5, 17.5, 35.0], colors='k')\n"
            "  plt.colorbar(cb, label='Concentration (kg/m3)')\n\n"
            "  # Save plot:\n"
            "  plt.savefig(gw_plot_io.out_path('my_plot.png'), dpi=150, bbox_inches='tight')\n\n"
            "The tool returns image URLs in an 'images' array. Embed them in your response "
            "as markdown: ![Description](url)"
        ),
    }


# ---------------------------------------------------------------------------
# File-type knowledge base
# ---------------------------------------------------------------------------

_EXT_KB: dict[str, FileTypeInfo] = {
    # ---- Flow packages (same as MF2005) ----
    ".nam": FileTypeInfo(
        kind="namefile",
        purpose="Master name file listing all flow + transport + VDF/VSC packages",
        what_to_look_for="FTYPE UNIT FNAME entries; check for VDF, VSC, BTN, ADV, DSP, SSM, GCG",
    ),
    ".dis": FileTypeInfo(
        kind="discretisation",
        purpose="Grid discretisation: NLAY, NROW, NCOL, NPER, cell sizes, top/bottom",
        what_to_look_for="Grid dimensions in header; DELR, DELC, TOP, BOTM arrays; stress period info",
    ),
    ".bas": FileTypeInfo(
        kind="basic",
        purpose="Basic package: IBOUND array, starting heads (STRT), HDRY value",
        what_to_look_for="IBOUND values (-1=CHD, 0=inactive, 1=active); STRT (initial head)",
    ),
    ".lpf": FileTypeInfo(
        kind="flow",
        purpose="Layer-Property Flow: HK, VKA, Ss, Sy, LAYTYP (confined/convertible)",
        what_to_look_for="HK arrays per layer; VKA; Ss/Sy for transient; LAYTYP flags",
    ),
    ".bcf": FileTypeInfo(
        kind="flow",
        purpose="Block-Centered Flow: TRAN or HY, Vcont, Sf1, Sf2, LAYCON",
        what_to_look_for="Transmissivity or K arrays; LAYCON; specific storage/yield",
    ),
    ".upw": FileTypeInfo(
        kind="flow",
        purpose="Upstream Weighting: HK, VKA, Ss, Sy (NWT-specific, like LPF)",
        what_to_look_for="HK arrays per layer; VKA; Ss/Sy; same layout as LPF",
    ),
    ".pcg": FileTypeInfo(
        kind="solver",
        purpose="Preconditioned Conjugate-Gradient solver settings",
        what_to_look_for="MXITER, ITER1; HCLOSE, RCLOSE (closure criteria)",
    ),
    ".nwt": FileTypeInfo(
        kind="solver",
        purpose="Newton solver settings (MODFLOW-NWT)",
        what_to_look_for="HEADTOL, FLUXTOL; MAXITEROUT; solver options",
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
        what_to_look_for="NSTRM, NSS; reach/segment data; segment flow",
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
    # ---- Transport packages (MT3DMS) ----
    ".btn": FileTypeInfo(
        kind="transport_basic",
        purpose="Basic Transport: porosity, ICBUND, initial concentrations, output timing",
        what_to_look_for="NCOMP, MCOMP (number of species); PRSITY (porosity) array; "
                         "ICBUND (-1=fixed conc, 0=inactive, 1=active); SCONC (starting concentration)",
    ),
    ".adv": FileTypeInfo(
        kind="transport",
        purpose="Advection: solution scheme for advective transport",
        what_to_look_for="MIXELM (MOC=1, MMOC=2, HMOC=3, TVD=0, FD=-1); PERCEL; NADVFD",
    ),
    ".dsp": FileTypeInfo(
        kind="transport",
        purpose="Dispersion: dispersivity and molecular diffusion coefficients",
        what_to_look_for="AL (longitudinal dispersivity per layer); TRPT (TH/TL ratio); "
                         "TRPV (TV/TL ratio); DMCOEF (molecular diffusion)",
    ),
    ".ssm": FileTypeInfo(
        kind="transport",
        purpose="Sink/Source Mixing: links transport to flow boundary conditions",
        what_to_look_for="MXSS (max point sources); ITYPE (source type); concentration for each BC",
    ),
    ".gcg": FileTypeInfo(
        kind="transport_solver",
        purpose="Generalized Conjugate Gradient solver for transport equation",
        what_to_look_for="MXITER, ITER1, ISOLVE; CCLOSE (concentration closure criterion)",
    ),
    ".rct": FileTypeInfo(
        kind="transport",
        purpose="Reaction: sorption, first-order decay, zero-order production, dual-domain",
        what_to_look_for="ISOTHM (sorption type); IREACT (reaction type); SP1, SP2 (sorption params); "
                         "RC1, RC2 (reaction rates)",
    ),
    ".tob": FileTypeInfo(
        kind="transport_observation",
        purpose="Transport Observation: concentration comparison targets",
        what_to_look_for="Observation well locations; observed concentrations; times",
    ),
    # ---- SEAWAT-specific packages ----
    ".vdf": FileTypeInfo(
        kind="density",
        purpose="Variable-Density Flow: density coupling between flow and transport",
        what_to_look_for="MTDNCONC (species for density); DENSEREF (reference density); "
                         "DRHODC (density slope); NSWTCPL (coupling iterations); "
                         "DNSCRIT (density convergence criterion); IWTABLE",
    ),
    ".vsc": FileTypeInfo(
        kind="viscosity",
        purpose="Viscosity: concentration/temperature-dependent viscosity",
        what_to_look_for="VISCREF (reference viscosity); DMUDC (viscosity slope); "
                         "NSMUEOS (species count); MTMUSPEC (species number); "
                         "VISCMIN/VISCMAX (limits); MUTEMPOPT (temperature option)",
    ),
    # ---- Output files ----
    ".hds": FileTypeInfo(
        kind="binary_output",
        purpose="Binary head output file -- computed heads for each saved timestep",
        what_to_look_for="Binary data -- use read_binary_output tool to extract statistics",
    ),
    ".cbc": FileTypeInfo(
        kind="binary_output",
        purpose="Binary cell-by-cell budget file -- flow terms per cell",
        what_to_look_for="Binary data -- use read_binary_output tool to extract budget breakdowns",
    ),
    ".ucn": FileTypeInfo(
        kind="binary_output",
        purpose="Binary concentration output file -- solute concentrations per timestep",
        what_to_look_for="Binary data -- use read_binary_output tool to extract concentration data",
    ),
    ".lst": FileTypeInfo(
        kind="listing",
        purpose="Listing file -- flow + transport solver convergence, budgets, warnings",
        what_to_look_for="Volumetric budget tables; convergence history; dry cell warnings; "
                         "transport mass balance; coupling iteration counts",
    ),
    ".mas": FileTypeInfo(
        kind="transport_output",
        purpose="Mass balance summary file for transport (MT3DMS output)",
        what_to_look_for="Cumulative mass IN/OUT; percent discrepancy per timestep",
    ),
    ".obs": FileTypeInfo(
        kind="transport_output",
        purpose="Transport observation output: simulated vs observed concentrations",
        what_to_look_for="Observation name; time; simulated concentration; observed concentration",
    ),
    ".cnf": FileTypeInfo(
        kind="transport_output",
        purpose="MT3DMS model configuration record file",
        what_to_look_for="Transport model configuration summary",
    ),
}


# ---------------------------------------------------------------------------
# Package property definitions
# ---------------------------------------------------------------------------

PACKAGE_PROPERTIES: dict[str, PackagePropertyInfo] = {
    # ---- Flow properties (same as MF2005) ----
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
    # ---- SEAWAT VDF parameters ----
    "VDF": PackagePropertyInfo(
        file_ext=".vdf",
        block="DATA",
        arrays={
            "MTDNCONC": PackageArrayInfo(label="MT3DMS species for density coupling", per_layer=False),
            "DENSEREF": PackageArrayInfo(label="Reference fluid density (kg/m3)", per_layer=False),
            "DRHODC": PackageArrayInfo(label="Density slope (drho/dC)", per_layer=False),
            "DENSEMIN": PackageArrayInfo(label="Minimum density limit", per_layer=False),
            "DENSEMAX": PackageArrayInfo(label="Maximum density limit", per_layer=False),
            "NSWTCPL": PackageArrayInfo(label="Max coupling iterations per timestep", per_layer=False),
            "DNSCRIT": PackageArrayInfo(label="Density convergence criterion", per_layer=False),
            "FIRSTDT": PackageArrayInfo(label="First transport timestep length", per_layer=False),
            "IWTABLE": PackageArrayInfo(label="Variable-density water table correction flag", per_layer=False),
        },
    ),
    # ---- SEAWAT VSC parameters ----
    "VSC": PackagePropertyInfo(
        file_ext=".vsc",
        block="DATA",
        arrays={
            "VISCREF": PackageArrayInfo(label="Reference dynamic viscosity (Pa-s)", per_layer=False),
            "DMUDC": PackageArrayInfo(label="Viscosity slope (dmu/dC)", per_layer=False),
            "VISCMIN": PackageArrayInfo(label="Minimum viscosity limit", per_layer=False),
            "VISCMAX": PackageArrayInfo(label="Maximum viscosity limit", per_layer=False),
            "NSMUEOS": PackageArrayInfo(label="Number of species affecting viscosity", per_layer=False),
            "MTMUSPEC": PackageArrayInfo(label="MT3DMS species for viscosity coupling", per_layer=False),
        },
    ),
    # ---- Transport properties ----
    "BTN": PackagePropertyInfo(
        file_ext=".btn",
        block="DATA",
        arrays={
            "PRSITY": PackageArrayInfo(label="Porosity", per_layer=True),
            "ICBUND": PackageArrayInfo(label="Transport boundary array (-1=fixed, 0=inactive, 1=active)", per_layer=True),
            "SCONC": PackageArrayInfo(label="Starting concentration", per_layer=True),
        },
    ),
    "DSP": PackagePropertyInfo(
        file_ext=".dsp",
        block="DATA",
        arrays={
            "AL": PackageArrayInfo(label="Longitudinal dispersivity", per_layer=True),
            "TRPT": PackageArrayInfo(label="Ratio of TH to TL dispersivity", per_layer=False),
            "TRPV": PackageArrayInfo(label="Ratio of TV to TL dispersivity", per_layer=False),
            "DMCOEF": PackageArrayInfo(label="Molecular diffusion coefficient", per_layer=True),
        },
    ),
    "RCT": PackagePropertyInfo(
        file_ext=".rct",
        block="DATA",
        arrays={
            "SP1": PackageArrayInfo(label="First sorption parameter", per_layer=True),
            "SP2": PackageArrayInfo(label="Second sorption parameter", per_layer=True),
            "RC1": PackageArrayInfo(label="First reaction rate", per_layer=True),
            "RC2": PackageArrayInfo(label="Second reaction rate", per_layer=True),
        },
    ),
}
