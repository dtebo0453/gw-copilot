"""MODFLOW 6 simulator-specific knowledge for LLM prompts and tools.

This module provides the MF6-specific text fragments that are injected into
the system prompt and tool definitions.  All MF6 domain knowledge that was
previously hardcoded across ``chat_agent.py`` and ``tool_loop.py`` is
consolidated here so the abstraction layer can swap it out per-simulator.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# System prompt fragment
# ---------------------------------------------------------------------------

def mf6_system_prompt_fragment() -> str:
    """Return the MF6-specific knowledge block for the LLM system prompt.

    Covers: file formats, capabilities, QA check descriptions, domain
    knowledge thresholds, and MF6-specific terminology.
    """
    return (
        "SIMULATOR: MODFLOW 6\n"
        "You understand MODFLOW 6 file formats deeply: NAM, DIS/DISV/DISU, TDIS,\n"
        "IMS, NPF, STO, IC, OC, WEL, CHD, GHB, RIV, DRN, RCH, EVT, UZF, etc.\n\n"

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
        "  and solver warnings (under-relaxation, backtracking, damping, timestep cuts).\n"
        "- pumping_summary: Analyze WEL package pumping rates by stress period.\n"
        "  Detects anomalous rate jumps (>2x between consecutive periods).\n"
        "- budget_timeseries: Extract IN/OUT per budget term across all timesteps.\n"
        "  Shows temporal trends in the water budget.\n"
        "- head_gradient: Compute cell-to-cell gradients per layer (final timestep).\n"
        "  Flags extreme gradients that may indicate grid resolution issues.\n"
        "- property_check: Check K, Kv/Kh, Ss, Sy ranges for unreasonable values.\n"
        "  Detects layer inversions (deeper layer more permeable than expected).\n"
        "- observation_comparison: Analyze MF6 observation output CSV files (*.obs.csv),\n"
        "  compute per-obs min/max/mean/trend statistics. If observed/target data CSV\n"
        "  files are present, computes RMSE, MAE, R², bias per observation.\n"
        "- listing_budget_detail: Deep listing file parse — per-package IN/OUT budget\n"
        "  tables, solver warnings (under-relaxation, backtracking, damping), dry-cell\n"
        "  location messages, timestep reductions.\n"
        "- property_zones: Spatial K zone analysis per layer — log-bins K into distinct\n"
        "  value clusters, computes coefficient of variation, flags >1000x adjacent-cell\n"
        "  contrasts that can cause numerical instability.\n"
        "- advanced_packages: Parse complex packages (SFR, LAK, MAW, UZF, CSUB, EVT)\n"
        "  into structured summaries: reach counts, widths, gradients, connections,\n"
        "  screens, outlets, options.\n"
        "- save_snapshot: Capture a lightweight snapshot of current model outputs\n"
        "  (per-layer head stats, budget totals, convergence status) for later comparison.\n"
        "  Run this after each model execution.\n"
        "- compare_runs: Compare two run snapshots side-by-side — shows changes in\n"
        "  mean heads, dry cells, budget, and convergence between model runs.\n\n"
        "For a comprehensive QA review, run multiple checks sequentially:\n"
        "1. mass_balance (overall model health)\n"
        "2. convergence (solver performance)\n"
        "3. dry_cells (stability issues)\n"
        "4. property_check (input data reasonableness)\n"
        "5. observation_comparison (calibration quality, if observations exist)\n"
        "6. listing_budget_detail (detailed budget + solver warnings)\n"
        "7. property_zones (spatial property distribution)\n"
        "8. advanced_packages (complex package review, if SFR/LAK/MAW/UZF present)\n\n"

        "QA/QC DOMAIN KNOWLEDGE:\n"
        "- Hydraulic conductivity (K) typical ranges:\n"
        "  * Gravel: 10-1000 m/d, Sand: 0.1-100 m/d, Silt: 1e-4 to 1 m/d\n"
        "  * Clay: 1e-8 to 1e-3 m/d, Sandstone: 1e-4 to 10 m/d\n"
        "- Kv/Kh ratio: typically 0.01-1.0 (vertical always less than horizontal)\n"
        "- Specific storage (Ss): typically 1e-6 to 1e-4 per meter\n"
        "- Specific yield (Sy): sand 0.15-0.30, gravel 0.20-0.35, clay 0.01-0.05\n"
        "- Mass balance: <0.5% discrepancy is good, 0.5-1% needs attention, >1% is problematic\n"
        "- Convergence: IMS outer iterations >50 suggest difficulty; non-convergence is critical\n"
        "- Dry cells: >10% of a layer dry warrants investigation; growing dry zones are concerning\n\n"

        "MODFLOW 6 DOMAIN KNOWLEDGE:\n"
        "- In WEL packages, a well with entries in multiple layers at the same (row,col)\n"
        "  or (node) position represents a multi-screened well.\n"
        "- Negative Q values = extraction (pumping), positive Q values = injection.\n"
        "- PERIOD blocks define stress-period-specific data; PACKAGEDATA defines defaults.\n"
        "- The .hds file contains computed heads per timestep; .cbc contains cell budgets.\n"
        "- The .lst file contains the solver convergence history and water budget.\n\n"

        "TARGETED DATA EXTRACTION:\n"
        "The read_binary_output tool supports targeted extraction:\n"
        "- To get heads at specific cells: use 'cells' parameter with a list of\n"
        "  {layer, row, col} objects (0-indexed). Returns time-series + per-cell stats.\n"
        "- To get budget flows for a specific component: use 'component' parameter\n"
        "  (e.g. 'WEL', 'CHD', 'STO-SS'). Returns top cells by flow magnitude.\n"
        "Use these for well analysis, boundary condition review, and targeted diagnostics.\n\n"

        "CROSS-ANALYSIS GUIDANCE:\n"
        "When investigating model behavior, combine tools for deeper insight:\n\n"
        "1. PUMPING vs HEAD DECLINE: Run pumping_summary to identify high-rate wells,\n"
        "   then use read_binary_output with cells=[well locations] to see head response.\n"
        "   Look for drawdown cones — if heads drop below well screens, pumping is\n"
        "   unsustainable.\n\n"
        "2. DRY CELLS vs STRESS CHANGES: Run dry_cells to find problematic areas,\n"
        "   then read the WEL/RCH package files to see if stresses changed in those\n"
        "   areas. Increasing pumping or decreasing recharge near dry zones is a red flag.\n\n"
        "3. MASS BALANCE vs BUDGET COMPONENTS: If mass_balance shows poor closure,\n"
        "   run listing_budget_detail to identify which packages contribute most to\n"
        "   the imbalance. Then use budget_timeseries to see temporal trends.\n\n"
        "4. CONVERGENCE vs PROPERTY CONTRASTS: If convergence shows high iterations,\n"
        "   run property_zones to check for extreme K contrasts. Adjacent cells with\n"
        "   >1000x K difference cause solver difficulty.\n\n"
        "5. WELL INTERFERENCE ANALYSIS: Use read_binary_output with cells for multiple\n"
        "   nearby wells to see if pumping from one well affects heads at another.\n"
        "   Compare head time-series to identify interference patterns.\n"
    )


# ---------------------------------------------------------------------------
# File-mention regex pattern (simulator-specific file extensions only)
# ---------------------------------------------------------------------------

def mf6_file_mention_regex() -> str:
    """Return the regex alternation for MF6-specific file extensions.

    These are combined with generic extensions (csv, json, txt, etc.) by
    the caller when building ``_FILE_MENTION_RE``.
    """
    return (
        "dis|disv|disu|nam|nams|tdis|ims|npf|sto|oc|ic|"
        "chd|wel|riv|drn|evt|rch|ghb|lak|sfr|uzf|maw|csub|gwt|grb|"
        "hds|cbc|lst"
    )


# ---------------------------------------------------------------------------
# Tool description overrides
# ---------------------------------------------------------------------------

def mf6_tool_description_overrides() -> dict[str, str]:
    """Return MF6-specific description text for tool definitions.

    Keys match tool names in ``TOOL_DEFINITIONS``.
    """
    return {
        "read_file": (
            "Read a text file from the model workspace. Returns the file content "
            "(truncated to max_bytes if the file is large). Use this for MF6 package "
            "files (.dis, .wel, .npf, etc.), CSV files, config files, listing files, "
            "or any other text-based file in the workspace."
        ),
        "read_binary_output": (
            "Extract numerical data from a MODFLOW 6 binary output file (.hds or .cbc) "
            "via FloPy. Returns per-layer statistics (min, max, mean, median, std) for "
            "sampled timesteps, drawdown analysis (for HDS), or per-component budget "
            "breakdowns (for CBC). Use this when the user asks about simulation results."
        ),
        "run_qa_check": (
            "Run a specialized QA/QC diagnostic check on the MODFLOW 6 model. "
            "Returns a detailed markdown report with analysis, tables, and recommendations.\n\n"
            "Available checks:\n"
            "- mass_balance: Parse listing file for volumetric budget, compute % discrepancy per stress period\n"
            "- dry_cells: Count cells with HDRY/1e30 per layer per timestep, identify spatial clusters\n"
            "- convergence: Parse listing file for solver iterations, convergence failures, and solver warnings\n"
            "- pumping_summary: Analyze WEL package rates by stress period, detect anomalous jumps\n"
            "- budget_timeseries: Extract IN/OUT per budget term across all timesteps\n"
            "- head_gradient: Compute cell-to-cell gradients per layer, flag extreme values\n"
            "- property_check: Check K/SS/SY ranges for unreasonable values and layer inversions\n"
            "- observation_comparison: Analyze observation output CSVs and calibration metrics (RMSE, MAE, R², bias)\n"
            "- listing_budget_detail: Per-package IN/OUT budget tables, solver warnings, dry-cell messages\n"
            "- property_zones: Spatial K zone analysis with adjacent-cell contrast detection\n"
            "- advanced_packages: Summarize complex packages (SFR, LAK, MAW, UZF, CSUB, EVT)\n"
            "- save_snapshot: Save current model output snapshot for later comparison\n"
            "- compare_runs: Compare two run snapshots (heads, budget, convergence)\n\n"
            "Use this tool when the user asks about model quality, mass balance, dry cells, "
            "convergence, pumping data, calibration, observations, property distribution, "
            "run comparison, or any QA/QC analysis."
        ),
        "generate_plot": (
            "Generate a plot from the model workspace data. You MUST provide a full Python "
            "script that reads real data from workspace files. NEVER use random/dummy data.\n\n"
            "The script runs in a sandbox with access to matplotlib, numpy, pandas, flopy, "
            "and helper modules (gw_plot_io, gw_mf6_io). Output images are saved to the "
            "plot output directory and returned as URLs.\n\n"
            "REQUIRED helper APIs in scripts:\n"
            "- import gw_plot_io — workspace file access\n"
            "- gw_plot_io.ws_path(rel) — get absolute path to a workspace file\n"
            "- gw_plot_io.out_path('name.png') — get output path for saving plots\n"
            "- gw_plot_io.find_one('.hds') — find a file by extension\n"
            "- import gw_mf6_io — MF6 package readers\n"
            "- Path(os.environ['GW_PLOT_OUTDIR']) — output directory\n\n"
            "COMMON PATTERNS:\n"
            "  # Read heads from HDS:\n"
            "  import flopy\n"
            "  hds = flopy.utils.HeadFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.hds'))))\n"
            "  data = hds.get_data(totim=hds.get_times()[-1])  # shape: (nlay, nrow, ncol)\n\n"
            "  # Save plot:\n"
            "  plt.savefig(gw_plot_io.out_path('my_plot.png'), dpi=150, bbox_inches='tight')\n\n"
            "SPATIAL ANALYSIS RECIPES:\n"
            "  # K distribution map (uses FloPy PlotMapView):\n"
            "  sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws_dir), verbosity_level=0)\n"
            "  gwf = sim.get_model(list(sim.model_names)[0])\n"
            "  fig, ax = plt.subplots(figsize=(10, 8))\n"
            "  pmv = flopy.plot.PlotMapView(model=gwf, layer=0, ax=ax)\n"
            "  arr = gwf.npf.k.array[0]\n"
            "  cb = pmv.plot_array(np.log10(arr), cmap='viridis')\n"
            "  plt.colorbar(cb, label='log10(K) [m/d]')\n"
            "  pmv.plot_grid(linewidth=0.3, color='gray')\n\n"
            "  # Head contour map:\n"
            "  hds_data = hds.get_data(totim=hds.get_times()[-1])\n"
            "  pmv = flopy.plot.PlotMapView(model=gwf, layer=0, ax=ax)\n"
            "  cb = pmv.plot_array(hds_data[0], cmap='coolwarm')\n"
            "  cs = pmv.contour_array(hds_data[0], levels=10, colors='black', linewidths=0.5)\n"
            "  plt.clabel(cs, inline=True, fontsize=8)\n\n"
            "  # Cross-section:\n"
            "  fig, ax = plt.subplots(figsize=(12, 4))\n"
            "  xsect = flopy.plot.PlotCrossSection(model=gwf, line={'row': nrow//2}, ax=ax)\n"
            "  xsect.plot_array(gwf.npf.k.array, cmap='viridis')\n"
            "  xsect.plot_grid(linewidth=0.3)\n\n"
            "  # Head time-series at specific cells:\n"
            "  cells = [(0, 10, 10), (0, 20, 20)]  # (lay, row, col)\n"
            "  for cell in cells:\n"
            "      ts = hds.get_ts(cell)  # shape: (ntimes, 2) → [time, head]\n"
            "      ax.plot(ts[:, 0], ts[:, 1], label=f'Cell {cell}')\n\n"
            "The tool returns image URLs in an 'images' array. Embed them in your response "
            "as markdown: ![Description](url)"
        ),
    }
