"""MODPATH 7 simulator-specific knowledge for LLM prompts and tools.

This module provides MODPATH-specific text fragments that are injected
into the system prompt and tool definitions.  MODPATH is a particle-
tracking post-processor — it reads MODFLOW head/budget output and
traces particle paths through the flow field.
"""

from __future__ import annotations

from gw.simulators.base import FileTypeInfo, PackagePropertyInfo, PackageArrayInfo


# ---------------------------------------------------------------------------
# System prompt fragment
# ---------------------------------------------------------------------------

def modpath_system_prompt_fragment() -> str:
    """Return the MODPATH 7 knowledge block for the LLM system prompt."""
    return (
        "SIMULATOR: MODPATH 7 (Particle Tracking Post-Processor)\n"
        "You understand MODPATH 7 file formats and particle tracking concepts deeply.\n\n"

        "MODPATH IS A POST-PROCESSOR:\n"
        "- MODPATH does NOT solve the groundwater flow equation.\n"
        "- It reads head and budget output from a MODFLOW run (MF6 or MF2005).\n"
        "- It traces particle paths through the computed flow field.\n"
        "- It does NOT produce head or budget files — only particle output.\n"
        "- The underlying MODFLOW model must be run FIRST before MODPATH.\n\n"

        "MODPATH FILE TYPES:\n"
        "- .mpsim — Simulation file: defines tracking type, direction, options\n"
        "- .mpnam — Name file: references the MODFLOW head/budget/DIS files\n"
        "- .mpbas — Basic package: porosity, IBOUND overrides\n"
        "- .mpend — Endpoint output: final particle positions and travel times\n"
        "- .mppth — Pathline output: particle tracks through time\n"
        "- .mpts  — Timeseries output: particle positions at fixed time intervals\n"
        "- .mplst — Listing file: simulation log and diagnostics\n\n"

        "SIMULATION TYPES:\n"
        "1. Endpoint analysis — Records only start and end positions.\n"
        "   Use for: capture zone delineation, travel time estimation,\n"
        "   wellhead protection area mapping.\n"
        "2. Pathline analysis — Records full particle trajectories.\n"
        "   Use for: flow path visualization, contaminant transport paths,\n"
        "   identifying preferential flow routes.\n"
        "3. Timeseries analysis — Records positions at fixed time intervals.\n"
        "   Use for: particle front tracking, plume evolution estimation,\n"
        "   time-of-travel zones.\n"
        "4. Combined analysis — Endpoint + pathline + timeseries together.\n\n"

        "TRACKING DIRECTION:\n"
        "- Forward tracking: Particles move in the direction of flow.\n"
        "  Use for: contaminant plume prediction, recharge zone tracing,\n"
        "  where does water FROM this location go?\n"
        "- Backward tracking: Particles move against the flow direction.\n"
        "  Use for: capture zone analysis, wellhead protection,\n"
        "  where does water AT this location come from?\n\n"

        "WEAK SINK/SOURCE HANDLING:\n"
        "- Weak sinks: cells where only PART of the flow exits through a\n"
        "  boundary condition (e.g., a partially penetrating well).\n"
        "  Options: pass through (default), stop at weak sink, stop based\n"
        "  on a user-specified threshold.\n"
        "- Weak sources: similar concept for flow entering through boundaries.\n"
        "- The weak sink/source option significantly affects capture zones.\n\n"

        "PARTICLE GROUPS AND RELEASE:\n"
        "- Particles are organized into groups with shared properties.\n"
        "- Release points can be specified at: cell faces, cell centers,\n"
        "  arbitrary (x,y,z) locations within cells.\n"
        "- Release can be at a single time or at multiple times (staggered).\n"
        "- For wellhead protection: place particles around the well screen.\n"
        "- For capture zones: release particles in a grid across the model.\n\n"

        "TERMINATION CONDITIONS:\n"
        "- Reached a boundary cell (e.g., well, drain, river, CHD)\n"
        "- Reached a specified zone number (zone stop)\n"
        "- Exceeded maximum tracking time\n"
        "- Entered an inactive or dry cell\n"
        "- Stranded (no flow to follow — typically a stagnation point)\n\n"

        "ZONE-BASED CAPTURE ANALYSIS:\n"
        "- MODPATH uses zone numbers (from IBOUND or zone array) to classify\n"
        "  where particles terminate.\n"
        "- Capture zone = the area from which particles reach a specific zone.\n"
        "- Zone 0 = inactive, Zone 1 = default active.\n"
        "- Higher zone numbers can mark wells, rivers, boundaries, etc.\n"
        "- Backward tracking from a well: particles trace where water comes from.\n\n"

        "TRAVEL TIME DISTRIBUTION:\n"
        "- Travel time is the time for a particle to reach its termination point.\n"
        "- Distribution analysis shows: short (days), medium (years), long paths.\n"
        "- Heavily dependent on porosity and K distribution.\n"
        "- Useful for vulnerability assessment and contaminant transport timing.\n\n"

        "CONNECTION TO MODFLOW:\n"
        "- MODPATH requires a successful MODFLOW run with saved head and budget.\n"
        "- Head file (.hds) provides the head distribution.\n"
        "- Budget file (.cbc) provides the cell-by-cell flows.\n"
        "- DIS/DISV file provides the grid geometry.\n"
        "- If the MODFLOW model has convergence issues or mass balance problems,\n"
        "  the particle tracking results will be unreliable.\n\n"

        "QA/QC FOR PARTICLE TRACKING:\n"
        "When the user asks about particle tracking quality, use run_qa_check.\n\n"
        "Available checks:\n"
        "- particle_termination: Analyze where and why particles terminate.\n"
        "  Red flags: many stranded particles, many entering inactive cells.\n"
        "  Good: most particles reach expected boundaries (wells, rivers).\n"
        "- travel_time_distribution: Statistical analysis of particle travel times.\n"
        "  Check for unreasonably short (<1 day) or long (>1000 yr) travel times.\n"
        "  May indicate porosity or K issues.\n"
        "- capture_zones: Analyze which zones particles terminate in.\n"
        "  Useful for wellhead protection and source-area identification.\n"
        "- dry_cell_encounters: Check if particles encounter dry or inactive cells.\n"
        "  Particles entering dry cells indicate model boundary issues.\n"
        "- mass_balance_prereq: Verify the underlying MODFLOW model has acceptable\n"
        "  mass balance. Poor mass balance invalidates particle tracking results.\n\n"

        "MODPATH QA DOMAIN KNOWLEDGE:\n"
        "- Porosity: sand 0.25-0.40, gravel 0.20-0.35, silt 0.35-0.50, clay 0.40-0.70\n"
        "- Effective porosity is typically lower than total porosity (0.5-0.8x).\n"
        "- Travel times are INVERSELY proportional to porosity — low porosity = fast travel.\n"
        "- Stranded particles (>5% of total) suggest stagnation zones or model issues.\n"
        "- Particles entering inactive cells (>2% of total) suggest IBOUND problems.\n"
        "- Capture zone should be contiguous — fragmented zones suggest numerical issues.\n"
        "- For wellhead protection, 10-year and 50-year travel time contours are standard.\n\n"

        "PLOTTING GUIDANCE:\n"
        "Common MODPATH visualizations:\n"
        "- Pathline map: plot particle tracks on model grid (colored by travel time)\n"
        "- Endpoint map: plot termination points (colored by status or zone)\n"
        "- Capture zone map: delineate area from which particles reach a target\n"
        "- Travel time histogram: distribution of particle travel times\n"
        "- Cross-section pathlines: pathlines in vertical section along a transect\n\n"

        "When generating MODPATH plots:\n"
        "  import flopy\n"
        "  from flopy.utils import EndpointFile, PathlineFile\n"
        "  epf = EndpointFile('model.mpend')\n"
        "  ep_data = epf.get_alldata()\n"
        "  plf = PathlineFile('model.mppth')\n"
        "  pl_data = plf.get_alldata()\n"
        "  # For pathline plots on a map view:\n"
        "  pmv = flopy.plot.PlotMapView(model=m)\n"
        "  pmv.plot_pathline(pl_data, layer='all', colors='blue')\n"
    )


# ---------------------------------------------------------------------------
# File-mention regex pattern
# ---------------------------------------------------------------------------

def modpath_file_mention_regex() -> str:
    """Return the regex alternation for MODPATH-specific file extensions."""
    return (
        "mpsim|mpnam|mpbas|mpend|mppth|mpts|mplst|mp7|"
        "modpath|particle|pathline|endpoint|timeseries|"
        "capture.zone|travel.time"
    )


# ---------------------------------------------------------------------------
# Tool description overrides
# ---------------------------------------------------------------------------

def modpath_tool_description_overrides() -> dict[str, str]:
    """Return MODPATH-specific description text for tool definitions."""
    return {
        "read_file": (
            "Read a text file from the model workspace. Returns the file content "
            "(truncated to max_bytes if the file is large). Use this for MODPATH "
            "input files (.mpsim, .mpnam, .mpbas), listing files (.mplst), "
            "or any other text-based file. MODPATH binary output files "
            "(.mpend, .mppth, .mpts) should be read via read_binary_output instead."
        ),
        "read_binary_output": (
            "Extract numerical data from MODPATH binary output files. "
            "Supports:\n"
            "- .mpend (endpoint file): Returns particle count, termination status "
            "breakdown (boundary, well, inactive, stranded), and travel time statistics.\n"
            "- .mppth (pathline file): Returns particle count, path length statistics, "
            "time range, and per-particle track information.\n"
            "- .mpts (timeseries file): Returns particle count and time point data.\n\n"
            "MODPATH does NOT produce head (.hds) or budget (.cbc) files. "
            "To read the underlying MODFLOW output, switch to the MODFLOW adapter."
        ),
        "run_qa_check": (
            "Run a specialized QA/QC diagnostic check on the MODPATH model. "
            "Returns a detailed markdown report.\n\n"
            "Available checks:\n"
            "- particle_termination: Where and why particles terminate\n"
            "- travel_time_distribution: Statistical analysis of travel times\n"
            "- capture_zones: Zone-based termination analysis\n"
            "- dry_cell_encounters: Particles entering dry/inactive cells\n"
            "- mass_balance_prereq: Verify underlying MODFLOW mass balance\n\n"
            "Use this when the user asks about particle tracking quality, "
            "capture zone analysis, travel time issues, or model reliability."
        ),
        "generate_plot": (
            "Generate a plot from MODPATH model data. You MUST provide a full Python "
            "script that reads real data from workspace files. NEVER use random/dummy data.\n\n"
            "The script runs in a sandbox with access to matplotlib, numpy, pandas, flopy, "
            "and helper modules (gw_plot_io).\n\n"
            "REQUIRED helper APIs in scripts:\n"
            "- import gw_plot_io\n"
            "- gw_plot_io.ws_path(rel) -- get absolute path to a workspace file\n"
            "- gw_plot_io.out_path('name.png') -- get output path for saving plots\n"
            "- gw_plot_io.find_one('.mpend') -- find a file by extension\n\n"
            "COMMON MODPATH PLOT PATTERNS:\n"
            "  # Read endpoints:\n"
            "  from flopy.utils import EndpointFile\n"
            "  epf = EndpointFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.mpend'))))\n"
            "  ep = epf.get_alldata()\n\n"
            "  # Read pathlines:\n"
            "  from flopy.utils import PathlineFile\n"
            "  plf = PathlineFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.mppth'))))\n"
            "  pl = plf.get_alldata()\n\n"
            "  # Plot pathlines on map:\n"
            "  fig, ax = plt.subplots(figsize=(10, 8))\n"
            "  for p in pl:\n"
            "      ax.plot(p['x'], p['y'], 'b-', alpha=0.3, linewidth=0.5)\n\n"
            "  # Endpoint scatter colored by travel time:\n"
            "  sc = ax.scatter(ep['x0'], ep['y0'], c=ep['time'], cmap='viridis', s=5)\n"
            "  plt.colorbar(sc, label='Travel time')\n\n"
            "  # Travel time histogram:\n"
            "  plt.hist(ep['time'], bins=50, edgecolor='black')\n"
            "  plt.xlabel('Travel Time')\n\n"
            "  plt.savefig(gw_plot_io.out_path('pathlines.png'), dpi=150, bbox_inches='tight')\n\n"
            "The tool returns image URLs in an 'images' array."
        ),
    }


# ---------------------------------------------------------------------------
# File-type knowledge base
# ---------------------------------------------------------------------------

_EXT_KB: dict[str, FileTypeInfo] = {
    ".mpsim": FileTypeInfo(
        kind="simulation",
        purpose="MODPATH simulation file: defines tracking type, direction, options, particle groups",
        what_to_look_for=(
            "Simulation type (1=endpoint, 2=pathline, 3=timeseries, 4=combined); "
            "tracking direction (1=forward, 2=backward); weak sink/source options; "
            "reference time; stop time; particle group definitions"
        ),
    ),
    ".mpnam": FileTypeInfo(
        kind="namefile",
        purpose="MODPATH name file: references the MODFLOW head, budget, and DIS files",
        what_to_look_for=(
            "HEAD file path (links to MODFLOW .hds); BUDGET file path (links to .cbc); "
            "DIS/DISV file path; GRB file path; MPBAS reference"
        ),
    ),
    ".mpbas": FileTypeInfo(
        kind="basic",
        purpose="MODPATH basic package: porosity array, IBOUND modifications, default zone values",
        what_to_look_for=(
            "Porosity values per layer (should be 0.01-0.70 for reasonable values); "
            "IBOUND overrides; default IFACE settings; zone array definitions"
        ),
    ),
    ".mpend": FileTypeInfo(
        kind="binary_output",
        purpose="MODPATH endpoint output: final particle positions, travel times, termination status",
        what_to_look_for=(
            "Binary data -- use read_binary_output tool. Contains particle ID, "
            "start/end coordinates, travel time, termination zone, status code"
        ),
    ),
    ".mppth": FileTypeInfo(
        kind="binary_output",
        purpose="MODPATH pathline output: full particle trajectories through the flow field",
        what_to_look_for=(
            "Binary data -- use read_binary_output tool. Contains particle tracks: "
            "x, y, z, time, layer, row, col at each tracked point"
        ),
    ),
    ".mpts": FileTypeInfo(
        kind="binary_output",
        purpose="MODPATH timeseries output: particle positions at fixed time intervals",
        what_to_look_for=(
            "Binary data -- use read_binary_output tool. Contains particle positions "
            "recorded at uniform time intervals for front-tracking analysis"
        ),
    ),
    ".mplst": FileTypeInfo(
        kind="listing",
        purpose="MODPATH listing file: simulation log, particle counts, warnings, diagnostics",
        what_to_look_for=(
            "Particle count; tracking statistics; warning messages about dry cells "
            "or inactive cells encountered; zone summary; execution time"
        ),
    ),
    ".mp7": FileTypeInfo(
        kind="configuration",
        purpose="Generic MODPATH 7 configuration or project file",
        what_to_look_for="Configuration parameters; file references; project metadata",
    ),
    ".sloc": FileTypeInfo(
        kind="particle_input",
        purpose="MODPATH starting locations file: defines particle release points",
        what_to_look_for=(
            "Particle group definitions; release point coordinates (local or global); "
            "cell face assignments; release times"
        ),
    ),
    ".mpzon": FileTypeInfo(
        kind="zone",
        purpose="MODPATH zone file: zone numbers for capture analysis and termination criteria",
        what_to_look_for="Zone arrays per layer; zone numbers corresponding to boundaries, wells, etc.",
    ),
}


# ---------------------------------------------------------------------------
# Package property definitions
# ---------------------------------------------------------------------------

PACKAGE_PROPERTIES: dict[str, PackagePropertyInfo] = {
    "MPBAS": PackagePropertyInfo(
        file_ext=".mpbas",
        block="DATA",
        arrays={
            "POROSITY": PackageArrayInfo(
                label="Effective porosity (dimensionless, typically 0.01-0.50)",
                per_layer=True,
            ),
            "IBOUND": PackageArrayInfo(
                label="Boundary array override (0=inactive, >0=active zone number)",
                per_layer=True,
            ),
        },
    ),
    "MPSIM": PackagePropertyInfo(
        file_ext=".mpsim",
        block="OPTIONS",
        arrays={
            "SIMULATION_TYPE": PackageArrayInfo(
                label="Tracking analysis type (1=endpoint, 2=pathline, 3=timeseries, 4=combined)",
                per_layer=False,
            ),
            "TRACKING_DIRECTION": PackageArrayInfo(
                label="Particle tracking direction (1=forward, 2=backward)",
                per_layer=False,
            ),
            "WEAK_SINK_OPTION": PackageArrayInfo(
                label="Weak sink handling (1=pass through, 2=stop at all, 3=threshold-based)",
                per_layer=False,
            ),
            "WEAK_SOURCE_OPTION": PackageArrayInfo(
                label="Weak source handling (1=pass through, 2=stop at all)",
                per_layer=False,
            ),
        },
    ),
}
