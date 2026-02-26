# MODFLOW 6 quick reference (GW Copilot fallback)

This file is a **small fallback** shipped with GW Copilot to help the assistant answer basic “what is this file/package?” questions when the workspace does not include full MODFLOW 6 documentation.

It is **not** a replacement for official USGS documentation. For best results, place the official MODFLOW 6 docs (or curated excerpts) in `<workspace>/docs/` so the local retriever can cite them.

## Name files (`*.nam`)

MODFLOW 6 uses *name files* to tell the program which input files to load.

You will commonly see:

* **Simulation name file** (often `mfsim.nam`) — lists simulation-level files (e.g., `TDIS`, solver, model name files).
* **Model name file** (often `something.nam`, e.g., `aoi_model.nam`) — lists model packages (e.g., `DIS`, `NPF`, `IC`, `STO`, `CHD`, `OC`) and their input filenames.

## Constant Head package (`*.chd`)

The CHD package applies **specified heads** at selected cells. Those cells behave like fixed-head boundaries for the stress periods where they are active.

Typical structure:

* `BEGIN OPTIONS ... END OPTIONS` — optional settings (aux variables, print/save options, etc.)
* `BEGIN DIMENSIONS ... END DIMENSIONS` — often `MAXBOUND` (max number of boundaries)
* `BEGIN PERIOD <per> ... END PERIOD` — lists boundary entries for that stress period

Boundary entries are commonly specified as a **cellid** (structured grids often use `layer row col`) plus one or more head values (package dependent). Some models include auxiliary columns.

## Discretization (`*.dis`)

The DIS package defines the **structured grid** (layers/rows/cols) and geometric arrays such as `TOP` and `BOTM` (layer bottoms). Arrays can be specified via keywords like `CONSTANT`, `INTERNAL`, or `OPEN/CLOSE`.
