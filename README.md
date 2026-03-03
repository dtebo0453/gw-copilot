# GW Copilot - AI-Powered Groundwater Modeling Assistant

An intelligent co-pilot for groundwater modelers that combines an agentic LLM backend with an interactive React frontend. GW Copilot helps you explore, analyze, visualize, and improve groundwater models through natural language conversation and automated diagnostics.

## Supported Simulators

| Simulator | Output Types | QA Checks | Key Capabilities |
|-----------|-------------|-----------|-------------------|
| **MODFLOW 6** | Heads, Budget | 13 | Full structured/unstructured grid support, GWF/GWT models |
| **MODFLOW-2005/NWT** | Heads, Budget | 11 | BCF/LPF/UPW flow packages, PCG/NWT solvers |
| **MODFLOW-USG** | Heads, Budget | 11 | Unstructured grids (DISU), SMS solver, CLN/GNC |
| **SEAWAT v4** | Heads, Budget, Concentration | 17 | Density-dependent flow, VDF/VSC coupling, saltwater intrusion |
| **MT3DMS / MT3D-USGS** | Concentration | 6 | Solute transport, advection/dispersion/reaction, multi-species |
| **MODPATH 7** | Pathlines, Endpoints | 5 | Particle tracking, capture zone analysis, travel time distributions |

GW Copilot automatically detects which simulator your model uses when you open a workspace folder.

## Features

### Conversational AI Assistant
- Natural language chat grounded in your model's actual data (grid, packages, stress periods, outputs)
- Agentic tool loop: the LLM can iteratively read files, inspect binary output, generate plots, run QA checks, and self-correct -- all without manual intervention
- Supports both **OpenAI** (Responses API) and **Anthropic** (Messages API) as LLM providers
- Simulator-specific domain knowledge injected into system prompts (hydraulic conductivity ranges, mass balance thresholds, solver convergence criteria, etc.)

### Automated QA/QC Diagnostics
- Up to 17 built-in quality checks per simulator, including:
  - **Mass balance** verification with configurable thresholds
  - **Dry cell** detection and spatial analysis
  - **Solver convergence** monitoring (IMS, PCG, NWT, SMS, GCG)
  - **Head gradient** evaluation for unrealistic slopes
  - **Pumping summary** with rate statistics
  - **Budget time series** analysis
  - **Property checks** (K ranges, storage, porosity)
  - **Density-coupling stability** (SEAWAT)
  - **Peclet/Courant number** estimation (MT3DMS)
  - **Particle termination** analysis (MODPATH)
- Quick-action chips in the chat panel for one-click QA runs

### Interactive Visualization
- AI-generated plots via sandboxed Python execution (Matplotlib + FloPy)
- The LLM writes, executes, and self-repairs plotting scripts automatically
- Supports cross-sections, time series, contour maps, head profiles, concentration plumes, and particle pathlines
- 3D model visualization powered by VTK.js

### Model Facts Panel
- At-a-glance snapshot of your model: grid dimensions, cell sizes, layer thicknesses, stress periods, packages, boundary condition summaries, head ranges, and budget terms
- Human-friendly formatting (no raw scientific notation or sentinel values)

### Model Patching
- AI-guided plan/validate/apply workflow for editing model input files
- Preview changes before applying them to your model

### File Explorer
- Browse all workspace files with simulator-aware file type descriptions
- Read and inspect any model input or output file through the chat

## Architecture

```
ui/                          React + TypeScript + Vite frontend
  src/components/
    ChatPanel.tsx            Conversational interface with action/QA chips
    PlotsTab.tsx             Plot generation and script editing
    ModelFactsPanel.tsx      Model summary panel
    Model3DTab.tsx           VTK.js 3D visualization
    MapTab.tsx               2D map view (Leaflet)
    WelcomeScreen.tsx        Landing page with folder picker

gw/api/                      FastAPI backend
  main.py                   App entry point and middleware
  routes.py                 Chat endpoint (/chat)
  plots.py                  Plot endpoints (/plots/plan-agentic, /plots/run)
  patches.py                Patch system (/patch/plan, /patch/validate, /patch/apply)
  model_snapshot.py          FloPy-first model snapshot extraction
  output_probes.py           Binary output readers (HDS, CBC, UCN, HeadUFile)
  qa_diagnostics.py          Shared QA check functions
  workspace_scan.py          Workspace file indexing

gw/llm/                      LLM integration layer
  chat_agent.py             System prompt construction, chat_reply()
  tool_loop.py              Agentic tool loop (5-7 tools, max 15 iterations)
  read_router.py            File reading dispatcher

gw/simulators/               Simulator Abstraction Layer (SAL)
  base.py                   SimulatorAdapter ABC (23+ methods) + OutputCapability
  registry.py               Auto-detection and name-based lookup
  mf6/                      MODFLOW 6 adapter, knowledge, I/O
  mf2005/                   MODFLOW-2005/NWT adapter, knowledge, I/O
  mfusg/                    MODFLOW-USG adapter, knowledge, I/O
  seawat/                   SEAWAT v4 adapter, knowledge
  mt3dms/                   MT3DMS / MT3D-USGS adapter, knowledge, I/O
  modpath/                  MODPATH 7 adapter, knowledge, I/O

gw/run/                      Model execution runners
  mf6_runner.py             MODFLOW 6 runner
  mf2005_runner.py          MODFLOW-2005/NWT runner
  mfusg_runner.py           MODFLOW-USG runner
  seawat_runner.py          SEAWAT runner
  mt3dms_runner.py          MT3DMS / MT3D-USGS runner
  modpath_runner.py         MODPATH 7 runner
```

## Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- An OpenAI or Anthropic API key

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Configure API Key

Set your LLM provider API key as an environment variable:

```bash
# OpenAI
set OPENAI_API_KEY=sk-...       # Windows
# export OPENAI_API_KEY=sk-...  # macOS/Linux

# Anthropic
set ANTHROPIC_API_KEY=sk-...
```

Or configure via the Settings panel in the UI.

### Run the Backend

```bash
uvicorn gw.api.main:app --reload --port 8000
```

### Run the Frontend

```bash
cd ui
npm install
npm run dev
```

Then open http://localhost:5173 in your browser, select a model folder, and start exploring.

## How It Works

### Simulator Detection
When you open a folder, GW Copilot checks for simulator-specific markers in priority order:
1. **MODFLOW 6** -- `mfsim.nam` present
2. **SEAWAT** -- `.nam` with VDF/VSC packages (checked before MF2005 since it's a superset)
3. **MODFLOW-USG** -- `.nam` with DISU/SMS/CLN packages (checked before MF2005 since it's a superset)
4. **MT3DMS** -- `.btn` file or `.nam` with BTN/ADV/DSP packages
5. **MODPATH** -- `.mpsim` or `.mpnam` files
6. **MODFLOW-2005/NWT** -- General `.nam` file (most permissive, checked last)

### Agentic Tool Loop
The LLM has access to 5-7 tools depending on the detected simulator:
- **read_file** -- Read any model input file
- **read_binary_output** -- Extract head/budget data from HDS/CBC files
- **list_files** -- Browse workspace contents
- **generate_plot** -- Write and execute Python plotting scripts
- **run_qa_check** -- Run specific QA diagnostics
- **read_concentration_output** -- *(MT3DMS, SEAWAT)* Extract concentration data from UCN files
- **read_particle_output** -- *(MODPATH)* Read endpoint/pathline data

The LLM can chain these tools across up to 15 iterations, self-correcting errors along the way.

### Model Snapshot
The snapshot system extracts model facts using FloPy (primary) with text parsing fallback:
- **Grid**: Type, dimensions, cell sizes, layer thicknesses
- **TDIS**: Stress periods, time units, total simulation time
- **Packages**: Full inventory with file paths
- **Outputs**: HDS, CBC, UCN, LST, endpoint/pathline file presence
- **Solver**: Convergence parameters
- **Stress data**: Well rates, drain/river/GHB conductances, boundary summaries

Facts are automatically injected into the chat context so the LLM's responses are grounded in your actual model data.

### Screenshots
<img width="562" height="710" alt="Welcome" src="https://github.com/user-attachments/assets/33372016-26ec-4bec-b83e-e35077ef0d20" />
<img width="1891" height="976" alt="Plots" src="https://github.com/user-attachments/assets/4095d2a2-2f5f-47e3-89cf-adc0d224d367" />
<img width="1891" height="974" alt="Model_Files" src="https://github.com/user-attachments/assets/df27f5d5-52bf-4483-a3d4-43c32e15d1cc" />
<img width="544" height="701" alt="LLM_Settings" src="https://github.com/user-attachments/assets/14eb9afa-26fd-4ef6-8194-5a64888ebeea" />
<img width="1892" height="977" alt="Copilot_ex1" src="https://github.com/user-attachments/assets/f99a4651-550a-4153-9bc0-76f4a42cfbe0" />
<img width="1892" height="974" alt="3D_Viewer" src="https://github.com/user-attachments/assets/2146051c-164c-463f-88dd-c57682e8f8aa" />


## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, TypeScript, Vite, VTK.js, Leaflet |
| Backend | FastAPI (Python), Uvicorn |
| Model I/O | FloPy, NumPy, Pandas |
| LLM Providers | OpenAI Responses API, Anthropic Messages API |
| Plotting | Matplotlib (sandboxed subprocess) |
