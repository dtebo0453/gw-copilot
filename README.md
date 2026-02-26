# GW Copilot - Groundwater Modeling Assistant

A hybrid CLI + UI tool for MODFLOW 6 model development with AI-assisted workflows.

## Features

### Core Architecture
- **Deterministic spine**: LLM handles planning + config drafting; model execution stays local and auditable
- **FloPy-first model parsing**: Reliable extraction of model facts (grid, packages, TDIS, solver settings)
- **Workspace security**: Path rooting, traversal protection, configurable allowlists

### UI Components
- **Model Facts Panel**: Real-time display of grid type, dimensions, packages, outputs
- **Chat Panel**: Natural language assistance grounded in model context
- **3D Visualization**: VTK.js-based structured grid rendering (DIS)
- **Plots**: LLM-assisted plot generation with repair loop
- **Model Files**: Browse and inspect workspace files

### API Endpoints
- `/workspace/facts` - Compact model facts for LLM grounding
- `/model/snapshot` - Full model snapshot with extraction details
- `/workspace/scan` - Workspace indexing with cache
- `/viz/mesh`, `/viz/scalars` - 3D mesh data
- `/chat` - Conversational interface

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure API Key
Set `OPENAI_API_KEY` as an environment variable for LLM features.

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

## CLI Usage

```bash
python gw_cli.py --help
python gw_cli.py llm-draft-config --help
```

### Example Workflow

```bash
# Build initial model from AOI
python gw_cli.py build-aoi --aoi example_aoi/aoi.geojson --cellsize 50 --nlay 3 --top 30 --botm 10 -10 -40 --outdir runs/aoi_demo

# Draft config with LLM assistance
python gw_cli.py llm-draft-config \
  --prompt "Build a transient site-scale model for a pumping test" \
  --base-config runs/aoi_demo/config.json \
  --inputs-dir runs/aoi_demo \
  --outdir runs/aoi_demo/llm

# Build the model
python gw_cli.py build --config runs/aoi_demo/llm/draft_config.json
```

## Model Snapshot System

The snapshot system extracts model facts using FloPy (primary) with text parsing fallback:

- **Grid**: Type (DIS/DISV/DISU), dimensions, cell sizes
- **TDIS**: Stress periods, time units, total simulation time
- **Packages**: Full inventory with file paths
- **Outputs**: HDS, CBC, LST, OBS presence
- **Solver**: IMS convergence parameters

Facts are automatically injected into chat context for grounded responses.

## Provider Interface

LLM providers live in `gw/llm/providers/`. Currently supports OpenAI; Claude/Azure can be added without changing the engine.
