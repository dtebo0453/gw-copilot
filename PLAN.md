# Implementation Plan: 3D Visualization Enhancements + Chat Improvements

## Overview

Two feature areas:
1. **3D Visualization** — Full 3D hex-cell model with X/Y/Z slicing and all grid properties
2. **Chat Routing** — Deterministic semantic hints so the LLM planner picks the right files on first ask

---

## Part A: Full 3D Visualization

### Current State
- Backend (`viz.py`) builds a single 2D quad surface per layer
- Frontend (`Model3DTab.tsx`) renders one Actor with one clipping plane (Z only)
- Only DIS GRIDDATA properties exposed (top, botm, idomain)
- Grayscale color map

### A1. Backend — Generic Package GRIDDATA Parser

**File: `gw/api/viz.py`** — New function `_parse_package_griddata()`

Reuse the existing `_parse_griddata_arrays` infrastructure to read GRIDDATA from NPF, STO, IC files:

```
_PACKAGE_ARRAYS = {
    "NPF":  {"K": nlay*nrow*ncol, "K33": nlay*nrow*ncol, "ICELLTYPE": nlay*nrow*ncol},
    "STO":  {"SS": nlay*nrow*ncol, "SY": nlay*nrow*ncol, "ICONVERT": nlay*nrow*ncol},
    "IC":   {"STRT": nlay*nrow*ncol},
}
```

- Parse the `.nam` file (using existing `_parse_nam_packages` from model_snapshot.py or a simpler local version) to discover package file paths
- For each known package with GRIDDATA (NPF, STO, IC), read its GRIDDATA block
- Store arrays on `DisInfo.extras` dict (e.g., `extras["k"]`, `extras["ss"]`, etc.)
- Cache alongside the DIS data; invalidate when any package file mtime changes

### A2. Backend — Full 3D Exterior Shell Mesh Builder

**File: `gw/api/viz.py`** — New function `_build_full_3d_mesh()`

Industry-standard approach: build exposed faces of active hexahedral cells.

Algorithm:
1. Compute vertex grid: `(nlay+1) × (nrow+1) × (ncol+1)` vertex positions
   - X from cumulative DELR, Y from cumulative DELC
   - Z[0] = TOP, Z[k] = BOTM[k-1] for k=1..nlay
   - Apply rotation + origin offset
2. Build IDOMAIN mask: `(nlay, nrow, ncol)` — cells where idomain > 0 are active
3. For each active cell, check 6 neighbors (±layer, ±row, ±col):
   - If neighbor is out-of-bounds or inactive → emit that face as a quad
   - This produces only the exterior shell (~20K faces for a 100×100×5 grid vs 300K for all hexahedra)
4. Return `{points: [...], polys: [...], cell_ids: [...], cell_count: N}`
   - `cell_ids` maps each emitted face back to its (lay,row,col) cell for scalar lookup

Performance optimizations:
- Use numpy vectorization for vertex grid construction
- Python loop only over active cells for face emission (unavoidable but fast for typical grid sizes)
- For very large grids (>500K cells), log a warning but still proceed

### A3. Backend — Update `/viz/summary` Endpoint

Expose all discovered properties:
- DIS base: top, botm, idomain
- DIS extras (if any)
- NPF: k, k33, icelltype (if NPF file found)
- STO: ss, sy (if STO file found)
- IC: strt (if IC file found)

Return friendly labels in the properties list:
```json
{"key": "k", "label": "Hydraulic Conductivity (K)", "source": "npf", "kind": "cell"}
```

### A4. Backend — Update `/viz/scalars` Endpoint

Generalize `_cell_scalars_for_surface()` into `_cell_scalars()`:
- For mode="top_surface": existing behavior (one layer slice)
- For mode="full_3d": return full 3D array indexed by cell_ids from the mesh
  - Scalars array must align with the face→cell mapping from `_build_full_3d_mesh`
  - Each face gets the scalar value of its parent cell

### A5. Backend — Update `/viz/mesh` Endpoint

- Accept `mode="full_3d"` in addition to `mode="top_surface"`
- Route to `_build_full_3d_mesh()` for full_3d mode
- Cache the result per session like top_surface meshes

### A6. Frontend — Multi-Axis Clipping Planes

**File: `ui/src/components/Model3DTab.tsx`**

Add three clipping planes (X, Y, Z) to the mapper:
- Create `vtkPlane` instances for X-normal, Y-normal, Z-normal
- Each has a slider (0–100%) controlling origin position along that axis
- `mapper.addClippingPlane(planeX)`, `.addClippingPlane(planeY)`, `.addClippingPlane(planeZ)`
- Update all three planes when any slider changes
- Default: all at 0% (no clipping)

UI: Three sliders labeled "Clip X", "Clip Y", "Clip Z" in the controls row.

### A7. Frontend — Scientific Colormap (Viridis)

Replace the grayscale 2-point LUT with a viridis-like colormap:
- Define ~10 RGB control points matching the viridis gradient
- Apply to `vtkColorTransferFunction` via `addRGBPoint()`
- This is the industry standard for scientific visualization (used by matplotlib, ParaView, ModelMuse)

### A8. Frontend — View Mode Toggle + Property Labels

- Add a toggle/select: "Surface" vs "Full 3D" (default to Full 3D)
- When "Full 3D" selected, call `vizMesh` with `mode="full_3d"` (no layer param)
- When "Surface" selected, existing behavior with layer selector
- Property dropdown: show friendly labels from summary (e.g., "Hydraulic Conductivity (K)" not just "k")
- Layer selector: only visible/enabled in Surface mode

### A9. Frontend — Update Header Text

Replace "MVP: DIS top-surface mesh + cell scalars" with cleaner description.

---

## Part B: Chat Semantic Routing

### Current State
- Stage 1 planner system prompt says "Prefer MODFLOW control/input files first" but gives no concept→file mapping
- `mf6_filetype_knowledge.py` has descriptions but isn't used in the planner prompt
- Result: LLM asks for clarification instead of reading the right file

### B1. Add Semantic Routing Hints to Planner Prompt

**File: `gw/api/plots.py`** — Modify `system_plan` construction (~line 2184)

Build a deterministic hint block injected into the system prompt:

```python
_CONCEPT_TO_FILE_HINTS = """
MODFLOW 6 Property-to-File Reference (use this to select files):
- Hydraulic conductivity, K, HK, K33, permeability, anisotropy → .npf file
- Storage, specific storage, SS, specific yield, SY, storativity → .sto file
- Initial heads, starting heads, STRT → .ic file
- Recharge, recharge rate → .rch or .rcha file
- Grid dimensions, NLAY, NROW, NCOL, cell size, DELR, DELC, TOP, BOTM, layer elevations → .dis file
- Well pumping rates, injection, extraction → .wel file
- Constant head boundaries, fixed head → .chd file
- River stage, river conductance → .riv file
- Drain elevation, drain conductance → .drn file
- General head boundary → .ghb file
- Time discretization, stress periods, NPER, PERLEN → .tdis file (or mfsim.tdis)
- Solver settings, convergence, OUTER_MAXIMUM, INNER_MAXIMUM → .ims file
- Output control, save heads, save budget → .oc file
- Model run log, warnings, errors, convergence history → .lst file
- Package inventory, which files are used → .nam file

IMPORTANT: When the question mentions any of the above concepts, ALWAYS include the corresponding file in read_requests. Do NOT ask for clarification about which file to read — use this reference.
"""
```

Append this to `system_plan` before the closing rules. This is:
- **Deterministic** — no extra LLM calls, just a string concat
- **Leverages existing knowledge** — mirrors `mf6_filetype_knowledge.py` but oriented for routing
- **Actionable** — tells the LLM exactly what to do

### B2. Enrich File Index with Type Hints

In the `_build_file_index` function, add the `mf6_filetype_knowledge.py` description to each file entry:

```python
from gw.llm.mf6_filetype_knowledge import guess_filetype

# In file index building:
info = guess_filetype(p.name)
if info:
    entry["kind"] = info.kind
    entry["purpose"] = info.purpose
```

This gives the planner structured knowledge like:
```json
{"path": "aoi_model.npf", "ext": ".npf", "kind": "Node Property Flow (NPF) package", "purpose": "Defines hydraulic properties..."}
```

---

## Implementation Order

1. **B1 + B2** (Chat routing) — Smallest change, highest impact, ~30 lines changed
2. **A1** (Package GRIDDATA parser) — Foundation for new properties
3. **A2** (Full 3D mesh builder) — Core new backend feature
4. **A3 + A4 + A5** (API updates) — Wire up new backend features
5. **A7** (Viridis colormap) — Quick frontend improvement
6. **A6** (Multi-axis clipping) — Frontend clipping planes
7. **A8 + A9** (UI controls + labels) — Final frontend polish

---

## Performance Considerations

- **JSON payload size**: A 200×200×10 grid with exterior shell ≈ 40K faces ≈ 800K floats for points + 200K ints for polys. At ~8 bytes/number in JSON ≈ 8MB. Acceptable for local API. Could add msgpack later if needed.
- **Backend mesh build time**: Vertex grid via numpy is instant. Face emission loop in Python: ~0.1s for 400K cells. Acceptable.
- **Frontend rendering**: vtk.js handles 40K quads easily. WebGL limit is ~millions of triangles.
- **Cache invalidation**: Track mtimes of DIS + all package files. If any change, invalidate session.

## Backwards Compatibility

- `mode="top_surface"` continues to work exactly as before
- New `mode="full_3d"` is additive
- All existing API contracts preserved
- Frontend defaults to Full 3D but Surface mode remains available
