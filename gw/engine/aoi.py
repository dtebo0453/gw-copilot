from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon, Point
from shapely.prepared import prep

@dataclass
class AOIGrid:
    nrow: int
    ncol: int
    delr: float
    delc: float
    x0: float
    y0: float
    top: float
    botm: List[float]

def load_geojson_polygon(path: str) -> Polygon:
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    geom=None
    if gj.get("type")=="FeatureCollection":
        for f in gj.get("features", []):
            g=f.get("geometry")
            if g and g.get("type") in ("Polygon","MultiPolygon"):
                geom=g; break
    elif gj.get("type")=="Feature":
        g=gj.get("geometry")
        if g and g.get("type") in ("Polygon","MultiPolygon"):
            geom=g
    elif gj.get("type") in ("Polygon","MultiPolygon"):
        geom=gj
    if geom is None:
        raise ValueError("No Polygon/MultiPolygon found in GeoJSON")
    shp = shape(geom)
    if isinstance(shp, Polygon): return shp
    if isinstance(shp, MultiPolygon): return shp.union_all()
    raise ValueError("Unsupported geometry type")

def grid_from_polygon(poly: Polygon, cellsize: float, nlay: int, top: float, botm: List[float], pad_cells: int=2) -> AOIGrid:
    minx, miny, maxx, maxy = poly.bounds
    pad = cellsize * pad_cells
    minx -= pad; miny -= pad; maxx += pad; maxy += pad
    width = maxx - minx; height = maxy - miny
    ncol = int(np.ceil(width / cellsize))
    nrow = int(np.ceil(height / cellsize))
    return AOIGrid(nrow=nrow, ncol=ncol, delr=float(cellsize), delc=float(cellsize),
                   x0=float(minx), y0=float(miny), top=float(top), botm=[float(b) for b in botm])

def rasterize_idomain(poly: Polygon, grid: AOIGrid) -> np.ndarray:
    prepared = prep(poly)
    idom = np.zeros((grid.nrow, grid.ncol), dtype=int)
    for i in range(grid.nrow):
        y = grid.y0 + (grid.nrow - i - 0.5) * grid.delc
        for j in range(grid.ncol):
            x = grid.x0 + (j + 0.5) * grid.delr
            if prepared.contains(Point(x,y)):
                idom[i,j]=1
    return idom

def write_idomain_csv(idomain2d: np.ndarray, out_csv: str) -> None:
    p = Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join([",".join(map(str,row.tolist())) for row in idomain2d]), encoding="utf-8")

def make_starter_config(model_name: str, workspace: str, grid: AOIGrid, nlay: int,
                        periods_nper: int=1, perlen: float=365.0, steady: bool=False) -> Dict[str, Any]:
    periods = {"nper": periods_nper, "perlen":[perlen]*periods_nper, "nstp":[1]*periods_nper,
               "tsmult":[1.0]*periods_nper, "steady":[steady]*periods_nper}
    return {
        "model_name": model_name,
        "workspace": workspace,
        "time_units": "days",
        "length_units": "meters",
        "grid": {"nlay": nlay, "nrow": grid.nrow, "ncol": grid.ncol, "delr": grid.delr, "delc": grid.delc,
                 "top": grid.top, "botm": grid.botm},
        "periods": periods,
        "solver": {"complexity":"SIMPLE"},
        "ic": {"strt": float(grid.top) - 5.0},
        "npf": {"icelltype": 1, "k": 10.0, "k33": 2.0},
        "sto": {"sy": 0.12, "ss": 1e-5},
        "recharge": {"rate": 0.0001},
        "inputs": {"idomain_csv": ""},
        "output_control": {"save_head": True, "save_budget": True},
    }
