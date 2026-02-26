from __future__ import annotations
import json, os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import flopy

from gw.utils.read_inputs import read_idomain_csv, read_boundary_csv

def _as_layered(x: Union[float, int, List[float]], nlay: int) -> np.ndarray:
    if isinstance(x, list):
        if len(x) != nlay:
            raise ValueError(f"Expected length {nlay}, got {len(x)}")
        return np.array([float(v) for v in x], dtype=float)
    return np.full(nlay, float(x), dtype=float)

@dataclass
class Cfg:
    d: Dict[str, Any]
    @property
    def model_name(self) -> str: return self.d["model_name"]
    @property
    def workspace(self) -> str: return self.d["workspace"]
    @property
    def grid(self) -> Dict[str, Any]: return self.d["grid"]
    @property
    def periods(self) -> Dict[str, Any]: return self.d["periods"]
    @property
    def ic(self) -> Dict[str, Any]: return self.d["ic"]
    @property
    def npf(self) -> Dict[str, Any]: return self.d["npf"]
    @property
    def sto(self) -> Dict[str, Any]: return self.d["sto"]
    @property
    def recharge(self) -> Dict[str, Any]: return self.d["recharge"]
    @property
    def inputs(self) -> Dict[str, Any]: return self.d.get("inputs", {})
    @property
    def output_control(self) -> Dict[str, Any]: return self.d.get("output_control", {})
    @property
    def solver(self) -> Dict[str, Any]: return self.d.get("solver", {"complexity":"SIMPLE"})
    @property
    def time_units(self) -> str: return self.d.get("time_units","days")
    @property
    def length_units(self) -> str: return self.d.get("length_units","meters")

def load_cfg(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        return Cfg(json.load(f))

def build_site_model(cfg: Cfg) -> str:
    os.makedirs(cfg.workspace, exist_ok=True)
    nlay, nrow, ncol = int(cfg.grid["nlay"]), int(cfg.grid["nrow"]), int(cfg.grid["ncol"])
    idomain2d = read_idomain_csv(cfg.inputs["idomain_csv"], nrow, ncol)
    idomain = np.repeat(idomain2d[np.newaxis,:,:], nlay, axis=0)

    sim = flopy.mf6.MFSimulation(sim_name=cfg.model_name, exe_name="mf6", sim_ws=cfg.workspace)

    flopy.mf6.ModflowTdis(
        sim, time_units=cfg.time_units, nper=cfg.periods["nper"],
        perioddata=list(zip(cfg.periods["perlen"], cfg.periods["nstp"], cfg.periods["tsmult"]))
    )

    flopy.mf6.ModflowIms(
        sim, print_option="SUMMARY", complexity=cfg.solver.get("complexity","SIMPLE"),
        outer_maximum=100, inner_maximum=300, inner_dvclose=1e-6, outer_dvclose=1e-6
    )

    gwf = flopy.mf6.ModflowGwf(sim, modelname=cfg.model_name, save_flows=True)

    botm = np.array(cfg.grid["botm"], dtype=float)
    if botm.shape[0] != nlay:
        raise ValueError("grid.botm must have length nlay")

    flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=nrow, ncol=ncol,
        delr=float(cfg.grid["delr"]), delc=float(cfg.grid["delc"]),
        top=float(cfg.grid["top"]), botm=botm, idomain=idomain,
        length_units=cfg.length_units
    )

    flopy.mf6.ModflowGwfic(gwf, strt=float(cfg.ic["strt"]))

    k = _as_layered(cfg.npf["k"], nlay)
    k33 = _as_layered(cfg.npf["k33"], nlay)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=int(cfg.npf.get("icelltype",1)), k=k, k33=k33,
                            save_specific_discharge=True)

    sy = _as_layered(cfg.sto["sy"], nlay)
    ss = _as_layered(cfg.sto["ss"], nlay)
    flopy.mf6.ModflowGwfsto(
        gwf, iconvert=1, sy=sy, ss=ss,
        steady_state=cfg.periods["steady"],
        transient=[not s for s in cfg.periods["steady"]]
    )

    rch = np.full((nrow,ncol), float(cfg.recharge["rate"]), dtype=float)
    flopy.mf6.ModflowGwfrcha(gwf, recharge=rch)

    def _group_to_spd(df: pd.DataFrame, kind: str):
        spd: Dict[int, list] = {}
        if df.empty:
            return spd
        for per, g in df.groupby("per"):
            per_i = int(per)
            rows=[]
            if kind=="WEL":
                for _, r in g.iterrows():
                    rows.append(((int(r.lay),int(r.row),int(r.col)), float(r.q)))
            elif kind=="GHB":
                for _, r in g.iterrows():
                    rows.append(((int(r.lay),int(r.row),int(r.col)), float(r.bhead), float(r.cond)))
            elif kind=="DRN":
                for _, r in g.iterrows():
                    rows.append(((int(r.lay),int(r.row),int(r.col)), float(r.elev), float(r.cond)))
            elif kind=="RIV":
                for _, r in g.iterrows():
                    rows.append(((int(r.lay),int(r.row),int(r.col)), float(r.stage), float(r.cond), float(r.rbot)))
            spd[per_i]=rows
        return spd

    wells = read_boundary_csv(cfg.inputs.get("wells_csv"))
    if not wells.empty:
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data=_group_to_spd(wells,"WEL"))

    ghb = read_boundary_csv(cfg.inputs.get("ghb_csv"))
    if not ghb.empty:
        flopy.mf6.ModflowGwfghb(gwf, stress_period_data=_group_to_spd(ghb,"GHB"))

    drn = read_boundary_csv(cfg.inputs.get("drn_csv"))
    if not drn.empty:
        flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=_group_to_spd(drn,"DRN"))

    riv = read_boundary_csv(cfg.inputs.get("riv_csv"))
    if not riv.empty:
        flopy.mf6.ModflowGwfriv(gwf, stress_period_data=_group_to_spd(riv,"RIV"))

    obsdf = read_boundary_csv(cfg.inputs.get("obs_heads_csv"))
    if not obsdf.empty:
        obs_file = f"{cfg.model_name}.obs.csv"
        obslist=[]
        for _, r in obsdf.iterrows():
            obslist.append((str(r["name"]), "HEAD", (int(r["lay"]), int(r["row"]), int(r["col"]))))
        flopy.mf6.ModflowUtlobs(gwf, pname="obs", digits=10, print_input=True, continuous={obs_file: obslist})

    saverec=[]; printrec=[]
    if cfg.output_control.get("save_head", True):
        saverec.append(("HEAD","ALL"))
        printrec.append(("HEAD","LAST"))
    if cfg.output_control.get("save_budget", True):
        saverec.append(("BUDGET","ALL"))
        printrec.append(("BUDGET","LAST"))

    flopy.mf6.ModflowGwfoc(
        gwf, saverecord=saverec, printrecord=printrec,
        head_filerecord=[f"{cfg.model_name}.hds"],
        budget_filerecord=[f"{cfg.model_name}.cbb"]
    )

    sim.write_simulation()
    return cfg.workspace

def run_mf6(workspace: str) -> bool:
    sim = flopy.mf6.MFSimulation.load(sim_ws=workspace, exe_name="mf6")
    ok, _ = sim.run_simulation()
    return bool(ok)

def compute_obs_residuals(cfg_path: str) -> str:
    cfg = load_cfg(cfg_path)
    obs_out_path = os.path.join(cfg.workspace, f"{cfg.model_name}.obs.csv")
    inp = cfg.inputs.get("obs_heads_csv")
    if not inp or not os.path.exists(obs_out_path):
        return ""
    sim_obs = pd.read_csv(obs_out_path)
    user_obs = pd.read_csv(inp)
    last = sim_obs.iloc[-1].to_dict()
    rows=[]
    for _, r in user_obs.iterrows():
        name=str(r["name"])
        obs=float(r["obs_head"])
        simv=float(last.get(name, float("nan")))
        rows.append({"name":name,"obs_head":obs,"sim_head":simv,"residual":simv-obs})
    out = os.path.join(cfg.workspace, "obs_residuals.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
