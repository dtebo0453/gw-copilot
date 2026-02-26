from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import json
import pandas as pd
import numpy as np
from pydantic import ValidationError
from gw.llm.schemas import DraftResponse
from gw.llm.registry import get_provider

def _write_debug_outputs(raw: Any, outdir: Optional[str]) -> None:
    """Write raw LLM output to disk for debugging."""
    if not outdir:
        return

    try:
        outp = Path(outdir)
        outp.mkdir(parents=True, exist_ok=True)

        # JSON dump (best effort)
        try:
            (outp / "raw_llm_output.json").write_text(
                json.dumps(raw, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        # Pretty text dump
        (outp / "raw_llm_output.txt").write_text(
            str(raw), encoding="utf-8"
        )
    except Exception:
        # Never fail because debug logging failed
        pass

def summarize_inputs(inputs_dir: Optional[str]) -> Dict[str, Any]:
    if not inputs_dir:
        return {}
    p = Path(inputs_dir)
    if not p.exists():
        return {"warning": f"inputs_dir not found: {inputs_dir}"}

    summary: Dict[str, Any] = {"files": []}

    idom = p / "idomain.csv"
    if idom.exists():
        arr = pd.read_csv(idom, header=None).values
        summary["idomain"] = {"nrow": int(arr.shape[0]), "ncol": int(arr.shape[1]), "active_fraction": float(np.mean(arr > 0))}
        summary["files"].append("idomain.csv")

    for fname, fields in [
        ("wells.csv", ["q"]),
        ("ghb.csv", ["bhead", "cond"]),
        ("drn.csv", ["elev", "cond"]),
        ("riv.csv", ["stage", "cond", "rbot"]),
        ("obs_heads.csv", ["obs_head"]),
    ]:
        f = p / fname
        if not f.exists():
            continue
        df = pd.read_csv(f)
        block: Dict[str, Any] = {"n": int(len(df))}
        if "per" in df.columns:
            block["periods"] = sorted({int(x) for x in df["per"].dropna().unique().tolist()})
        for col in fields:
            if col in df.columns and len(df) > 0:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) > 0:
                    block[f"{col}_min"] = float(s.min())
                    block[f"{col}_max"] = float(s.max())
        summary[fname.replace(".csv","")] = block
        summary["files"].append(fname)

    return summary

def _coerce_draft_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common model output shapes into DraftResponse shape.

    Accepts these variants:
      - Proper: {"config": {...}, "questions": [...], "conceptual_model": "..."}
      - Flattened config: {"model_name":..., "grid":..., "questions":..., "conceptual_model":...}
      - questions as objects: [{"id":1,"text":"..."}, ...]
    """
    # If the model returned the config fields at top level, wrap them into "config".
    if "config" not in raw:
        likely_cfg_keys = {"model_name", "workspace", "grid", "periods", "ic", "npf", "sto", "recharge", "inputs", "output_control"}
        if any(k in raw for k in likely_cfg_keys):
            cfg = {k: raw[k] for k in list(raw.keys()) if k in likely_cfg_keys or k in {"time_units","length_units","solver"}}
            # Keep only cfg keys that exist
            cfg = {k: v for k, v in cfg.items()}
            raw["config"] = cfg
            # Remove them from top-level to avoid confusion
            for k in list(raw.keys()):
                if k in cfg and k != "config":
                    raw.pop(k, None)

    # If model returns questions as objects like {"id":1,"text":"..."}, coerce to list[str]
    q = raw.get("questions")
    if isinstance(q, dict):
        raw["questions"] = list(q.values())
    elif isinstance(q, list):
        fixed = []
        for item in q:
            if isinstance(item, str):
                fixed.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    fixed.append(item["text"])
                elif "question" in item and isinstance(item["question"], str):
                    fixed.append(item["question"])
                else:
                    fixed.append(json.dumps(item))
            else:
                fixed.append(str(item))
        raw["questions"] = fixed

    # Ensure conceptual_model key exists (even if empty) so we can write the file.
    # if "conceptual_model" not in raw or raw.get("conceptual_model") is None:
    #     raw["conceptual_model"] = ""
    cm = raw.get("conceptual_model")
    if isinstance(cm, dict):
        # Common pattern: {"memo": "..."} or {"text": "..."}
        if "memo" in cm and isinstance(cm["memo"], str):
            raw["conceptual_model"] = cm["memo"]
        elif "text" in cm and isinstance(cm["text"], str):
            raw["conceptual_model"] = cm["text"]
        else:
            raw["conceptual_model"] = json.dumps(cm, indent=2)
    elif cm is None:
        raw["conceptual_model"] = ""    
        

    return raw


def draft_config(prompt: str, provider: str="openai", model: Optional[str]=None,
                 base_config_path: Optional[str]=None, inputs_dir: Optional[str]=None) -> DraftResponse:
    base_cfg = json.loads(Path(base_config_path).read_text(encoding="utf-8")) if base_config_path else None
    context: Dict[str, Any] = {
        "base_config": base_cfg,
        "inputs_summary": summarize_inputs(inputs_dir),
        "constraints": {"no_raw_data": True, "produce_builder_compatible_config": True},
    }
    schema = DraftResponse.model_json_schema()
    prov = get_provider(provider, model=model)
    raw = prov.draft(prompt=prompt, context=context, schema=schema)
    raw = prov.draft(prompt=prompt, context=context, schema=schema)
    
    # Always write raw output for debugging
    _write_debug_outputs(raw, outdir=Path(base_config_path).parent / "llm")
    
    raw = _coerce_draft_payload(raw)
    try:
        return DraftResponse.model_validate(raw)
    except ValidationError as e:
    	raise RuntimeError(
        		"LLM output failed schema validation.\n"
        		"The model did not return the expected top-level keys:\n"
        		"  - config\n"
        		"  - questions\n"
        		"  - conceptual_model\n\n"
        		"Try re-running with a prompt like:\n"
        		"  Return JSON with top-level keys: config, questions, conceptual_model.\n"
        		"  Put ALL model fields under config.\n\n"
        		f"Validation errors:\n{e}"
    	) from e