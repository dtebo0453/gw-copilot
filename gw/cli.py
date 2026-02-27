from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from gw.engine.site_builder import load_cfg, build_site_model, run_mf6, compute_obs_residuals
from gw.engine.aoi import (
    load_geojson_polygon,
    grid_from_polygon,
    rasterize_idomain,
    write_idomain_csv,
    make_starter_config,
)
from gw.engine.critique import critique_model, write_report
from gw.fix_apply import apply_fix_plan
from gw.llm.fix_plan import FixPlan  # adjust import to where FixPlan lives
from gw.revalidate import run_revalidate

app = typer.Typer(add_completion=False, help="GW Hybrid CLI V1 (deterministic engine + helpers)")
console = Console()

from gw.revalidate_cli import register_revalidate
register_revalidate(app)

from gw.pipeline_cli import register_pipeline
register_pipeline(app)


def _resolve_fix_plan_path(inputs_dir: str, fix_plan: Optional[str]) -> Path:
    """
    Resolve fix_plan.json path.
    Priority:
      1) explicit --fix-plan
      2) <inputs_dir>/workspace/run_artifacts/fix_plan.json
      3) <inputs_dir>/run_artifacts/fix_plan.json
      4) ./fix_plan.json (cwd)
    """
    if fix_plan:
        p = Path(fix_plan)
        if p.is_dir():
            p = p / "fix_plan.json"
        return p

    base = Path(inputs_dir)
    candidates = [
        base / "workspace" / "run_artifacts" / "fix_plan.json",
        base / "run_artifacts" / "fix_plan.json",
        Path("fix_plan.json"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # Return the most-likely default even if missing (for a nicer error message)
    return candidates[0]


@app.command("build")
def build(
    config: str = typer.Option(..., "--config", "-c"),
    run: bool = typer.Option(False, "--run"),
    residuals: bool = typer.Option(False, "--residuals"),
):
    cfg = load_cfg(config)
    ws = build_site_model(cfg)
    console.print(f"[green]Wrote model to:[/green] {ws}")
    if run:
        ok = run_mf6(ws)
        console.print(f"[bold]MF6 run success:[/bold] {ok}")
    if residuals:
        out = compute_obs_residuals(config)
        if out:
            console.print(f"[green]Wrote residuals:[/green] {out}")
        else:
            console.print("[yellow]No residuals written (missing obs input or MF6 obs output).[/yellow]")


@app.command("build-aoi")
def build_aoi(
    aoi: str = typer.Option(..., "--aoi"),
    outdir: str = typer.Option(..., "--outdir"),
    cellsize: float = typer.Option(..., "--cellsize"),
    nlay: int = typer.Option(3, "--nlay"),
    top: float = typer.Option(30.0, "--top"),
    botm: List[float] = typer.Argument(...),
    model_name: str = typer.Option("aoi_model", "--model-name"),
    pad_cells: int = typer.Option(2, "--pad-cells"),
):
    if len(botm) != nlay:
        raise typer.BadParameter(f"botm must have {nlay} values")
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)
    poly = load_geojson_polygon(aoi)
    grid = grid_from_polygon(poly, cellsize=cellsize, nlay=nlay, top=top, botm=botm, pad_cells=pad_cells)
    idom = rasterize_idomain(poly, grid)
    idomain_path = outp / "idomain.csv"
    write_idomain_csv(idom, str(idomain_path))
    ws = str((outp / "workspace").as_posix())
    cfg = make_starter_config(model_name=model_name, workspace=ws, grid=grid, nlay=nlay)
    cfg["inputs"]["idomain_csv"] = str(idomain_path.as_posix())
    cfg_path = outp / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    table = Table(title="AOI Build Outputs")
    table.add_column("Item")
    table.add_column("Path / Value")
    table.add_row("config", str(cfg_path))
    table.add_row("idomain", str(idomain_path))
    table.add_row("grid", f"{grid.nrow} x {grid.ncol} @ {cellsize}")
    table.add_row("active fraction", f"{float(idom.mean()):.3f}")
    console.print(table)


@app.command("critique")
def critique_cmd(
    config: str = typer.Option(..., help="Path to model config JSON"),
    inputs_dir: str = typer.Option(None, help="Run directory with idomain.csv"),
    outdir: str = typer.Option(None, help="Output directory for QA report"),
):
    from gw.qa.engine import run_qa

    report = run_qa(config, inputs_dir, outdir)
    if report["summary"]["errors"] > 0:
        raise typer.Exit(code=2)


@app.command("llm-draft-config")
def llm_draft_config_cmd(
    prompt: str = typer.Option(..., "--prompt"),
    outdir: str = typer.Option(..., "--outdir"),
    base_config: Optional[str] = typer.Option(None, "--base-config"),
    inputs_dir: Optional[str] = typer.Option(None, "--inputs-dir"),
    provider: str = typer.Option("openai", "--provider"),
    model: Optional[str] = typer.Option(None, "--model"),
):
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)
    # Lazy import: only require LLM deps when this command is invoked
    try:
        from gw.llm.draft import draft_config
    except ImportError as e:
        console.print("[red]LLM features are not available.[/red] Install deps with: pip install -r requirements.txt")
        raise typer.Exit(code=2) from e

    draft = draft_config(
        prompt=prompt,
        provider=provider,
        model=model,
        base_config_path=base_config,
        inputs_dir=inputs_dir,
    )
    (outp / "draft_config.json").write_text(json.dumps(draft.config, indent=2), encoding="utf-8")
    (outp / "questions.json").write_text(json.dumps({"questions": draft.questions}, indent=2), encoding="utf-8")
    (outp / "conceptual_model.md").write_text(draft.conceptual_model.strip() + "\n", encoding="utf-8")
    table = Table(title="LLM Draft Outputs")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("draft_config.json", str(outp / "draft_config.json"))
    table.add_row("questions.json", str(outp / "questions.json"))
    table.add_row("conceptual_model.md", str(outp / "conceptual_model.md"))
    console.print(table)


@app.command("llm-review-memo")
def llm_review_memo_cmd(
    config: str = typer.Option(..., help="Path to model config JSON (e.g., draft_config.json)"),
    qa_report: str = typer.Option(None, help="Path to qa_report.json (optional)"),
    out_md: str = typer.Option(..., help="Output markdown path for review memo"),
    provider: str = typer.Option("openai", help="LLM provider name (e.g., openai)"),
    model: str = typer.Option(None, help="Model name override (optional)"),
    style: str = typer.Option(
        "peer-review",
        help="Review style: peer-review|regulatory|calibration|numerics|executive",
    ),
):
    try:
        from gw.qa.llm_reviewer import write_llm_review_memo
    except ImportError as e:
        console.print("[red]LLM review is not available.[/red] Install deps and retry.")
        raise typer.Exit(code=2) from e

    write_llm_review_memo(
        config_path=config,
        qa_report_path=qa_report,
        out_md_path=out_md,
        provider=provider,
        model=model,
        style=style,
    )
    console.print(f"[green]Wrote LLM review memo:[/green] {out_md}")


@app.command("run-mf6")
def run_mf6_cmd(
    config: str = typer.Option(..., help="Path to model config JSON (draft or base)."),
    mf6_path: str = typer.Option(None, help="Optional path to mf6 executable (otherwise uses PATH)."),
    timeout_sec: int = typer.Option(None, help="Optional timeout in seconds."),
    outdir: str = typer.Option(
        None,
        help="Output directory for run artifacts/report (default: <workspace>/run_artifacts).",
    ),
    max_pct_discrepancy: float = typer.Option(1.0, help="Fail if abs(percent discrepancy) exceeds this value."),
):
    from gw.run.mf6_runner import run_mf6 as _run_mf6, write_run_artifacts
    from gw.run.parse_listing import default_listing_path, parse_listing
    from gw.run.report import write_run_report

    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    workspace = cfg.get("workspace")
    if not workspace:
        console.print("[red]Config missing 'workspace'.[/red]")
        raise typer.Exit(code=2)

    outp = Path(outdir) if outdir else Path(workspace) / "run_artifacts"
    outp.mkdir(parents=True, exist_ok=True)

    rc, stdout, stderr = _run_mf6(workspace=workspace, mf6_path=mf6_path, timeout_sec=timeout_sec)
    write_run_artifacts(str(outp), rc, stdout, stderr)

    lst_path = default_listing_path(workspace) or str(Path(workspace) / "mfsim.lst")
    listing = parse_listing(lst_path)

    notes: List[str] = []
    status = "PASS"
    if rc != 0:
        status = "FAIL"
        notes.append(f"mf6 returned non-zero exit code: {rc}")
    if not listing.get("normal_termination"):
        status = "FAIL"
        notes.append("Listing did not show normal termination.")
    if listing.get("convergence_failure"):
        status = "FAIL"
        notes.append("Convergence failure detected in listing.")
    pct = listing.get("percent_discrepancy")
    if pct is not None and abs(pct) > float(max_pct_discrepancy):
        status = "FAIL"
        notes.append(f"Percent discrepancy {pct} exceeds threshold {max_pct_discrepancy}.")

    report = {
        "status": status,
        "return_code": rc,
        "workspace": workspace,
        "listing": listing,
        "notes": notes,
    }
    write_run_report(report, str(outp))

    if status != "PASS":
        console.print(f"[red]MF6 run checks failed.[/red] See: {outp / 'run_report.md'}")
        raise typer.Exit(code=2)
    console.print(f"[green]MF6 run checks passed.[/green] Report: {outp / 'run_report.md'}")


@app.command("materialize-mf6")
def materialize_mf6_cmd(
    config: str = typer.Option(..., help="Path to model config JSON (draft or base)."),
    inputs_dir: str = typer.Option(None, help="Directory containing optional inputs like idomain.csv."),
    overwrite: bool = typer.Option(
        False,
        help="If set, delete existing files in workspace before writing new ones.",
    ),
):
    try:
        from gw.build.mf6_materialize import materialize_mf6
    except ImportError as e:
        console.print("[red]MF6 materialization not available.[/red]")
        raise typer.Exit(code=2) from e

    ws, files = materialize_mf6(config_path=config, inputs_dir=inputs_dir, overwrite=overwrite)
    console.print(f"[green]MF6 inputs written to workspace:[/green] {ws}")
    console.print(f"[green]Files:[/green] {', '.join(files)}")


@app.command("make-templates")
def make_templates_cmd(
    config: str = typer.Option(..., help="Path to model config JSON (draft or base)."),
    outdir: str = typer.Option(None, help="Where to write templates (default: runs/<case>/ based on workspace parent)."),
    include: str = typer.Option("rch,wel,chd", help="Comma-separated list: rch,wel,chd"),
    recharge_value: float = typer.Option(0.0, help="Fill value for recharge.csv grid (e.g., 1e-4)."),
):
    """Generate template stress CSVs (recharge.csv, wells.csv, chd.csv)."""
    inc = [x.strip() for x in include.split(",") if x.strip()]
    try:
        from gw.build.templates import write_templates
    except ImportError as e:
        console.print("[red]Template generator not available.[/red]")
        raise typer.Exit(code=2) from e

    paths = write_templates(config_path=config, outdir=outdir, include=inc, recharge_value=recharge_value)
    if not paths:
        console.print("[yellow]No templates written (check --include).[/yellow]")
        raise typer.Exit(code=0)
    console.print("[green]Wrote template CSVs:[/green]")
    for k, v in paths.items():
        console.print(f"  - {k}: {v}")


@app.command("validate-stresses")
def validate_stresses_cmd(
    config: str = typer.Option(..., help="Path to model config JSON (draft or base)."),
    inputs_dir: str = typer.Option(
        None,
        help="Directory that may contain wells.csv/chd.csv/recharge.csv/idomain.csv.",
    ),
    workspace: str = typer.Option(
        None,
        help="Workspace directory (optional). Used to locate idomain/workspace artifacts.",
    ),
    outpath: str = typer.Option(
        None,
        help="Where to write the validation report (default: <workspace>/run_artifacts/stress_validation_report.md).",
    ),
):
    """Validate stress CSVs against grid/idomain and write a report."""
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    ws_raw = workspace or cfg.get("workspace") or "."
    # Resolve workspace relative to config's directory if it's a relative path
    ws_path = Path(ws_raw)
    if not ws_path.is_absolute():
        ws_path = (Path(config).parent / ws_raw).resolve()
    ws = str(ws_path)

    # Ensure run_artifacts directory exists before writing the report
    report_dir = Path(ws) / "run_artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = outpath or str(report_dir / "stress_validation_report.md")

    from gw.build.stress_validate import (
        validate_stress_inputs,
        write_stress_validation_report,
        has_errors,
    )

    findings = validate_stress_inputs(cfg=cfg, inputs_dir=inputs_dir, workspace=ws)
    write_stress_validation_report(findings, report)

    if has_errors(findings):
        console.print(f"[red]Stress validation failed.[/red] See: {report}")
        raise typer.Exit(code=2)
    console.print(f"[green]Stress validation passed.[/green] Report: {report}")


# ✅ FIXED: single validate alias (no imports of missing modules)
@app.command("validate")
def validate_alias_cmd(
    inputs_dir: str = typer.Option(
        ...,
        "--inputs-dir",
        help="Inputs directory containing stress CSVs and config.json.",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to config.json. Defaults to <inputs-dir>/config.json.",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace",
        help="Workspace dir (optional). Defaults to cfg['workspace'] if present.",
    ),
    outpath: Optional[str] = typer.Option(
        None,
        "--outpath",
        help="Where to write the validation report (default: <workspace>/run_artifacts/stress_validation_report.md).",
    ),
):
    """
    Alias for validate-stresses (keeps CLI and GUI simpler: 'validate').
    """
    inputs_path = Path(inputs_dir)
    if not inputs_path.exists():
        raise typer.BadParameter(f"inputs-dir does not exist: {inputs_dir}")

    # Resolve config default
    if config is None:
        cfg_candidate = inputs_path / "config.json"
        if cfg_candidate.exists():
            config = str(cfg_candidate)
        elif workspace and (Path(workspace) / "config.json").exists():
            config = str(Path(workspace) / "config.json")
        else:
            raise typer.BadParameter("Could not find config.json. Provide --config explicitly.")

    # Call the existing validate-stresses command function directly
    return validate_stresses_cmd(
        config=config,
        inputs_dir=str(inputs_path),
        workspace=workspace,
        outpath=outpath,
    )


@app.command("llm-suggest-fixes")
def llm_suggest_fixes_cmd(
    config: str = typer.Option(..., help="Path to model config JSON."),
    inputs_dir: str = typer.Option(
        None,
        help="Directory containing stress CSVs and optionally idomain.csv.",
    ),
    report: str = typer.Option(
        None,
        help="Path to stress_validation_report.md (default: <workspace>/run_artifacts/stress_validation_report.md).",
    ),
    provider: str = typer.Option(None, help="LLM provider name (default: from saved config or 'openai')."),
    model: str = typer.Option(None, help="Model override (default: from saved config)."),
    outpath: str = typer.Option(
        None,
        help="Where to write fix_plan.json (default: alongside report).",
    ),
):
    # Resolve provider/model from user's saved LLM config when not explicit
    if not provider or not model:
        try:
            from gw.api.llm_config import get_active_provider, get_model
            if not provider:
                provider = get_active_provider()
            if not model:
                model = get_model()
        except Exception:
            pass
    provider = provider or "openai"

    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    ws_raw = cfg.get("workspace") or "."
    # Resolve workspace relative to the config file's directory if it's relative
    ws_path = Path(ws_raw)
    if not ws_path.is_absolute():
        ws_path = (Path(config).parent / ws_raw).resolve()
    ws = str(ws_path)
    rep = report or str(Path(ws) / "run_artifacts" / "stress_validation_report.md")
    outp = outpath or str(Path(rep).with_name("fix_plan.json"))
    from gw.llm.fix_plan import suggest_fix_plan

    plan = suggest_fix_plan(
        provider=provider,
        model=model,
        config_path=config,
        inputs_dir=inputs_dir,
        stress_validation_report_path=rep,
    )
    Path(outp).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Wrote FixPlan:[/green] {outp}")


@app.command("apply-fixes")
def apply_fixes_cmd(
    fix_plan: Optional[str] = typer.Option(
        None,
        "--fix-plan",
        "-p",
        help=(
            "Path to FixPlan JSON (file or directory). "
            "If omitted, auto-discovers under inputs-dir."
        ),
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to config.json (base config). Defaults to <inputs-dir>/config.json or workspace config if you use one.",
    ),
    inputs_dir: str = typer.Option(
        ...,
        "--inputs-dir",
        help="Inputs directory containing stress CSVs (wells.csv, chd.csv, recharge.csv, etc.)",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace",
        help="Workspace dir (used for default artifact output folder workspace/run_artifacts).",
    ),
    out_dir: Optional[str] = typer.Option(
        None,
        "--out-dir",
        help="Where to write audit artifacts (applied_fixes*.json/md). Overrides workspace/run_artifacts.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Actually apply changes. If omitted, runs in dry-run mode.",
    ),
    confirm: List[str] = typer.Option(
        [],
        "--confirm",
        help="Provide confirmation strings (repeatable). Used to allow severity='manual' actions.",
    ),
):
    """
    Apply a FixPlan deterministically (dry-run by default).
    """
    inputs_path = Path(inputs_dir)
    if not inputs_path.exists():
        raise typer.BadParameter(f"inputs-dir does not exist: {inputs_dir}")

    # ---- Defaults / auto-derivation ----
    if workspace is None:
        ws_candidate = inputs_path / "workspace"
        workspace = str(ws_candidate) if ws_candidate.exists() else None

    # Use helper resolver (supports file or directory and multiple default locations)
    plan_path = _resolve_fix_plan_path(inputs_dir=str(inputs_path), fix_plan=fix_plan)

    if not plan_path.exists():
        raise typer.BadParameter(
            "Could not find fix_plan.json. Provide --fix-plan (file or directory), "
            "or generate it with `llm-suggest-fixes`.\n"
            f"Tried default: {plan_path}"
        )

    if config is None:
        cfg_candidate = inputs_path / "config.json"
        if cfg_candidate.exists():
            config = str(cfg_candidate)
        elif workspace and (Path(workspace) / "config.json").exists():
            config = str(Path(workspace) / "config.json")
        else:
            raise typer.BadParameter("Could not find config.json. Provide --config explicitly.")

    cfg_path = Path(config)
    if not cfg_path.exists():
        raise typer.BadParameter(f"Config file not found: {config}")

    # ---- Load inputs ----
    plan = FixPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    base_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # ---- Apply (or dry-run) ----
    updated_cfg, res = apply_fix_plan(
        fix_plan=plan,
        base_config=base_cfg,
        inputs_dir=str(inputs_path),
        workspace=workspace,
        out_dir=out_dir,
        dry_run=(not apply),
        user_confirmations=confirm,
        max_actions=base_cfg.get("constraints", {}).get("max_actions", 8),
    )

    # ---- Persist updated config copy if changed ----
    if apply and res.changed_config:
        target_dir = (
            Path(out_dir)
            if out_dir
            else (Path(workspace) / "run_artifacts" if workspace else inputs_path / "run_artifacts")
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        new_cfg_path = target_dir / "config.updated.json"
        new_cfg_path.write_text(json.dumps(updated_cfg, indent=2), encoding="utf-8")
        typer.echo(f"Wrote updated config: {new_cfg_path}")

    typer.echo(f"FixPlan:    {plan_path}")
    typer.echo(f"Audit JSON: {res.audit_json_path}")
    typer.echo(f"Audit MD:   {res.audit_md_path}")


@app.command("apply-fix-plan")
def apply_fix_plan_cmd(
    plan: str = typer.Option(..., help="Path to fix_plan.json."),
    config: str = typer.Option(..., help="Path to model config JSON."),
    inputs_dir: str = typer.Option(..., help="Directory containing stress CSVs to modify."),
    workspace: str = typer.Option(None, help="Workspace override (default from config)."),
    apply_risky: bool = typer.Option(False, help="Allow risky actions like shift_indexing."),
    dry_run: bool = typer.Option(False, help="Do not modify files; just report what would happen."),
):
    from gw.build.stress_fixes import apply_fix_plan as _apply_fix_plan

    res = _apply_fix_plan(
        plan_path=plan,
        config_path=config,
        inputs_dir=inputs_dir,
        workspace=workspace,
        apply_risky=apply_risky,
        dry_run=dry_run,
    )
    console.print(f"[cyan]Stress validation report:[/cyan] {res.report_path}")
    if res.changed_files:
        console.print("[green]Modified files:[/green]")
        for p in res.changed_files:
            console.print(f"  - {p}")
    if res.notes:
        console.print("[yellow]Notes:[/yellow]")
        for n in res.notes:
            console.print(f"  - {n}")
