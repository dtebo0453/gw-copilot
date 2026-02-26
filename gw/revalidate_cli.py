from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from gw.revalidate import run_revalidate


def register_revalidate(app: typer.Typer) -> None:
    @app.command("revalidate")
    def revalidate_cmd(
        inputs_dir: str = typer.Option(
            ...,
            "--inputs-dir",
            help="Inputs directory containing stress CSVs and config.json.",
        ),
        config: Optional[str] = typer.Option(
            None,
            "--config",
            help="Path to config.json (base config). Defaults to <inputs-dir>/config.json.",
        ),
        workspace: Optional[str] = typer.Option(
            None,
            "--workspace",
            help="Workspace dir (default <inputs-dir>/workspace). Artifacts default to <workspace>/run_artifacts.",
        ),
        out_dir: Optional[str] = typer.Option(
            None,
            "--out-dir",
            help="Where to write artifacts. Overrides <workspace>/run_artifacts.",
        ),
        previous_json: Optional[str] = typer.Option(
            None,
            "--previous-json",
            help="Path to previous stress_validation.json for diffing. Defaults to <workspace>/run_artifacts/stress_validation.json if present.",
        ),
    ):
        """Re-run deterministic stress input validation and write new artifacts."""
        inputs_path = Path(inputs_dir)
        if not inputs_path.exists():
            raise typer.BadParameter(f"inputs-dir does not exist: {inputs_dir}")

        if workspace is None:
            workspace = str(inputs_path / "workspace")

        if config is None:
            cfg_candidate = inputs_path / "config.json"
            if cfg_candidate.exists():
                config = str(cfg_candidate)
            elif (Path(workspace) / "config.json").exists():
                config = str(Path(workspace) / "config.json")
            else:
                raise typer.BadParameter("Could not find config.json. Provide --config explicitly.")

        cfg_path = Path(config)
        if not cfg_path.exists():
            raise typer.BadParameter(f"Config file not found: {config}")

        base_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        if previous_json is None and workspace:
            candidate = Path(workspace) / "run_artifacts" / "stress_validation.json"
            if candidate.exists():
                previous_json = str(candidate)

        res = run_revalidate(
            inputs_dir=str(inputs_path),
            base_config=base_cfg,
            workspace=workspace,
            out_dir=out_dir,
            previous_json_path=previous_json,
        )

        typer.echo(res.summary_line)
        typer.echo(f"Stress MD:   {res.stress_report_md_path}")
        if res.stress_report_json_path:
            typer.echo(f"Stress JSON: {res.stress_report_json_path}")
        if res.diff_md_path:
            typer.echo(f"Diff MD:     {res.diff_md_path}")

        if not res.ok:
            raise typer.Exit(code=res.exit_code)
