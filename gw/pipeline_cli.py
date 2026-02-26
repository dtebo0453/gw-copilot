from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from gw.pipeline import run_pipeline


def register_pipeline(app: typer.Typer) -> None:
    @app.command("pipeline")
    def pipeline_cmd(
        inputs_dir: str = typer.Option(..., "--inputs-dir", help="Inputs directory containing stress CSVs and config.json."),
        config: Optional[str] = typer.Option(None, "--config", help="Path to config.json. Defaults to <inputs-dir>/config.json."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace dir (default <inputs-dir>/workspace)."),
        out_dir: Optional[str] = typer.Option(None, "--out-dir", help="Where to write artifacts. Overrides <workspace>/run_artifacts."),
        apply: bool = typer.Option(False, "--apply", help="Actually apply changes. If omitted, runs in dry-run mode."),
        provider: str = typer.Option("openai", "--provider", help="LLM provider key (e.g. openai)."),
        model: Optional[str] = typer.Option(None, "--model", help="LLM model override."),
        max_actions: int = typer.Option(8, "--max-actions", help="Maximum actions in FixPlan."),
    ):
        """Run validate -> suggest-fix-plan -> apply-fixes -> revalidate as one command."""
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

        res = run_pipeline(
            inputs_dir=str(inputs_path),
            base_config=base_cfg,
            workspace=workspace,
            out_dir=out_dir,
            apply=apply,
            provider=provider,
            model=model,
            max_actions=max_actions,
        )

        typer.echo(f"{res.message} step={res.step} ok={res.ok}")
        typer.echo(f"Artifacts dir: {res.artifacts_dir}")
        typer.echo(f"Dashboard:     {res.dashboard_json}")
        if not res.ok:
            raise typer.Exit(code=res.exit_code)
