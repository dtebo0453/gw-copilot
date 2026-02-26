"""Entry point for running CLI commands as a subprocess.

Usage::

    python -m gw.cli_entry <command> [args...]

This replaces the legacy ``python gw_cli.py <command>`` pattern and works
correctly when GW Copilot is installed via pip.
"""
from gw.cli import app

if __name__ == "__main__":
    app()
