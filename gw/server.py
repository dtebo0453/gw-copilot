"""GW Copilot server launcher.

Entry point for the ``gw-copilot`` console script.
Starts the FastAPI server and opens the default browser.

Usage::

    gw-copilot                  # start on port 8000, open browser
    gw-copilot --port 9000      # custom port
    gw-copilot --no-browser     # headless (CI / remote)
    gw-copilot --reload         # dev mode with auto-reload
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser


def _open_browser(url: str, delay: float = 1.5) -> None:
    """Open *url* in the default browser after a short delay."""

    def _go() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass  # best-effort; don't crash the server

    t = threading.Thread(target=_go, daemon=True)
    t.start()


def main() -> None:
    """CLI entry point for ``gw-copilot``."""

    parser = argparse.ArgumentParser(
        prog="gw-copilot",
        description="Start the GW Copilot server and open the UI.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GW_PORT", "8000")),
        help="Port to listen on (default: 8000, or GW_PORT env var)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open the browser automatically",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    # Import version lazily so argparse --help is fast
    from gw import __version__

    print(f"\n  GW Copilot v{__version__}")
    print(f"  Server:  {url}")
    print(f"  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        _open_browser(url)

    import uvicorn

    uvicorn.run(
        "gw.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
