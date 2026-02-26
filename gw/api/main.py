from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from gw import __version__
from gw.api.routes import router as core_router
from gw.api.plots import router as plots_router
from gw.api.viz import router as viz_router
from gw.api.docs import router as docs_router
from gw.api.model_snapshot import router as snapshot_router
from gw.api.workspace_scan import router as scan_router
from gw.api.patches import router as patches_router
from gw.api.llm_config import router as llm_config_router
from gw.api.model_profile import router as profile_router

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security middleware
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies larger than *max_bytes* (default 10 MB)."""

    def __init__(self, app, max_bytes: int = 10 * 1024 * 1024):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Bundled frontend discovery
# ---------------------------------------------------------------------------

def _frontend_dir() -> Path | None:
    """Locate the bundled frontend directory.

    Checks two locations:
    1. ``gw/_frontend/`` — present when pip-installed (package data)
    2. ``ui/dist/``      — present in dev mode after ``npm run build``

    Returns *None* if no built frontend is found (pure dev mode with Vite).
    """
    # 1. Package-internal _frontend/ (pip-installed or prepare_release)
    pkg_dir = Path(__file__).resolve().parent.parent / "_frontend"
    if (pkg_dir / "index.html").is_file():
        return pkg_dir

    # 2. ui/dist/ relative to project root (dev mode)
    project_root = Path(__file__).resolve().parent.parent.parent
    dev_dir = project_root / "ui" / "dist"
    if (dev_dir / "index.html").is_file():
        return dev_dir

    return None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(title="GW Copilot", version=__version__)

    # --- CORS (needed for dev mode when Vite runs on a separate port) ---
    # Harmless in integrated mode (same-origin requests skip CORS).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        max_age=3600,
    )

    # --- Security headers ---
    app.add_middleware(SecurityHeadersMiddleware)

    # --- Request body size cap (10 MB) ---
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=10 * 1024 * 1024)

    # --- API Routers (registered BEFORE the SPA catch-all) ---
    app.include_router(core_router)
    app.include_router(plots_router)
    app.include_router(viz_router)
    app.include_router(docs_router)
    app.include_router(snapshot_router)
    app.include_router(scan_router)
    app.include_router(patches_router)
    app.include_router(llm_config_router)
    app.include_router(profile_router)

    # --- Health check ---
    @app.get("/health")
    def health():
        """Lightweight health-check endpoint for monitoring."""
        return {"status": "ok", "version": __version__}

    # --- Global unhandled-exception handler ---
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Log full details server-side; return a safe message to the client.
        logger.error("Unhandled exception on %s %s", request.method, request.url, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # --- Serve bundled frontend (if available) ---
    frontend = _frontend_dir()
    if frontend is not None:
        logger.info("Serving frontend from %s", frontend)

        # Mount Vite's hashed asset bundles as static files
        assets_dir = frontend / "assets"
        if assets_dir.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="frontend-assets",
            )

        # SPA catch-all: return index.html for any unmatched path.
        # Registered LAST so all API routes take priority.
        @app.get("/{full_path:path}")
        async def _spa_catchall(full_path: str):
            # Serve actual files (e.g., favicon.ico, manifest.json)
            if full_path:
                file_path = frontend / full_path
                if (
                    file_path.is_file()
                    and file_path.resolve().is_relative_to(frontend.resolve())
                ):
                    return FileResponse(str(file_path))
            # Everything else → index.html (React router handles it)
            return FileResponse(str(frontend / "index.html"))
    else:
        logger.info(
            "No bundled frontend found. Run 'npm run build' in ui/ "
            "or use the Vite dev server on port 5173."
        )

    return app


app = create_app()
