from __future__ import annotations

"""LLM configuration API endpoints.

Allows the UI to configure which LLM provider to use and set API keys.
Keys are stored in environment variables (session-only) or a local config file.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Config file location (in user's home directory)
CONFIG_DIR = Path.home() / ".gw_copilot"
CONFIG_FILE = CONFIG_DIR / "llm_config.json"


class LLMConfigRequest(BaseModel):
    provider: str  # "openai" or "anthropic"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    model: Optional[str] = None


def _ensure_config_dir():
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_config() -> Dict[str, Any]:
    """Load config from file."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def _save_config(config: Dict[str, Any]):
    """Save config to file with restricted permissions.

    On POSIX systems the file is set to owner-only (0o600).  On Windows
    the standard ACLs for the user's home directory provide adequate
    protection; explicit chmod is a best-effort no-op.
    """
    _ensure_config_dir()
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    # Restrict file permissions so API keys aren't world-readable
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass  # Windows — home dir ACLs apply


@router.get("/llm/config")
def get_llm_config() -> Dict[str, Any]:
    """Get current LLM configuration (without exposing full API keys)."""
    config = _load_config()
    
    # Check environment variables
    openai_env = os.environ.get("OPENAI_API_KEY", "")
    anthropic_env = os.environ.get("ANTHROPIC_API_KEY", "")
    
    # Check config file
    openai_file = config.get("openai_api_key", "")
    anthropic_file = config.get("anthropic_api_key", "")
    
    return {
        "provider": config.get("provider", "openai"),
        "model": config.get("model"),
        "openai_configured": bool(openai_env or openai_file),
        "anthropic_configured": bool(anthropic_env or anthropic_file),
    }


@router.post("/llm/config")
def set_llm_config(req: LLMConfigRequest) -> Dict[str, Any]:
    """Update LLM configuration."""
    config = _load_config()
    
    # Update provider
    if req.provider in ("openai", "anthropic"):
        config["provider"] = req.provider
    else:
        raise HTTPException(status_code=400, detail="Invalid provider. Use 'openai' or 'anthropic'.")
    
    # Update API keys if provided
    if req.openai_api_key:
        config["openai_api_key"] = req.openai_api_key
        # Also set environment variable for current session
        os.environ["OPENAI_API_KEY"] = req.openai_api_key
    
    if req.anthropic_api_key:
        config["anthropic_api_key"] = req.anthropic_api_key
        os.environ["ANTHROPIC_API_KEY"] = req.anthropic_api_key
    
    # Update model if provided
    if req.model:
        config["model"] = req.model
    elif req.model == "":
        config.pop("model", None)
    
    _save_config(config)
    
    return {
        "status": "ok",
        "provider": config.get("provider"),
        "model": config.get("model"),
    }


def get_active_provider() -> str:
    """Get the currently active LLM provider."""
    config = _load_config()
    return config.get("provider", "openai")


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for the specified provider."""
    # Check environment first
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key
    elif provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key
    
    # Fall back to config file
    config = _load_config()
    if provider == "openai":
        return config.get("openai_api_key")
    elif provider == "anthropic":
        return config.get("anthropic_api_key")
    
    return None


def get_model() -> Optional[str]:
    """Get the configured model name."""
    config = _load_config()
    return config.get("model")
