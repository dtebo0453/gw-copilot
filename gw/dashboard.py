from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_dashboard(
    *,
    dashboard_path: str,
    step: str,
    ok: bool,
    counts: Optional[Dict[str, int]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "ts_utc": _utc_now_iso(),
        "step": step,
        "ok": ok,
        "counts": counts or {},
        "artifacts": artifacts or {},
    }
    if message:
        payload["message"] = message
    if extra:
        payload["extra"] = extra
    Path(dashboard_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
