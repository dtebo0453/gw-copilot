from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from gw.api.workspace_files import resolve_workspace_root
from gw.llm.docs_retriever import search_docs, search_web_docs

router = APIRouter()

@router.get("/docs/search")
def docs_search(
    inputs_dir: str,
    q: str,
    workspace: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    ws_root = resolve_workspace_root(inputs_dir, workspace)

    # Prefer workspace/docs then workspace/documentation
    docs_root = None
    for cand in (ws_root / "docs", ws_root / "documentation"):
        if cand.exists() and cand.is_dir():
            docs_root = cand
            break

    if docs_root is None:
        raise HTTPException(status_code=404, detail="No docs folder found in workspace (expected ./docs or ./documentation)")

    hits = search_docs(ws_root, docs_root, q, top_k=int(top_k))
    return {
        "query": q,
        "docs_root": str(docs_root),
        "count": len(hits),
        "results": [h.__dict__ for h in hits],
    }


@router.get("/docs/web_search")
def docs_web_search(
    inputs_dir: str,
    q: str,
    workspace: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Search official web docs (controlled).

    This endpoint only works when GW_ENABLE_WEB_DOCS=1 on the server.
    Results are cached under <workspace>/.gw_copilot/web_docs_cache.sqlite.
    """
    ws_root = resolve_workspace_root(inputs_dir, workspace)
    hits = search_web_docs(ws_root, q, k=int(top_k), explicit_files=None)
    return {
        "query": q,
        "count": len(hits),
        "results": hits,
    }
