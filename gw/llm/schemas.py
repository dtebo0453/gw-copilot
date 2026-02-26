from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class DraftResponse(BaseModel):
    config: Dict[str, Any] = Field(..., description="Draft MODFLOW 6 config JSON for the deterministic builder.")
    questions: List[str] = Field(default_factory=list, description="Clarifying questions / missing inputs.")
    conceptual_model: str = Field(..., description="Conceptual model memo, assumptions, and rationale.")
