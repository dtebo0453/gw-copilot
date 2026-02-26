from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

class LLMProvider(ABC):
    @abstractmethod
    def draft(self, prompt: str, context: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
