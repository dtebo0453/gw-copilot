"""Simulator Abstraction Layer — base classes.

Every groundwater simulator plugin implements the ``SimulatorAdapter`` ABC.
The interface is intentionally broad: it covers model I/O, binary output
reading, QA diagnostics, model execution, and LLM knowledge fragments.

The adapter's primary job is to *feed the LLM the right data and context*
for whatever model format the user opens.  As LLMs improve, the tool
improves — the simulator layer just ensures accurate file contents, binary
output summaries, and domain-specific vocabulary are available.

Design pattern mirrors ``gw.llm.providers.base.LLMProvider``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulatorInfo:
    """Metadata about a simulator plugin."""
    name: str               # short key, e.g. "mf6", "mf2005", "mfnwt"
    display_name: str       # human-readable, e.g. "MODFLOW 6"
    version: Optional[str] = None


@dataclass
class GridInfo:
    """Unified grid metadata returned by all adapters."""
    grid_type: str           # "dis", "disv", "disu", "structured", etc.
    nlay: int
    ncpl: int                # cells per layer
    total_cells: int
    nrow: Optional[int] = None
    ncol: Optional[int] = None
    nvert: Optional[int] = None
    xorigin: float = 0.0
    yorigin: float = 0.0
    angrot: float = 0.0


@dataclass(frozen=True)
class FileTypeInfo:
    """Metadata about a simulator-specific file type."""
    kind: str
    purpose: str
    what_to_look_for: str


@dataclass(frozen=True)
class PackageArrayInfo:
    """Descriptor for a single array within a package."""
    label: str
    per_layer: bool = True


@dataclass(frozen=True)
class PackagePropertyInfo:
    """Descriptor for a package's griddata / period arrays."""
    file_ext: str
    block: str
    arrays: Dict[str, PackageArrayInfo] = field(default_factory=dict)


@dataclass(frozen=True)
class QACheck:
    """Descriptor for an available QA check."""
    name: str
    description: str


@dataclass(frozen=True)
class OutputCapability:
    """Describes what output types a simulator can produce/read."""
    heads: bool = True
    budget: bool = True
    concentration: bool = False    # MT3DMS, SEAWAT — UCN files
    pathlines: bool = False        # MODPATH — pathline files
    endpoints: bool = False        # MODPATH — endpoint files


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SimulatorAdapter(ABC):
    """Abstract interface that every simulator plugin must implement.

    The adapter is a *thin delegation layer* — it wires existing simulator-
    specific modules behind a common interface so the rest of the application
    (chat agent, tool loop, QA system, patches, viz) can work with any
    simulator without knowing its internals.
    """

    # ── Identity ──────────────────────────────────────────────────────────

    @abstractmethod
    def info(self) -> SimulatorInfo:
        """Return metadata about this simulator."""
        ...

    # ── Detection ─────────────────────────────────────────────────────────

    @abstractmethod
    def detect(self, ws_root: Path) -> float:
        """Return a confidence score (0.0–1.0) that *ws_root* contains this
        simulator's model.

        The registry calls ``detect`` on every registered adapter and picks
        the highest scorer.  Return 0.0 for no match, 1.0 for certainty.
        """
        ...

    # ── Model loading ─────────────────────────────────────────────────────

    @abstractmethod
    def build_snapshot(self, ws_root: Path) -> Dict[str, Any]:
        """Build a rich model snapshot dict.

        The snapshot captures grid geometry, time discretisation, packages,
        output file metadata, and workspace facts.  Its structure is
        consumed by the chat agent for LLM grounding.
        """
        ...

    @abstractmethod
    def build_model_brief(self, snapshot: Dict[str, Any]) -> str:
        """Build a compact text summary of the model for LLM prompt injection.

        Typically 300-500 characters summarising grid, TDIS, packages, and
        binary output availability.
        """
        ...

    # ── Stress I/O ────────────────────────────────────────────────────────

    @abstractmethod
    def read_stress_package(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        value_cols: int = 1,
        keep_aux: bool = True,
    ) -> Any:
        """Read a stress-package file and return a DataFrame."""
        ...

    @abstractmethod
    def read_time_discretization(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        """Read time discretisation; return DataFrame with per, perlen,
        t_start, t_end, t_mid columns."""
        ...

    @abstractmethod
    def read_namefile(
        self,
        ws_root: Path,
        rel_path: Optional[str] = None,
    ) -> Any:
        """Read the name/sim file; return DataFrame with ftype, fname,
        pkgname columns."""
        ...

    # ── Binary output ─────────────────────────────────────────────────────

    @abstractmethod
    def probe_head_file(
        self, ws_root: Path, rel_path: str
    ) -> Dict[str, Any]:
        """Probe a head output file for metadata (times, shape, value range)."""
        ...

    @abstractmethod
    def probe_budget_file(
        self, ws_root: Path, rel_path: str
    ) -> Dict[str, Any]:
        """Probe a budget output file for metadata (record names, times)."""
        ...

    @abstractmethod
    def extract_head_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_times: int = 5,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        """Extract head data with per-layer statistics and drawdown analysis."""
        ...

    @abstractmethod
    def extract_budget_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_records: int = 8,
        max_chars: int = 30_000,
    ) -> Dict[str, Any]:
        """Extract budget data with per-component breakdowns."""
        ...

    @abstractmethod
    def extract_head_timeseries(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        cells: List[Dict[str, int]],
        max_chars: int = 50_000,
    ) -> Dict[str, Any]:
        """Extract head time-series at specific cells across all timesteps.

        *cells*: list of ``{"layer": int, "row": int, "col": int}`` (1-based).
        Returns dict with ``ok``, ``summary_text``, ``metadata``.
        """
        ...

    @abstractmethod
    def extract_budget_cells(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        component: str,
        max_cells: int = 50,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        """Extract per-cell budget flows for a specific component.

        Returns the top cells by absolute flow magnitude with their cellid
        and flow values for the final timestep.
        """
        ...

    # ── Model execution ───────────────────────────────────────────────────

    @abstractmethod
    def run_model(
        self,
        ws_root: Path,
        *,
        exe_path: Optional[str] = None,
        timeout_sec: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        """Execute the simulator.  Returns ``(returncode, stdout, stderr)``."""
        ...

    @abstractmethod
    def find_executable(self, exe_path: Optional[str] = None) -> str:
        """Locate the simulator executable on PATH or at *exe_path*."""
        ...

    # ── QA diagnostics ────────────────────────────────────────────────────

    @abstractmethod
    def available_qa_checks(self) -> List[QACheck]:
        """Return descriptors for every QA check this simulator supports."""
        ...

    @abstractmethod
    def run_qa_check(
        self, ws_root: Path, check_name: str, **kwargs: Any
    ) -> str:
        """Run a specific QA check and return a Markdown report."""
        ...

    # ── File-type knowledge ───────────────────────────────────────────────

    @abstractmethod
    def file_type_knowledge(self) -> Dict[str, FileTypeInfo]:
        """Return dict mapping file extensions (e.g. ``".npf"``) to
        ``FileTypeInfo`` metadata."""
        ...

    @abstractmethod
    def package_properties(self) -> Dict[str, PackagePropertyInfo]:
        """Return dict mapping package type names (e.g. ``"NPF"``) to
        their array property descriptors."""
        ...

    @abstractmethod
    def file_extensions(self) -> Set[str]:
        """Return the set of all recognised file extensions for this
        simulator (e.g. ``{".dis", ".wel", ".npf", ...}``)."""
        ...

    # ── LLM knowledge ─────────────────────────────────────────────────────

    @abstractmethod
    def system_prompt_fragment(self) -> str:
        """Return the simulator-specific knowledge fragment for the LLM
        system prompt.

        Includes package names, file types, solver names, domain knowledge,
        QA thresholds, and any other text the LLM needs to reason about
        models built with this simulator.
        """
        ...

    @abstractmethod
    def tool_descriptions(self) -> Dict[str, str]:
        """Return simulator-specific description overrides for tool
        definitions.

        Keys are tool names (``"read_binary_output"``, ``"run_qa_check"``,
        etc.).  Values are the description text to splice into the tool
        schema.
        """
        ...

    @abstractmethod
    def file_mention_pattern(self) -> str:
        """Return a regex alternation pattern (no delimiters) for detecting
        simulator-specific file mentions in user messages.

        Example for MF6: ``"dis|disv|disu|nam|tdis|ims|npf|..."``
        """
        ...

    # ── Output capabilities ───────────────────────────────────────────

    def output_capabilities(self) -> OutputCapability:
        """Describe what output types this simulator produces.

        Used by the tool loop to conditionally expose concentration and
        particle-tracking tools.  Default: heads + budget only.
        """
        return OutputCapability()

    # ── Concentration output (MT3DMS, SEAWAT) ─────────────────────────
    # Optional — default implementations return "not supported".

    def probe_concentration_file(
        self, ws_root: Path, rel_path: str
    ) -> Dict[str, Any]:
        """Probe a UCN concentration file for metadata (times, shape, range).

        Only implemented by transport simulators (MT3DMS, SEAWAT).
        """
        return {"ok": False, "error": "not supported by this simulator"}

    def extract_concentration_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_times: int = 5,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        """Extract concentration data with per-layer statistics."""
        return {"ok": False, "error": "not supported by this simulator"}

    def extract_concentration_timeseries(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        cells: List[Dict[str, int]],
        max_chars: int = 50_000,
    ) -> Dict[str, Any]:
        """Extract concentration time-series at specific cells."""
        return {"ok": False, "error": "not supported by this simulator"}

    # ── Particle tracking output (MODPATH) ────────────────────────────
    # Optional — default implementations return "not supported".

    def probe_pathline_file(
        self, ws_root: Path, rel_path: str
    ) -> Dict[str, Any]:
        """Probe a MODPATH pathline/endpoint file for metadata."""
        return {"ok": False, "error": "not supported by this simulator"}

    def extract_endpoint_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_chars: int = 40_000,
    ) -> Dict[str, Any]:
        """Extract endpoint data — termination zones, travel times."""
        return {"ok": False, "error": "not supported by this simulator"}

    def extract_pathline_data(
        self,
        ws_root: Path,
        rel_path: str,
        *,
        max_chars: int = 50_000,
    ) -> Dict[str, Any]:
        """Extract pathline data — particle tracks, durations."""
        return {"ok": False, "error": "not supported by this simulator"}

    # ── Grid ──────────────────────────────────────────────────────────────

    @abstractmethod
    def get_grid_info(self, ws_root: Path) -> Optional[GridInfo]:
        """Load and return grid information for the model, or ``None`` if
        unavailable."""
        ...

    # ── FloPy bridge (optional) ───────────────────────────────────────────
    # Not every simulator has FloPy support.  Defaults return "unsupported".

    def get_simulation(
        self, ws_root: Path
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Load simulation via FloPy.  Returns ``(sim, error_msg)``.

        Default implementation returns ``(None, "not supported")``.
        """
        return None, f"{self.info().name} does not support FloPy simulation loading"

    def clear_simulation_cache(self, ws_root: Optional[Path] = None) -> None:
        """Clear cached simulation objects.  Default: no-op."""
        pass
