from __future__ import annotations

"""Agentic tool-use loop for the GW Copilot chat agent.

This module implements an iterative tool-calling loop that lets the LLM
request additional workspace data mid-conversation (file reads, binary
extraction, directory listing), trigger plot generation, and run QA/QC
diagnostic checks.

Supported tools:
  - read_file         — Read any text file from the workspace
  - read_binary_output — Extract numerical data from HDS/CBC via FloPy
  - list_files        — List workspace files matching an optional glob pattern
  - generate_plot     — Generate a plot in the sandbox (reuses plots.py infra)
  - run_qa_check      — Run specialized QA/QC diagnostics (mass balance, dry cells, etc.)

Provider-specific implementations:
  - Anthropic: uses client.messages.create(tools=[...]) with tool_use / tool_result
  - OpenAI: uses client.responses.create(tools=[...]) with function tool type

Design:
  - Maximum 15 iterations to allow multi-step QA investigations.
  - Each tool call is logged to an audit trail for debugging.
  - The loop exits when the LLM emits a text-only response (no further tools).
  - If max iterations are reached, accumulated text is returned.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (provider-agnostic JSON schemas)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a text file from the model workspace. Returns the file content "
            "(truncated to max_bytes if the file is large). Use this for MF6 package "
            "files (.dis, .wel, .npf, etc.), CSV files, config files, listing files, "
            "or any other text-based file in the workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path within the workspace (e.g., 'model.dis', "
                        "'gwf_model.wel', 'mfsim.nam'). Must be a file that exists "
                        "in the workspace."
                    ),
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Maximum bytes to read (default 200000). Use a smaller value for very large files.",
                    "default": 200000,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_binary_output",
        "description": (
            "Extract numerical data from a MODFLOW 6 binary output file (.hds or .cbc) "
            "via FloPy. Returns per-layer statistics (min, max, mean, median, std) for "
            "sampled timesteps, drawdown analysis (for HDS), or per-component budget "
            "breakdowns (for CBC). Use this when the user asks about simulation results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the .hds or .cbc file within the workspace.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files in the workspace, optionally filtered by a glob pattern. "
            "Returns file paths, sizes, and types. Use this to discover what files "
            "are available before reading specific ones."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Optional glob pattern to filter files (e.g., '*.wel', '**/*.csv', "
                        "'*.hds'). If omitted, lists all files."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_plot",
        "description": (
            "Generate a plot from the model workspace data. You MUST provide a full Python "
            "script that reads real data from workspace files. NEVER use random/dummy data.\n\n"
            "The script runs in a sandbox with access to matplotlib, numpy, pandas, flopy, "
            "and helper modules (gw_plot_io, gw_mf6_io). Output images are saved to the "
            "plot output directory and returned as URLs.\n\n"
            "REQUIRED helper APIs in scripts:\n"
            "- import gw_plot_io — workspace file access\n"
            "- gw_plot_io.ws_path(rel) — get absolute path to a workspace file\n"
            "- gw_plot_io.out_path('name.png') — get output path for saving plots\n"
            "- gw_plot_io.find_one('.hds') — find a file by extension\n"
            "- import gw_mf6_io — MF6 package readers\n"
            "- Path(os.environ['GW_PLOT_OUTDIR']) — output directory\n\n"
            "COMMON PATTERNS:\n"
            "  # Read heads from HDS:\n"
            "  import flopy\n"
            "  hds = flopy.utils.HeadFile(str(gw_plot_io.ws_path(gw_plot_io.find_one('.hds'))))\n"
            "  data = hds.get_data(totim=hds.get_times()[-1])  # shape: (nlay, nrow, ncol)\n\n"
            "  # Save plot:\n"
            "  plt.savefig(gw_plot_io.out_path('my_plot.png'), dpi=150, bbox_inches='tight')\n\n"
            "The tool returns image URLs in an 'images' array. Embed them in your response "
            "as markdown: ![Description](url)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Natural language description of the plot to generate.",
                },
                "script": {
                    "type": "string",
                    "description": (
                        "Full Python script to execute. If provided, this is run directly "
                        "instead of generating a script from the prompt."
                    ),
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "run_qa_check",
        "description": (
            "Run a specialized QA/QC diagnostic check on the MODFLOW 6 model. "
            "Returns a detailed markdown report with analysis, tables, and recommendations.\n\n"
            "Available checks:\n"
            "- mass_balance: Parse listing file for volumetric budget, compute % discrepancy per stress period\n"
            "- dry_cells: Count cells with HDRY/1e30 per layer per timestep, identify spatial clusters\n"
            "- convergence: Parse listing file for solver iterations and convergence failures\n"
            "- pumping_summary: Analyze WEL package rates by stress period, detect anomalous jumps\n"
            "- budget_timeseries: Extract IN/OUT per budget term across all timesteps\n"
            "- head_gradient: Compute cell-to-cell gradients per layer, flag extreme values\n"
            "- property_check: Check K/SS/SY ranges for unreasonable values and layer inversions\n\n"
            "Use this tool when the user asks about model quality, mass balance, dry cells, "
            "convergence, pumping data issues, or any QA/QC analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "check_name": {
                    "type": "string",
                    "description": (
                        "Name of the QA check to run. One of: mass_balance, dry_cells, "
                        "convergence, pumping_summary, budget_timeseries, head_gradient, "
                        "property_check."
                    ),
                },
            },
            "required": ["check_name"],
        },
    },
]


def _tool_definitions_anthropic() -> List[Dict[str, Any]]:
    """Convert tool definitions to Anthropic format."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in TOOL_DEFINITIONS
    ]


def _tool_definitions_openai() -> List[Dict[str, Any]]:
    """Convert tool definitions to OpenAI responses API function tool format."""
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        }
        for t in TOOL_DEFINITIONS
    ]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _execute_tool(
    name: str,
    args: Dict[str, Any],
    *,
    ws_root: Path,
    inputs_dir: str,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """Dispatch a tool call and return its result.

    Returns a dict with at minimum: {ok: bool, ...result_fields...}
    On error: {ok: False, error: str}
    """
    try:
        if name == "read_file":
            return _tool_read_file(args, ws_root=ws_root)
        elif name == "read_binary_output":
            return _tool_read_binary(args, ws_root=ws_root)
        elif name == "list_files":
            return _tool_list_files(args, ws_root=ws_root)
        elif name == "generate_plot":
            return _tool_generate_plot(
                args, ws_root=ws_root, inputs_dir=inputs_dir, workspace=workspace,
            )
        elif name == "run_qa_check":
            return _tool_run_qa_check(args, ws_root=ws_root)
        else:
            return {"ok": False, "error": f"Unknown tool: {name}"}
    except Exception as e:
        logger.warning("Tool %s failed: %s: %s", name, type(e).__name__, e)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _safe_resolve(ws_root: Path, rel_path: str) -> Optional[Path]:
    """Resolve a relative path safely within ws_root. Returns None if unsafe."""
    rel = (rel_path or "").replace("\\", "/").strip().strip("/")
    if not rel or ".." in rel.split("/"):
        return None
    import re
    if re.match(r"^[A-Za-z]:/", rel):
        return None
    p = (ws_root / rel).resolve()
    try:
        p.relative_to(ws_root.resolve())
    except ValueError:
        return None
    return p


def _tool_read_file(args: Dict[str, Any], *, ws_root: Path) -> Dict[str, Any]:
    """Read a text file from the workspace."""
    from gw.api.workspace_files import read_file_text

    rel = args.get("path", "")
    max_bytes = int(args.get("max_bytes", 200_000))
    max_bytes = min(max_bytes, 500_000)  # hard cap

    p = _safe_resolve(ws_root, rel)
    if p is None:
        return {"ok": False, "error": f"Invalid or unsafe path: {rel}"}
    if not p.exists() or not p.is_file():
        return {"ok": False, "error": f"File not found: {rel}"}

    try:
        content, truncated, sha, size, kind = read_file_text(p, max_bytes=max_bytes)
    except Exception as e:
        return {"ok": False, "error": f"Failed to read: {type(e).__name__}: {e}"}

    if kind != "text" or content is None:
        return {
            "ok": False,
            "error": f"File is binary ({kind}), size={size}. Use read_binary_output for .hds/.cbc files.",
        }

    return {
        "ok": True,
        "path": rel,
        "content": content,
        "size": size,
        "truncated": truncated,
    }


def _tool_read_binary(args: Dict[str, Any], *, ws_root: Path) -> Dict[str, Any]:
    """Extract data from an HDS or CBC binary file."""
    from gw.api.output_probes import extract_hds_data, extract_cbc_data

    rel = args.get("path", "")
    p = _safe_resolve(ws_root, rel)
    if p is None:
        return {"ok": False, "error": f"Invalid or unsafe path: {rel}"}
    if not p.exists() or not p.is_file():
        return {"ok": False, "error": f"File not found: {rel}"}

    ext = p.suffix.lower()
    if ext == ".hds":
        result = extract_hds_data(ws_root, rel, max_chars=50_000)
        if result.get("ok") and result.get("summary_text"):
            return {
                "ok": True,
                "path": rel,
                "type": "hds",
                "data": result["summary_text"],
                "metadata": result.get("metadata", {}),
            }
        return {"ok": False, "error": result.get("error", "Failed to extract HDS data")}

    elif ext == ".cbc":
        result = extract_cbc_data(ws_root, rel, max_chars=40_000)
        if result.get("ok") and result.get("summary_text"):
            return {
                "ok": True,
                "path": rel,
                "type": "cbc",
                "data": result["summary_text"],
                "metadata": result.get("metadata", {}),
            }
        return {"ok": False, "error": result.get("error", "Failed to extract CBC data")}

    else:
        return {"ok": False, "error": f"Unsupported binary format: {ext}. Only .hds and .cbc are supported."}


def _tool_list_files(args: Dict[str, Any], *, ws_root: Path) -> Dict[str, Any]:
    """List workspace files with optional glob filtering."""
    from gw.api.workspace_files import list_workspace_files

    pattern = args.get("pattern") or None
    try:
        files, truncated = list_workspace_files(ws_root, glob=pattern, max_files=200, include_hash=False)
    except Exception as e:
        return {"ok": False, "error": f"Failed to list files: {type(e).__name__}: {e}"}

    file_list = [
        {"path": f.path_rel, "size": f.size, "kind": f.kind}
        for f in files
    ]
    return {
        "ok": True,
        "files": file_list,
        "count": len(file_list),
        "truncated": truncated,
    }


def _tool_run_qa_check(args: Dict[str, Any], *, ws_root: Path) -> Dict[str, Any]:
    """Run a QA/QC diagnostic check on the model."""
    from gw.api.qa_diagnostics import run_qa_check

    check_name = str(args.get("check_name", "")).strip()
    if not check_name:
        return {"ok": False, "error": "check_name is required"}

    try:
        report = run_qa_check(str(ws_root), check_name)
        return {
            "ok": True,
            "check": check_name,
            "report": report,
        }
    except Exception as e:
        return {"ok": False, "error": f"QA check failed: {type(e).__name__}: {e}"}


def _tool_generate_plot(
    args: Dict[str, Any],
    *,
    ws_root: Path,
    inputs_dir: str,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a plot using the sandbox infrastructure.

    If a script is provided, execute it directly.
    If only a prompt is provided, use the plot planning LLM to generate a script first.
    """
    from gw.api.plots import execute_plot_in_sandbox

    prompt = str(args.get("prompt", "")).strip()
    script = args.get("script")

    if script and isinstance(script, str) and script.strip():
        # Direct script execution
        try:
            result = execute_plot_in_sandbox(
                ws_root=ws_root,
                script=script.strip(),
                inputs_dir=inputs_dir,
                prompt=prompt,
                workspace=workspace,
                timeout_sec=120,
            )
            return _format_plot_result(result, inputs_dir, workspace)
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": f"Plot execution failed: {type(e).__name__}: {e}"}

    if not prompt:
        return {"ok": False, "error": "Either prompt or script must be provided."}

    # Use the plot planner to generate a script from the prompt
    try:
        script = _plan_plot_script(prompt, ws_root=ws_root, inputs_dir=inputs_dir, workspace=workspace)
    except Exception as e:
        return {"ok": False, "error": f"Failed to plan plot: {type(e).__name__}: {e}"}

    if not script:
        return {"ok": False, "error": "Plot planner could not generate a script for this request."}

    # Execute the generated script
    try:
        result = execute_plot_in_sandbox(
            ws_root=ws_root,
            script=script,
            inputs_dir=inputs_dir,
            prompt=prompt,
            workspace=workspace,
            timeout_sec=120,
        )
        return _format_plot_result(result, inputs_dir, workspace)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"Plot execution failed: {type(e).__name__}: {e}"}


def _format_plot_result(result: Dict[str, Any], inputs_dir: str, workspace: Optional[str]) -> Dict[str, Any]:
    """Format sandbox result into a tool response with image URLs."""
    from urllib.parse import quote

    outputs = result.get("outputs", [])
    run_id = result.get("run_id", "")
    exit_code = result.get("exit_code", -1)
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")

    image_outputs = []
    for out in outputs:
        name = out.get("name", "")
        ext = Path(name).suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".gif"):
            # Build a URL that the existing /plots/run/output endpoint can serve
            # NOTE: endpoint param is "path", not "filename"
            url = (
                f"/plots/run/output"
                f"?inputs_dir={quote(inputs_dir, safe='')}"
                f"&run_id={quote(run_id, safe='')}"
                f"&path={quote(name, safe='')}"
            )
            if workspace:
                url += f"&workspace={quote(workspace, safe='')}"
            image_outputs.append({"name": name, "url": url})

    return {
        "ok": exit_code == 0,
        "run_id": run_id,
        "images": image_outputs,
        "exit_code": exit_code,
        "stdout": stdout[:2000] if stdout else "",
        "stderr": stderr[:2000] if stderr else "",
    }


def _plan_plot_script(
    prompt: str,
    *,
    ws_root: Path,
    inputs_dir: str,
    workspace: Optional[str] = None,
) -> Optional[str]:
    """Use the plot planning LLM call to generate a script from a prompt.

    This is a lightweight wrapper that calls the same planning logic
    as the /plots/plan endpoint but returns just the script string.
    """
    from gw.api.plots import _llm_is_configured

    if not _llm_is_configured():
        return None

    # Use the LLM planner via the /plots/plan logic (no HTTP overhead)
    from gw.api.plots import plots_plan as _plots_plan_endpoint
    try:
        plan_result = _plots_plan_endpoint({
            "prompt": prompt,
            "inputs_dir": inputs_dir,
            "workspace": workspace,
            "selected_files": [],
        })
        if isinstance(plan_result, dict):
            script = plan_result.get("script")
            if script and isinstance(script, str) and script.strip():
                return script.strip()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Provider-specific tool loops
# ---------------------------------------------------------------------------

def _anthropic_tool_loop(
    *,
    client: Any,
    model: str,
    system: str,
    messages: List[Dict[str, Any]],
    ws_root: Path,
    inputs_dir: str,
    workspace: Optional[str] = None,
    max_iterations: int = 15,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the tool-use loop using the Anthropic messages API.

    Returns: (final_text, audit_log)
    """
    tools = _tool_definitions_anthropic()
    audit: List[Dict[str, Any]] = []
    accumulated_text = ""

    # Convert messages to Anthropic format (ensure proper structure)
    conv_messages = []
    for m in messages:
        conv_messages.append({"role": m["role"], "content": m["content"]})

    for iteration in range(max_iterations):
        t0 = time.time()

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                messages=conv_messages,
                tools=tools,
            )
        except Exception as e:
            logger.error("Anthropic tool loop iteration %d failed: %s", iteration, e)
            if accumulated_text:
                return accumulated_text, audit
            raise

        dt = time.time() - t0
        stop_reason = getattr(resp, "stop_reason", None)

        # Extract text and tool_use blocks from response
        text_parts = []
        tool_calls = []
        response_content = []

        for block in getattr(resp, "content", []):
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
                response_content.append({"type": "text", "text": block.text})
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                response_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        text_so_far = "".join(text_parts).strip()
        if text_so_far:
            accumulated_text = text_so_far

        # If no tool calls, we're done
        if not tool_calls:
            logger.info(
                "Anthropic tool loop done at iteration %d (stop=%s, text_len=%d, elapsed=%.2fs)",
                iteration, stop_reason, len(accumulated_text), dt,
            )
            return accumulated_text or "(no text returned)", audit

        # Append assistant response to conversation
        conv_messages.append({"role": "assistant", "content": response_content})

        # Execute each tool call and build tool_result messages
        tool_results = []
        for tc in tool_calls:
            tc_t0 = time.time()
            result = _execute_tool(
                tc["name"], tc["input"],
                ws_root=ws_root, inputs_dir=inputs_dir, workspace=workspace,
            )
            tc_dt = time.time() - tc_t0

            # Serialize result for the LLM
            result_text = json.dumps(result, indent=2, default=str)
            # Truncate very large results
            if len(result_text) > 80_000:
                result_text = result_text[:80_000] + "\n... (truncated)"

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": result_text,
            })

            audit.append({
                "iteration": iteration,
                "tool": tc["name"],
                "args": tc["input"],
                "ok": result.get("ok", False),
                "elapsed_sec": round(tc_dt, 3),
                "result_chars": len(result_text),
            })

            logger.info(
                "Tool call: %s(%s) → ok=%s chars=%d elapsed=%.2fs",
                tc["name"],
                json.dumps(tc["input"], default=str)[:200],
                result.get("ok"),
                len(result_text),
                tc_dt,
            )

        # Append tool results as user message
        conv_messages.append({"role": "user", "content": tool_results})

    # Max iterations reached
    logger.warning("Anthropic tool loop hit max iterations (%d)", max_iterations)
    return accumulated_text or "(max tool iterations reached)", audit


def _openai_tool_loop(
    *,
    client: Any,
    model: str,
    system: str,
    messages: List[Dict[str, Any]],
    ws_root: Path,
    inputs_dir: str,
    workspace: Optional[str] = None,
    max_iterations: int = 15,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the tool-use loop using the OpenAI responses API.

    Returns: (final_text, audit_log)
    """
    tools = _tool_definitions_openai()
    audit: List[Dict[str, Any]] = []
    accumulated_text = ""

    # Build the initial input
    input_msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        *[{"role": m["role"], "content": m["content"]} for m in messages],
    ]

    for iteration in range(max_iterations):
        t0 = time.time()

        try:
            resp = client.responses.create(
                model=model,
                input=input_msgs,
                tools=tools,
            )
        except Exception as e:
            logger.error("OpenAI tool loop iteration %d failed: %s", iteration, e)
            if accumulated_text:
                return accumulated_text, audit
            raise

        dt = time.time() - t0

        # Extract text and function calls from response output
        text_parts = []
        function_calls = []

        for item in getattr(resp, "output", []) or []:
            item_type = getattr(item, "type", "")

            if item_type == "message":
                # Message content blocks
                for c in getattr(item, "content", []) or []:
                    c_type = getattr(c, "type", "")
                    if c_type in ("output_text", "text"):
                        text_parts.append(getattr(c, "text", ""))

            elif item_type == "function_call":
                function_calls.append({
                    "call_id": getattr(item, "call_id", ""),
                    "name": getattr(item, "name", ""),
                    "arguments": getattr(item, "arguments", "{}"),
                })

        text_so_far = "".join(text_parts).strip()
        if text_so_far:
            accumulated_text = text_so_far

        # If no function calls, we're done
        if not function_calls:
            logger.info(
                "OpenAI tool loop done at iteration %d (text_len=%d, elapsed=%.2fs)",
                iteration, len(accumulated_text), dt,
            )
            return accumulated_text or "(no text returned)", audit

        # Add the model's output to the conversation
        # For the OpenAI responses API, we pass the whole response output
        # items back to the next call, plus function results
        next_input = list(input_msgs)

        # Add function call outputs from the model's response
        for item in getattr(resp, "output", []) or []:
            next_input.append(item)

        # Execute each function call and add results
        for fc in function_calls:
            tc_t0 = time.time()

            try:
                parsed_args = json.loads(fc["arguments"]) if isinstance(fc["arguments"], str) else fc["arguments"]
            except (json.JSONDecodeError, TypeError):
                parsed_args = {}

            result = _execute_tool(
                fc["name"], parsed_args,
                ws_root=ws_root, inputs_dir=inputs_dir, workspace=workspace,
            )
            tc_dt = time.time() - tc_t0

            result_text = json.dumps(result, indent=2, default=str)
            if len(result_text) > 80_000:
                result_text = result_text[:80_000] + "\n... (truncated)"

            # Add function result to the conversation
            next_input.append({
                "type": "function_call_output",
                "call_id": fc["call_id"],
                "output": result_text,
            })

            audit.append({
                "iteration": iteration,
                "tool": fc["name"],
                "args": parsed_args,
                "ok": result.get("ok", False),
                "elapsed_sec": round(tc_dt, 3),
                "result_chars": len(result_text),
            })

            logger.info(
                "Tool call: %s(%s) → ok=%s chars=%d elapsed=%.2fs",
                fc["name"],
                json.dumps(parsed_args, default=str)[:200],
                result.get("ok"),
                len(result_text),
                tc_dt,
            )

        input_msgs = next_input

    # Max iterations reached
    logger.warning("OpenAI tool loop hit max iterations (%d)", max_iterations)
    return accumulated_text or "(max tool iterations reached)", audit


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def tool_loop(
    *,
    messages: List[Dict[str, str]],
    system: str,
    ws_root: Path,
    inputs_dir: str,
    workspace: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_iterations: int = 15,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the agentic tool-use loop with the configured LLM provider.

    Parameters
    ----------
    messages : conversation history (role/content dicts)
    system : system prompt
    ws_root : resolved workspace root Path
    inputs_dir : raw inputs_dir string (for plot URLs)
    workspace : optional workspace name
    provider : 'anthropic' or 'openai' (auto-detected if None)
    model : model name override
    max_iterations : max tool-calling rounds (default 15)

    Returns
    -------
    (final_text, audit_log) where audit_log is a list of tool call records
    """
    import os

    # Auto-detect provider if not specified
    if provider is None:
        from gw.llm.chat_agent import _get_active_provider
        provider = _get_active_provider()

    if provider == "anthropic":
        from gw.llm.chat_agent import _anthropic_client, _get_configured_model, _select_model_for_provider
        client = _anthropic_client()
        if client is None:
            raise RuntimeError("Anthropic not configured. Set ANTHROPIC_API_KEY or configure in Settings.")

        env_default = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        use_model = _select_model_for_provider("anthropic", model or _get_configured_model(), env_default)

        return _anthropic_tool_loop(
            client=client,
            model=use_model,
            system=system,
            messages=messages,
            ws_root=ws_root,
            inputs_dir=inputs_dir,
            workspace=workspace,
            max_iterations=max_iterations,
        )

    else:
        # OpenAI (default)
        from gw.llm.chat_agent import _openai_client, _get_configured_model, _select_model_for_provider
        client = _openai_client()
        if client is None:
            raise RuntimeError("OpenAI not configured. Set OPENAI_API_KEY or configure in Settings.")

        env_default = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        use_model = _select_model_for_provider("openai", model or _get_configured_model(), env_default)

        return _openai_tool_loop(
            client=client,
            model=use_model,
            system=system,
            messages=messages,
            ws_root=ws_root,
            inputs_dir=inputs_dir,
            workspace=workspace,
            max_iterations=max_iterations,
        )
