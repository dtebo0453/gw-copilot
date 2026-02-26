import React, { useEffect, useMemo, useState } from "react";

const API_BASE =
  (import.meta as any).env?.VITE_API_BASE || "";

type RunSummary = {
  run_id: string;
  ts_utc?: string | null;
  prompt?: string | null;
  exit_code?: number | null;
};

type RunDetail = {
  run_id: string;
  ts_utc?: string | null;
  prompt: string;
  script: string;
  stdout: string;
  stderr: string;
  exit_code: number | null;
  duration_sec?: number | null;
  context_hash: string;
  files_used: string[];
  outputs: { name: string; path: string }[];
};

type PlanResp = {
  status: "ok" | "needs_clarification";
  notes?: string;
  questions?: string[];
  recommendations?: { path: string; reason: string }[];
  files_used?: string[];
  script?: string;
  code?: string;
  fixed_script?: string;
  plot_script?: string;
  context_hash?: string;
  validation_warnings?: { level: string; message: string; fix_hint?: string }[];
};

type RunResp = {
  run_id: string;
  run_dir: string;
  outputs: { name: string; path: string }[];
  stdout: string;
  stderr: string;
  exit_code: number;
};

// Plot templates — detailed, LLM-friendly prompts that produce good scripts
const PLOT_TEMPLATES = [
  {
    label: "Head Contour Map",
    prompt:
      "Create a filled-contour map of simulated hydraulic heads for model layer 1 at the final time step. " +
      "Use flopy.utils.HeadFile to read the .hds binary output, mask inactive/dry cells (values >= 1e+29), " +
      "and plot with 20 contour levels using the 'viridis' colormap. Add a colorbar labeled 'Head (m)' " +
      "and label the axes with Row and Column. If the model uses DISV or DISU, use flopy.plot.PlotMapView " +
      "to handle the unstructured grid. Include a title with the layer number and simulation time.",
  },
  {
    label: "Head Time Series",
    prompt:
      "Plot a time series of simulated hydraulic head at a monitoring cell near the center of the model grid " +
      "for layer 1. Use flopy.utils.HeadFile to read the .hds output, extract head values at every saved " +
      "time step for that cell, and plot head (y-axis, in meters) vs simulation time (x-axis). " +
      "Mark each time step with a small circle marker. Add axis labels, a grid, and a legend showing " +
      "the cell (layer, row, col). If the model has multiple layers, overlay a second line for the deepest layer.",
  },
  {
    label: "Water Budget",
    prompt:
      "Create a horizontal bar chart showing the volumetric water budget for the final stress period. " +
      "Use flopy.utils.CellBudgetFile to open the .cbc output (try double precision first, then single). " +
      "For each unique budget record name, sum the inflows (positive, blue bars) and outflows (negative, " +
      "red bars) and display them side-by-side. Include a vertical line at zero, label the y-axis with " +
      "the budget component names, and add a title showing the simulation time. Show the percent discrepancy " +
      "in the title if available.",
  },
  {
    label: "Drawdown Map",
    prompt:
      "Calculate drawdown by subtracting simulated heads at the final time step from heads at the first " +
      "time step (initial - final) for layer 1, then display as a filled contour map. Use the 'RdBu_r' " +
      "colormap centered on zero so areas of decline are red and areas of rise are blue. Mask inactive " +
      "or dry cells (values >= 1e+29). Add a colorbar labeled 'Drawdown (m)', axis labels, and a title " +
      "showing the time range. Use flopy.plot.PlotMapView if the grid is DISV/DISU.",
  },
  {
    label: "Cross Section",
    prompt:
      "Create a cross-section plot through a row near the center of the model showing the simulated head " +
      "distribution across all layers. Use flopy.plot.PlotCrossSection with line={'row': center_row} to " +
      "plot the grid geometry and overlay the head data as a color-filled array. Add a colorbar labeled " +
      "'Head (m)'. Label the x-axis as 'Distance along row (m)' and y-axis as 'Elevation (m)'. " +
      "Draw the model grid lines in light gray. Include a title identifying the row index.",
  },
  {
    label: "Pumping Rates",
    prompt:
      "Plot the well (WEL package) pumping rates vs simulation time as a line chart. Read the .wel " +
      "package file to get stress-period data. If the model has a TDIS file, convert stress period " +
      "indices to actual model time for the x-axis. Plot each well as a separate line (up to 20 wells; " +
      "if more, group by total pumping and show the top 20). Also include a bold dashed line for the " +
      "total pumping rate summed across all wells. Label axes as 'Time' and 'Pumping Rate (m\u00b3/d)'. " +
      "Add a legend and grid lines.",
  },
  {
    label: "K Distribution",
    prompt:
      "Create a plan-view map of horizontal hydraulic conductivity (K) for layer 1. Read the NPF " +
      "package file to extract the K array from the GRIDDATA block, reshape to (nlay, nrow, ncol), " +
      "and display the layer 1 slice using imshow or contourf with a logarithmic color scale if values " +
      "span more than one order of magnitude. Add a colorbar labeled 'K (m/d)', axis labels for " +
      "Row and Column, and a descriptive title. Use a 'plasma' or 'cividis' colormap.",
  },
  {
    label: "Flow Vectors",
    prompt:
      "Create a plan-view plot of groundwater flow direction for layer 1 using flow vectors (quiver plot). " +
      "Load the simulation with flopy, get the model object, and use flopy.plot.PlotMapView to create " +
      "the map. Read the .cbc file with CellBudgetFile and use plot_vector() or the specific discharge " +
      "postprocessing to compute qx and qy components. Overlay the vectors on a head contour map " +
      "as background context. Color the vectors by magnitude. Add a title, colorbar, and axis labels.",
  },
  {
    label: "Recharge Map",
    prompt:
      "Create a plan-view map showing the spatial distribution of recharge rates applied to the model. " +
      "Read the RCH or RCHA package file to extract the recharge array. If recharge varies by stress " +
      "period, show the values for the first stress period. Display using imshow or contourf with an " +
      "appropriate colormap (e.g., 'Blues'). Add a colorbar labeled 'Recharge (m/d)', row/column axis " +
      "labels, and a descriptive title. If the model uses list-based RCH, aggregate by cell.",
  },
  {
    label: "Boundary Conditions",
    prompt:
      "Create a map showing all boundary condition locations for layer 1 on a single figure. " +
      "Load the simulation with flopy and iterate through the model packages. Plot the model grid " +
      "in light gray using PlotMapView, then overlay colored markers or shaded cells for each " +
      "boundary type (e.g., WEL=red circles, CHD=blue squares, RIV=green line, DRN=orange triangles, " +
      "GHB=purple diamonds). Add a legend identifying each package, axis labels, and a title. " +
      "Use distinct colors and markers so boundaries are easily distinguishable.",
  },
  {
    label: "SFR Streamflow",
    prompt:
      "Plot the simulated streamflow for all SFR (Streamflow Routing) reaches over time. " +
      "Read the SFR output from the .cbc file or SFR observation file to extract reach flows " +
      "at each time step. Plot streamflow (y-axis, m\u00b3/s or m\u00b3/d) vs simulation time (x-axis) " +
      "with a separate line for each reach or grouped by segment. If there are many reaches, " +
      "show only the downstream-most reach and label it. Add axis labels, grid lines, a legend, " +
      "and a title. Highlight any reaches with zero flow in a different style.",
  },
  {
    label: "Layer Comparison",
    prompt:
      "Create a 2x2 subplot figure comparing simulated heads across different model layers at the " +
      "final time step. Show layer 1 (top-left), layer 2 (top-right), and the deepest two layers " +
      "in the bottom row. Use the same color scale (vmin/vmax) across all four subplots so they " +
      "are directly comparable. Add a shared colorbar, label each subplot with the layer number, " +
      "and mask inactive/dry cells. If the model has fewer than 4 layers, use as many as are available.",
  },
];

async function fetchJson(url: string, init?: RequestInit) {
  let r: Response;
  try {
    r = await fetch(url, init);
  } catch (networkErr: any) {
    // Actual network failure (server down, CORS, DNS, etc.)
    throw new Error(`[network] ${networkErr?.message || networkErr}`);
  }
  const txt = await r.text();
  let data: any = null;
  try {
    data = txt ? JSON.parse(txt) : null;
  } catch {
    // keep txt for debugging
  }
  if (!r.ok) {
    const detail =
      (data && (data.detail || data.message)) || txt || `HTTP ${r.status}`;
    throw new Error(String(detail));
  }
  return data;
}

function extractScript(obj: any): string | null {
  if (!obj || typeof obj !== "object") return null;

  for (const k of ["script", "code", "fixed_script", "plot_script"]) {
    const v = obj[k];
    if (typeof v === "string" && v.trim()) return v.trim();
  }

  // Fallback: fenced python in text-ish fields
  for (const k of ["text", "content", "message", "notes"]) {
    const v = obj[k];
    if (typeof v === "string" && v.includes("```")) {
      const m = v.match(/```(?:python)?\s*\n([\s\S]*?)```/i);
      if (m?.[1]?.trim()) return m[1].trim();
    }
  }

  return null;
}

function outputUrl(
  inputsDir: string,
  runId: string,
  outPath: string,
  workspace?: string | null
) {
  const params = new URLSearchParams();
  params.set("inputs_dir", inputsDir);
  params.set("run_id", runId);
  params.set("path", outPath);
  if (workspace) params.set("workspace", workspace);
  return `${API_BASE}/plots/run/output?${params.toString()}`;
}

/** Format a UTC ISO timestamp as a human-friendly local string. */
function fmtTime(ts?: string | null): string {
  if (!ts) return "";
  try {
    const d = new Date(ts);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return ts;
  }
}

export function PlotsTab(props: {
  inputsDir: string;
  workspace?: string | null;
}) {
  const { inputsDir, workspace } = props;

  const [prompt, setPrompt] = useState<string>("");
  const [script, setScript] = useState<string>("");
  const [notes, setNotes] = useState<string>("");
  const [questions, setQuestions] = useState<string[]>([]);
  const [filesUsed, setFilesUsed] = useState<string[]>([]);
  const [contextHash, setContextHash] = useState<string>("");

  const [planStatus, setPlanStatus] = useState<"ok" | "needs_clarification">("ok");
  const [validationWarnings, setValidationWarnings] = useState<{level: string; message: string; fix_hint?: string}[]>([]);
  const [planning, setPlanning] = useState(false);
  const [running, setRunning] = useState(false);
  const [repairing, setRepairing] = useState(false);
  const [autoFixing, setAutoFixing] = useState(false);
  const [autoFixAttempt, setAutoFixAttempt] = useState(0);

  const [stdout, setStdout] = useState<string>("");
  const [stderr, setStderr] = useState<string>("");
  const [exitCode, setExitCode] = useState<number | null>(null);
  const [outputs, setOutputs] = useState<{ name: string; path: string }[]>([]);

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [showScript, setShowScript] = useState(false);
  const [showTemplates, setShowTemplates] = useState(true);
  const [showRunsPanel, setShowRunsPanel] = useState(false);

  const [err, setErr] = useState<string | null>(null);
  const [repairCount, setRepairCount] = useState(0);
  const MAX_REPAIRS = 5;

  // Track whether we are viewing a historical run (read-only)
  const [viewingHistory, setViewingHistory] = useState(false);

  // Image outputs for inline display
  const imageOutputs = useMemo(() => {
    return outputs.filter(
      (o) =>
        o.name.endsWith(".png") ||
        o.name.endsWith(".jpg") ||
        o.name.endsWith(".jpeg") ||
        o.name.endsWith(".svg")
    );
  }, [outputs]);

  async function refreshRuns() {
    try {
      const resp = await fetchJson(
        `${API_BASE}/plots/runs?inputs_dir=${encodeURIComponent(inputsDir)}${
          workspace ? `&workspace=${encodeURIComponent(workspace)}` : ""
        }`
      );
      setRuns(resp?.runs ?? []);
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    refreshRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputsDir, workspace]);

  /** Load full detail for a historical run and populate all fields. */
  async function loadRunDetail(runId: string) {
    setErr(null);
    try {
      const detail: RunDetail = await fetchJson(
        `${API_BASE}/plots/run/detail?inputs_dir=${encodeURIComponent(inputsDir)}&run_id=${encodeURIComponent(runId)}${
          workspace ? `&workspace=${encodeURIComponent(workspace)}` : ""
        }`
      );
      setSelectedRunId(detail.run_id);
      setPrompt(detail.prompt || "");
      setScript(detail.script || "");
      setStdout(detail.stdout || "");
      setStderr(detail.stderr || "");
      setExitCode(detail.exit_code);
      setOutputs(detail.outputs || []);
      setFilesUsed(detail.files_used || []);
      setContextHash(detail.context_hash || "");
      setNotes("");
      setQuestions([]);
      setValidationWarnings([]);
      setPlanStatus("ok");
      setRepairCount(0);
      setShowScript(true);
      setShowTemplates(false);
      setViewingHistory(true);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    }
  }

  /** Delete a run and refresh the list. */
  async function deleteRun(runId: string) {
    try {
      await fetchJson(
        `${API_BASE}/plots/run/${encodeURIComponent(runId)}?inputs_dir=${encodeURIComponent(inputsDir)}${
          workspace ? `&workspace=${encodeURIComponent(workspace)}` : ""
        }`,
        { method: "DELETE" }
      );
      // If we were viewing the deleted run, clear the view
      if (selectedRunId === runId) {
        clearState();
      }
      await refreshRuns();
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    }
  }

  function clearState() {
    setPrompt("");
    setScript("");
    setNotes("");
    setQuestions([]);
    setFilesUsed([]);
    setContextHash("");
    setStdout("");
    setStderr("");
    setExitCode(null);
    setOutputs([]);
    setSelectedRunId("");
    setShowScript(false);
    setShowTemplates(true);
    setViewingHistory(false);
    setErr(null);
    setRepairCount(0);
    setAutoFixing(false);
    setAutoFixAttempt(0);
    setValidationWarnings([]);
    setPlanStatus("ok");
  }

  async function doPlan() {
    setErr(null);
    setPlanning(true);
    setViewingHistory(false);
    // Reset run state since the agentic endpoint also executes
    setStdout("");
    setStderr("");
    setExitCode(null);
    setOutputs([]);
    setSelectedRunId("");

    try {
      // Use the agentic endpoint which plans + executes + self-repairs
      const resp: any = await fetchJson(`${API_BASE}/plots/plan-agentic`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          inputs_dir: inputsDir,
          workspace: workspace ?? null,
        }),
      });

      setPlanStatus(resp.status ?? "ok");
      setNotes(resp.notes ?? "");
      setQuestions(resp.questions ?? []);
      setFilesUsed(resp.files_used ?? []);
      setScript(resp.script ?? "");
      setContextHash(resp.context_hash ?? "");
      setValidationWarnings(resp.validation_warnings ?? []);
      setShowScript(true);
      setShowTemplates(false);
      setRepairCount(0);

      // The agentic endpoint also returns run results
      if (resp.run_id) {
        setSelectedRunId(resp.run_id);
        setExitCode(resp.exit_code ?? null);
        setStdout(resp.stdout ?? "");
        setStderr(resp.stderr ?? "");
        setOutputs(resp.outputs ?? []);
        await refreshRuns();
      }
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setPlanning(false);
    }
  }

  /** Run the current script. Returns the response (or null on error). */
  async function doRunInner(currentScript: string): Promise<RunResp | null> {
    setErr(null);
    setRunning(true);
    setViewingHistory(false);
    setStdout("");
    setStderr("");
    setExitCode(null);
    setOutputs([]);

    try {
      const resp: RunResp = await fetchJson(`${API_BASE}/plots/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          inputs_dir: inputsDir,
          workspace: workspace ?? null,
          script: currentScript,
          confirm: "confirm:run_plot",
          context_hash: contextHash,
          files_used: filesUsed,
        }),
      });

      setStdout(resp.stdout ?? "");
      setStderr(resp.stderr ?? "");
      setExitCode(resp.exit_code ?? null);
      setOutputs(resp.outputs ?? []);

      if (resp.run_id) setSelectedRunId(resp.run_id);
      await refreshRuns();
      return resp;
    } catch (e: any) {
      setErr(e?.message ?? String(e));
      return null;
    } finally {
      setRunning(false);
    }
  }

  /** Repair the current script. Returns the fixed script (or null on error). */
  async function doRepairInner(
    currentScript: string,
    currentStdout: string,
    currentStderr: string,
  ): Promise<string | null> {
    setErr(null);
    setRepairing(true);
    setViewingHistory(false);

    try {
      const resp: PlanResp = await fetchJson(`${API_BASE}/plots/repair`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          inputs_dir: inputsDir,
          workspace: workspace ?? null,
          script: currentScript,
          stderr: currentStderr,
          stdout: currentStdout,
          files_used: filesUsed,
        }),
      });

      setPlanStatus(resp.status);
      setNotes(resp.notes ?? "");
      setQuestions(resp.questions ?? []);
      setFilesUsed(resp.files_used ?? []);
      setValidationWarnings(resp.validation_warnings ?? []);
      setContextHash(resp.context_hash ?? contextHash);

      const repaired = extractScript(resp);
      if (repaired) {
        setScript(repaired);
        return repaired;
      }
      return null;
    } catch (e: any) {
      setErr(e?.message ?? String(e));
      return null;
    } finally {
      setRepairing(false);
    }
  }

  /** Public doRun: run script, then auto-repair loop if it fails. */
  async function doRun() {
    setRepairCount(0);
    setAutoFixAttempt(0);
    const resp = await doRunInner(script);
    if (!resp) return; // network error

    // If script failed, kick off auto-repair loop
    if (resp.exit_code !== 0) {
      await doAutoRepairLoop(script, resp.stdout ?? "", resp.stderr ?? "");
    }
  }

  /** Auto-repair loop: repair → run → repeat until success or limit. */
  async function doAutoRepairLoop(
    currentScript: string,
    currentStdout: string,
    currentStderr: string,
  ) {
    setAutoFixing(true);
    let s = currentScript;
    let out = currentStdout;
    let errOut = currentStderr;

    for (let attempt = 1; attempt <= MAX_REPAIRS; attempt++) {
      setAutoFixAttempt(attempt);
      setRepairCount(attempt);

      // Step 1: Repair
      const fixed = await doRepairInner(s, out, errOut);
      if (!fixed) {
        // Repair itself failed (network error etc.)
        break;
      }
      s = fixed;

      // Step 2: Run the fixed script
      const resp = await doRunInner(s);
      if (!resp) break; // network error

      // Step 3: Check result
      if (resp.exit_code === 0) {
        // Success!
        break;
      }

      // Prepare for next iteration
      out = resp.stdout ?? "";
      errOut = resp.stderr ?? "";
    }

    setAutoFixing(false);
    setAutoFixAttempt(0);
  }

  /** Manual repair: fix once then re-run with auto-repair loop. */
  async function doRepairManual() {
    setRepairCount((c) => c + 1);
    const fixed = await doRepairInner(script, stdout, stderr);
    if (!fixed) return;

    // Run the fixed script and auto-repair if it fails again
    const resp = await doRunInner(fixed);
    if (!resp) return;
    if (resp.exit_code !== 0) {
      await doAutoRepairLoop(fixed, resp.stdout ?? "", resp.stderr ?? "");
    }
  }

  function useTemplate(template: typeof PLOT_TEMPLATES[0]) {
    setPrompt(template.prompt);
    setShowTemplates(false);
    setViewingHistory(false);
  }

  const canRun = script.trim().length > 0 && planStatus !== "needs_clarification";
  const canRepair =
    !repairing &&
    !running &&
    !autoFixing &&
    exitCode != null &&
    exitCode !== 0 &&
    ((stderr?.trim().length ?? 0) > 0 || (stdout?.trim().length ?? 0) > 0) &&
    script.trim().length > 0 &&
    repairCount < MAX_REPAIRS;

  const isBusy = planning || running || repairing || autoFixing;

  return (
    <div className="plots-container">
      {/* Left Panel: Prompt & Script */}
      <div className="plots-left">
        <div className="plots-section">
          <div className="plots-section-header">
            <span className="label">Plot Request</span>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              {viewingHistory && (
                <button className="btn-link" onClick={clearState}>
                  New Plot
                </button>
              )}
              <button
                className="btn-link"
                onClick={() => setShowTemplates(!showTemplates)}
              >
                {showTemplates ? "Hide templates" : "Show templates"}
              </button>
            </div>
          </div>

          {showTemplates && (
            <div className="plots-templates">
              {PLOT_TEMPLATES.map((t, i) => (
                <button
                  key={i}
                  className="plots-template-btn"
                  onClick={() => useTemplate(t)}
                >
                  {t.label}
                </button>
              ))}
            </div>
          )}

          <textarea
            className="input plots-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the plot you want to create..."
            rows={3}
          />

          <div className="plots-actions">
            <button
              className="btn primary"
              disabled={planning || !prompt.trim()}
              onClick={doPlan}
            >
              {planning ? "Generating\u2026" : "Generate Plot"}
            </button>
            <button
              className="btn"
              disabled={!canRun || running}
              onClick={doRun}
            >
              {running ? "Running\u2026" : "Run Script"}
            </button>
            {canRepair && (
              <button className="btn" onClick={doRepairManual} disabled={repairing}>
                {repairing ? "Fixing\u2026" : `Auto-Fix (${MAX_REPAIRS - repairCount} left)`}
              </button>
            )}
          </div>
          {repairCount >= MAX_REPAIRS && exitCode != null && exitCode !== 0 && (
            <div className="plots-error" style={{ marginTop: 8 }}>
              Auto-fix limit reached ({MAX_REPAIRS} attempts). Review the script and error output manually, or regenerate with a different prompt.
            </div>
          )}
        </div>

        {err && (
          <div className="plots-error">
            {err.includes("OPENAI_API_KEY") ? (
              <>
                <strong>API Key Missing:</strong> Set your OPENAI_API_KEY environment variable on the server to enable LLM-driven plotting.
              </>
            ) : err.includes("[network]") ? (
              <>
                <strong>Connection Error:</strong> Could not reach the backend server. Make sure it is running at {API_BASE}.
              </>
            ) : (
              err
            )}
          </div>
        )}

        {notes && (
          <div className="plots-section">
            <div className="label">Notes</div>
            <div className="plots-notes">{notes}</div>
          </div>
        )}

        {questions.length > 0 && (
          <div className="plots-section plots-questions-box">
            <div className="label">Clarification Needed</div>
            <ul className="plots-questions">
              {questions.map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>
          </div>
        )}

        {validationWarnings.length > 0 && (
          <div className="plots-section" style={{ padding: "8px 12px" }}>
            {validationWarnings.map((w, i) => (
              <div
                key={i}
                style={{
                  padding: "6px 10px",
                  marginBottom: 4,
                  borderRadius: 6,
                  fontSize: 12,
                  background: w.level === "error" ? "#fff0f0" : "#fffbe6",
                  border: `1px solid ${w.level === "error" ? "#ffa0a0" : "#ffe58f"}`,
                  color: w.level === "error" ? "#a8071a" : "#ad6800",
                }}
              >
                <strong>{w.level === "error" ? "Error" : "Warning"}:</strong> {w.message}
                {w.fix_hint && (
                  <div style={{ marginTop: 2, opacity: 0.8 }}>{w.fix_hint}</div>
                )}
              </div>
            ))}
          </div>
        )}

        {(showScript || script) && (
          <div className="plots-section">
            <div className="plots-section-header">
              <span className="label">Script</span>
              <button
                className="btn-link"
                onClick={() => setShowScript(!showScript)}
              >
                {showScript ? "Hide" : "Show"}
              </button>
            </div>
            {showScript && (
              <textarea
                className="input plots-script"
                value={script}
                onChange={(e) => setScript(e.target.value)}
                rows={10}
              />
            )}
          </div>
        )}

        {filesUsed.length > 0 && (
          <div className="plots-section">
            <div className="label">Files Used</div>
            <div className="plots-files">
              {filesUsed.map((f) => (
                <span key={f} className="plots-file-tag">
                  {f.split("/").pop()}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right Panel: Output & Results */}
      <div className="plots-right">
        <div className="plots-section plots-output-section">
          <div className="label">Plot Output</div>

          {imageOutputs.length > 0 && selectedRunId ? (
            <div className="plots-images">
              {imageOutputs.map((o) => (
                <div key={o.path} className="plots-image-container">
                  <img
                    src={outputUrl(inputsDir, selectedRunId, o.path, workspace)}
                    alt={o.name}
                    className="plots-image"
                    onError={(e) => {
                      // Retry once with just the filename (fallback for path mismatch)
                      const img = e.currentTarget;
                      const fname = o.path.split(/[\\/]/).pop() || o.name;
                      const retryUrl = outputUrl(inputsDir, selectedRunId, fname, workspace);
                      if (img.src !== retryUrl) {
                        img.src = retryUrl;
                      } else {
                        // Show broken image placeholder
                        img.style.display = "none";
                        const notice = img.parentElement?.querySelector(".plots-image-error");
                        if (notice) (notice as HTMLElement).style.display = "block";
                      }
                    }}
                  />
                  <div className="plots-image-error" style={{ display: "none", padding: "20px", textAlign: "center", color: "#888", fontSize: 12 }}>
                    Image could not be loaded. Try downloading the run outputs.
                  </div>
                  <div className="plots-image-name">{o.name}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="plots-placeholder">
              {autoFixing ? (
                <div className="plots-loading">
                  <div className="plots-spinner" />
                  <span>
                    {repairing
                      ? `Fixing script (attempt ${autoFixAttempt}/${MAX_REPAIRS})...`
                      : running
                        ? `Re-running fixed script (attempt ${autoFixAttempt}/${MAX_REPAIRS})...`
                        : `Auto-repair in progress...`}
                  </span>
                </div>
              ) : running ? (
                <div className="plots-loading">
                  <div className="plots-spinner" />
                  <span>Running script...</span>
                </div>
              ) : planning ? (
                <div className="plots-loading">
                  <div className="plots-spinner" />
                  <span>Generating plot (reading data, writing script, executing)...</span>
                </div>
              ) : repairing ? (
                <div className="plots-loading">
                  <div className="plots-spinner" />
                  <span>Fixing script...</span>
                </div>
              ) : (
                <>
                  <div className="plots-placeholder-icon">📊</div>
                  {exitCode !== null && exitCode === 0 && imageOutputs.length === 0 ? (
                    <>
                      <div>Script ran successfully but produced no image files</div>
                      <div className="muted">
                        Check the stdout output below, or try a different prompt
                      </div>
                    </>
                  ) : exitCode !== null && exitCode !== 0 ? (
                    <>
                      <div>Script encountered an error</div>
                      <div className="muted">
                        {canRepair
                          ? 'Click "Auto-Fix" to let the LLM try to repair the script'
                          : "Review the error below and try regenerating with a different prompt"}
                      </div>
                    </>
                  ) : (
                    <>
                      <div>Your plot will appear here</div>
                      <div className="muted">
                        Enter a prompt and click "Generate Plot" to start
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        {/* Stdout / stderr details (shown when there is output) */}
        {(stdout || stderr) && (
          <div className="plots-section">
            <details className="plots-output-details">
              <summary>
                Output {exitCode !== null && (
                  <span className={exitCode === 0 ? "status-ok" : "status-error"}>
                    (exit {exitCode})
                  </span>
                )}
              </summary>
              {stderr && (
                <pre className="plots-output-pre" style={{ borderLeft: "3px solid #c5221f" }}>
                  {stderr}
                </pre>
              )}
              {stdout && (
                <pre className="plots-output-pre">
                  {stdout}
                </pre>
              )}
            </details>
          </div>
        )}

        {/* Recent Runs Panel */}
        <div className="plots-section">
          <div className="plots-section-header">
            <span className="label">Recent Runs ({runs.length})</span>
            {runs.length > 0 && (
              <button
                className="btn-link"
                onClick={() => setShowRunsPanel(!showRunsPanel)}
              >
                {showRunsPanel ? "Collapse" : "Expand"}
              </button>
            )}
          </div>

          {runs.length === 0 ? (
            <div style={{ fontSize: 12, color: "#999", padding: "4px 0" }}>
              No runs yet. Generate and run a script to see history here.
            </div>
          ) : !showRunsPanel ? (
            /* Collapsed: show compact chips */
            <div className="plots-runs">
              {runs.slice(0, 5).map((r) => (
                <button
                  key={r.run_id}
                  className={`plots-run-chip ${selectedRunId === r.run_id ? "active" : ""}`}
                  onClick={() => loadRunDetail(r.run_id)}
                  title={r.prompt || r.run_id}
                >
                  <span className={r.exit_code === 0 ? "status-ok" : "status-error"}>
                    {r.exit_code === 0 ? "\u2713" : "\u2717"}
                  </span>
                  <span className="plots-run-chip-label">
                    {r.prompt
                      ? r.prompt.length > 24
                        ? r.prompt.slice(0, 24) + "\u2026"
                        : r.prompt
                      : r.run_id.slice(0, 8)}
                  </span>
                </button>
              ))}
              {runs.length > 5 && (
                <button
                  className="btn-link"
                  style={{ fontSize: 11 }}
                  onClick={() => setShowRunsPanel(true)}
                >
                  +{runs.length - 5} more
                </button>
              )}
            </div>
          ) : (
            /* Expanded: full run list with details and delete */
            <div className="plots-runs-list">
              {runs.map((r) => (
                <div
                  key={r.run_id}
                  className={`plots-run-row ${selectedRunId === r.run_id ? "active" : ""}`}
                >
                  <button
                    className="plots-run-row-main"
                    onClick={() => loadRunDetail(r.run_id)}
                  >
                    <span className={`plots-run-status ${r.exit_code === 0 ? "status-ok" : "status-error"}`}>
                      {r.exit_code === 0 ? "\u2713" : "\u2717"}
                    </span>
                    <div className="plots-run-info">
                      <div className="plots-run-prompt-text">
                        {r.prompt || <span className="muted">No prompt</span>}
                      </div>
                      <div className="plots-run-meta">
                        <span className="plots-run-id-text">{r.run_id.slice(0, 12)}</span>
                        {r.ts_utc && (
                          <span className="plots-run-time">{fmtTime(r.ts_utc)}</span>
                        )}
                      </div>
                    </div>
                  </button>
                  <button
                    className="plots-run-delete"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteRun(r.run_id);
                    }}
                    title="Delete this run"
                  >
                    &times;
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
