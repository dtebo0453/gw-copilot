import React, { useEffect, useState } from "react";
import { workspaceFacts } from "../api";

type Props = {
  inputsDir: string;
  workspace?: string | null;
};

type Fact = {
  key: string;
  value: any;
  source: string;
};

type LayerInfo = {
  layer: number;
  thickness: { min: number | null; mean: number | null; max: number | null };
  active_frac: number;
};

type StressSummary = {
  records?: number;
  periods?: number;
  range?: [number, number];
};

type OutputMeta = {
  hds?: {
    times?: number;
    shape?: number[];
    head_range?: [number, number];
  };
  cbc?: {
    times?: number;
    components?: string[];
    precision?: string;
  };
};

type SolverInfo = {
  outer_dvclose?: number;
  inner_dvclose?: number;
  outer_maximum?: number;
  inner_maximum?: number;
  linear_acceleration?: string;
  file?: string;
};

type FactsResponse = {
  ok: boolean;
  workspace_root: string;
  simulator?: string | null;
  grid: {
    type: string;
    nlay?: number;
    nrow?: number;
    ncol?: number;
    ncpl?: number;
    nodes?: number;
    file?: string;
    delr_range?: [number, number];
    delc_range?: [number, number];
  } | null;
  tdis: {
    nper: number | null;
    time_units: string | null;
    total_time: number | null;
  } | null;
  packages: string[];
  outputs_present: {
    lst: boolean;
    hds: boolean;
    cbc: boolean;
    obs: boolean;
  };
  facts: Fact[];
  extraction_method: string;
  layers?: LayerInfo[];
  stress_summaries?: Record<string, StressSummary>;
  output_metadata?: OutputMeta;
  solver?: SolverInfo | null;
};

// ── Formatting helpers ──────────────────────────────────────────────

/** Format a number in a human-friendly way: avoid scientific notation for
 *  everyday values, use compact notation for large/tiny values. */
function fmtNum(v: number, decimals = 2): string {
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  // Very tiny values — show scientific
  if (abs < 0.001) return v.toExponential(decimals);
  // Normal range — use locale formatting
  if (abs < 1_000_000) return v.toLocaleString(undefined, { maximumFractionDigits: decimals });
  // Large — compact with K/M/B
  if (abs >= 1e9) return (v / 1e9).toFixed(1) + "B";
  if (abs >= 1e6) return (v / 1e6).toFixed(1) + "M";
  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

/** Format a range as "lo – hi" using consistent decimal places. */
function fmtRange(lo: number, hi: number): string {
  return `${fmtNum(lo, 2)} to ${fmtNum(hi, 2)}`;
}

/** MODFLOW time-unit codes: 0=undef, 1=sec, 2=min, 3=hr, 4=day, 5=yr */
const TIME_UNIT_SECONDS: Record<string, number> = {
  seconds: 1,
  minutes: 60,
  hours: 3600,
  days: 86400,
  years: 365.25 * 86400,
};

/** Convert a raw total_time + time_units string to a human-friendly label.
 *  E.g. 15552000 "seconds" → "180 days" */
function fmtTime(totalTime: number, units: string | null): string {
  if (!units) return fmtNum(totalTime);
  const ul = units.toLowerCase();
  const secPerUnit = TIME_UNIT_SECONDS[ul] ?? 1;
  const totalSec = totalTime * secPerUnit;

  // Try to express in the most human-readable unit
  const candidates: [string, number][] = [
    ["years", 365.25 * 86400],
    ["days", 86400],
    ["hours", 3600],
    ["minutes", 60],
    ["seconds", 1],
  ];

  for (const [label, factor] of candidates) {
    const val = totalSec / factor;
    if (val >= 1 && Number.isFinite(val)) {
      // Prefer whole numbers
      if (Math.abs(val - Math.round(val)) < 0.01) {
        return `${Math.round(val).toLocaleString()} ${label}`;
      }
      if (val >= 10) {
        return `${Math.round(val).toLocaleString()} ${label}`;
      }
      return `${val.toFixed(1)} ${label}`;
    }
  }
  return `${fmtNum(totalTime)} ${units}`;
}

/** Format cell-size ranges for DELR/DELC into something clean.
 *  If uniform: "1500 × 1500 m"
 *  If variable: "1,500–8,000 × 1,500–7,000 m" */
function fmtCellSize(
  delr?: [number, number] | null,
  delc?: [number, number] | null,
): string | null {
  if (!delr) return null;
  const rUniform = Math.abs(delr[0] - delr[1]) < 0.01;
  const cUniform = delc ? Math.abs(delc[0] - delc[1]) < 0.01 : true;

  const fmtDim = (range: [number, number], uniform: boolean) => {
    if (uniform) return fmtNum(range[0], 0);
    return `${fmtNum(range[0], 0)}–${fmtNum(range[1], 0)}`;
  };

  const r = fmtDim(delr, rUniform);
  const c = delc ? fmtDim(delc, cUniform) : null;

  if (rUniform && cUniform && c) {
    return `${r} × ${c}`;
  }
  if (c) {
    return `${r} × ${c} (variable)`;
  }
  return r;
}

// ── Component ───────────────────────────────────────────────────────

export function ModelFactsPanel({ inputsDir, workspace }: Props) {
  const [facts, setFacts] = useState<FactsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(true);

  useEffect(() => {
    if (!inputsDir) return;

    setLoading(true);
    setError(null);

    workspaceFacts({ inputs_dir: inputsDir, workspace: workspace ?? undefined })
      .then((data) => {
        setFacts(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err?.message || String(err));
        setLoading(false);
      });
  }, [inputsDir, workspace]);

  if (!inputsDir) {
    return null;
  }

  const grid = facts?.grid;
  const tdis = facts?.tdis;
  const outputs = facts?.outputs_present;
  const packages = facts?.packages || [];
  const stress = facts?.stress_summaries || {};
  const outMeta = facts?.output_metadata || {};
  const solver = facts?.solver;

  // Format grid dimensions
  const gridDims = grid
    ? grid.type === "dis"
      ? `${grid.nlay} × ${grid.nrow} × ${grid.ncol}`
      : grid.type === "disv"
      ? `${grid.nlay} layers, ${grid.ncpl?.toLocaleString()} cells/layer`
      : grid.type === "disu"
      ? `${grid.nodes?.toLocaleString()} nodes`
      : "Unknown"
    : "—";

  // Compute total cells for display
  const totalCells = grid
    ? grid.type === "dis" && grid.nlay && grid.nrow && grid.ncol
      ? (grid.nlay * grid.nrow * grid.ncol).toLocaleString()
      : grid.type === "disv" && grid.nlay && grid.ncpl
      ? (grid.nlay * grid.ncpl).toLocaleString()
      : grid.type === "disu" && grid.nodes
      ? grid.nodes.toLocaleString()
      : null
    : null;

  const cellSize = fmtCellSize(grid?.delr_range, grid?.delc_range);

  return (
    <div className="facts-panel">
      <div
        className="facts-header"
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: "pointer", userSelect: "none" }}
      >
        <span className="facts-title">
          {expanded ? "▼" : "▶"} Model Facts
        </span>
        {facts?.extraction_method && (
          <span className="facts-method">
            via {facts.extraction_method === "flopy" ? "FloPy" : "text parsing"}
          </span>
        )}
      </div>

      {expanded && (
        <div className="facts-body">
          {loading && <div className="muted">Loading facts...</div>}

          {error && (
            <div className="facts-error">
              <strong>Error:</strong> {error}
            </div>
          )}

          {facts && !loading && (
            <>
              {/* ── Simulator ── */}
              {facts.simulator && (
                <div className="facts-section">
                  <div className="facts-row">
                    <span className="facts-label">Simulator</span>
                    <span className="facts-value">{facts.simulator}</span>
                  </div>
                </div>
              )}

              {/* ── Grid Info ── */}
              <div className="facts-section">
                <div className="facts-row">
                  <span className="facts-label">Grid</span>
                  <span className="facts-value">
                    {grid?.type?.toUpperCase() || "—"}{" "}
                    {gridDims !== "—" && `(${gridDims})`}
                  </span>
                </div>
                {totalCells && (
                  <div className="facts-row">
                    <span className="facts-label">Total Cells</span>
                    <span className="facts-value">{totalCells}</span>
                  </div>
                )}
                {cellSize && (
                  <div className="facts-row">
                    <span className="facts-label">Cell Size</span>
                    <span className="facts-value">{cellSize}</span>
                  </div>
                )}
              </div>

              {/* ── Layer Thickness ── */}
              {facts?.layers && facts.layers.length > 0 && (
                <div className="facts-section">
                  <details>
                    <summary className="facts-detail-summary">
                      Layer Thickness ({facts.layers.length} layers)
                    </summary>
                    <div className="facts-detail-body">
                      {facts.layers.map((l) => (
                        <div key={l.layer} className="facts-row facts-row-sm">
                          <span className="facts-label">Layer {l.layer}</span>
                          <span className="facts-value">
                            {l.thickness.min != null
                              ? `${fmtNum(l.thickness.min, 1)} / ${fmtNum(l.thickness.mean ?? 0, 1)} / ${fmtNum(l.thickness.max ?? 0, 1)}`
                              : "N/A"}
                            {l.active_frac < 1.0 &&
                              ` · ${(l.active_frac * 100).toFixed(0)}% active`}
                          </span>
                        </div>
                      ))}
                      <div className="facts-detail-note">min / avg / max</div>
                    </div>
                  </details>
                </div>
              )}

              {/* ── Temporal Info ── */}
              <div className="facts-section">
                <div className="facts-row">
                  <span className="facts-label">Stress Periods</span>
                  <span className="facts-value">{tdis?.nper ?? "—"}</span>
                </div>
                {tdis?.total_time != null && (
                  <div className="facts-row">
                    <span className="facts-label">Total Time</span>
                    <span className="facts-value">
                      {fmtTime(tdis.total_time, tdis.time_units)}
                    </span>
                  </div>
                )}
              </div>

              {/* ── Packages ── */}
              {packages.length > 0 && (
                <div className="facts-section">
                  <div className="facts-row" style={{ alignItems: "flex-start" }}>
                    <span className="facts-label">Packages</span>
                    <span className="facts-value facts-tag-wrap">
                      {packages.map((pkg) => (
                        <span key={pkg} className="facts-tag">
                          {pkg}
                        </span>
                      ))}
                    </span>
                  </div>
                </div>
              )}

              {/* ── Outputs ── */}
              {outputs && (
                <div className="facts-section">
                  <div className="facts-row">
                    <span className="facts-label">Outputs</span>
                    <span className="facts-value facts-tag-wrap">
                      {outputs.hds && <span className="facts-tag ok">HDS</span>}
                      {outputs.cbc && <span className="facts-tag ok">CBC</span>}
                      {outputs.lst && <span className="facts-tag ok">LST</span>}
                      {outputs.obs && <span className="facts-tag ok">OBS</span>}
                      {!outputs.hds && <span className="facts-tag missing">no HDS</span>}
                      {!outputs.cbc && <span className="facts-tag missing">no CBC</span>}
                    </span>
                  </div>
                </div>
              )}

              {/* ── Output Details (collapsible) ── */}
              {(outMeta.hds || outMeta.cbc) && (
                <div className="facts-section">
                  <details>
                    <summary className="facts-detail-summary">
                      Output Details
                    </summary>
                    <div className="facts-detail-body">
                      {outMeta.hds && (
                        <>
                          <div className="facts-row facts-row-sm">
                            <span className="facts-label">HDS Timesteps</span>
                            <span className="facts-value">{outMeta.hds.times ?? "—"}</span>
                          </div>
                          {outMeta.hds.head_range && (
                            <div className="facts-row facts-row-sm">
                              <span className="facts-label">Head Range</span>
                              <span className="facts-value">
                                {fmtRange(outMeta.hds.head_range[0], outMeta.hds.head_range[1])}
                              </span>
                            </div>
                          )}
                        </>
                      )}
                      {outMeta.cbc && (
                        <>
                          <div className="facts-row facts-row-sm">
                            <span className="facts-label">CBC Timesteps</span>
                            <span className="facts-value">{outMeta.cbc.times ?? "—"}</span>
                          </div>
                          {outMeta.cbc.components && outMeta.cbc.components.length > 0 && (
                            <div className="facts-row facts-row-sm" style={{ alignItems: "flex-start" }}>
                              <span className="facts-label">Budget Terms</span>
                              <span className="facts-value facts-tag-wrap">
                                {outMeta.cbc.components.map((c) => (
                                  <span key={c} className="facts-tag">{c}</span>
                                ))}
                              </span>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </details>
                </div>
              )}

              {/* ── Stress Package Summaries (collapsible) ── */}
              {Object.keys(stress).length > 0 && (
                <div className="facts-section">
                  <details>
                    <summary className="facts-detail-summary">
                      Stress Data ({Object.keys(stress).length} packages)
                    </summary>
                    <div className="facts-detail-body">
                      {Object.entries(stress).map(([pkg, s]) => (
                        <div key={pkg} className="facts-row facts-row-sm">
                          <span className="facts-label facts-pkg-label">{pkg}</span>
                          <span className="facts-value">
                            {s.records != null && (
                              <span>{s.records.toLocaleString()} rec</span>
                            )}
                            {s.periods != null && (
                              <span className="facts-sep"> · {s.periods} per</span>
                            )}
                            {s.range && (
                              <span className="facts-sep">
                                {" "}· [{fmtNum(s.range[0], 3)} .. {fmtNum(s.range[1], 3)}]
                              </span>
                            )}
                          </span>
                        </div>
                      ))}
                    </div>
                  </details>
                </div>
              )}

              {/* ── Solver Info ── */}
              {solver && (
                <div className="facts-section">
                  <details>
                    <summary className="facts-detail-summary">
                      Solver
                    </summary>
                    <div className="facts-detail-body">
                      {solver.outer_maximum != null && (
                        <div className="facts-row facts-row-sm">
                          <span className="facts-label">Max Outer Iter</span>
                          <span className="facts-value">{solver.outer_maximum}</span>
                        </div>
                      )}
                      {solver.inner_maximum != null && (
                        <div className="facts-row facts-row-sm">
                          <span className="facts-label">Max Inner Iter</span>
                          <span className="facts-value">{solver.inner_maximum}</span>
                        </div>
                      )}
                      {solver.outer_dvclose != null && (
                        <div className="facts-row facts-row-sm">
                          <span className="facts-label">Outer Close</span>
                          <span className="facts-value">{fmtNum(solver.outer_dvclose)}</span>
                        </div>
                      )}
                      {solver.inner_dvclose != null && (
                        <div className="facts-row facts-row-sm">
                          <span className="facts-label">Inner Close</span>
                          <span className="facts-value">{fmtNum(solver.inner_dvclose)}</span>
                        </div>
                      )}
                      {solver.linear_acceleration && (
                        <div className="facts-row facts-row-sm">
                          <span className="facts-label">Acceleration</span>
                          <span className="facts-value">{solver.linear_acceleration}</span>
                        </div>
                      )}
                    </div>
                  </details>
                </div>
              )}

              {/* ── Health warning ── */}
              {!facts.ok && (
                <div className="facts-warning">
                  Snapshot extraction incomplete — some features may be limited.
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
