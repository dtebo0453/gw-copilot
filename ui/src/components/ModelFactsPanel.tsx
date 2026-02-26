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

type FactsResponse = {
  ok: boolean;
  workspace_root: string;
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
};

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

  // Format grid dimensions
  const gridDims = grid
    ? grid.type === "dis"
      ? `${grid.nlay} × ${grid.nrow} × ${grid.ncol}`
      : grid.type === "disv"
      ? `${grid.nlay} layers, ${grid.ncpl} cells/layer`
      : grid.type === "disu"
      ? `${grid.nodes} nodes`
      : "Unknown"
    : "—";

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
              {/* Grid Info */}
              <div className="facts-section">
                <div className="facts-row">
                  <span className="facts-label">Grid Type</span>
                  <span className="facts-value">
                    {grid?.type?.toUpperCase() || "—"}
                  </span>
                </div>
                <div className="facts-row">
                  <span className="facts-label">Dimensions</span>
                  <span className="facts-value">{gridDims}</span>
                </div>
                {grid?.nlay != null && (
                  <div className="facts-row">
                    <span className="facts-label">Number of Layers</span>
                    <span className="facts-value">{grid.nlay}</span>
                  </div>
                )}
                {grid?.delr_range && (
                  <div className="facts-row">
                    <span className="facts-label">Cell Size (DELR)</span>
                    <span className="facts-value">
                      {Math.abs(grid.delr_range[0] - grid.delr_range[1]) < 0.01
                        ? `Uniform: ${grid.delr_range[0].toFixed(2)}`
                        : `Variable: ${grid.delr_range[0].toFixed(2)} – ${grid.delr_range[1].toFixed(2)}`}
                    </span>
                  </div>
                )}
                {grid?.delc_range && (
                  <div className="facts-row">
                    <span className="facts-label">Cell Size (DELC)</span>
                    <span className="facts-value">
                      {Math.abs(grid.delc_range[0] - grid.delc_range[1]) < 0.01
                        ? `Uniform: ${grid.delc_range[0].toFixed(2)}`
                        : `Variable: ${grid.delc_range[0].toFixed(2)} – ${grid.delc_range[1].toFixed(2)}`}
                    </span>
                  </div>
                )}
              </div>

              {/* Layer Thickness Summary */}
              {facts?.layers && facts.layers.length > 0 && (
                <div className="facts-section">
                  <details>
                    <summary className="facts-label" style={{ cursor: "pointer" }}>
                      Layer Thickness ({facts.layers.length} layers)
                    </summary>
                    <div style={{ marginTop: 6 }}>
                      {facts.layers.map((l) => (
                        <div key={l.layer} className="facts-row" style={{ fontSize: 10 }}>
                          <span className="facts-label">Layer {l.layer}</span>
                          <span className="facts-value">
                            {l.thickness.min != null
                              ? `${l.thickness.min.toFixed(1)} / ${l.thickness.mean?.toFixed(1)} / ${l.thickness.max?.toFixed(1)} (min/avg/max)`
                              : "N/A"}
                            {l.active_frac < 1.0 &&
                              ` | ${(l.active_frac * 100).toFixed(0)}% active`}
                          </span>
                        </div>
                      ))}
                    </div>
                  </details>
                </div>
              )}

              {/* Time Info */}
              <div className="facts-section">
                <div className="facts-row">
                  <span className="facts-label">Stress Periods</span>
                  <span className="facts-value">{tdis?.nper ?? "—"}</span>
                </div>
                {tdis?.time_units && (
                  <div className="facts-row">
                    <span className="facts-label">Time Units</span>
                    <span className="facts-value">{tdis.time_units}</span>
                  </div>
                )}
                {tdis?.total_time != null && (
                  <div className="facts-row">
                    <span className="facts-label">Total Time</span>
                    <span className="facts-value">
                      {tdis.total_time.toLocaleString()}
                    </span>
                  </div>
                )}
              </div>

              {/* Packages */}
              {packages.length > 0 && (
                <div className="facts-section">
                  <div className="facts-row">
                    <span className="facts-label">Packages</span>
                    <span className="facts-value facts-packages">
                      {packages.map((pkg) => (
                        <span key={pkg} className="facts-tag">
                          {pkg}
                        </span>
                      ))}
                    </span>
                  </div>
                </div>
              )}

              {/* Outputs */}
              {outputs && (
                <div className="facts-section">
                  <div className="facts-row">
                    <span className="facts-label">Outputs</span>
                    <span className="facts-value facts-outputs">
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

              {/* Health status */}
              {!facts.ok && (
                <div className="facts-warning">
                  ⚠️ Snapshot extraction incomplete — some features may be limited.
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
