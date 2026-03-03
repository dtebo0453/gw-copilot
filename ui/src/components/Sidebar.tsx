import React, { useState } from "react";
import { fsBrowse } from "../api";
import { ModelFactsPanel } from "./ModelFactsPanel";

export function Sidebar({
  inputsDir,
  setInputsDir,
  onInspect,
  scanStatus,
  workspace,
  onOpenSettings,
}: {
  inputsDir: string;
  setInputsDir: (v: string) => void;
  onInspect: () => void;
  scanStatus?: string | null;
  workspace?: string | null;
  onOpenSettings?: () => void;
}) {
  const [browsing, setBrowsing] = useState(false);

  async function handleBrowse() {
    setBrowsing(true);
    try {
      const path = await fsBrowse();
      if (path) setInputsDir(path);
    } catch {
      // ignore - user may have cancelled or tkinter unavailable
    } finally {
      setBrowsing(false);
    }
  }

  return (
    <div className="sidebar">
      <div className="sidebarHeader">
        <div className="brand">GW Copilot</div>
        {onOpenSettings && (
          <button className="btn-icon" onClick={onOpenSettings} title="LLM Settings">
            ⚙️
          </button>
        )}
      </div>

      <div className="sidebarSection">
        <label className="label">Inputs dir</label>
        <div className="row" style={{ gap: 4 }}>
          <input
            className="input"
            value={inputsDir}
            onChange={(e) => setInputsDir(e.target.value)}
            placeholder="runs/aoi_demo"
            style={{ flex: 1 }}
          />
          <button
            className="btn"
            onClick={handleBrowse}
            disabled={browsing}
            title="Browse for folder"
            style={{ minWidth: 32, padding: "4px 8px", fontSize: 16, fontWeight: "bold" }}
          >
            +
          </button>
        </div>
        <button className="btn" onClick={onInspect}>
          Inspect
        </button>
        {scanStatus ? <div className="muted" style={{ marginTop: 6 }}>{scanStatus}</div> : null}
      </div>

      {/* Model Facts Panel */}
      <ModelFactsPanel inputsDir={inputsDir} workspace={workspace} />
    </div>
  );
}
