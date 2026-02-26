import React, { useCallback, useEffect, useRef, useState } from "react";
import { inspectProject, importProject, setupProject, workspaceScan } from "./api";
import { Sidebar } from "./components/Sidebar";
import { TopTabs, TopTab } from "./components/TopTabs";
import ChatPanel from "./components/ChatPanel";
import { ArtifactsTab } from "./components/ArtifactsTab";
import { PlotsTab } from "./components/PlotsTab";
import { ModelFilesTab } from "./components/ModelFilesTab";
import { Model3DTab } from "./components/Model3DTab";
import { MapTab } from "./components/MapTab";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { LLMSettingsModal } from "./components/LLMSettingsModal";
import { ErrorBoundary } from "./components/ErrorBoundary";

function normalizeWorkspace(workspace: string | null | undefined): string | null {
  if (!workspace) return null;
  const w = String(workspace).replace(/\\/g, "/");
  // If backend returns an absolute Windows path, reduce to relative "workspace" when possible.
  if (w.includes(":") || w.startsWith("/")) {
    if (w.endsWith("/workspace") || w.endsWith("\\workspace")) return "workspace";
    return null;
  }
  // Common case: "runs/aoi_demo/workspace" -> "workspace"
  if (w.endsWith("/workspace")) return "workspace";
  return workspace;
}

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "";

export function App() {
  const [inputsDir, setInputsDir] = useState("");
  const [workspace, setWorkspace] = useState<string | null>(null);
  const [artifactsDir, setArtifactsDir] = useState<string | null>(null);
  const [recentArtifacts, setRecentArtifacts] = useState<string[]>([]);
  const [tab, setTab] = useState<TopTab>("Artifacts");
  const [importPlan, setImportPlan] = useState<any | null>(null);
  const [importing, setImporting] = useState(false);
  const [scanStatus, setScanStatus] = useState<string | null>(null);
  const [showLLMSettings, setShowLLMSettings] = useState(false);
  const [hasValidProject, setHasValidProject] = useState(false);
  const [backendOk, setBackendOk] = useState<boolean | null>(null); // null = checking

  // Periodic backend health check
  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 3000);
        const r = await fetch(`${API_BASE}/docs`, { method: "HEAD", signal: ctrl.signal });
        clearTimeout(timer);
        if (!cancelled) setBackendOk(r.ok || r.status === 200 || r.status === 307);
      } catch {
        if (!cancelled) setBackendOk(false);
      }
    }
    check();
    const interval = setInterval(check, 30000); // re-check every 30s
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  // Resizable split between top (tabs) and bottom (chat)
  const mainRef = useRef<HTMLDivElement | null>(null);
  const [topPx, setTopPx] = useState<number>(420);
  const [isDragging, setIsDragging] = useState(false);
  const draggingRef = useRef(false);

  const handleSplitterPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    draggingRef.current = true;
    setIsDragging(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
  }, []);

  const handleSplitterPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    const el = mainRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const minTop = 220;
    const minBottom = 260;
    const maxTop = Math.max(minTop, rect.height - minBottom);
    const clamped = Math.min(maxTop, Math.max(minTop, y));
    setTopPx(clamped);
  }, []);

  const handleSplitterPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current = false;
    setIsDragging(false);
    try { (e.target as HTMLElement).releasePointerCapture(e.pointerId); } catch {}
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  // Warm the workspace scan cache whenever we successfully resolve a workspace.
  // This improves chat + plots grounding and avoids first-use latency.
  useEffect(() => {
    if (!inputsDir) return;
    // workspace can be null for legacy projects; still attempt scan (backend will infer).
    const t = window.setTimeout(async () => {
      try {
        setScanStatus("Scanning workspace…");
        await workspaceScan({ inputs_dir: inputsDir, workspace: workspace ?? null, force: false });
        setScanStatus("Workspace indexed");
      } catch (e: any) {
        // Non-fatal — the app should still be usable.
        setScanStatus(null);
      }
    }, 250);
    return () => window.clearTimeout(t);
  }, [inputsDir, workspace]);

  async function doInspect() {
    if (!inputsDir.trim()) {
      setHasValidProject(false);
      return;
    }
    
    try {
      const resp = await inspectProject(inputsDir);
      if (resp?.status === "ok" && resp?.workspace_info) {
        const info = resp.workspace_info;
        setWorkspace(normalizeWorkspace(info.workspace ?? null));
        setArtifactsDir(info.artifacts_dir ?? null);
        setRecentArtifacts(info.recent_artifacts ?? []);
        setImportPlan(null);
        setHasValidProject(true);
        return;
      }
      if ((resp?.status === "needs_import" || resp?.status === "needs_setup") && resp?.import_plan) {
        setImportPlan({ ...resp.import_plan, _setup_mode: resp.status === "needs_setup" });
        setHasValidProject(false);
        return;
      }
      setImportPlan(null);
      setHasValidProject(false);
      alert(resp?.message ?? "Could not inspect that folder.");
    } catch (e: any) {
      setHasValidProject(false);
      // Don't alert on initial load
      if (inputsDir.trim()) {
        console.warn("Inspect failed:", e);
      }
    }
  }

  async function doImport() {
    if (!importPlan) return;
    try {
      setImporting(true);

      let res: any;
      if (importPlan._setup_mode) {
        // New flow: create GW_Copilot/ in-place (no file copying)
        res = await setupProject(importPlan.source_path);
      } else {
        // Legacy flow: copy files into runs/imported/
        res = await importProject(importPlan);
      }

      const info = res?.workspace_info;
      if (res?.status === "ok" && (res?.project_path || importPlan.source_path) && info) {
        setInputsDir(res.project_path || importPlan.source_path);
        setWorkspace(normalizeWorkspace(info.workspace ?? null));
        setArtifactsDir(info.artifacts_dir ?? null);
        setRecentArtifacts(info.recent_artifacts ?? []);
        setImportPlan(null);
        setHasValidProject(true);
      } else {
        alert("Setup did not return expected project info.");
      }
    } catch (e: any) {
      alert(e?.message ?? String(e));
    } finally {
      setImporting(false);
    }
  }

  return (
    <div className="layout">
      {/* Backend connection warning */}
      {backendOk === false && (
        <div className="connection-banner">
          <span>⚠️ Cannot reach the backend server at {API_BASE}.</span>
          <span className="muted" style={{ marginLeft: 8 }}>
            Make sure it is running, then refresh.
          </span>
        </div>
      )}

      {/* LLM Settings Modal */}
      {showLLMSettings && (
        <LLMSettingsModal onClose={() => setShowLLMSettings(false)} />
      )}

      {/* Welcome Screen when no project selected */}
      {!inputsDir.trim() && (
        <WelcomeScreen
          onSelectFolder={(path) => {
            setInputsDir(path);
            // Auto-inspect after selection
            setTimeout(() => doInspect(), 100);
          }}
          onOpenSettings={() => setShowLLMSettings(true)}
        />
      )}

      {/* Main App Layout */}
      {inputsDir.trim() && (
        <>
          <Sidebar
            inputsDir={inputsDir}
            setInputsDir={setInputsDir}
            onInspect={doInspect}
            recentArtifacts={recentArtifacts}
            scanStatus={scanStatus}
            workspace={workspace}
            onOpenSettings={() => setShowLLMSettings(true)}
          />

          <div className="main" ref={mainRef}>
            {/* Transparent overlay during splitter drag to prevent iframes stealing events */}
            {isDragging && (
              <div style={{
                position: "absolute", inset: 0, zIndex: 9999,
                cursor: "row-resize", background: "transparent",
              }} />
            )}
            {importPlan && (
              <div className="modalOverlay">
                <div className="modal">
                  <div className="modalTitle">
                    {importPlan._setup_mode
                      ? "Set up GW Copilot in this folder?"
                      : "Create Copilot project wrapper?"}
                  </div>
                  <div className="modalBody">
                    {importPlan._setup_mode ? (
                      <div className="muted" style={{ marginBottom: 8 }}>
                        A MODFLOW 6 workspace was detected. A <code>GW_Copilot/</code> folder
                        will be created inside this directory to store validation reports, plots,
                        and other tool outputs. <strong>Your original model files will not be modified.</strong>
                      </div>
                    ) : (
                      <div className="muted" style={{ marginBottom: 8 }}>
                        No Copilot project was found in the selected folder. I detected a MODFLOW 6 workspace and can create a
                        wrapper project under <code>runs/imported/</code>.
                      </div>
                    )}

                    <div style={{ marginBottom: 8 }}>
                      <div>
                        <strong>Model folder:</strong> <code>{importPlan.source_path}</code>
                      </div>
                      {importPlan._setup_mode ? (
                        <div>
                          <strong>GW Copilot folder:</strong> <code>{importPlan.source_path}/GW_Copilot/</code>
                        </div>
                      ) : (
                        <div>
                          <strong>Destination:</strong> <code>{importPlan.proposed_project_path}</code>
                        </div>
                      )}
                    </div>

                    {importPlan.warnings?.length ? (
                      <div className="modalWarnings">
                        <div className="label">Warnings</div>
                        <ul>
                          {importPlan.warnings.map((w: string, i: number) => (
                            <li key={i}>{w}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}

                    {!importPlan._setup_mode && importPlan.actions?.length > 0 && (
                      <details>
                        <summary>Planned actions</summary>
                        <pre className="logPre">{JSON.stringify(importPlan.actions ?? [], null, 2)}</pre>
                      </details>
                    )}
                  </div>
                  <div className="modalActions">
                    <button className="btn" onClick={() => setImportPlan(null)} disabled={importing}>
                      Cancel
                    </button>
                    <button className="btn primary" onClick={doImport} disabled={importing}>
                      {importing
                        ? "Setting up…"
                        : importPlan._setup_mode
                          ? "Set up GW Copilot"
                          : "Create project"}
                    </button>
                  </div>
                </div>
              </div>
            )}

            <div className="topPane" style={{ height: topPx }}>
              <TopTabs tab={tab} setTab={setTab}>
                {tab === "Map" && (
                  <ErrorBoundary fallbackLabel="Map">
                    <MapTab inputsDir={inputsDir} workspace={workspace ?? null} />
                  </ErrorBoundary>
                )}

                {tab === "3D" && (
                  <ErrorBoundary fallbackLabel="3D View">
                    <Model3DTab inputsDir={inputsDir} workspace={workspace ?? null} />
                  </ErrorBoundary>
                )}

                {tab === "Artifacts" && (
                  <ErrorBoundary fallbackLabel="Artifacts">
                    <ArtifactsTab artifactsDir={artifactsDir} recentArtifacts={recentArtifacts} />
                  </ErrorBoundary>
                )}

                {tab === "Model Files" && (
                  <ErrorBoundary fallbackLabel="Model Files">
                    <ModelFilesTab inputsDir={inputsDir} workspace={workspace ?? undefined} />
                  </ErrorBoundary>
                )}

                {tab === "Plots" && (
                  <ErrorBoundary fallbackLabel="Plots">
                    <PlotsTab inputsDir={inputsDir} workspace={workspace} />
                  </ErrorBoundary>
                )}
              </TopTabs>
            </div>

            <div
              className="splitter"
              role="separator"
              aria-label="Resize panels"
              onPointerDown={handleSplitterPointerDown}
              onPointerMove={handleSplitterPointerMove}
              onPointerUp={handleSplitterPointerUp}
              style={{ touchAction: "none" }}
            />

            <div className="bottomPane">
              <ErrorBoundary fallbackLabel="Chat">
                <ChatPanel inputsDir={inputsDir} workspace={workspace ?? undefined} onJobDone={doInspect} />
              </ErrorBoundary>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
