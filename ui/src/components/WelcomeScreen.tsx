import React, { useState } from "react";
import { fsFind, fsBrowse } from "../api";

type Props = {
  onSelectFolder: (path: string) => void;
  onOpenSettings: () => void;
};

export function WelcomeScreen({ onSelectFolder, onOpenSettings }: Props) {
  const [search, setSearch] = useState("");
  const [searching, setSearching] = useState(false);
  const [browsing, setBrowsing] = useState(false);
  const [manualPath, setManualPath] = useState("");
  const [matches, setMatches] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function doSearch() {
    const q = search.trim();
    if (q.length < 2) {
      setError("Enter at least 2 characters to search");
      return;
    }

    setSearching(true);
    setError(null);

    try {
      const res = await fsFind({ query: q, kind: "dir", max_results: 20 });
      setMatches(res?.matches ?? []);
      if ((res?.matches ?? []).length === 0) {
        setError("No matching folders found");
      }
    } catch (e: any) {
      setError(e?.message || String(e));
      setMatches([]);
    } finally {
      setSearching(false);
    }
  }

  async function handleBrowse() {
    setBrowsing(true);
    setError(null);
    try {
      const path = await fsBrowse();
      if (path) onSelectFolder(path);
    } catch (e: any) {
      setError("Could not open folder picker: " + (e?.message || String(e)));
    } finally {
      setBrowsing(false);
    }
  }

  function handleManualOpen() {
    const p = manualPath.trim();
    if (!p) {
      setError("Please enter a folder path");
      return;
    }
    onSelectFolder(p);
  }

  return (
    <div className="welcome-screen">
      <div className="welcome-content">
        <div className="welcome-header">
          <div className="welcome-logo">🌊</div>
          <h1 className="welcome-title">GW Copilot</h1>
          <p className="welcome-subtitle">
            AI-assisted groundwater modeling for MODFLOW 6
          </p>
        </div>

        <div className="welcome-section">
          <h2>Get Started</h2>
          <p className="muted">
            Select a MODFLOW 6 model folder to begin exploring and improving your groundwater model.
          </p>

          {/* Primary action: Browse for folder */}
          <button
            className="btn primary welcome-browse-btn"
            onClick={handleBrowse}
            disabled={browsing}
          >
            {browsing ? "Opening..." : "Browse for Model Folder"}
          </button>

          {/* Manual path entry */}
          <div className="welcome-divider">
            <span>or paste a folder path</span>
          </div>

          <div className="welcome-search-row">
            <input
              className="input welcome-search-input"
              type="text"
              placeholder="C:\path\to\your\model\folder"
              value={manualPath}
              onChange={(e) => setManualPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleManualOpen();
              }}
            />
            <button
              className="btn primary"
              onClick={handleManualOpen}
            >
              Open
            </button>
          </div>

          {/* Search section */}
          <div className="welcome-divider">
            <span>or search by name</span>
          </div>

          <div className="welcome-search">
            <div className="welcome-search-row">
              <input
                className="input welcome-search-input"
                type="text"
                placeholder="Search for a model folder (e.g., 'aoi_demo' or 'gwf')..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") doSearch();
                }}
              />
              <button
                className="btn"
                onClick={doSearch}
                disabled={searching}
              >
                {searching ? "Searching..." : "Search"}
              </button>
            </div>

            {error && <div className="welcome-error">{error}</div>}

            {matches.length > 0 && (
              <div className="welcome-results">
                {matches.map((m) => (
                  <button
                    key={m}
                    className="welcome-result-btn"
                    onClick={() => onSelectFolder(m)}
                  >
                    <span className="welcome-result-icon">📁</span>
                    <span className="welcome-result-path">{m}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="welcome-section">
          <h3>Features</h3>
          <div className="welcome-features">
            <div className="welcome-feature">
              <span className="welcome-feature-icon">💬</span>
              <div>
                <strong>Chat Assistant</strong>
                <p>Ask questions about your model and get AI-powered insights</p>
              </div>
            </div>
            <div className="welcome-feature">
              <span className="welcome-feature-icon">📊</span>
              <div>
                <strong>Auto-Generated Plots</strong>
                <p>Create visualizations with natural language prompts</p>
              </div>
            </div>
            <div className="welcome-feature">
              <span className="welcome-feature-icon">🔍</span>
              <div>
                <strong>Model Inspector</strong>
                <p>Browse and understand your model files</p>
              </div>
            </div>
            <div className="welcome-feature">
              <span className="welcome-feature-icon">🎛️</span>
              <div>
                <strong>3D Visualization</strong>
                <p>Explore your model grid in 3D</p>
              </div>
            </div>
          </div>
        </div>

        <div className="welcome-footer">
          <button className="btn-link" onClick={onOpenSettings}>
            ⚙️ Configure LLM Settings
          </button>
        </div>
      </div>
    </div>
  );
}
