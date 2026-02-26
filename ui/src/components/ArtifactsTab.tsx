import React, { useEffect, useState } from "react";
import { readArtifact } from "../api";

export function ArtifactsTab({
  artifactsDir,
  recentArtifacts,
}: {
  artifactsDir: string | null;
  recentArtifacts: string[];
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [text, setText] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setSelected(null);
    setText("");
  }, [artifactsDir]);

  async function open(name: string) {
    if (!artifactsDir) return;
    const p = `${artifactsDir}/${name}`.split("\\").join("/");
    setSelected(name);
    setLoading(true);
    try {
      const t = await readArtifact(p);
      setText(t);
    } catch (e: any) {
      setText(`Error loading artifact: ${e?.message ?? String(e)}`);
    } finally {
      setLoading(false);
    }
  }

  const ext = selected ? selected.split(".").pop()?.toLowerCase() ?? "" : "";
  const isImage = ["png", "jpg", "jpeg", "svg", "gif", "bmp", "webp"].includes(ext);

  return (
    <div className="artifacts-container">
      {/* Left: File list */}
      <div className="artifacts-list-panel">
        <div className="artifacts-list-header">
          <span className="label">Artifacts</span>
          <span className="artifacts-count">
            {recentArtifacts.length} file{recentArtifacts.length !== 1 ? "s" : ""}
          </span>
        </div>

        <div className="artifacts-list-scroll">
          {recentArtifacts.length === 0 ? (
            <div className="artifacts-empty">
              <div className="artifacts-empty-icon">📁</div>
              <div>No artifacts yet</div>
              <div className="muted">Run a job to generate artifacts</div>
            </div>
          ) : (
            recentArtifacts.map((a) => {
              const aExt = a.split(".").pop()?.toLowerCase() ?? "";
              const icon = ["png", "jpg", "jpeg", "svg"].includes(aExt)
                ? "🖼️"
                : ["md"].includes(aExt)
                ? "📝"
                : ["csv", "txt", "log"].includes(aExt)
                ? "📄"
                : ["json"].includes(aExt)
                ? "📋"
                : "📎";
              return (
                <button
                  key={a}
                  className={`artifacts-file-btn ${a === selected ? "active" : ""}`}
                  onClick={() => open(a)}
                >
                  <span className="artifacts-file-icon">{icon}</span>
                  <span className="artifacts-file-name">{a}</span>
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Right: Preview */}
      <div className="artifacts-view-panel">
        {selected ? (
          <>
            <div className="artifacts-view-header">
              <span className="artifacts-view-title">{selected}</span>
              <span className="artifacts-view-badge">{ext.toUpperCase()}</span>
            </div>

            {loading ? (
              <div className="artifacts-loading">
                <div className="plots-spinner" />
                <span>Loading…</span>
              </div>
            ) : isImage ? (
              <div className="artifacts-image-wrap">
                <img
                  src={`${artifactsDir}/${selected}`.split("\\").join("/")}
                  alt={selected}
                  className="artifacts-image"
                />
              </div>
            ) : (
              <pre className="artifacts-pre">{text}</pre>
            )}
          </>
        ) : (
          <div className="artifacts-placeholder">
            <div className="plots-placeholder-icon">📂</div>
            <div>Select an artifact to preview</div>
            <div className="muted">
              Choose a file from the list on the left
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
