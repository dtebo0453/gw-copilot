import React, { useEffect, useMemo, useState } from "react";
import { API } from "../api";

type FileEntry = {
  path_rel: string;
  size: number;
  mtime: string;
  kind: "text" | "binary" | string;
  sha256?: string | null;
};

type FilesResponse = {
  root?: string;
  files: FileEntry[];
  truncated?: boolean;
};

type FileReadResponse = {
  path_rel: string;
  content: string | null;
  truncated: boolean;
  sha256: string;
  size: number;
  kind: "text" | "binary" | string;
  mime?: string;
};

function formatBytes(n: number): string {
  if (!Number.isFinite(n)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function fileIcon(kind: string, name: string): string {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  if (["png", "jpg", "jpeg", "svg", "gif", "bmp"].includes(ext)) return "\uD83D\uDDBC\uFE0F";
  if (["csv", "dat"].includes(ext)) return "\uD83D\uDCCA";
  if (["nam", "mfn"].includes(ext)) return "\uD83D\uDCE6";
  if (["dis", "dis6", "disu", "disv"].includes(ext)) return "\uD83D\uDDFA\uFE0F";
  if (["hds", "cbc", "ucn"].includes(ext)) return "\uD83D\uDCA7";
  if (["lst", "list"].includes(ext)) return "\uD83D\uDCCB";
  if (kind === "binary") return "\uD83D\uDCCE";
  return "\uD83D\uDCC4";
}

export function ModelFilesTab(props: { inputsDir: string; workspace?: string }) {
  const { inputsDir, workspace } = props;

  const [q, setQ] = useState<string>("");
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [selected, setSelected] = useState<FileEntry | null>(null);
  const [content, setContent] = useState<FileReadResponse | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingFile, setLoadingFile] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [truncated, setTruncated] = useState<boolean>(false);
  const [rootShown, setRootShown] = useState<string>("");

  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    if (!qq) return files;
    return files.filter((f) => f.path_rel.toLowerCase().includes(qq));
  }, [files, q]);

  async function loadList() {
    setErr(null);
    setLoadingList(true);
    try {
      const params = new URLSearchParams();
      params.set("inputs_dir", inputsDir);
      if (workspace) params.set("workspace", workspace);
      params.set("glob", "**/*");
      params.set("max", "5000");
      params.set("include_hash", "false");

      const r = await fetch(`${API}/workspace/files?${params.toString()}`);
      const txt = await r.text();
      if (!r.ok) throw new Error(`List failed: ${r.status} ${txt.slice(0, 300)}`);

      const data = JSON.parse(txt) as FilesResponse;
      setFiles(data.files ?? []);
      setTruncated(!!data.truncated);
      setRootShown(data.root ?? "");

      if (selected) {
        const still = (data.files ?? []).find((f) => f.path_rel === selected.path_rel) || null;
        setSelected(still);
      }
    } catch (e: any) {
      setErr(e?.message ?? String(e));
      setFiles([]);
      setSelected(null);
      setContent(null);
      setTruncated(false);
      setRootShown("");
    } finally {
      setLoadingList(false);
    }
  }

  async function loadFile(path_rel: string) {
    setErr(null);
    setLoadingFile(true);
    setContent(null);
    try {
      const params = new URLSearchParams();
      params.set("inputs_dir", inputsDir);
      if (workspace) params.set("workspace", workspace);
      params.set("path_rel", path_rel);
      params.set("max_bytes", "2000000");

      const r = await fetch(`${API}/workspace/file?${params.toString()}`);
      const txt = await r.text();
      if (!r.ok) throw new Error(`Read failed: ${r.status} ${txt.slice(0, 300)}`);

      const data = JSON.parse(txt) as FileReadResponse;
      setContent(data);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setLoadingFile(false);
    }
  }

  function downloadUrl(path_rel: string) {
    const params = new URLSearchParams();
    params.set("inputs_dir", inputsDir);
    if (workspace) params.set("workspace", workspace);
    params.set("path_rel", path_rel);
    return `${API}/workspace/file/download?${params.toString()}`;
  }

  useEffect(() => {
    loadList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputsDir, workspace]);

  useEffect(() => {
    if (selected?.path_rel) loadFile(selected.path_rel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.path_rel]);

  return (
    <div className="mf-container">
      {/* Left: File list */}
      <div className="mf-list-panel">
        <div className="mf-list-header">
          <span className="label">Model Files</span>
          <span className="mf-file-count">
            {filtered.length}{q.trim() ? ` / ${files.length}` : ""} file{filtered.length !== 1 ? "s" : ""}
          </span>
        </div>

        <div className="mf-controls">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Filter files\u2026"
            className="mf-filter-input"
          />
          <button
            onClick={() => loadList()}
            className="btn mf-refresh-btn"
            disabled={loadingList}
          >
            {loadingList ? "\u2026" : "\u21BB"}
          </button>
        </div>

        {(rootShown || truncated) && (
          <div className="mf-info-bar">
            {rootShown && (
              <div className="mf-root-path">
                <span className="muted">Root:</span>{" "}
                <code>{rootShown}</code>
              </div>
            )}
            {truncated && (
              <span className="mf-truncated-badge">List truncated (max 5000)</span>
            )}
          </div>
        )}

        <div className="mf-list-scroll">
          {filtered.map((f) => {
            const active = selected?.path_rel === f.path_rel;
            return (
              <button
                key={f.path_rel}
                className={`mf-file-row ${active ? "active" : ""}`}
                onClick={() => setSelected(f)}
              >
                <div className="mf-file-row-top">
                  <span className="mf-file-icon">{fileIcon(f.kind, f.path_rel)}</span>
                  <span className="mf-file-name">{f.path_rel}</span>
                </div>
                <div className="mf-file-meta">
                  {formatBytes(f.size)}
                  <span className="mf-file-meta-sep">&middot;</span>
                  {f.kind}
                  <span className="mf-file-meta-sep">&middot;</span>
                  {f.mtime}
                </div>
              </button>
            );
          })}
          {!loadingList && filtered.length === 0 && (
            <div className="mf-empty">
              <div className="muted">No files found.</div>
            </div>
          )}
        </div>

        {err && <div className="mf-error">{err}</div>}
      </div>

      {/* Right: Preview */}
      <div className="mf-view-panel">
        {selected ? (
          <>
            <div className="mf-view-header">
              <div className="mf-view-title-row">
                <span className="mf-view-title">{selected.path_rel}</span>
                <a
                  href={downloadUrl(selected.path_rel)}
                  className="btn mf-download-btn"
                >
                  Download
                </a>
              </div>
            </div>

            {loadingFile ? (
              <div className="mf-loading">
                <div className="plots-spinner" />
                <span>Loading file\u2026</span>
              </div>
            ) : content ? (
              <>
                <div className="mf-file-info">
                  <span>{formatBytes(content.size)}</span>
                  <span className="mf-file-meta-sep">&middot;</span>
                  <span>{content.kind}</span>
                  <span className="mf-file-meta-sep">&middot;</span>
                  <code className="mf-sha">{content.sha256}</code>
                  {content.truncated && (
                    <span className="mf-truncated-badge">Truncated preview</span>
                  )}
                </div>

                {content.kind !== "text" || content.content === null ? (
                  <div className="mf-binary-notice">
                    This file is binary or cannot be displayed inline. Use the Download button above.
                  </div>
                ) : (
                  <pre className="mf-code-pre">{content.content}</pre>
                )}
              </>
            ) : null}
          </>
        ) : (
          <div className="mf-placeholder">
            <div className="plots-placeholder-icon">{"\uD83D\uDCC1"}</div>
            <div>Select a file to preview</div>
            <div className="muted">
              Choose a model file from the list on the left
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
