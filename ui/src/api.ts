// ui/src/api.ts

export const API =
  (import.meta as any).env?.VITE_API_BASE || "";

type Json = any;

/**
 * Build a URL object for an API path with optional query params.
 * Works with both absolute API bases (dev: "http://127.0.0.1:8000")
 * and relative paths (production: "").
 */
function apiUrl(path: string): URL {
  const base = API || window.location.origin;
  return new URL(`${base}${path}`);
}

/** Default timeout for viz / data fetches (30 s). */
const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Create an AbortSignal that fires after `ms` milliseconds.
 * If a caller-provided signal is given, the two are combined so that
 * either one aborting will cancel the request.
 */
function withTimeout(ms: number, outerSignal?: AbortSignal): AbortSignal {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException("Request timed out", "TimeoutError")), ms);

  // If the outer signal already aborted, forward immediately
  if (outerSignal?.aborted) {
    clearTimeout(timer);
    ctrl.abort(outerSignal.reason);
  } else if (outerSignal) {
    outerSignal.addEventListener("abort", () => {
      clearTimeout(timer);
      ctrl.abort(outerSignal.reason);
    }, { once: true });
  }

  // Cleanup timer when the controller fires for any reason
  ctrl.signal.addEventListener("abort", () => clearTimeout(timer), { once: true });

  return ctrl.signal;
}

async function fetchText(url: string, init?: RequestInit): Promise<string> {
  const r = await fetch(url, init);
  const txt = await r.text();
  if (!r.ok) {
    // Try to surface FastAPI-style errors
    try {
      const j = txt ? JSON.parse(txt) : null;
      const detail = j?.detail || j?.message || txt || `HTTP ${r.status}`;
      throw new Error(String(detail));
    } catch {
      throw new Error(txt || `HTTP ${r.status}`);
    }
  }
  return txt;
}

async function fetchJson(url: string, init?: RequestInit): Promise<Json> {
  const r = await fetch(url, init);
  const txt = await r.text();
  let data: any = null;

  try {
    data = txt ? JSON.parse(txt) : null;
  } catch {
    // Not JSON; treat as raw text
  }

  if (!r.ok) {
    const detail = data?.detail || data?.message || txt || `HTTP ${r.status}`;
    throw new Error(String(detail));
  }
  return data;
}

/** -----------------------------
 *  Project inspection / import
 *  ----------------------------- */

export async function inspectProject(path: string) {
  return fetchJson(`${API}/projects/inspect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
}

export async function importProject(plan: any) {
  return fetchJson(`${API}/projects/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan }),
  });
}

export async function setupProject(path: string) {
  return fetchJson(`${API}/projects/setup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
}

/** -----------------------------
 *  Artifacts
 *  ----------------------------- */

export async function readArtifact(path: string): Promise<string> {
  const u = apiUrl(`/artifacts/read`);
  u.searchParams.set("path", path);
  return fetchText(u.toString());
}

/** -----------------------------
 *  Filesystem search
 *  ----------------------------- */

export async function fsFind(payload: {
  query: string;
  kind?: "dir" | "file" | "any";
  max_results?: number;
  roots?: string[] | null;
}) {
  return fetchJson(`${API}/fs/find`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fsBrowse(): Promise<string | null> {
  const data = await fetchJson(`${API}/fs/browse`);
  return data?.path ?? null;
}

/** -----------------------------
 *  Deterministic jobs runner
 *  (SSE log streaming; falls back gracefully)
 *  ----------------------------- */

export async function apiRun(
  action: string,
  payload: any,
  onLine?: (line: string) => void
) {
  // Start job
  const start = await fetchJson(`${API}/run/${encodeURIComponent(action)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });

  const jobId = start?.job_id;
  if (!jobId) throw new Error("API did not return job_id.");

  // Stream events
  const eventsUrl = `${API}/jobs/${encodeURIComponent(jobId)}/events`;

  await new Promise<void>((resolve) => {
    let settled = false;

    try {
      const es = new EventSource(eventsUrl);

      es.onmessage = (ev) => {
        if (!ev.data) return;
        try {
          const obj = JSON.parse(ev.data);
          if (obj?.line && typeof onLine === "function") onLine(String(obj.line));
          if (obj?.final) {
            settled = true;
            es.close();
            resolve();
          }
        } catch {
          // ignore malformed chunks
        }
      };

      es.onerror = () => {
        try {
          es.close();
        } catch {}
        resolve();
      };
    } catch {
      resolve();
    }

    // Avoid hanging forever if SSE never finalizes
    setTimeout(() => {
      if (!settled) resolve();
    }, 60_000);
  });

  // Final status
  const final = await fetchJson(`${API}/jobs/${encodeURIComponent(jobId)}`);
  return final;
}

/** -----------------------------
 *  Chat (freeform)
 *  ----------------------------- */

export async function chat(payload: {
  message: string;
  inputs_dir: string;
  workspace?: string | null;
  history?: { role: string; content: string }[];
}) {
  return fetchJson(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** -----------------------------
 *  Workspace Q&A (read-only, grounded in workspace files)
 *  NOTE: Deprecated — all chat now routes through /chat endpoint.
 *  Kept for backward compatibility / potential Stage 3 plots usage.
 *  ----------------------------- */

export async function workspaceAsk(payload: {
  question: string;
  inputs_dir: string;
  workspace?: string | null;
  max_read_files?: number;
  max_bytes_each?: number;
  total_byte_cap?: number;
  history?: { role: string; content: string }[];
}) {
  return fetchJson(`${API}/workspace/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** -----------------------------
 *  Workspace scan (warm cache)
 *  ----------------------------- */

export async function workspaceScan(payload: {
  inputs_dir: string;
  workspace?: string | null;
  force?: boolean;
}) {
  return fetchJson(`${API}/workspace/scan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
/** -----------------------------
 *  3D Viz endpoints
 *  ----------------------------- */

export async function vizSummary(params: {
  inputs_dir: string;
  workspace?: string | null;
}, signal?: AbortSignal) {
  const u = apiUrl(`/viz/summary`);
  u.searchParams.set("inputs_dir", params.inputs_dir);
  if (params.workspace) u.searchParams.set("workspace", params.workspace);
  return fetchJson(u.toString(), { signal: withTimeout(DEFAULT_TIMEOUT_MS, signal) });
}

export async function vizMesh(params: {
  inputs_dir: string;
  workspace?: string | null;
  mode?: string;   // e.g. "top_surface"
  layer?: number;  // default 0
}, signal?: AbortSignal) {
  const u = apiUrl(`/viz/mesh`);
  u.searchParams.set("inputs_dir", params.inputs_dir);
  if (params.workspace) u.searchParams.set("workspace", params.workspace);
  u.searchParams.set("mode", params.mode ?? "top_surface");
  u.searchParams.set("layer", String(params.layer ?? 0));
  return fetchJson(u.toString(), { signal: withTimeout(DEFAULT_TIMEOUT_MS, signal) });
}

export async function vizScalars(params: {
  inputs_dir: string;
  workspace?: string | null;
  key: string;     // e.g. "top" | "botm" | "idomain" | "hk" | "k33"
  layer?: number;  // default 0
  mode?: string;   // e.g. "top_surface" | "block_model" | "all_layers_surface"
}, signal?: AbortSignal) {
  const u = apiUrl(`/viz/scalars`);
  u.searchParams.set("inputs_dir", params.inputs_dir);
  if (params.workspace) u.searchParams.set("workspace", params.workspace);
  u.searchParams.set("key", params.key);
  u.searchParams.set("layer", String(params.layer ?? 0));
  if (params.mode) u.searchParams.set("mode", params.mode);
  return fetchJson(u.toString(), { signal: withTimeout(DEFAULT_TIMEOUT_MS, signal) });
}

/** -----------------------------
 *  Viz boundary (model domain polygon)
 *  ----------------------------- */

export async function vizBoundary(params: {
  inputs_dir: string;
  workspace?: string | null;
}) {
  const u = apiUrl(`/viz/boundary`);
  u.searchParams.set("inputs_dir", params.inputs_dir);
  if (params.workspace) u.searchParams.set("workspace", params.workspace);
  return fetchJson(u.toString(), { signal: withTimeout(DEFAULT_TIMEOUT_MS) });
}

/** -----------------------------
 *  Spatial reference management
 *  ----------------------------- */

export async function getSpatialRef(inputs_dir: string) {
  const u = apiUrl(`/viz/spatial-ref`);
  u.searchParams.set("inputs_dir", inputs_dir);
  return fetchJson(u.toString());
}

export async function setSpatialRef(
  inputs_dir: string,
  ref: {
    epsg?: number | null;
    xorigin?: number;
    yorigin?: number;
    angrot?: number;
    crs_name?: string;
    centroid_lat?: number;
    centroid_lon?: number;
  }
) {
  const u = apiUrl(`/viz/spatial-ref`);
  u.searchParams.set("inputs_dir", inputs_dir);
  return fetchJson(u.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(ref),
  });
}

export async function clearSpatialRef(inputs_dir: string) {
  const u = apiUrl(`/viz/spatial-ref`);
  u.searchParams.set("inputs_dir", inputs_dir);
  return fetchJson(u.toString(), { method: "DELETE" });
}

/** -----------------------------
 *  Viz bounds (3D bounding box for clip planes)
 *  ----------------------------- */

export async function vizBounds(params: {
  inputs_dir: string;
  workspace?: string | null;
}) {
  const u = apiUrl(`/viz/bounds`);
  u.searchParams.set("inputs_dir", params.inputs_dir);
  if (params.workspace) u.searchParams.set("workspace", params.workspace);
  return fetchJson(u.toString(), { signal: withTimeout(DEFAULT_TIMEOUT_MS) });
}

/** -----------------------------
 *  Workspace facts (compact model info for grounding)
 *  ----------------------------- */

export async function workspaceFacts(params: {
  inputs_dir: string;
  workspace?: string | null;
}) {
  const u = apiUrl(`/workspace/facts`);
  u.searchParams.set("inputs_dir", params.inputs_dir);
  if (params.workspace) u.searchParams.set("workspace", params.workspace);
  return fetchJson(u.toString());
}

/** -----------------------------
 *  Model snapshot (full snapshot for debugging)
 *  ----------------------------- */

export async function modelSnapshot(params: {
  inputs_dir: string;
  workspace?: string | null;
}) {
  const u = apiUrl(`/model/snapshot`);
  u.searchParams.set("inputs_dir", params.inputs_dir);
  if (params.workspace) u.searchParams.set("workspace", params.workspace);
  return fetchJson(u.toString());
}

