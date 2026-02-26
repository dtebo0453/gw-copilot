import React, { useEffect, useMemo, useRef, useState } from "react";
import { vizMesh, vizScalars, vizSummary, vizBounds } from "../api";

type Props = {
  inputsDir: string;
  workspace?: string | null;
};

// vtk.js imports (lazy) to keep initial bundle light and avoid hard crashes
type VTK = {
  vtkGenericRenderWindow: any;
  vtkPolyData: any;
  vtkPoints: any;
  vtkCellArray: any;
  vtkDataArray: any;
  vtkMapper: any;
  vtkActor: any;
  vtkColorTransferFunction: any;
  vtkPlane: any;
};

async function loadVTK(): Promise<VTK> {
  const [
    ,  // Geometry profile — side-effect import that registers WebGL concrete
       // implementations (OpenGL Mapper, Actor, etc.) required by the render
       // pipeline's ForwardPass.traverse().  Without it vtk.js throws
       // "Cannot read properties of undefined (reading 'traverse')".
    { default: vtkGenericRenderWindow },
    { default: vtkPolyData },
    { default: vtkPoints },
    { default: vtkCellArray },
    { default: vtkDataArray },
    { default: vtkMapper },
    { default: vtkActor },
    { default: vtkColorTransferFunction },
    { default: vtkPlane },
  ] = await Promise.all([
    import("@kitware/vtk.js/Rendering/Profiles/Geometry"),
    import("@kitware/vtk.js/Rendering/Misc/GenericRenderWindow"),
    import("@kitware/vtk.js/Common/DataModel/PolyData"),
    import("@kitware/vtk.js/Common/Core/Points"),
    import("@kitware/vtk.js/Common/Core/CellArray"),
    import("@kitware/vtk.js/Common/Core/DataArray"),
    import("@kitware/vtk.js/Rendering/Core/Mapper"),
    import("@kitware/vtk.js/Rendering/Core/Actor"),
    import("@kitware/vtk.js/Rendering/Core/ColorTransferFunction"),
    import("@kitware/vtk.js/Common/DataModel/Plane"),
  ]);

  return {
    vtkGenericRenderWindow,
    vtkPolyData,
    vtkPoints,
    vtkCellArray,
    vtkDataArray,
    vtkMapper,
    vtkActor,
    vtkColorTransferFunction,
    vtkPlane,
  };
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

function getNlay(summary: any): number {
  if (!summary) return 0;
  if (typeof summary.nlay === "number") return summary.nlay;
  if (typeof summary?.grid?.nlay === "number") return summary.grid.nlay;
  return 0;
}

type PropertyDescriptor = { key: string; label: string; source: string };

function getAvailableProps(summary: any): PropertyDescriptor[] {
  const base: PropertyDescriptor[] = [
    { key: "top", label: "Top Elevation", source: "DIS" },
    { key: "botm", label: "Bottom Elevation", source: "DIS" },
    { key: "idomain", label: "IDOMAIN", source: "DIS" },
  ];

  if (!summary) return base;

  if (Array.isArray(summary.properties)) {
    const seen = new Set<string>();
    const out: PropertyDescriptor[] = [];
    for (const p of summary.properties) {
      if (p?.key && !seen.has(p.key)) {
        seen.add(p.key);
        out.push({ key: p.key, label: p.label || p.key, source: p.source || "" });
      }
    }
    // Ensure base properties exist
    for (const b of base) {
      if (!seen.has(b.key)) {
        out.unshift(b);
      }
    }
    return out;
  }

  return base;
}

// Coolwarm scientific colormap (blue → white → red)
// ---------- Colormap definitions ----------
type ColormapName = "coolwarm" | "viridis" | "jet" | "terrain" | "blues" | "reds" | "greyscale";

const COLORMAPS: Record<ColormapName, [number, number, number, number][]> = {
  coolwarm: [
    [0.0,  0.23, 0.30, 0.75],
    [0.25, 0.40, 0.55, 0.90],
    [0.5,  0.87, 0.87, 0.87],
    [0.75, 0.90, 0.45, 0.30],
    [1.0,  0.70, 0.02, 0.15],
  ],
  viridis: [
    [0.0,  0.267, 0.004, 0.329],
    [0.25, 0.282, 0.140, 0.458],
    [0.5,  0.127, 0.567, 0.551],
    [0.75, 0.454, 0.810, 0.337],
    [1.0,  0.993, 0.906, 0.144],
  ],
  jet: [
    [0.0,  0.0,  0.0,  0.5],
    [0.15, 0.0,  0.0,  1.0],
    [0.35, 0.0,  1.0,  1.0],
    [0.5,  0.0,  1.0,  0.0],
    [0.65, 1.0,  1.0,  0.0],
    [0.85, 1.0,  0.0,  0.0],
    [1.0,  0.5,  0.0,  0.0],
  ],
  terrain: [
    [0.0,  0.20, 0.30, 0.60],
    [0.25, 0.0,  0.60, 0.40],
    [0.5,  0.85, 0.85, 0.45],
    [0.75, 0.55, 0.35, 0.17],
    [1.0,  1.0,  1.0,  1.0],
  ],
  blues: [
    [0.0,  0.97, 0.98, 1.0],
    [0.25, 0.62, 0.79, 0.88],
    [0.5,  0.26, 0.57, 0.78],
    [0.75, 0.13, 0.37, 0.66],
    [1.0,  0.03, 0.19, 0.42],
  ],
  reds: [
    [0.0,  1.0,  0.96, 0.94],
    [0.25, 0.99, 0.73, 0.63],
    [0.5,  0.99, 0.42, 0.29],
    [0.75, 0.84, 0.15, 0.11],
    [1.0,  0.40, 0.0,  0.05],
  ],
  greyscale: [
    [0.0,  0.0,  0.0,  0.0],
    [0.5,  0.5,  0.5,  0.5],
    [1.0,  1.0,  1.0,  1.0],
  ],
};

const COLORMAP_LABELS: Record<ColormapName, string> = {
  coolwarm: "Cool–Warm",
  viridis: "Viridis",
  jet: "Jet (Rainbow)",
  terrain: "Terrain",
  blues: "Blues",
  reds: "Reds",
  greyscale: "Greyscale",
};

function applyColormap(lut: any, mn: number, mx: number, cmName: ColormapName = "coolwarm") {
  lut.removeAllPoints();
  const range = mx - mn || 1;
  const stops = COLORMAPS[cmName] || COLORMAPS.coolwarm;
  for (const [t, r, g, b] of stops) {
    lut.addRGBPoint(mn + t * range, r, g, b);
  }
}

function colormapCSS(cmName: ColormapName = "coolwarm"): string {
  const stops = (COLORMAPS[cmName] || COLORMAPS.coolwarm).map(
    ([t, r, g, b]) =>
      `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${(t * 100).toFixed(0)}%`
  );
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

type MeshMode = "top_surface" | "all_layers_surface" | "block_model";

const MODE_LABELS: Record<MeshMode, string> = {
  top_surface: "Layer Surface",
  all_layers_surface: "All Layers",
  block_model: "3D Block Model",
};

type Bounds3D = { xmin: number; xmax: number; ymin: number; ymax: number; zmin: number; zmax: number };

export function Model3DTab({ inputsDir, workspace }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  const vtkRef = useRef<any>(null);
  const actorRef = useRef<any>(null);
  const mapperRef = useRef<any>(null);
  const polyRef = useRef<any>(null);
  const planeXRef = useRef<any>(null);
  const planeYRef = useRef<any>(null);
  const planeZRef = useRef<any>(null);

  const didInitCameraRef = useRef(false);
  const isVtkReadyRef = useRef(false);
  const meshReqSeqRef = useRef(0);
  const meshAbortRef = useRef<AbortController | null>(null);
  const expandedBoundsRef = useRef<number[] | null>(null);

  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [vtkReady, setVtkReady] = useState(false);

  const [summary, setSummary] = useState<any | null>(null);
  const [bounds, setBounds] = useState<Bounds3D | null>(null);
  const [layer, setLayer] = useState<number>(0);
  const [propKey, setPropKey] = useState<string>("top");
  const [meshMode, setMeshMode] = useState<MeshMode>("top_surface");
  const [clipX, setClipX] = useState<number>(0);
  const [clipY, setClipY] = useState<number>(0);
  const [clipZ, setClipZ] = useState<number>(0);
  const [showGrid, setShowGrid] = useState<boolean>(true);

  // Track current scalar range for legend
  const [scalarMin, setScalarMin] = useState<number>(0);
  const [scalarMax, setScalarMax] = useState<number>(1);
  const [scalarLabel, setScalarLabel] = useState<string>("top");

  // Colormap + user range override
  const [colormapName, setColormapName] = useState<ColormapName>("coolwarm");
  const [userMin, setUserMin] = useState<string>("");  // empty = auto
  const [userMax, setUserMax] = useState<string>("");  // empty = auto
  const [dataMin, setDataMin] = useState<number>(0);   // actual data range
  const [dataMax, setDataMax] = useState<number>(1);

  const availableProps = useMemo(() => getAvailableProps(summary), [summary]);
  const maxLayer = useMemo(() => {
    const nlay = getNlay(summary);
    return nlay > 0 ? Math.max(0, nlay - 1) : 0;
  }, [summary]);

  const showLayerSelector = meshMode === "top_surface";

  // Init VTK once — deferred until container has actual dimensions
  useEffect(() => {
    let disposed = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    async function tryInit(attempt: number) {
      if (disposed) return;
      try {
        const el = containerRef.current;
        if (!el) {
          if (attempt < 20) retryTimer = setTimeout(() => tryInit(attempt + 1), 200);
          return;
        }

        const rect = el.getBoundingClientRect();
        if (rect.width < 10 || rect.height < 10) {
          if (attempt < 20) retryTimer = setTimeout(() => tryInit(attempt + 1), 200);
          return;
        }

        const vtk = await loadVTK();
        if (disposed) return;

        const grw = vtk.vtkGenericRenderWindow.newInstance({
          background: [1.0, 1.0, 1.0, 1.0],
          defaultViewAPI: "WebGL",
        });
        grw.setContainer(el);

        await new Promise<void>((resolve) => {
          requestAnimationFrame(() => {
            if (disposed) { resolve(); return; }
            try { grw.resize(); } catch {}
            resolve();
          });
        });
        if (disposed) return;

        const renderer = grw.getRenderer();
        const renderWindow = grw.getRenderWindow();

        if (!renderer || !renderWindow) {
          throw new Error("Failed to obtain renderer/renderWindow from GenericRenderWindow");
        }

        const poly = vtk.vtkPolyData.newInstance();
        const mapper = vtk.vtkMapper.newInstance();

        const actor = vtk.vtkActor.newInstance();
        actor.setMapper(mapper);
        actor.setVisibility(false);

        const lut = vtk.vtkColorTransferFunction.newInstance();
        applyColormap(lut, 0, 1);

        mapper.setLookupTable(lut);
        mapper.setUseLookupTableScalarRange(false);
        mapper.setScalarRange(0, 1);
        mapper.setScalarModeToUseCellData();
        mapper.setScalarVisibility(true);
        mapper.setColorModeToMapScalars();

        // Create 3 clipping planes (X, Y, Z)
        const planeX = vtk.vtkPlane.newInstance();
        const planeY = vtk.vtkPlane.newInstance();
        const planeZ = vtk.vtkPlane.newInstance();
        mapper.addClippingPlane(planeX);
        mapper.addClippingPlane(planeY);
        mapper.addClippingPlane(planeZ);

        renderer.addActor(actor);

        vtkRef.current = { vtk, grw, renderer, renderWindow, lut };
        actorRef.current = actor;
        mapperRef.current = mapper;
        polyRef.current = poly;
        planeXRef.current = planeX;
        planeYRef.current = planeY;
        planeZRef.current = planeZ;

        didInitCameraRef.current = false;
        isVtkReadyRef.current = true;
        setVtkReady(true);

        // Keep the render window in sync with container size (splitters, window resize, etc.)
        try {
          const ro = new ResizeObserver(() => {
            const cur = vtkRef.current;
            if (!cur?.grw) return;
            try { cur.grw.resize(); } catch {}
            // As the viewport changes, update clipping range so the model stays visible.
            try { cur.renderer?.resetCameraClippingRange?.(); } catch {}
            try { cur.renderWindow?.render?.(); } catch {}
          });
          ro.observe(el);
          (vtkRef.current as any).resizeObserver = ro;
        } catch {
          // ResizeObserver not available (very old browsers) – ignore.
        }
      } catch (e: any) {
        setErr(`VTK init failed: ${e?.message ?? String(e)}`);
      }
    }

    tryInit(0);

    return () => {
      disposed = true;
      if (retryTimer) clearTimeout(retryTimer);
      isVtkReadyRef.current = false;
      setVtkReady(false);
      try { (vtkRef.current as any)?.resizeObserver?.disconnect?.(); } catch {}
      try { vtkRef.current?.grw?.delete?.(); } catch {}
      vtkRef.current = null;
      actorRef.current = null;
      mapperRef.current = null;
      polyRef.current = null;
      planeXRef.current = null;
      planeYRef.current = null;
      planeZRef.current = null;
    };
  }, []);

  // Load summary + bounds when inputs/workspace changes
  useEffect(() => {
    (async () => {
      setErr(null);
      setSummary(null);
      setBounds(null);
      didInitCameraRef.current = false;
      try {
        const [s, b] = await Promise.all([
          vizSummary({ inputs_dir: inputsDir, workspace: workspace ?? null }),
          vizBounds({ inputs_dir: inputsDir, workspace: workspace ?? null }).catch(() => null),
        ]);
        setSummary(s);
        setBounds(b);
        setLayer(0);
        setPropKey("top");
        setMeshMode("top_surface");
        // Default clips to 0% (no clipping). 100% would clip away the entire model.
        setClipX(0);
        setClipY(0);
        setClipZ(0);
      } catch (e: any) {
        setErr(e?.message ?? String(e));
      }
    })();
  }, [inputsDir, workspace]);

  async function loadMeshAndScalars(nextLayer: number, nextProp: string, nextMode: MeshMode) {
    try { meshAbortRef.current?.abort(); } catch (_) {}
    meshAbortRef.current = new AbortController();
    const signal = meshAbortRef.current.signal;
    const reqSeq = ++meshReqSeqRef.current;

    if (!isVtkReadyRef.current) return;

    const vtkState = vtkRef.current;
    const vtk = vtkState?.vtk;
    const poly = polyRef.current;
    const mapper = mapperRef.current;

    if (!vtkState || !vtk || !poly || !mapper) return;

    setLoading(true);
    setErr(null);

    try {
      const mesh = await vizMesh(
        { inputs_dir: inputsDir, workspace: workspace ?? null, mode: nextMode, layer: nextLayer },
        signal,
      );
      if (signal.aborted || reqSeq !== meshReqSeqRef.current) return;

      const scal = await vizScalars(
        { inputs_dir: inputsDir, key: nextProp, workspace: workspace ?? null, layer: nextLayer, mode: nextMode },
      );
      if (signal.aborted || reqSeq !== meshReqSeqRef.current) return;

      if (!mesh?.points || !mesh?.polys) throw new Error("Viz mesh response missing points/polys.");
      if (!Array.isArray(mesh.points) || mesh.points.length === 0) throw new Error(`Invalid points array`);
      if (mesh.points.length % 3 !== 0) throw new Error(`Points array length not a multiple of 3`);
      if (!Array.isArray(mesh.polys) || mesh.polys.length < 5) throw new Error(`Invalid polys array`);
      if (!Array.isArray(scal?.values) || scal.values.length === 0) throw new Error("Viz scalars empty.");

      // Size guard — refuse to render payloads that would crash WebGL
      const MAX_POINTS_FRONTEND = 3_000_000; // 1M vertices × 3 floats
      if (mesh.points.length > MAX_POINTS_FRONTEND) {
        throw new Error(
          `Mesh too large for browser (${(mesh.points.length / 3).toLocaleString()} vertices). ` +
          `Try 'Layer Surface' mode or a single layer.`
        );
      }

      // Clean up previous VTK objects to prevent WebGL memory leaks
      try { poly.getPoints()?.delete?.(); } catch {}
      try { poly.getPolys()?.delete?.(); } catch {}
      try { poly.getCellData()?.getScalars()?.delete?.(); } catch {}

      const pts = vtk.vtkPoints.newInstance();
      pts.setData(Float32Array.from(mesh.points), 3);

      const polys = vtk.vtkCellArray.newInstance();
      polys.setData(Uint32Array.from(mesh.polys), 1);

      // Build scalar data array
      const scalarArr = vtk.vtkDataArray.newInstance({
        name: nextProp,
        values: Float32Array.from(scal.values),
        numberOfComponents: 1,
      });

      // Set geometry and scalars on polydata BEFORE calling setInputData
      // (vtk.js snapshots pipeline state on setInputData — scalars must be present)
      poly.setPoints(pts);
      poly.setPolys(polys);
      poly.getCellData().setScalars(scalarArr);
      poly.getCellData().setActiveScalars(nextProp);

      // Compute effective min/max (data range with user override)
      let rawMn = typeof scal.min === "number" ? scal.min : 0;
      let rawMx = typeof scal.max === "number" ? scal.max : 1;
      if (rawMn === rawMx) {
        const delta = Math.abs(rawMn) * 0.01 || 0.5;
        rawMn = rawMn - delta;
        rawMx = rawMx + delta;
      }
      setDataMin(rawMn);
      setDataMax(rawMx);

      // Apply user min/max override if provided
      const mn = userMin !== "" && isFinite(parseFloat(userMin)) ? parseFloat(userMin) : rawMn;
      const mx = userMax !== "" && isFinite(parseFloat(userMax)) ? parseFloat(userMax) : rawMx;

      // Apply colormap to lookup table
      applyColormap(vtkState.lut, mn, mx, colormapName);

      // Configure mapper for scalar coloring
      mapper.setScalarVisibility(true);
      mapper.setScalarModeToUseCellData();
      mapper.setColorModeToMapScalars();
      mapper.setUseLookupTableScalarRange(false);
      mapper.setScalarRange(mn, mx);

      // Now connect poly → mapper (with scalars already present)
      mapper.setInputData(poly);

      setScalarMin(mn);
      setScalarMax(mx);
      setScalarLabel(scal.label || nextProp);

      poly.modified?.();
      mapper.modified?.();
      actorRef.current?.setVisibility(true);

      // Apply grid (edge) visibility
      try {
        const prop = actorRef.current?.getProperty();
        if (prop) {
          prop.setEdgeVisibility(showGrid);
          prop.setEdgeColor(0.6, 0.6, 0.6);
        }
      } catch {}


      const rawBounds = poly.getBounds?.();
      if (!rawBounds || rawBounds.length !== 6) throw new Error(`Invalid bounds: ${JSON.stringify(rawBounds)}`);
      const [bxmin, bxmax, bymin, bymax, bzmin, bzmax] = rawBounds;
      if (!isFinite(bxmin) || !isFinite(bxmax) || !isFinite(bymin) || !isFinite(bymax) || !isFinite(bzmin) || !isFinite(bzmax)) {
        throw new Error(`Non-finite bounds: ${JSON.stringify(rawBounds)}`);
      }

      // Expand degenerate bounds (e.g. flat models where zmin == zmax) so
      // clip planes and camera clipping range work correctly
      const b = [...rawBounds] as number[];
      for (let dim = 0; dim < 3; dim++) {
        const lo = dim * 2, hi = dim * 2 + 1;
        if (b[hi] - b[lo] < 1e-6) {
          const mid = (b[lo] + b[hi]) / 2;
          const pad = Math.max(Math.abs(mid) * 0.1, 1.0);
          b[lo] = mid - pad;
          b[hi] = mid + pad;
        }
      }

      // Store expanded bounds for clip-plane slider updates
      expandedBoundsRef.current = b;

      // Update clip planes
      updateClipPlanes(b);

      if (!didInitCameraRef.current) {
        try {
          vtkState.renderer.resetCamera(b as any);
        } catch (e: any) {
          throw new Error(`resetCamera failed: ${e?.message}`);
        }
        didInitCameraRef.current = true;
      } else {
        try { vtkState.renderer.resetCameraClippingRange?.(); } catch {}
      }

      try { vtkState.renderWindow.render(); } catch (e: any) {
        throw new Error(`Render failed: ${e?.message ?? String(e)}`);
      }
    } catch (e: any) {
      if (e?.name === "AbortError" || e?.name === "TimeoutError" || signal.aborted) return;
      setErr(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  /** Re-apply colormap + range without reloading mesh/scalars. */
  function updateColormapAndRange() {
    const vtkState = vtkRef.current;
    const mapper = mapperRef.current;
    if (!vtkState || !mapper) return;

    const mn = userMin !== "" && isFinite(parseFloat(userMin)) ? parseFloat(userMin) : dataMin;
    const mx = userMax !== "" && isFinite(parseFloat(userMax)) ? parseFloat(userMax) : dataMax;

    applyColormap(vtkState.lut, mn, mx, colormapName);
    mapper.setScalarRange(mn, mx);
    mapper.modified?.();
    setScalarMin(mn);
    setScalarMax(mx);

    try { vtkState.renderWindow.render(); } catch {}
  }

  // Re-apply colormap/range when user changes colormap or min/max
  useEffect(() => {
    if (!vtkReady || !mapperRef.current) return;
    updateColormapAndRange();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colormapName, userMin, userMax]);

  function updateClipPlanes(b?: number[]) {
    const vtkState = vtkRef.current;
    const poly = polyRef.current;
    if (!vtkState || !poly) return;

    const polyBounds = b || poly.getBounds?.();
    if (!polyBounds || polyBounds.length !== 6 || !isFinite(polyBounds[0])) return;

    const [xmin, xmax, ymin, ymax, zmin, zmax] = polyBounds;

    // X clip: normal points in +X direction so 0% = no clipping, 100% = fully clipped.
    // VTK clips the negative half-space (dot(normal, point-origin) < 0), so with
    // normal (+1,0,0) points where px < origin are clipped.
    if (planeXRef.current) {
      const xOrigin = xmin + (xmax - xmin) * (clipX / 100);
      planeXRef.current.setNormal(1, 0, 0);
      planeXRef.current.setOrigin(xOrigin, 0, 0);
    }

    // Y clip: normal points in +Y direction (same logic as X)
    if (planeYRef.current) {
      const yOrigin = ymin + (ymax - ymin) * (clipY / 100);
      planeYRef.current.setNormal(0, 1, 0);
      planeYRef.current.setOrigin(0, yOrigin, 0);
    }

    // Z clip: normal points in +Z direction
    if (planeZRef.current) {
      const zOrigin = zmin + (zmax - zmin) * (clipZ / 100);
      planeZRef.current.setNormal(0, 0, 1);
      planeZRef.current.setOrigin(0, 0, zOrigin);
    }
  }

  // Load mesh+scalars when layer, property, or mode changes
  useEffect(() => {
    if (!summary || !vtkReady) return;
    loadMeshAndScalars(layer, propKey, meshMode);
  }, [summary, layer, propKey, meshMode, vtkReady]);

  // Update clipping planes live
  useEffect(() => {
    const vtkState = vtkRef.current;
    if (!vtkState || !didInitCameraRef.current) return;

    // Use expanded bounds (handles flat models) if available
    const b = expandedBoundsRef.current;
    if (!b || b.length !== 6 || !isFinite(b[4]) || !isFinite(b[5])) return;

    updateClipPlanes(b);

    try { vtkState.renderer.resetCameraClippingRange?.(); } catch {}
    try { vtkState.renderWindow.render(); } catch {}
  }, [clipX, clipY, clipZ]);

  // Toggle grid (edge) visibility on actor
  useEffect(() => {
    const actor = actorRef.current;
    const vtkState = vtkRef.current;
    if (!actor || !vtkState) return;
    try {
      const prop = actor.getProperty();
      prop.setEdgeVisibility(showGrid);
      prop.setEdgeColor(0.6, 0.6, 0.6);
      vtkState.renderWindow.render();
    } catch {}
  }, [showGrid, vtkReady]);

  const currentPropLabel = useMemo(() => {
    const found = availableProps.find((p) => p.key === propKey);
    return found?.label || propKey;
  }, [propKey, availableProps]);

  return (
    <div className="panel">
      <div className="row" style={{ alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div className="label">3D Model Viewer</div>
          <div className="muted" style={{ marginTop: 2 }}>
            {summary?.grid_type ? summary.grid_type.toUpperCase() : "Grid"} visualization with cell properties.
            {summary?.size_class === "large" && " Large model — use Layer Surface mode."}
            {summary?.total_cells ? ` ${summary.total_cells.toLocaleString()} cells.` : ""}
          </div>
        </div>
        <div className="muted">
          Inputs: <code>{inputsDir}</code>
        </div>
      </div>

      {err ? (
        <div className="errorBox">
          <strong>Viz error:</strong> {err}
          <div className="muted" style={{ marginTop: 6 }}>
            Tip: ensure your MODFLOW 6 workspace includes a discretization file (<code>.dis</code>, <code>.disv</code>, or <code>.disu</code>).
            For DISV/DISU models, FloPy must be installed.
          </div>
        </div>
      ) : null}

      {/* Controls row 1: Mode, Layer, Property */}
      <div className="viz-controls">
        <div className="field">
          <div className="label">View Mode</div>
          <select
            className="input"
            value={meshMode}
            onChange={(e) => {
              setMeshMode(e.target.value as MeshMode);
              didInitCameraRef.current = false; // Reset camera for new mode
            }}
            disabled={!summary || loading}
          >
            {(summary?.mesh_modes || ["top_surface"]).map((m: string) => (
              <option key={m} value={m}>
                {MODE_LABELS[m as MeshMode] || m}
              </option>
            ))}
          </select>
        </div>

        {showLayerSelector && (
          <div className="field">
            <div className="label">Layer</div>
            <input
              className="input"
              type="number"
              min={0}
              max={maxLayer}
              value={layer}
              onChange={(e) => setLayer(clamp(parseInt(e.target.value || "0", 10), 0, maxLayer))}
              disabled={!summary || loading}
              style={{ width: 80 }}
            />
          </div>
        )}

        <div className="field" style={{ minWidth: 200 }}>
          <div className="label">Property</div>
          <select
            className="input"
            value={propKey}
            onChange={(e) => { setPropKey(e.target.value); setUserMin(""); setUserMax(""); }}
            disabled={!summary || loading}
          >
            {availableProps.map((p) => (
              <option key={p.key} value={p.key}>
                {p.label}{p.source ? ` [${p.source}]` : ""}
              </option>
            ))}
          </select>
        </div>

        <button className="btn" onClick={() => loadMeshAndScalars(layer, propKey, meshMode)} disabled={!summary || loading}>
          {loading ? "Loading..." : "Reload"}
        </button>

        <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={showGrid}
            onChange={(e) => setShowGrid(e.target.checked)}
          />
          Grid
        </label>
      </div>

      {/* Controls row 2: Colormap + Scale */}
      <div className="viz-controls" style={{ marginTop: 6 }}>
        <div className="field">
          <div className="label">Colormap</div>
          <select
            className="input"
            value={colormapName}
            onChange={(e) => setColormapName(e.target.value as ColormapName)}
            disabled={!summary || loading}
          >
            {(Object.keys(COLORMAPS) as ColormapName[]).map((cm) => (
              <option key={cm} value={cm}>{COLORMAP_LABELS[cm]}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <div className="label">Min</div>
          <input
            className="input"
            type="number"
            step="any"
            placeholder={dataMin.toPrecision(4)}
            value={userMin}
            onChange={(e) => setUserMin(e.target.value)}
            disabled={!summary || loading}
            style={{ width: 100 }}
          />
        </div>
        <div className="field">
          <div className="label">Max</div>
          <input
            className="input"
            type="number"
            step="any"
            placeholder={dataMax.toPrecision(4)}
            value={userMax}
            onChange={(e) => setUserMax(e.target.value)}
            disabled={!summary || loading}
            style={{ width: 100 }}
          />
        </div>

        <button
          className="btn"
          onClick={() => { setUserMin(""); setUserMax(""); }}
          disabled={!summary || loading || (userMin === "" && userMax === "")}
          title="Reset to auto range"
          style={{ fontSize: 11, padding: "4px 10px" }}
        >
          Auto
        </button>
      </div>

      {/* Controls row 3: X/Y/Z clip sliders */}
      <div className="viz-controls" style={{ marginTop: 6 }}>
        <div className="viz-clip-group">
          <div className="label">Clip X</div>
          <input
            type="range"
            min={0}
            max={100}
            value={clipX}
            onChange={(e) => setClipX(parseFloat(e.target.value))}
            disabled={!summary}
          />
          <span className="muted">{clipX.toFixed(0)}%</span>
        </div>
        <div className="viz-clip-group">
          <div className="label">Clip Y</div>
          <input
            type="range"
            min={0}
            max={100}
            value={clipY}
            onChange={(e) => setClipY(parseFloat(e.target.value))}
            disabled={!summary}
          />
          <span className="muted">{clipY.toFixed(0)}%</span>
        </div>
        <div className="viz-clip-group">
          <div className="label">Clip Z</div>
          <input
            type="range"
            min={0}
            max={100}
            value={clipZ}
            onChange={(e) => setClipZ(parseFloat(e.target.value))}
            disabled={!summary}
          />
          <span className="muted">{clipZ.toFixed(0)}%</span>
        </div>
      </div>

      {/* 3D viewport + anchored legend */}
      <div className="viz3dWrap">
        <div className="viz3d" ref={containerRef} />
        {didInitCameraRef.current && (
          <div className="viz-legend viz-legend-anchored">
            <div className="viz-legend-label">{scalarLabel}</div>
            <div className="viz-legend-bar">
              <span className="viz-legend-val">{scalarMin.toPrecision(4)}</span>
              <div className="viz-legend-gradient" style={{ background: colormapCSS(colormapName) }} />
              <span className="viz-legend-val">{scalarMax.toPrecision(4)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
