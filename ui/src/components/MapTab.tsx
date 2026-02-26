import React, { useEffect, useRef, useState, useCallback } from "react";
import { vizBoundary, setSpatialRef, clearSpatialRef } from "../api";
import proj4 from "proj4";

type Props = {
  inputsDir: string;
  workspace?: string | null;
};

type SpatialRef = {
  epsg?: number | null;
  xorigin?: number;
  yorigin?: number;
  angrot?: number;
  crs_name?: string;
  proj4_def?: string;   // cached proj4 string from backend
};

type LocationContext = {
  centroid_lat: number;
  centroid_lon: number;
  epsg?: number;
  crs_name?: string;
};

type BoundaryData = {
  corners: [number, number][];
  xorigin: number;
  yorigin: number;
  angrot: number;
  x_total: number;
  y_total: number;
  has_real_coords: boolean;
  nrow: number;
  ncol: number;
  nlay: number;
  delr_range: [number, number];
  delc_range: [number, number];
  delr?: number[];
  delc?: number[];
  grid_type?: string;     // "dis" | "disv" | "disu"
  ncpl?: number;
  cell_polygons?: number[][][]; // DISV/DISU cell vertex coords [[x,y],...]
  spatial_ref?: SpatialRef;
  location_context?: LocationContext;
};

// ============================================================
// EPSG → proj4 resolution: built-in fast-path + epsg.io fetch
// ============================================================

/** Module-level cache: EPSG code → proj4 def string (persists across renders). */
const _proj4Cache = new Map<number, string>();

/** Module-level negative cache: codes we already tried and failed. */
const _proj4Failed = new Set<number>();

/** Module-level name cache. */
const _nameCache = new Map<number, string>();

/**
 * Synchronous fast-path: returns a proj4 def if we already know it
 * (built-in UTM/State-Plane formulas or previously fetched).
 */
function getProj4DefSync(epsg: number): string | null {
  if (_proj4Cache.has(epsg)) return _proj4Cache.get(epsg)!;

  // WGS84 UTM zones: 326xx = North, 327xx = South
  if (epsg >= 32601 && epsg <= 32660) {
    const zone = epsg - 32600;
    const d = `+proj=utm +zone=${zone} +datum=WGS84 +units=m +no_defs`;
    _proj4Cache.set(epsg, d);
    _nameCache.set(epsg, `WGS 84 / UTM zone ${zone}N`);
    return d;
  }
  if (epsg >= 32701 && epsg <= 32760) {
    const zone = epsg - 32700;
    const d = `+proj=utm +zone=${zone} +south +datum=WGS84 +units=m +no_defs`;
    _proj4Cache.set(epsg, d);
    _nameCache.set(epsg, `WGS 84 / UTM zone ${zone}S`);
    return d;
  }
  // NAD83 UTM zones: 269xx
  if (epsg >= 26901 && epsg <= 26923) {
    const zone = epsg - 26900;
    const d = `+proj=utm +zone=${zone} +datum=NAD83 +units=m +no_defs`;
    _proj4Cache.set(epsg, d);
    _nameCache.set(epsg, `NAD83 / UTM zone ${zone}N`);
    return d;
  }
  // WGS84 geographic
  if (epsg === 4326) {
    const d = "+proj=longlat +datum=WGS84 +no_defs";
    _proj4Cache.set(epsg, d);
    _nameCache.set(epsg, "WGS 84 (lat/lon)");
    return d;
  }

  return null;
}

/**
 * Async resolver: tries built-in first, then fetches from epsg.io.
 * Returns { proj: string, name: string } or null on failure.
 */
async function resolveEpsg(epsg: number): Promise<{ proj: string; name: string } | null> {
  // Fast-path: already resolved
  const cached = getProj4DefSync(epsg);
  if (cached) {
    return { proj: cached, name: _nameCache.get(epsg) || `EPSG:${epsg}` };
  }
  // Already tried and failed
  if (_proj4Failed.has(epsg)) return null;

  // Fetch from epsg.io  (returns a proj4 string at /XXXX.proj4)
  try {
    const resp = await fetch(`https://epsg.io/${epsg}.proj4`, { signal: AbortSignal.timeout(8000) });
    if (!resp.ok) {
      _proj4Failed.add(epsg);
      return null;
    }
    const projStr = (await resp.text()).trim();
    if (!projStr || projStr.startsWith("<") || projStr.length < 10) {
      // Got an HTML error page or garbage
      _proj4Failed.add(epsg);
      return null;
    }
    // Validate that proj4 can parse it
    try {
      proj4(projStr);
    } catch {
      _proj4Failed.add(epsg);
      return null;
    }
    _proj4Cache.set(epsg, projStr);

    // Also try to get a human-readable name
    let name = `EPSG:${epsg}`;
    try {
      const infoResp = await fetch(`https://epsg.io/${epsg}.json`, { signal: AbortSignal.timeout(5000) });
      if (infoResp.ok) {
        const info = await infoResp.json();
        if (info?.results?.[0]?.name) {
          name = info.results[0].name;
        }
      }
    } catch {
      // name lookup failed, use fallback
    }
    _nameCache.set(epsg, name);
    return { proj: projStr, name };
  } catch {
    _proj4Failed.add(epsg);
    return null;
  }
}

function getCrsNameSync(epsg: number): string {
  if (_nameCache.has(epsg)) return _nameCache.get(epsg)!;
  return `EPSG:${epsg}`;
}

/**
 * Project model coordinates to WGS84 lat/lon.
 * Returns [lon, lat] pairs.
 */
function projectToLatLon(
  points: [number, number][],
  projDef: string
): [number, number][] | null {
  try {
    const transformer = proj4(projDef, "EPSG:4326");
    return points.map(([x, y]) => {
      const [lon, lat] = transformer.forward([x, y]);
      return [lon, lat] as [number, number];
    });
  } catch {
    return null;
  }
}

export function MapTab({ inputsDir, workspace }: Props) {
  const [boundary, setBoundary] = useState<BoundaryData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showRefPanel, setShowRefPanel] = useState(false);
  const [basemap, setBasemap] = useState<"satellite" | "osm" | "none">("satellite");

  // Resolved proj4 def for current boundary's EPSG (async-resolved)
  const [resolvedProj, setResolvedProj] = useState<string | null>(null);
  const [resolvedName, setResolvedName] = useState<string | null>(null);

  // Spatial ref form state
  const [formEpsg, setFormEpsg] = useState("");
  const [formXorigin, setFormXorigin] = useState("");
  const [formYorigin, setFormYorigin] = useState("");
  const [formAngrot, setFormAngrot] = useState("0");
  const [saving, setSaving] = useState(false);
  const [refStatus, setRefStatus] = useState<string | null>(null);

  // EPSG lookup status for the form field
  const [epsgLookup, setEpsgLookup] = useState<"idle" | "loading" | "found" | "not_found">("idle");
  const [epsgLookupName, setEpsgLookupName] = useState("");

  const mapRef = useRef<HTMLDivElement>(null);
  const leafletMapRef = useRef<any>(null);
  const gridLayerRef = useRef<any>(null);
  const basemapLayerRef = useRef<any>(null);

  // Load boundary data
  const loadBoundary = useCallback(() => {
    if (!inputsDir) return;
    setLoading(true);
    setErr(null);
    setResolvedProj(null);
    setResolvedName(null);
    vizBoundary({ inputs_dir: inputsDir, workspace: workspace ?? null })
      .then(async (data: any) => {
        setBoundary(data);
        // Populate form from spatial_ref if present
        if (data.spatial_ref) {
          const sr = data.spatial_ref;
          if (sr.epsg) setFormEpsg(String(sr.epsg));
          if (sr.xorigin != null) setFormXorigin(String(sr.xorigin));
          if (sr.yorigin != null) setFormYorigin(String(sr.yorigin));
          if (sr.angrot != null) setFormAngrot(String(sr.angrot));
        } else if (data.has_real_coords) {
          setFormXorigin(String(data.xorigin));
          setFormYorigin(String(data.yorigin));
          setFormAngrot(String(data.angrot));
        }

        // Resolve proj4 def for the EPSG code (async)
        const epsg = data.spatial_ref?.epsg;
        if (epsg) {
          // If the backend stored a proj4_def, use it directly
          if (data.spatial_ref?.proj4_def) {
            _proj4Cache.set(epsg, data.spatial_ref.proj4_def);
            setResolvedProj(data.spatial_ref.proj4_def);
            setResolvedName(data.spatial_ref.crs_name || `EPSG:${epsg}`);
          } else {
            const result = await resolveEpsg(epsg);
            if (result) {
              setResolvedProj(result.proj);
              setResolvedName(result.name);
            }
          }
        }
        setLoading(false);
      })
      .catch((e: any) => {
        setErr(e?.message ?? String(e));
        setLoading(false);
      });
  }, [inputsDir, workspace]);

  useEffect(() => { loadBoundary(); }, [loadBoundary]);

  // Debounced EPSG lookup as user types
  useEffect(() => {
    const raw = formEpsg.trim();
    if (!raw) {
      setEpsgLookup("idle");
      setEpsgLookupName("");
      return;
    }
    const code = parseInt(raw, 10);
    if (isNaN(code) || code < 1000) {
      setEpsgLookup("idle");
      setEpsgLookupName("");
      return;
    }

    // Check sync cache first
    const sync = getProj4DefSync(code);
    if (sync) {
      setEpsgLookup("found");
      setEpsgLookupName(getCrsNameSync(code));
      return;
    }
    if (_proj4Failed.has(code)) {
      setEpsgLookup("not_found");
      setEpsgLookupName("");
      return;
    }

    // Async lookup with debounce
    setEpsgLookup("loading");
    const timer = setTimeout(async () => {
      const result = await resolveEpsg(code);
      // Only update if the input hasn't changed
      if (formEpsg.trim() === raw) {
        if (result) {
          setEpsgLookup("found");
          setEpsgLookupName(result.name);
        } else {
          setEpsgLookup("not_found");
          setEpsgLookupName("");
        }
      }
    }, 400);
    return () => clearTimeout(timer);
  }, [formEpsg]);

  // Toggle grid layer visibility
  useEffect(() => {
    const map = leafletMapRef.current;
    const gridLayer = gridLayerRef.current;
    if (!map || !gridLayer) return;
    if (showGrid) {
      if (!map.hasLayer(gridLayer)) map.addLayer(gridLayer);
    } else {
      if (map.hasLayer(gridLayer)) map.removeLayer(gridLayer);
    }
  }, [showGrid]);

  // Render map when boundary data + proj are available
  useEffect(() => {
    if (!boundary || !mapRef.current) return;

    // Clean up previous map if any
    if (leafletMapRef.current) {
      try { leafletMapRef.current.remove(); } catch {}
      leafletMapRef.current = null;
    }

    import("leaflet").then((L) => {
      const el = mapRef.current;
      if (!el) return;

      const projDef = resolvedProj;

      // Determine if we should show a geographic map (with tiles)
      const hasProjection = projDef !== null && boundary.has_real_coords;

      // Also check if corners are already geographic
      const coords = boundary.corners;
      const xs = coords.map((c) => c[0]);
      const ys = coords.map((c) => c[1]);
      const xMin = Math.min(...xs);
      const xMax = Math.max(...xs);
      const yMin = Math.min(...ys);
      const yMax = Math.max(...ys);
      const isAlreadyGeographic =
        boundary.has_real_coords &&
        xMin >= -180 && xMax <= 180 && yMin >= -90 && yMax <= 90;

      if (hasProjection || isAlreadyGeographic) {
        // Georeferenced: project to lat/lon and display on tiles
        let latLngs: [number, number][];

        if (isAlreadyGeographic) {
          latLngs = coords.map((c) => [c[1], c[0]]);
        } else if (projDef) {
          const projected = projectToLatLon(coords as [number, number][], projDef);
          if (!projected) {
            renderSimpleMap(L, el, boundary);
            return;
          }
          latLngs = projected.map(([lon, lat]) => [lat, lon] as [number, number]);
        } else {
          renderSimpleMap(L, el, boundary);
          return;
        }

        const map = L.map(el, {
          zoomControl: true,
          attributionControl: true,
          minZoom: 2,
        });

        addBasemapLayer(L, map, basemap);

        const polygon = L.polygon(latLngs, {
          color: "#1a73e8",
          weight: 3,
          fillColor: "#1a73e8",
          fillOpacity: 0.08,
          dashArray: "6 3",
        }).addTo(map);

        // Cell grid lines / polygons
        const gLayer = L.layerGroup();
        const gt = boundary.grid_type || "dis";
        if (gt === "dis" && boundary.delr?.length && boundary.delc?.length) {
          const angRad = (boundary.angrot * Math.PI) / 180;
          const cosA = Math.cos(angRad);
          const sinA = Math.sin(angRad);
          const ox = boundary.xorigin;
          const oy = boundary.yorigin;

          const toWorld = (x: number, y: number): [number, number] => [
            ox + x * cosA - y * sinA,
            oy + x * sinA + y * cosA,
          ];

          const worldToLatLng = (wx: number, wy: number): [number, number] => {
            if (isAlreadyGeographic) return [wy, wx];
            if (projDef) {
              const [lon, lat] = proj4(projDef, "EPSG:4326", [wx, wy]);
              return [lat, lon];
            }
            return [wy, wx];
          };

          let cx = 0;
          for (let c = 0; c <= boundary.ncol; c++) {
            const [wx1, wy1] = toWorld(cx, 0);
            const [wx2, wy2] = toWorld(cx, boundary.y_total);
            const p1 = worldToLatLng(wx1, wy1);
            const p2 = worldToLatLng(wx2, wy2);
            L.polyline([p1, p2], { color: "#666", weight: 0.5, opacity: 0.5 }).addTo(gLayer);
            if (c < boundary.ncol) cx += boundary.delr[c];
          }

          let ry = 0;
          for (let r = 0; r <= boundary.nrow; r++) {
            const [wx1, wy1] = toWorld(0, ry);
            const [wx2, wy2] = toWorld(boundary.x_total, ry);
            const p1 = worldToLatLng(wx1, wy1);
            const p2 = worldToLatLng(wx2, wy2);
            L.polyline([p1, p2], { color: "#666", weight: 0.5, opacity: 0.5 }).addTo(gLayer);
            if (r < boundary.nrow) ry += boundary.delc[r];
          }
        } else if (boundary.cell_polygons?.length) {
          for (const cellVerts of boundary.cell_polygons) {
            if (cellVerts.length < 3) continue;
            let cellLatLngs: [number, number][];
            if (isAlreadyGeographic) {
              cellLatLngs = cellVerts.map((v) => [v[1], v[0]] as [number, number]);
            } else if (projDef) {
              const projected = projectToLatLon(cellVerts as [number, number][], projDef);
              cellLatLngs = projected
                ? projected.map(([lon, lat]) => [lat, lon] as [number, number])
                : cellVerts.map((v) => [v[1], v[0]] as [number, number]);
            } else {
              cellLatLngs = cellVerts.map((v) => [v[1], v[0]] as [number, number]);
            }
            L.polygon(cellLatLngs, {
              color: "#666",
              weight: 0.5,
              fillColor: "transparent",
              fillOpacity: 0,
              opacity: 0.5,
            }).addTo(gLayer);
          }
        }
        gridLayerRef.current = gLayer;
        if (showGrid) gLayer.addTo(map);

        map.fitBounds(polygon.getBounds(), { padding: [60, 60] });

        setTimeout(() => {
          try { map.invalidateSize(true); } catch {}
        }, 0);

        leafletMapRef.current = map;
      } else {
        renderSimpleMap(L, el, boundary);
      }
    });

    return () => {
      if (leafletMapRef.current) {
        try { leafletMapRef.current.remove(); } catch {}
        leafletMapRef.current = null;
      }
    };
  }, [boundary, basemap, resolvedProj]);

  function addBasemapLayer(L: any, map: any, type: string) {
    if (basemapLayerRef.current) {
      try { map.removeLayer(basemapLayerRef.current); } catch {}
    }
    let layer: any = null;
    if (type === "satellite") {
      layer = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        {
          attribution: "Tiles &copy; Esri",
          maxZoom: 19,
        }
      );
    } else if (type === "osm") {
      layer = L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
          attribution: "&copy; OpenStreetMap contributors",
          maxZoom: 19,
        }
      );
    }
    if (layer) {
      layer.addTo(map);
      basemapLayerRef.current = layer;
    }
  }

  function renderSimpleMap(L: any, el: HTMLElement, b: BoundaryData) {
    const map = L.map(el, {
      crs: L.CRS.Simple,
      zoomControl: true,
      attributionControl: false,
      minZoom: -12,
    });

    el.style.background = "#fff";

    const bounds: [number, number][] = [
      [0, 0],
      [0, b.x_total],
      [b.y_total, b.x_total],
      [b.y_total, 0],
    ];

    const polygon = L.polygon(bounds, {
      color: "#000",
      weight: 3,
      fillColor: "transparent",
      fillOpacity: 0,
    }).addTo(map);

    const gLayer = L.layerGroup();
    const gt = b.grid_type || "dis";
    if (gt === "dis" && b.delr?.length && b.delc?.length) {
      let cx = 0;
      for (let c = 0; c <= b.ncol; c++) {
        L.polyline(
          [[0, cx], [b.y_total, cx]],
          { color: "#999", weight: 0.5, opacity: 0.6 }
        ).addTo(gLayer);
        if (c < b.ncol) cx += b.delr[c];
      }

      let ry = 0;
      for (let r = 0; r <= b.nrow; r++) {
        L.polyline(
          [[ry, 0], [ry, b.x_total]],
          { color: "#999", weight: 0.5, opacity: 0.6 }
        ).addTo(gLayer);
        if (r < b.nrow) ry += b.delc[r];
      }
    } else if (b.cell_polygons?.length) {
      for (const cellVerts of b.cell_polygons) {
        if (cellVerts.length < 3) continue;
        const latLngs: [number, number][] = cellVerts.map((v) => [v[1], v[0]]);
        L.polygon(latLngs, {
          color: "#999",
          weight: 0.5,
          fillColor: "transparent",
          fillOpacity: 0,
          opacity: 0.6,
        }).addTo(gLayer);
      }
    }
    gridLayerRef.current = gLayer;
    if (showGrid) gLayer.addTo(map);

    const midX = b.x_total / 2;
    const midY = b.y_total / 2;

    L.marker([0 - b.y_total * 0.05, midX], {
      icon: L.divIcon({
        className: "map-dim-label",
        html: `<span>${b.x_total.toFixed(1)} (${b.ncol} cols)</span>`,
        iconSize: [200, 20],
        iconAnchor: [100, 10],
      }),
    }).addTo(map);

    L.marker([midY, 0 - b.x_total * 0.05], {
      icon: L.divIcon({
        className: "map-dim-label",
        html: `<span>${b.y_total.toFixed(1)} (${b.nrow} rows)</span>`,
        iconSize: [200, 20],
        iconAnchor: [100, 10],
      }),
    }).addTo(map);

    bounds.forEach((ll) => {
      L.circleMarker(ll, {
        radius: 4,
        color: "#000",
        fillColor: "#fff",
        fillOpacity: 1,
        weight: 2,
      }).addTo(map);
    });

    map.fitBounds(polygon.getBounds(), { padding: [60, 60] });

    setTimeout(() => {
      try { map.invalidateSize(true); } catch {}
    }, 0);

    leafletMapRef.current = map;
  }

  // Handle spatial reference save
  const handleSaveRef = async () => {
    const epsg = formEpsg.trim() ? parseInt(formEpsg.trim(), 10) : null;
    const xo = formXorigin.trim() ? parseFloat(formXorigin.trim()) : 0;
    const yo = formYorigin.trim() ? parseFloat(formYorigin.trim()) : 0;
    const ang = formAngrot.trim() ? parseFloat(formAngrot.trim()) : 0;

    if (epsg && isNaN(epsg)) {
      setRefStatus("Invalid EPSG code");
      return;
    }

    // Resolve the EPSG to a proj4 def (async)
    let projStr: string | undefined;
    let crsName: string | undefined;
    if (epsg) {
      setSaving(true);
      setRefStatus("Looking up EPSG code...");
      const result = await resolveEpsg(epsg);
      if (!result) {
        setRefStatus(`EPSG:${epsg} could not be resolved. Check the code and ensure you have internet access.`);
        setSaving(false);
        return;
      }
      projStr = result.proj;
      crsName = result.name;
    }

    // Compute centroid lat/lon for location context
    let centroidLat: number | undefined;
    let centroidLon: number | undefined;
    if (projStr && boundary) {
      try {
        // Centroid in model coordinates
        const cx = boundary.x_total / 2;
        const cy = boundary.y_total / 2;
        // Apply rotation
        const angRad = (ang * Math.PI) / 180;
        const cosA = Math.cos(angRad);
        const sinA = Math.sin(angRad);
        const wx = xo + cx * cosA - cy * sinA;
        const wy = yo + cx * sinA + cy * cosA;
        // Project to WGS84
        const [lon, lat] = proj4(projStr, "EPSG:4326", [wx, wy]);
        centroidLat = lat;
        centroidLon = lon;
      } catch {
        // projection failed, skip centroid
      }
    }

    setSaving(true);
    setRefStatus(null);
    try {
      await setSpatialRef(inputsDir, {
        epsg,
        xorigin: xo,
        yorigin: yo,
        angrot: ang,
        crs_name: crsName,
        centroid_lat: centroidLat,
        centroid_lon: centroidLon,
      });
      setRefStatus("Saved! Reloading map...");
      setShowRefPanel(false);
      loadBoundary();
    } catch (e: any) {
      setRefStatus("Error: " + (e?.message ?? String(e)));
    } finally {
      setSaving(false);
    }
  };

  const handleClearRef = async () => {
    setSaving(true);
    try {
      await clearSpatialRef(inputsDir);
      setFormEpsg("");
      setFormXorigin("");
      setFormYorigin("");
      setFormAngrot("0");
      setRefStatus(null);
      setResolvedProj(null);
      setResolvedName(null);
      setShowRefPanel(false);
      loadBoundary();
    } catch (e: any) {
      setRefStatus("Error: " + (e?.message ?? String(e)));
    } finally {
      setSaving(false);
    }
  };

  // Determine display state
  const spatialRef = boundary?.spatial_ref;
  const locationCtx = boundary?.location_context;
  const isGeoReferenced = boundary?.has_real_coords || false;
  const hasCrs = resolvedProj !== null;

  // Format lat/lon for display
  const locationLabel = locationCtx
    ? `${Math.abs(locationCtx.centroid_lat).toFixed(4)}\u00b0${locationCtx.centroid_lat >= 0 ? "N" : "S"}, ${Math.abs(locationCtx.centroid_lon).toFixed(4)}\u00b0${locationCtx.centroid_lon >= 0 ? "E" : "W"}`
    : null;

  // Grid info summary
  const gridInfo = boundary
    ? [
        boundary.grid_type === "dis"
          ? `${boundary.nrow} rows x ${boundary.ncol} cols x ${boundary.nlay} layers`
          : boundary.grid_type === "disv"
            ? `${boundary.ncpl ?? "?"} cells/layer x ${boundary.nlay} layers (DISV)`
            : boundary.grid_type === "disu"
              ? `${boundary.ncpl ?? "?"} nodes (DISU)`
              : `${boundary.nlay} layers`,
        `Extent: ${boundary.x_total.toFixed(1)} x ${boundary.y_total.toFixed(1)}`,
        isGeoReferenced
          ? `Origin: (${boundary.xorigin.toFixed(2)}, ${boundary.yorigin.toFixed(2)})`
          : null,
        boundary.angrot !== 0
          ? `Rotation: ${boundary.angrot.toFixed(1)}\u00b0`
          : null,
        locationLabel
          ? `Location: ${locationLabel}`
          : null,
      ].filter(Boolean)
    : [];

  return (
    <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Info bar */}
      <div className="plots-section" style={{ flex: "0 0 auto" }}>
        <div className="plots-section-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <strong style={{ fontSize: 13 }}>Model Boundary Map</strong>
            {boundary && hasCrs && (
              <span className="facts-tag ok" style={{ fontSize: 11 }}>
                {resolvedName || spatialRef?.crs_name || `EPSG:${spatialRef?.epsg}`}
              </span>
            )}
            {boundary && isGeoReferenced && !hasCrs && (
              <span className="facts-tag ok" style={{ fontSize: 11 }}>
                Georeferenced
              </span>
            )}
            {boundary && !isGeoReferenced && !hasCrs && (
              <span className="facts-tag" style={{ fontSize: 11 }}>
                No georeference
              </span>
            )}
          </div>
          {boundary && (
            <button
              className="chip"
              style={{ fontSize: 11, padding: "3px 10px" }}
              onClick={() => setShowRefPanel(!showRefPanel)}
            >
              {showRefPanel ? "Close" : isGeoReferenced ? "Edit Reference" : "Set Spatial Reference"}
            </button>
          )}
        </div>

        {gridInfo.length > 0 && (
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            {gridInfo.map((info, i) => (
              <span key={i} className="muted">
                {info}
              </span>
            ))}
            <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, cursor: "pointer", marginLeft: 8 }}>
              <input
                type="checkbox"
                checked={showGrid}
                onChange={(e) => setShowGrid(e.target.checked)}
              />
              Grid
            </label>
            {/* Basemap selector (only when georeferenced with projection) */}
            {(isGeoReferenced && hasCrs) && (
              <select
                value={basemap}
                onChange={(e) => setBasemap(e.target.value as any)}
                style={{ fontSize: 11, padding: "2px 6px", borderRadius: 4, border: "1px solid #ddd" }}
              >
                <option value="satellite">Satellite</option>
                <option value="osm">Street Map</option>
                <option value="none">No Basemap</option>
              </select>
            )}
          </div>
        )}

        {/* Spatial Reference Panel */}
        {showRefPanel && (
          <div className="spatial-ref-panel" style={{
            marginTop: 8,
            padding: 12,
            background: "#f8f9fa",
            borderRadius: 8,
            border: "1px solid #e0e0e0",
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
              Define Spatial Reference
            </div>
            <div style={{ fontSize: 11, color: "#666", marginBottom: 10 }}>
              Enter the EPSG code for your model's coordinate reference system and
              the grid origin in that CRS. Any valid EPSG code is supported
              (definitions are fetched automatically from epsg.io).
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "6px 10px", alignItems: "center", fontSize: 12 }}>
              <label style={{ fontWeight: 500 }}>EPSG Code:</label>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input
                  type="text"
                  value={formEpsg}
                  onChange={(e) => setFormEpsg(e.target.value)}
                  placeholder="e.g. 2229, 32614"
                  style={{ padding: "4px 8px", borderRadius: 4, border: "1px solid #ccc", fontSize: 12, width: 160 }}
                />
                {epsgLookup === "loading" && (
                  <span style={{ fontSize: 11, color: "#999" }}>Looking up...</span>
                )}
              </div>

              <label style={{ fontWeight: 500 }}>X Origin (Easting):</label>
              <input
                type="text"
                value={formXorigin}
                onChange={(e) => setFormXorigin(e.target.value)}
                placeholder="e.g. 500000"
                style={{ padding: "4px 8px", borderRadius: 4, border: "1px solid #ccc", fontSize: 12, width: 160 }}
              />

              <label style={{ fontWeight: 500 }}>Y Origin (Northing):</label>
              <input
                type="text"
                value={formYorigin}
                onChange={(e) => setFormYorigin(e.target.value)}
                placeholder="e.g. 3500000"
                style={{ padding: "4px 8px", borderRadius: 4, border: "1px solid #ccc", fontSize: 12, width: 160 }}
              />

              <label style={{ fontWeight: 500 }}>Rotation (deg):</label>
              <input
                type="text"
                value={formAngrot}
                onChange={(e) => setFormAngrot(e.target.value)}
                placeholder="0"
                style={{ padding: "4px 8px", borderRadius: 4, border: "1px solid #ccc", fontSize: 12, width: 160 }}
              />
            </div>

            {epsgLookup === "found" && epsgLookupName && (
              <div style={{ marginTop: 6, fontSize: 11, color: "#1a73e8" }}>
                {epsgLookupName}
              </div>
            )}
            {epsgLookup === "not_found" && formEpsg.trim() && (
              <div style={{ marginTop: 6, fontSize: 11, color: "#d93025" }}>
                EPSG:{formEpsg.trim()} not found. Please check the code.
              </div>
            )}

            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <button
                className="chip chip-ok"
                disabled={saving}
                onClick={handleSaveRef}
                style={{ fontSize: 11, padding: "4px 12px" }}
              >
                {saving ? "Saving..." : "Apply & Reload"}
              </button>
              {(spatialRef?.epsg || isGeoReferenced) && (
                <button
                  className="chip"
                  disabled={saving}
                  onClick={handleClearRef}
                  style={{ fontSize: 11, padding: "4px 12px" }}
                >
                  Clear Reference
                </button>
              )}
              <button
                className="chip"
                onClick={() => setShowRefPanel(false)}
                style={{ fontSize: 11, padding: "4px 12px" }}
              >
                Cancel
              </button>
            </div>

            {refStatus && (
              <div style={{ marginTop: 6, fontSize: 11, color: refStatus.startsWith("Error") || refStatus.includes("could not") ? "#d93025" : "#137333" }}>
                {refStatus}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Map area */}
      <div
        className="plots-section plots-output-section"
        style={{ flex: 1, minHeight: 0, padding: 0, overflow: "hidden" }}
      >
        {loading && (
          <div className="plots-placeholder">
            <div className="plots-loading">
              <div className="plots-spinner" />
              <div className="muted">Loading boundary data...</div>
            </div>
          </div>
        )}

        {err && (
          <div style={{ padding: 12 }}>
            <div className="plots-error">{err}</div>
            <div className="muted" style={{ marginTop: 8 }}>
              Ensure your MODFLOW 6 workspace includes a valid discretization file
              (.dis, .disv, or .disu). DISV/DISU models require FloPy.
            </div>
          </div>
        )}

        {!loading && !err && !boundary && (
          <div className="plots-placeholder">
            <div style={{ fontSize: 48, opacity: 0.5, marginBottom: 12 }}>
              {"\uD83D\uDDFA\uFE0F"}
            </div>
            <div className="muted">
              Select a workspace to view the model boundary
            </div>
          </div>
        )}

        <div
          ref={mapRef}
          style={{
            width: "100%",
            flex: 1,
            minHeight: 0,
            borderRadius: "0 0 12px 12px",
            display: loading || err || !boundary ? "none" : "flex",
          }}
        />
      </div>
    </div>
  );
}
