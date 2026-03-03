"""MODPATH 7 file I/O helpers.

Parsers for the MODPATH simulation file (.mpsim), name file (.mpnam),
and binary output files (endpoints, pathlines, timeseries).  These are
used by the MODPATH adapter for model introspection and QA diagnostics.

MODPATH 7 is a particle-tracking post-processor for MODFLOW.  It does
NOT have a FloPy ``load()`` method, so all parsing is done directly
from the text/binary files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation type / tracking direction enums (for clarity)
# ---------------------------------------------------------------------------

_SIM_TYPE_MAP = {
    1: "endpoint",
    2: "pathline",
    3: "timeseries",
    4: "combined",
}

_TRACKING_DIR_MAP = {
    1: "forward",
    2: "backward",
}


# ---------------------------------------------------------------------------
# .mpsim parser
# ---------------------------------------------------------------------------

def parse_mpsim(path: Path) -> Dict[str, Any]:
    """Parse a MODPATH 7 simulation file (.mpsim).

    Extracts simulation type, tracking direction, reference time, weak
    sink/source options, and other simulation flags.

    Returns
    -------
    dict
        Keys: ``ok``, ``simulation_type``, ``tracking_direction``,
        ``reference_time``, ``stop_time``, ``option_flags``,
        ``budget_output_option``, ``trace_mode``, ``raw_lines``.
    """
    result: Dict[str, Any] = {"ok": False, "file": str(path)}

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("parse_mpsim: cannot read %s: %s", path, exc)
        result["error"] = f"Cannot read file: {exc}"
        return result

    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]

    if len(lines) < 4:
        result["error"] = "File too short to be a valid .mpsim"
        return result

    result["ok"] = True
    result["raw_line_count"] = len(lines)

    # Line 1: name file reference (often the .mpnam path)
    result["name_file_ref"] = lines[0]

    # Line 2: listing file
    result["listing_file"] = lines[1] if len(lines) > 1 else ""

    # Line 3: simulation type (1=endpoint, 2=pathline, 3=timeseries, 4=combined)
    try:
        sim_type_int = int(lines[2].split()[0])
        result["simulation_type"] = _SIM_TYPE_MAP.get(sim_type_int,
                                                       f"unknown({sim_type_int})")
        result["simulation_type_code"] = sim_type_int
    except (ValueError, IndexError):
        result["simulation_type"] = "unknown"

    # Line 4: tracking direction (1=forward, 2=backward)
    try:
        track_int = int(lines[3].split()[0])
        result["tracking_direction"] = _TRACKING_DIR_MAP.get(track_int,
                                                              f"unknown({track_int})")
        result["tracking_direction_code"] = track_int
    except (ValueError, IndexError):
        result["tracking_direction"] = "unknown"

    # Line 5: weak sink option
    if len(lines) > 4:
        try:
            ws_opt = int(lines[4].split()[0])
            result["weak_sink_option"] = ws_opt
        except (ValueError, IndexError):
            pass

    # Line 6: weak source option
    if len(lines) > 5:
        try:
            wsr_opt = int(lines[5].split()[0])
            result["weak_source_option"] = wsr_opt
        except (ValueError, IndexError):
            pass

    # Line 7: reference time option
    if len(lines) > 6:
        try:
            ref_opt = int(lines[6].split()[0])
            result["reference_time_option"] = ref_opt
        except (ValueError, IndexError):
            pass

    # Line 8: stop time option (if applicable)
    if len(lines) > 7:
        try:
            parts = lines[7].split()
            result["stop_time_option"] = int(parts[0])
            if len(parts) > 1:
                result["stop_time_value"] = float(parts[1])
        except (ValueError, IndexError):
            pass

    # Line 9: zone data option
    if len(lines) > 8:
        try:
            result["zone_data_option"] = int(lines[8].split()[0])
        except (ValueError, IndexError):
            pass

    # Line 10: retardation factor option
    if len(lines) > 9:
        try:
            result["retardation_option"] = int(lines[9].split()[0])
        except (ValueError, IndexError):
            pass

    # Scan remaining lines for budget output option / particle group count
    option_flags: Dict[str, Any] = {}
    for ln in lines[10:]:
        parts = ln.split()
        if not parts:
            continue
        # Try to detect particle group count (usually a small integer on its own line)
        if len(parts) == 1:
            try:
                val = int(parts[0])
                if 1 <= val <= 1000 and "particle_group_count" not in option_flags:
                    option_flags["particle_group_count"] = val
            except ValueError:
                pass
    result["option_flags"] = option_flags

    return result


# ---------------------------------------------------------------------------
# .mpnam parser
# ---------------------------------------------------------------------------

def parse_mpnam(path: Path) -> Dict[str, str]:
    """Parse a MODPATH 7 name file (.mpnam).

    The .mpnam file lists the MODFLOW files that MODPATH needs:
    head file, budget file, and DIS/DISV/DISU file.

    Returns
    -------
    dict
        Keys may include: ``head_file``, ``budget_file``, ``dis_file``,
        ``tdis_file``, ``mpbas_file``, and any other referenced files.
        All values are relative paths as written in the file.
    """
    result: Dict[str, str] = {}

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("parse_mpnam: cannot read %s: %s", path, exc)
        return result

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        ftype = parts[0].upper()
        fname = parts[1]

        # Map MODPATH name file keywords to result keys
        if ftype in ("HEAD", "HEADFILE", "HDS"):
            result["head_file"] = fname
        elif ftype in ("BUDGET", "BUDGETFILE", "CBC", "CBB"):
            result["budget_file"] = fname
        elif ftype in ("DIS", "DISV", "DISU"):
            result["dis_file"] = fname
            result["dis_type"] = ftype
        elif ftype == "TDIS":
            result["tdis_file"] = fname
        elif ftype in ("GRB", "GRBFILE"):
            result["grb_file"] = fname
        elif ftype in ("MPBAS", "BAS"):
            result["mpbas_file"] = fname
        elif ftype in ("MPSIM", "SIM"):
            result["mpsim_file"] = fname
        else:
            # Store any other file type under its keyword
            result[ftype.lower() + "_file"] = fname

    return result


# ---------------------------------------------------------------------------
# Endpoint file reader
# ---------------------------------------------------------------------------

def read_endpoints(path: Path) -> Dict[str, Any]:
    """Read a MODPATH endpoint file (.mpend) via FloPy.

    Returns a summary with particle count, termination status breakdown,
    and travel time statistics.
    """
    result: Dict[str, Any] = {"ok": False, "file": str(path)}

    if not path.exists():
        result["error"] = f"Endpoint file not found: {path.name}"
        return result

    try:
        import numpy as np
        from flopy.utils import EndpointFile
    except ImportError as exc:
        result["error"] = f"FloPy not available: {exc}"
        return result

    try:
        epf = EndpointFile(str(path))
        ep_data = epf.get_alldata()
    except Exception as exc:
        result["error"] = f"Failed to read endpoint file: {type(exc).__name__}: {exc}"
        return result

    n_particles = len(ep_data)
    result["ok"] = True
    result["particle_count"] = n_particles

    if n_particles == 0:
        result["summary"] = "No particles found in endpoint file."
        return result

    # Termination status breakdown
    # Status codes: 1=normally terminated (reached boundary/zone),
    # 2=terminated at weak sink, 3=zone stop, 5=still active,
    # 6=stranded, 7=inactive cell
    status_field = None
    for candidate in ("status", "Status", "istatus", "ISTATUS"):
        if candidate in ep_data.dtype.names:
            status_field = candidate
            break

    if status_field is not None:
        statuses = ep_data[status_field]
        unique, counts = np.unique(statuses, return_counts=True)
        status_labels = {
            1: "normally_terminated",
            2: "terminated_at_weak_sink",
            3: "zone_stop",
            4: "terminated_at_weak_source",
            5: "still_active",
            6: "stranded",
            7: "entered_inactive_cell",
        }
        status_breakdown: Dict[str, int] = {}
        for val, cnt in zip(unique, counts):
            label = status_labels.get(int(val), f"status_{int(val)}")
            status_breakdown[label] = int(cnt)
        result["termination_status"] = status_breakdown

    # Travel time statistics
    time_field = None
    for candidate in ("time", "Time", "traveltime", "TRAVELTIME",
                       "timeoftravel", "TIMEOFTRAVEL"):
        if candidate in ep_data.dtype.names:
            time_field = candidate
            break

    if time_field is not None:
        times = ep_data[time_field].astype(float)
        valid = times[np.isfinite(times) & (times >= 0)]
        if len(valid) > 0:
            result["travel_time_stats"] = {
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean": float(np.mean(valid)),
                "median": float(np.median(valid)),
                "std": float(np.std(valid)),
                "p10": float(np.percentile(valid, 10)),
                "p90": float(np.percentile(valid, 90)),
            }

    # Starting / ending zones if available
    for zone_field in ("zone", "Zone", "zone0", "finalzone"):
        if zone_field in ep_data.dtype.names:
            zones = ep_data[zone_field]
            unique_z, counts_z = np.unique(zones, return_counts=True)
            result[f"{zone_field}_distribution"] = {
                str(int(z)): int(c) for z, c in zip(unique_z, counts_z)
            }

    # Available field names for reference
    result["available_fields"] = list(ep_data.dtype.names)

    return result


# ---------------------------------------------------------------------------
# Pathline file reader
# ---------------------------------------------------------------------------

def read_pathlines(path: Path) -> Dict[str, Any]:
    """Read a MODPATH pathline file (.mppth) via FloPy.

    Returns a summary with particle count, time range, and path length
    statistics.
    """
    result: Dict[str, Any] = {"ok": False, "file": str(path)}

    if not path.exists():
        result["error"] = f"Pathline file not found: {path.name}"
        return result

    try:
        import numpy as np
        from flopy.utils import PathlineFile
    except ImportError as exc:
        result["error"] = f"FloPy not available: {exc}"
        return result

    try:
        plf = PathlineFile(str(path))
        pl_data = plf.get_alldata()
    except Exception as exc:
        result["error"] = f"Failed to read pathline file: {type(exc).__name__}: {exc}"
        return result

    # pl_data is a list of numpy arrays (one per particle)
    n_particles = len(pl_data)
    result["ok"] = True
    result["particle_count"] = n_particles

    if n_particles == 0:
        result["summary"] = "No particles found in pathline file."
        return result

    # Aggregate statistics across all particles
    total_points = 0
    all_times: List[float] = []
    path_lengths: List[float] = []
    travel_times: List[float] = []

    for i, pdata in enumerate(pl_data):
        if pdata is None or len(pdata) == 0:
            continue
        total_points += len(pdata)

        # Extract time data
        time_field = None
        for candidate in ("time", "Time", "t"):
            if candidate in pdata.dtype.names:
                time_field = candidate
                break

        if time_field is not None:
            t_vals = pdata[time_field].astype(float)
            if len(t_vals) > 0:
                all_times.extend(t_vals.tolist())
                travel_times.append(float(t_vals[-1] - t_vals[0]))

        # Compute path length from x, y coordinates if available
        x_field = y_field = None
        for xc in ("x", "X"):
            if xc in pdata.dtype.names:
                x_field = xc
                break
        for yc in ("y", "Y"):
            if yc in pdata.dtype.names:
                y_field = yc
                break

        if x_field is not None and y_field is not None and len(pdata) > 1:
            x = pdata[x_field].astype(float)
            y = pdata[y_field].astype(float)
            dx = np.diff(x)
            dy = np.diff(y)
            segment_lengths = np.sqrt(dx**2 + dy**2)
            path_lengths.append(float(np.sum(segment_lengths)))

    result["total_pathline_points"] = total_points
    result["points_per_particle"] = {
        "mean": total_points / n_particles if n_particles > 0 else 0,
    }

    if all_times:
        result["time_range"] = {
            "min": float(min(all_times)),
            "max": float(max(all_times)),
        }

    if travel_times:
        tt = np.array(travel_times)
        result["travel_time_stats"] = {
            "min": float(np.min(tt)),
            "max": float(np.max(tt)),
            "mean": float(np.mean(tt)),
            "median": float(np.median(tt)),
            "std": float(np.std(tt)),
        }

    if path_lengths:
        pl = np.array(path_lengths)
        result["path_length_stats"] = {
            "min": float(np.min(pl)),
            "max": float(np.max(pl)),
            "mean": float(np.mean(pl)),
            "median": float(np.median(pl)),
        }

    # Capture available fields from first particle
    if pl_data and len(pl_data[0]) > 0:
        result["available_fields"] = list(pl_data[0].dtype.names)

    return result


# ---------------------------------------------------------------------------
# Timeseries file reader
# ---------------------------------------------------------------------------

def read_timeseries(path: Path) -> Dict[str, Any]:
    """Read a MODPATH timeseries file (.mpts) via FloPy.

    Returns a summary with particle count and time point information.
    """
    result: Dict[str, Any] = {"ok": False, "file": str(path)}

    if not path.exists():
        result["error"] = f"Timeseries file not found: {path.name}"
        return result

    try:
        import numpy as np
        from flopy.utils import TimeseriesFile
    except ImportError as exc:
        result["error"] = f"FloPy not available: {exc}"
        return result

    try:
        tsf = TimeseriesFile(str(path))
        ts_data = tsf.get_alldata()
    except Exception as exc:
        result["error"] = f"Failed to read timeseries file: {type(exc).__name__}: {exc}"
        return result

    n_particles = len(ts_data)
    result["ok"] = True
    result["particle_count"] = n_particles

    if n_particles == 0:
        result["summary"] = "No particles found in timeseries file."
        return result

    # Aggregate time point count
    total_points = sum(len(d) for d in ts_data if d is not None)
    result["total_timeseries_points"] = total_points
    result["points_per_particle"] = {
        "mean": total_points / n_particles if n_particles > 0 else 0,
    }

    # Time range from first particle
    if ts_data and len(ts_data[0]) > 0:
        first = ts_data[0]
        time_field = None
        for candidate in ("time", "Time", "t"):
            if candidate in first.dtype.names:
                time_field = candidate
                break
        if time_field is not None:
            t_vals = first[time_field].astype(float)
            result["time_range"] = {
                "min": float(np.min(t_vals)),
                "max": float(np.max(t_vals)),
                "n_times": len(t_vals),
            }
        result["available_fields"] = list(first.dtype.names)

    return result


# ---------------------------------------------------------------------------
# Linked flow model finder
# ---------------------------------------------------------------------------

def find_linked_flow_model(ws_root: Path) -> Optional[str]:
    """From the .mpnam file, determine which MODFLOW model this MODPATH
    run is linked to.

    Searches for .mpnam files in the workspace, parses them, and returns
    the name file path of the linked MODFLOW model (if determinable).

    Returns
    -------
    str or None
        Relative path to the linked MODFLOW name file, or None.
    """
    mpnam_files = list(ws_root.glob("*.mpnam"))
    if not mpnam_files:
        # Try finding via .mpsim
        mpsim_files = list(ws_root.glob("*.mpsim"))
        for mpsim in mpsim_files:
            try:
                text = mpsim.read_text(encoding="utf-8", errors="replace")
                first_line = text.strip().splitlines()[0].strip()
                # First line of .mpsim often references the .mpnam file
                candidate = ws_root / first_line
                if candidate.exists() and candidate.suffix.lower() == ".mpnam":
                    mpnam_files = [candidate]
                    break
            except Exception:
                continue

    if not mpnam_files:
        return None

    nam_data = parse_mpnam(mpnam_files[0])

    # The head file or DIS file reference tells us the linked model.
    # Look for a .nam file in the same directory that corresponds.
    head_ref = nam_data.get("head_file", "")
    dis_ref = nam_data.get("dis_file", "")

    # Try to find a MODFLOW name file from the head file reference
    if head_ref:
        # Head file is usually something like model.hds — look for model.nam
        stem = Path(head_ref).stem
        # Check common name file extensions
        for ext in [".nam", ".mfsim.nam"]:
            candidate = ws_root / (stem + ext)
            if candidate.exists():
                return str(candidate.relative_to(ws_root))

    # Try the DIS file reference
    if dis_ref:
        stem = Path(dis_ref).stem
        for ext in [".nam"]:
            candidate = ws_root / (stem + ext)
            if candidate.exists():
                return str(candidate.relative_to(ws_root))

    # Fallback: just return any .nam file we can find
    all_nams = [f for f in ws_root.glob("*.nam")
                if f.name.lower() != "mfsim.nam"]
    if all_nams:
        return all_nams[0].name

    # Check for mfsim.nam (MF6)
    if (ws_root / "mfsim.nam").exists():
        return "mfsim.nam"

    return None
