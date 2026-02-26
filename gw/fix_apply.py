
from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

# Import your FixPlan + action models
# Adjust these imports to match your project structure.
from gw.llm.fix_plan import FixPlan  # <- change if needed


# -------------------------
# Helpers / policy controls
# -------------------------

DEFAULT_ALLOWED_CONFIG_PATH_PREFIXES = (
    "solver.",
    "ic.",
    "npf.",
    "sto.",
    "stress_options.",  # example namespace if you have it
)

DEFAULT_ALLOWED_CONFIG_EXACT_PATHS = {
    # examples; add as you see fit
    "periods.nper",
    "periods.perlen",
    "periods.nstp",
    "periods.tsmult",
    "periods.steady",
}

# If you want to hard-ban certain edits:
DEFAULT_DENY_CONFIG_PATH_PREFIXES = (
    "workspace",
    "model_name",
    "grid.",  # many teams disallow model geometry edits from auto-fixes
)


@dataclass
class ApplyResult:
    changed_files: List[str]
    changed_config: bool
    notes: List[str]
    audit_json_path: str
    audit_md_path: str


class FixApplyError(RuntimeError):
    pass


class ConfirmationError(FixApplyError):
    pass


class PolicyError(FixApplyError):
    pass


# -------------------------
# Core executor
# -------------------------

def apply_fix_plan(
    *,
    fix_plan: FixPlan,
    base_config: Dict[str, Any],
    inputs_dir: str,
    workspace: Optional[str] = None,
    out_dir: Optional[str] = None,
    dry_run: bool = True,
    user_confirmations: Optional[List[str]] = None,
    max_actions: int = 8,
    allow_config_path_prefixes: Tuple[str, ...] = DEFAULT_ALLOWED_CONFIG_PATH_PREFIXES,
    allow_config_exact_paths: Optional[set[str]] = None,
    deny_config_path_prefixes: Tuple[str, ...] = DEFAULT_DENY_CONFIG_PATH_PREFIXES,
) -> Tuple[Dict[str, Any], ApplyResult]:
    """
    Deterministically apply a FixPlan.

    Returns:
        (updated_config, ApplyResult)

    Behavior:
      - Applies actions in order.
      - Enforces severity gating:
          * 'manual' requires explicit user confirmation string match.
      - Enforces policy gating for set_config_value (whitelist / blacklist).
      - Writes an audit JSON + Markdown summary to out_dir (defaults to workspace/run_artifacts or inputs_dir).
      - If dry_run=True: does not modify any files; still produces audit with "planned" operations.
    """
    allow_config_exact_paths = allow_config_exact_paths or set(DEFAULT_ALLOWED_CONFIG_EXACT_PATHS)
    user_confirmations = user_confirmations or []

    if len(fix_plan.actions) > max_actions:
        raise PolicyError(f"FixPlan contains {len(fix_plan.actions)} actions, exceeds max_actions={max_actions}")

    inputs_path = Path(inputs_dir)
    if not inputs_path.exists():
        raise FixApplyError(f"inputs_dir does not exist: {inputs_dir}")

    # Determine output directory for artifacts
    if out_dir:
        out_path = Path(out_dir)
    elif workspace:
        out_path = Path(workspace) / "run_artifacts"
    else:
        out_path = inputs_path / "run_artifacts"
    out_path.mkdir(parents=True, exist_ok=True)

    # Work on a copy of config in-memory
    cfg = copy.deepcopy(base_config)

    changed_files: List[str] = []
    changed_config = False
    notes: List[str] = []

    audit_events: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    def _require_confirmation(action_idx: int, action_obj: Any) -> None:
        """
        For 'manual' severity, require that a matching confirmation string is present.
        We use the FixPlan.user_confirmations_needed list as the canonical prompts.
        """
        severity = getattr(action_obj, "severity", None)
        if severity != "manual":
            return

        needed = fix_plan.user_confirmations_needed or []
        # Simple strategy: require ANY confirmation_needed string to be included in user_confirmations.
        # More strict: require a per-action confirmation id; you can extend later.
        for prompt in needed:
            if prompt in user_confirmations:
                return

        raise ConfirmationError(
            f"Action {action_idx} is severity='manual' but no required user confirmation provided.\n"
            f"Needed one of: {needed}\n"
            f"Provided: {user_confirmations}"
        )

    def _policy_check_config_path(path: str) -> None:
        # deny first
        for pref in deny_config_path_prefixes:
            if path.startswith(pref):
                raise PolicyError(f"Config path '{path}' is denied by policy prefix '{pref}'")

        # allow exact
        if path in allow_config_exact_paths:
            return

        # allow prefix
        for pref in allow_config_path_prefixes:
            if path.startswith(pref):
                return

        raise PolicyError(
            f"Config path '{path}' not allowed by policy. "
            f"Allowed prefixes: {allow_config_path_prefixes}, allowed exact paths: {sorted(allow_config_exact_paths)}"
        )

    def _set_by_dotted_path(obj: Dict[str, Any], path: str, value: Any) -> None:
        """
        Deterministic dotted-path setter: "npf.k" -> cfg["npf"]["k"] = value
        """
        parts = path.split(".")
        cur: Any = obj
        for key in parts[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                # Create intermediate dicts deterministically (or raise if you prefer)
                cur[key] = {}
            cur = cur[key]
        cur[parts[-1]] = value

    def _drop_csv_rows(csv_file: Path, row_indices: List[int]) -> Dict[str, Any]:
        """
        Drop rows by 0-based indices from a CSV file while preserving header.
        Deterministic: stable ordering, stable output.
        """
        if not csv_file.exists():
            raise FixApplyError(f"CSV file does not exist: {csv_file}")

        # Normalize indices (unique, sorted)
        drop_set = set(int(i) for i in row_indices)
        sorted_drops = sorted(drop_set)

        with csv_file.open("r", encoding="utf-8", newline="") as f:
            reader = list(csv.reader(f))

        if not reader:
            raise FixApplyError(f"CSV file is empty: {csv_file}")

        header = reader[0]
        data = reader[1:]

        kept = []
        dropped = 0
        for idx, row in enumerate(data):
            if idx in drop_set:
                dropped += 1
            else:
                kept.append(row)

        new_rows = [header] + kept

        return {
            "file": str(csv_file),
            "dropped_indices": sorted_drops,
            "dropped_count": dropped,
            "kept_count": len(kept),
            "original_count": len(data),
            "new_count": len(kept),
        }, new_rows

    def _atomic_write_csv(csv_file: Path, rows: List[List[str]]) -> None:
        tmp = csv_file.with_suffix(csv_file.suffix + ".tmp")
        bak = csv_file.with_suffix(csv_file.suffix + ".bak")

        # backup original
        if csv_file.exists():
            if bak.exists():
                bak.unlink()
            csv_file.replace(bak)

        # write tmp then move into place
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)

        if csv_file.exists():
            csv_file.unlink()
        tmp.replace(csv_file)

    # -------------------------
    # Apply actions in order
    # -------------------------
    for i, act in enumerate(fix_plan.actions):
        _require_confirmation(i, act)

        action_type = getattr(act, "action", None)
        rationale = getattr(act, "rationale", "")
        severity = getattr(act, "severity", "safe")
        params = getattr(act, "params", None)

        event: Dict[str, Any] = {
            "ts_utc": now,
            "idx": i,
            "action": action_type,
            "severity": severity,
            "rationale": rationale,
            "dry_run": dry_run,
            "status": "planned",
            "details": {},
        }

        if action_type == "note":
            # No-op; just record
            msg = None
            try:
                msg = getattr(params, "message", None) if params is not None else None
            except Exception:
                msg = None
            notes.append(msg or rationale or "note")
            event["status"] = "applied"  # still counts as "done"
            event["details"] = {"message": msg or rationale or "note"}
            audit_events.append(event)
            continue

        if action_type == "set_config_value":
            if params is None:
                raise FixApplyError("set_config_value missing params")

            path = getattr(params, "path", None)
            value = getattr(params, "value", None)

            if not isinstance(path, str) or not path:
                raise FixApplyError("set_config_value.params.path must be a non-empty string")

            _policy_check_config_path(path)

            event["details"] = {"path": path, "value": value}

            if not dry_run:
                _set_by_dotted_path(cfg, path, value)
                changed_config = True
                event["status"] = "applied"
            else:
                event["status"] = "planned"

            audit_events.append(event)
            continue

        if action_type == "shift_indexing":
            if params is None:
                raise FixApplyError("shift_indexing missing params")

            from_idx = getattr(params, "from_indexing", None)
            to_idx = getattr(params, "to_indexing", None)
            if from_idx not in ("zero", "one") or to_idx not in ("zero", "one"):
                raise FixApplyError("shift_indexing params must be from_indexing/to_indexing in {'zero','one'}")

            # Deterministic behavior: we do NOT guess which files/fields to shift here.
            # Typically you store an indexing convention in config (or metadata) and have validators interpret it.
            # So we only update that convention.
            path = "stress_options.indexing"  # <- choose your canonical location
            _policy_check_config_path(path)  # ensure it’s allowed (or add to allowed exact paths)

            event["details"] = {"from": from_idx, "to": to_idx, "config_path": path}

            if not dry_run:
                _set_by_dotted_path(cfg, path, to_idx)
                changed_config = True
                event["status"] = "applied"
            else:
                event["status"] = "planned"

            audit_events.append(event)
            continue

        if action_type == "drop_rows":
            if params is None:
                raise FixApplyError("drop_rows missing params")

            package = getattr(params, "package", None)
            row_indices = getattr(params, "row_indices", None)
            filename = getattr(params, "filename", None)

            if package not in ("wel", "chd", "rch"):
                raise FixApplyError(f"drop_rows.params.package must be one of wel/chd/rch, got: {package}")
            if not isinstance(row_indices, list) or not all(isinstance(x, int) for x in row_indices):
                raise FixApplyError("drop_rows.params.row_indices must be a list[int]")

            # Determine target file deterministically.
            # If filename present, trust it (but confine it to inputs_dir).
            # Else fall back to a deterministic mapping (wel->wells.csv etc).
            default_map = {"wel": "wells.csv", "chd": "chd.csv", "rch": "recharge.csv"}
            target_name = filename or default_map[package]
            target = inputs_path / target_name

            info, new_rows = _drop_csv_rows(target, row_indices)

            event["details"] = info

            if not dry_run:
                _atomic_write_csv(target, new_rows)
                changed_files.append(str(target))
                event["status"] = "applied"
            else:
                event["status"] = "planned"

            audit_events.append(event)
            continue

        raise FixApplyError(f"Unknown action type: {action_type}")

    # -------------------------
    # Write audit artifacts
    # -------------------------
    audit = {
        "generated_utc": now,
        "dry_run": dry_run,
        "changed_files": changed_files,
        "changed_config": changed_config,
        "notes": notes,
        "events": audit_events,
        "checks_to_rerun": fix_plan.checks_to_rerun,
        "user_confirmations_needed": fix_plan.user_confirmations_needed,
        "user_confirmations_provided": user_confirmations,
    }

    audit_json_path = out_path / ("applied_fixes.dryrun.json" if dry_run else "applied_fixes.json")
    audit_md_path = out_path / ("applied_fixes.dryrun.md" if dry_run else "applied_fixes.md")

    audit_json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Simple Markdown report (deterministic)
    md_lines = []
    md_lines.append("# Applied Fixes Report")
    md_lines.append("")
    md_lines.append(f"- Generated (UTC): {now}")
    md_lines.append(f"- Dry run: `{dry_run}`")
    md_lines.append(f"- Changed config: `{changed_config}`")
    md_lines.append(f"- Changed files: `{len(changed_files)}`")
    for f in changed_files:
        md_lines.append(f"  - {f}")
    md_lines.append("")
    if notes:
        md_lines.append("## Notes")
        for n in notes:
            md_lines.append(f"- {n}")
        md_lines.append("")

    md_lines.append("## Actions")
    for ev in audit_events:
        md_lines.append(f"### {ev['idx']}. {ev['action']} ({ev['severity']}) — {ev['status']}")
        if ev.get("rationale"):
            md_lines.append(f"- Rationale: {ev['rationale']}")
        if ev.get("details"):
            md_lines.append("- Details:")
            md_lines.append("```json")
            md_lines.append(json.dumps(ev["details"], indent=2))
            md_lines.append("```")
        md_lines.append("")

    if fix_plan.checks_to_rerun:
        md_lines.append("## Checks to re-run")
        for c in fix_plan.checks_to_rerun:
            md_lines.append(f"- {c}")
        md_lines.append("")

    audit_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    result = ApplyResult(
        changed_files=changed_files,
        changed_config=changed_config,
        notes=notes,
        audit_json_path=str(audit_json_path),
        audit_md_path=str(audit_md_path),
    )

    return cfg, result
