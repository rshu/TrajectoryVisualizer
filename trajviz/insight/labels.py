"""Label file loading and aggregation."""

import json
import os


def load_labeled_json(path: str) -> dict:
    """Load and validate a *_labeled.json file.

    Returns the parsed dict.  Raises ValueError on invalid structure.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("steps"), list):
        raise ValueError("Labeled JSON missing 'steps' array")
    if not data.get("taxonomy_version"):
        raise ValueError("Labeled JSON missing 'taxonomy_version'")
    # Remember where we loaded from so aggregate_labels can resolve a sibling
    # trajectory file if the baked-in trajectory_file path is stale.
    data["_labeled_path"] = os.path.abspath(path)
    return data


def aggregate_labels(data: dict) -> dict:
    """Compute label aggregation statistics from labeled JSON data.

    Returns a dict with keys:
        total, classified, unknown, classification_rate,
        taxonomy_version, model,
        phase_counts, action_counts, action_to_phase,
        phase_durations, action_durations, steps
    """
    all_steps = data.get("steps", [])
    # v1 sidecars contain assistant labels only and user steps are recovered
    # from the trajectory below.  v2 sidecars contain every original index,
    # including deterministic user labels.  Keep classification KPIs
    # assistant-focused while using embedded user records in the timeline.
    embedded_user_steps = [
        s for s in all_steps
        if isinstance(s, dict) and s.get("role") == "user"
    ]
    # KPI denominators stay assistant-focused: v1 records carry no role or
    # role == "assistant"; v2 additionally emits deterministic records for
    # user and other roles (system/developer/"?"), which must not count as
    # unclassified assistant work.
    steps = [
        s for s in all_steps
        if isinstance(s, dict) and s.get("role", "assistant") == "assistant"
    ]
    total = len(steps)
    unknown = sum(
        1 for s in steps
        if s.get("phase") == "unknown" or s.get("action") == "unknown"
    )
    classified = total - unknown

    # Phase (level 1) counts and durations
    phase_counts: dict[str, int] = {}
    phase_durations: dict[str, float] = {}
    # Action (level 2) counts and durations, with phase mapping
    action_counts: dict[str, int] = {}
    action_durations: dict[str, float] = {}
    action_to_phase: dict[str, str] = {}

    for s in steps:
        phase = s.get("phase", "unknown")
        action = s.get("action", "unknown")
        dur = s.get("duration_s") or 0

        # Skip "unknown" labels — they are counted in the KPI strip
        # but excluded from distribution charts and timeline
        if phase == "unknown" or action == "unknown":
            continue

        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        phase_durations[phase] = phase_durations.get(phase, 0) + dur

        action_counts[action] = action_counts.get(action, 0) + 1
        action_durations[action] = action_durations.get(action, 0) + dur
        if action not in action_to_phase:
            action_to_phase[action] = phase

    # Filter timeline steps to exclude unknown labels
    classified_steps = [
        s for s in steps
        if s.get("phase", "unknown") != "unknown"
        and s.get("action", "unknown") != "unknown"
    ]

    # Load user steps from the original trajectory for the timeline
    timeline_steps = list(classified_steps)
    timeline_steps.extend(embedded_user_steps)
    traj_path = data.get("trajectory_file", "")
    # If the recorded path is not on this machine (e.g. labeled on Linux,
    # viewed on macOS), fall back to a sibling file in the same directory
    # as the labeled JSON.
    if traj_path and not os.path.isfile(traj_path):
        labeled_path = data.get("_labeled_path", "")
        if labeled_path and os.path.isfile(labeled_path):
            sibling = os.path.join(os.path.dirname(labeled_path),
                                   os.path.basename(traj_path))
            if os.path.isfile(sibling):
                traj_path = sibling
    session_end_ms: float | None = None
    # v2 sidecars embed their user records (and their own timing), so the
    # trajectory re-read below is only needed for v1 sidecars.
    if not embedded_user_steps and traj_path and os.path.isfile(traj_path):
        try:
            from .loaders import load_trajectory
            from .parser import parse_steps

            raw = load_trajectory(traj_path)
            if "_error" not in raw:
                traj_steps = parse_steps(raw)
                embedded_user_indices = {
                    s.get("index")
                    for s in timeline_steps
                    if s.get("role") == "user"
                }
                for s in traj_steps:
                    if (s.get("role") == "user"
                            and s.get("index") not in embedded_user_indices):
                        timeline_steps.append({
                            "index": s.get("index", 0),
                            "role": "user",
                            "phase": "user",
                            "action": "user_prompt",
                            "duration_s": s.get("duration") or 0,
                            "tokens_total": s.get("tokens", {}).get("total", 0),
                            "tool_calls": [],
                            "finish": "",
                            "agent": "",
                            "model_id": "",
                            "text_preview": s.get("text_preview", ""),
                        })
                        embedded_user_indices.add(s.get("index"))
                # Extract trajectory-end timestamp so we can backfill the last
                # labeled step's duration if its completion time was missing.
                timing = raw.get("timing") if isinstance(raw.get("timing"), dict) else {}
                finished_at = timing.get("finished_at")
                if isinstance(finished_at, str) and finished_at:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                        session_end_ms = dt.timestamp() * 1000
                    except (ValueError, TypeError):
                        session_end_ms = None
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                "Could not load trajectory for timeline: %s", exc)
    timeline_steps.sort(key=lambda s: s.get("index", 0))

    # Backfill the last step's duration if its completion timestamp is missing
    # and we have a session-end anchor to work from.
    if timeline_steps and session_end_ms is not None:
        last = timeline_steps[-1]
        if not last.get("duration_s"):
            start_ms = last.get("time_created_ms")
            if isinstance(start_ms, (int, float)) and session_end_ms > start_ms:
                last["duration_s"] = round((session_end_ms - start_ms) / 1000.0, 2)

    return {
        "total": total,
        "classified": classified,
        "unknown": unknown,
        "classification_rate": round(classified / total * 100, 1) if total else 0,
        "taxonomy_version": data.get("taxonomy_version", "?"),
        "model": data.get("model", "?"),
        "phase_counts": phase_counts,
        "action_counts": action_counts,
        "action_to_phase": action_to_phase,
        "phase_durations": phase_durations,
        "action_durations": action_durations,
        "steps": timeline_steps,
    }
