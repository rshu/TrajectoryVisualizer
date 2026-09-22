"""Pattern detection for recurring tool sequences, workflow phases, and anti-patterns."""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter

from trajviz.insight.parser import spawned_child_session_id
from trajviz.insight.shell_cmd import shell_runs_search
from trajviz.insight.tool_failure import tool_call_failed
from trajviz.tool_vocab import (
    BASH_TOOL_NAMES,
    WRITE_TOOL_NAMES as _WRITE_TOOL_NAMES,
    write_target_path,
)


# ---------------------------------------------------------------------------
# Expected phase ordering for anomaly detection
# ---------------------------------------------------------------------------

_PHASE_ORDER = ["understand", "plan", "implement", "debug", "validate", "report"]
_PHASE_RANK: dict[str, int] = {name: i for i, name in enumerate(_PHASE_ORDER)}

# Cap to avoid pathological inputs
_MAX_STEPS = 2000

_PLAN_TOOL_NAMES = {
    "TodoWrite", "todowrite", "TodoUpdate", "TaskCreate", "TaskUpdate",
    "TaskList", "EnterPlanMode",
}
_READ_TOOL_NAMES = {"Read", "read", "WebFetch"}
_SEARCH_TOOL_NAMES = {
    *BASH_TOOL_NAMES,
    "Grep", "Glob", "grep", "glob", "find", "ToolSearch", "WebSearch",
}
_VALIDATION_COMMAND_PATTERNS = (
    "pytest", "python -m pytest", "unittest", "tox", "nox", "go test",
    "cargo test", "npm test", "pnpm test", "yarn test", "jest", "vitest",
    "mvn test", "gradle test", "bazel test", "make test", "ctest", "ruff",
    "flake8", "pylint", "mypy", "eslint", "lint", "check", "verify",
)


def _has_plan_snapshot(step: dict) -> bool:
    """Return True when a step carries an explicit TODO/task snapshot."""
    for part in step.get("parts", []):
        if part.get("type") != "snapshot":
            continue
        data = part.get("data", {})
        if isinstance(data, dict) and ("todos" in data or "items" in data):
            return True
    return False


def _is_validation_command(command: str) -> bool:
    """Return True when a shell command appears to run validation."""
    lowered = " ".join(str(command).lower().split())
    if not lowered:
        return False
    return any(pattern in lowered for pattern in _VALIDATION_COMMAND_PATTERNS)


def classify_structural_phase(step: dict) -> str:
    """Classify one assistant step into a deterministic workflow phase.

    The classifier is intentionally conservative. It maps directly observable
    tool behavior to one of the six workflow phases used by the semantic
    labeler: understand, plan, implement, debug, validate, report.
    """
    if step.get("role") != "assistant":
        return "unknown"

    tool_calls = step.get("tool_calls", [])
    if _has_plan_snapshot(step):
        return "plan"

    has_plan_tool = False
    has_write = False
    has_read_or_search = False
    has_validation = False
    has_failure = False

    for tc in tool_calls:
        tool_name = tc.get("tool_name", "")
        if tool_name in _PLAN_TOOL_NAMES:
            has_plan_tool = True
        if tool_name in _WRITE_TOOL_NAMES:
            has_write = True
        if tool_name in _READ_TOOL_NAMES or tool_name in _SEARCH_TOOL_NAMES:
            has_read_or_search = True
        if tool_call_failed(tc):
            has_failure = True

        inp = tc.get("input", {})
        if isinstance(inp, dict):
            command = inp.get("command", "")
            if command and _is_validation_command(command):
                has_validation = True

    if has_plan_tool:
        return "plan"
    if has_write:
        return "implement"
    if has_validation:
        return "validate"
    if has_failure:
        return "debug"
    if has_read_or_search:
        return "understand"
    if step.get("finish") in ("stop", "end_turn") or not tool_calls:
        return "report"
    return "unknown"


def build_structural_phase_segments(steps: list[dict]) -> list[dict]:
    """Group assistant steps into contiguous structural workflow phases."""
    segments: list[dict] = []
    current: dict | None = None

    for step in steps[:_MAX_STEPS]:
        phase = classify_structural_phase(step)
        if phase == "unknown":
            continue

        step_idx = step.get("index", 0)
        if current is None or current["name"] != phase:
            if current is not None:
                segments.append(current)
            current = {
                "name": phase,
                "start_idx": step_idx,
                "end_idx": step_idx,
            }
        else:
            current["end_idx"] = step_idx

    if current is not None:
        segments.append(current)

    return segments


# ---------------------------------------------------------------------------
# 1. Tool Sequence Detection
# ---------------------------------------------------------------------------

def _extract_tool_names(step: dict) -> list[str]:
    """Extract tool-call names from a single step."""
    names: list[str] = []
    for tc in step.get("tool_calls", []):
        name = tc.get("tool_name") or tc.get("name", "")
        if name:
            names.append(name)
    return names


def detect_tool_sequences(
    steps: list[dict],
    min_freq: int = 3,
    max_n: int = 5,
) -> list[dict]:
    """Find recurring tool-call n-grams across the trajectory.

    Parameters
    ----------
    steps : list[dict]
        Parsed trajectory steps, each with a ``tool_calls`` list.
    min_freq : int
        Minimum frequency for an n-gram to be reported.
    max_n : int
        Maximum n-gram length to consider (inclusive).

    Returns
    -------
    list[dict]
        Each entry: ``{"sequence": [str], "frequency": int, "step_indices": [int]}``,
        sorted by frequency descending.
    """
    if not steps:
        return []

    # Build flat list of (tool_name, originating_step_index) pairs
    tool_stream: list[tuple[str, int]] = []
    for step in steps[:_MAX_STEPS]:
        idx = step.get("index", 0)
        for name in _extract_tool_names(step):
            tool_stream.append((name, idx))

    if len(tool_stream) < 2:
        return []

    results: list[dict] = []

    for n in range(2, max_n + 1):
        if n > len(tool_stream):
            break

        # Count n-grams and track their start indices in the stream
        ngram_indices: dict[tuple[str, ...], list[int]] = {}
        for i in range(len(tool_stream) - n + 1):
            gram = tuple(tool_stream[i + k][0] for k in range(n))
            ngram_indices.setdefault(gram, []).append(i)

        for gram, stream_positions in ngram_indices.items():
            # Count NON-OVERLAPPING occurrences so a single contiguous burst of
            # one tool (Read×4 -> three overlapping (Read,Read)) is not inflated
            # into a frequent sequence.
            occ_positions: list[int] = []
            last_end = -1
            for pos in sorted(stream_positions):
                if pos >= last_end:
                    occ_positions.append(pos)
                    last_end = pos + n
            if len(occ_positions) < min_freq:
                continue
            step_indices = sorted({tool_stream[pos][1] for pos in occ_positions})
            results.append({
                "sequence": list(gram),
                "frequency": len(occ_positions),
                "step_indices": step_indices,
            })

    # Sort by frequency descending, then by shorter sequences first for ties
    results.sort(key=lambda r: (-r["frequency"], len(r["sequence"])))
    return results


# ---------------------------------------------------------------------------
# 2. Failure Pattern Detection
# ---------------------------------------------------------------------------

def _step_has_success(step: dict) -> bool:
    """Return True if the step has at least one successful tool call."""
    for tc in step.get("tool_calls", []):
        status = tc.get("status", "")
        if status and status not in ("error", "failed", "failure", "cancelled", "timeout"):
            return True
        # No explicit error status and no bad exit code -> treat as success
        meta = tc.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        if status == "" and meta.get("exit", 0) in (None, 0):
            return True
    return False


def detect_failure_patterns(steps: list[dict]) -> list[dict]:
    """Detect recurring failure patterns and their recovery paths.

    Reuses :func:`.diagnostics.cluster_errors` to group errors, then for
    each cluster finds the most common sequence of tool calls between the
    error and the next successful step.

    Returns
    -------
    list[dict]
        Each entry: ``{"cluster_label": str, "count": int,
        "example_error": str, "recovery_path": [str] | None}``.
    """
    if not steps:
        return []

    from .diagnostics import cluster_errors

    clusters = cluster_errors(steps)
    if not clusters:
        return []

    # Build step-index lookup
    step_map: dict[int, dict] = {s.get("index", i): s for i, s in enumerate(steps)}
    all_indices = sorted(step_map.keys())

    results: list[dict] = []

    for cluster in clusters:
        # Collect recovery paths for every error occurrence in this cluster
        recovery_paths: list[tuple[str, ...]] = []

        for err_step_idx in cluster["steps"]:
            # Find steps between the error and the next success
            path_names: list[str] = []
            found_recovery = False

            # Walk forward from the step *after* the error
            pos = _bisect_right(all_indices, err_step_idx)
            for j in range(pos, min(pos + 20, len(all_indices))):
                next_idx = all_indices[j]
                next_step = step_map[next_idx]
                names = _extract_tool_names(next_step)
                path_names.extend(names)
                if _step_has_success(next_step):
                    found_recovery = True
                    break

            if found_recovery and path_names:
                recovery_paths.append(tuple(path_names))

        # Most common recovery path
        recovery_path: list[str] | None = None
        if recovery_paths:
            counter = Counter(recovery_paths)
            most_common = counter.most_common(1)[0][0]
            recovery_path = list(most_common)

        results.append({
            "cluster_label": f"{cluster['tool']}: {cluster['pattern']}",
            "count": cluster["count"],
            "example_error": cluster["pattern"],
            "recovery_path": recovery_path,
            "steps": list(cluster.get("steps", [])),
            "error_class": cluster.get("error_class", "tool"),
        })

    return results


_bisect_right = bisect_right


# ---------------------------------------------------------------------------
# 3. Phase Anomaly Detection
# ---------------------------------------------------------------------------

def detect_phase_anomalies(
    steps: list[dict],
    phases: list[dict],
) -> list[dict]:
    """Detect backward phase transitions that may indicate rework or regression.

    Parameters
    ----------
    steps : list[dict]
        Full parsed trajectory steps (used to compute total step count).
    phases : list[dict]
        Phase list with keys ``name``, ``start_idx``, ``end_idx`` (e.g.
        from manual annotation or an external classifier).

    Returns
    -------
    list[dict]
        Each entry: ``{"from_phase": str, "to_phase": str, "step_idx": int,
        "confidence": float, "category": str, "explanation": str}``.
        ``category`` is ``"intentional_iteration"`` or ``"unintentional_drift"``.
    """
    if not phases or not steps:
        return []

    total_steps = len(steps[:_MAX_STEPS])
    if total_steps == 0:
        return []

    anomalies: list[dict] = []

    for i in range(1, len(phases)):
        prev_phase = phases[i - 1]
        curr_phase = phases[i]

        prev_name = prev_phase.get("name", "").lower().strip()
        curr_name = curr_phase.get("name", "").lower().strip()

        prev_rank = _PHASE_RANK.get(prev_name)
        curr_rank = _PHASE_RANK.get(curr_name)

        # Skip phases not in the expected ordering
        if prev_rank is None or curr_rank is None:
            continue

        # Backward transition
        if curr_rank < prev_rank:
            transition_step = curr_phase.get("start_idx", 0)
            regressed_steps = curr_phase.get("end_idx", transition_step) - transition_step + 1
            confidence = round(regressed_steps / total_steps, 4) if total_steps > 0 else 0.0

            # Categorize: check if a planning step precedes the regression
            category = "unintentional_drift"
            window_start = max(0, transition_step - 3)
            for s in steps[window_start:transition_step]:
                for tc in s.get("tool_calls", []):
                    if tc.get("tool_name") in _PLAN_TOOL_NAMES:
                        category = "intentional_iteration"
                        break
                if category == "intentional_iteration":
                    break

            if category == "intentional_iteration":
                explanation = (
                    f"Backward transition from '{prev_name}' to '{curr_name}' "
                    f"at step {transition_step}. "
                    f"A planning step precedes the transition, suggesting "
                    f"intentional iteration (spans {regressed_steps} step(s), "
                    f"{confidence * 100:.1f}% of trajectory)."
                )
            else:
                explanation = (
                    f"Backward transition from '{prev_name}' to '{curr_name}' "
                    f"at step {transition_step}. "
                    f"The regressed phase spans {regressed_steps} step(s) "
                    f"({confidence * 100:.1f}% of trajectory), "
                    f"suggesting rework or unexpected context switch."
                )

            anomalies.append({
                "from_phase": prev_name,
                "to_phase": curr_name,
                "step_idx": transition_step,
                "confidence": confidence,
                "category": category,
                "explanation": explanation,
            })

    return anomalies


# ---------------------------------------------------------------------------
# 4. Plan Progress Analysis (TodoWrite tracking)
# ---------------------------------------------------------------------------

def extract_plan_history(steps: list[dict]) -> list[dict]:
    """Extract plan snapshots from TodoWrite tool calls.

    Returns a list of plan snapshots:
    ``[{"step": int, "items": [{"content": str, "status": str}]}]``
    """
    history: list[dict] = []
    for s in steps[:_MAX_STEPS]:
        for tc in s.get("tool_calls", []):
            name = tc.get("tool_name") or tc.get("name", "")
            # Match TodoWrite (Claude Code), todowrite (OpenCode), TaskCreate, etc.
            if name.lower() not in ("todowrite", "todo_write", "taskcreate", "taskupdate"):
                continue
            inp = tc.get("input", tc.get("arguments", {}))
            if not isinstance(inp, dict):
                continue
            todos = inp.get("todos", [])
            if not todos:
                continue
            items = [
                {"content": t.get("content", ""), "status": t.get("status", "?")}
                for t in todos if isinstance(t, dict)
            ]
            history.append({"step": s.get("index", 0), "items": items})
    return history


def detect_plan_phases(plan_history: list[dict]) -> list[dict]:
    """Group plan snapshots into phases.

    A phase boundary is detected when the item set changes completely
    (no content overlap with the previous snapshot).
    """
    if not plan_history:
        return []

    phases: list[dict] = []
    current_contents: set[str] = set()
    phase_start = 0

    for i, snapshot in enumerate(plan_history):
        new_contents = {item["content"] for item in snapshot["items"]}
        if i > 0 and current_contents and not current_contents & new_contents:
            # Complete content change — new phase
            phases.append({
                "phase_index": len(phases),
                "start_snapshot": phase_start,
                "end_snapshot": i - 1,
                "start_step": plan_history[phase_start]["step"],
                "end_step": plan_history[i - 1]["step"],
                "item_count": len(current_contents),
            })
            phase_start = i
        current_contents = new_contents

    # Final phase
    if plan_history:
        phases.append({
            "phase_index": len(phases),
            "start_snapshot": phase_start,
            "end_snapshot": len(plan_history) - 1,
            "start_step": plan_history[phase_start]["step"],
            "end_step": plan_history[-1]["step"],
            "item_count": len(current_contents),
        })

    return phases


def compute_plan_metrics(plan_history: list[dict]) -> dict:
    """Compute plan-level metrics from plan history.

    Returns dict with: item durations, stalled items, plan reset count.
    """
    if not plan_history:
        return {"items": [], "stalled": [], "plan_resets": 0, "total_items": 0}

    # Track per-item first in_progress and completed steps
    item_first_progress: dict[str, int] = {}
    item_completed: dict[str, int] = {}

    for snapshot in plan_history:
        step = snapshot["step"]
        for item in snapshot["items"]:
            content = item["content"]
            status = item["status"]
            if status == "in_progress" and content not in item_first_progress:
                item_first_progress[content] = step
            if status == "completed" and content not in item_completed:
                item_completed[content] = step

    items = []
    stalled = []
    all_contents: set[str] = set()
    for snapshot in plan_history:
        for item in snapshot["items"]:
            all_contents.add(item["content"])

    for content in all_contents:
        start = item_first_progress.get(content)
        end = item_completed.get(content)
        duration = (end - start) if start is not None and end is not None else None
        entry = {"content": content, "start_step": start, "end_step": end, "duration_steps": duration}
        items.append(entry)
        if start is not None and end is None or duration is not None and duration > 20:
            stalled.append(entry)

    phases = detect_plan_phases(plan_history)
    return {
        "items": items,
        "stalled": stalled,
        "plan_resets": max(0, len(phases) - 1),
        "total_items": len(all_contents),
    }


# ---------------------------------------------------------------------------
# 5. Sub-Agent Delegation Analysis
# ---------------------------------------------------------------------------

def _get_step_info(step: dict, trajectory: list[dict]) -> dict:
    """Retrieve the trajectory 'info' dict corresponding to a parsed step.

    Matches by the step's 'raw_index' (original position in trajectory);
    returns {} if raw_index is missing or out of range.
    """
    raw_idx = step.get("raw_index")
    if raw_idx is not None and 0 <= raw_idx < len(trajectory):
        entry = trajectory[raw_idx]
        info = entry.get("info", {}) if isinstance(entry, dict) else {}
        return info if isinstance(info, dict) else {}
    return {}


def extract_subagent_sessions(
    steps: list[dict], trajectory: list[dict],
) -> list[dict]:
    """Identify sub-agent sessions by grouping consecutive isSubAgent steps.

    Returns list of sessions:
    ``[{"session_id": str, "start_step": int, "end_step": int,
        "step_count": int, "spawn_step": int | None}]``
    """
    sessions: list[dict] = []
    current_session: dict | None = None

    for s in steps[:_MAX_STEPS]:
        info = _get_step_info(s, trajectory)
        # Prefer step fields (may be inferred from Task spawn metadata).
        is_sub = bool(s.get("is_sub_agent")) or bool(info.get("isSubAgent", False))
        session_id = s.get("session_id") or info.get("sessionID", "")
        step_idx = s.get("index", 0)

        if is_sub and session_id:
            if current_session is None or current_session["session_id"] != session_id:
                if current_session is not None:
                    sessions.append(current_session)
                current_session = {
                    "session_id": session_id,
                    "start_step": step_idx,
                    "end_step": step_idx,
                    "step_count": 1,
                    "spawn_step": None,
                }
            else:
                current_session["end_step"] = step_idx
                current_session["step_count"] += 1
        else:
            if current_session is not None:
                sessions.append(current_session)
                current_session = None

    if current_session is not None:
        sessions.append(current_session)

    # OpenCode-style exports record the spawned child session on the parent
    # task tool part's metadata rather than on the child messages.  Build the
    # child-session -> delegation-step map in one pass so consolidated
    # CodeArts/OpenCode exports can identify the actual delegation step.
    spawn_by_child: dict[str, int] = {}
    for step in steps[:_MAX_STEPS]:
        raw_idx = step.get("raw_index")
        if not isinstance(raw_idx, int) or not (0 <= raw_idx < len(trajectory)):
            continue
        entry = trajectory[raw_idx]
        parts = entry.get("parts", []) if isinstance(entry, dict) else []
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict) or part.get("type") != "tool":
                continue
            state = part.get("state", {})
            metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
            child_id = spawned_child_session_id(
                metadata,
                caller_session_id=str(step.get("session_id") or ""),
            )
            if child_id:
                spawn_by_child.setdefault(child_id, step.get("index", 0))

    for session in sessions:
        if session["spawn_step"] is None:
            session["spawn_step"] = spawn_by_child.get(session["session_id"])

    return sessions


def compute_subagent_metrics(
    sessions: list[dict], steps: list[dict],
) -> list[dict]:
    """Compute per-session metrics: tokens, tools, duration."""
    results = []
    for session in sessions:
        start = session["start_step"]
        end = session["end_step"]
        session_steps = [s for s in steps if start <= s.get("index", 0) <= end]

        total_tokens = sum(s["tokens"]["total"] for s in session_steps)
        total_tools = sum(s["tool_call_count"] for s in session_steps)
        durations = [s["duration"] for s in session_steps if s.get("duration") is not None]
        total_duration = sum(durations)

        results.append({
            **session,
            "total_tokens": total_tokens,
            "total_tools": total_tools,
            "total_duration": round(total_duration, 2),
        })
    return results


# ---------------------------------------------------------------------------
# 6. Fruitless Search & Anti-Pattern Detection
# ---------------------------------------------------------------------------


def _is_search_call(tc: dict) -> bool:
    """A tool call counts as a search only if it actually searches.

    Bash is in ``_SEARCH_TOOL_NAMES`` because agents run grep/rg/find through
    it, but ordinary shell commands (mkdir, cp, chmod, git add) that legitimately
    emit no output must not be mistaken for empty searches.
    """
    name = tc.get("tool_name")
    if name not in _SEARCH_TOOL_NAMES:
        return False
    if name in BASH_TOOL_NAMES:
        inp = tc.get("input", {})
        cmd = inp.get("command", "") if isinstance(inp, dict) else ""
        return shell_runs_search(cmd) if isinstance(cmd, str) else False
    return True


def _is_fruitless_step(step: dict) -> bool:
    """Check if a step's search/grep tool calls all returned empty results.

    Supports two detection methods:
    1. tool_call output/result is empty or contains no matches
    2. tool_call status indicates no results
    """
    tool_calls = step.get("tool_calls", [])
    if not tool_calls:
        return False

    search_calls = [tc for tc in tool_calls if _is_search_call(tc)]
    if not search_calls:
        return False

    # Method 1: Check individual tool call outputs/results
    for tc in search_calls:
        output = tc.get("output", tc.get("result", ""))
        if isinstance(output, str):
            stripped = output.strip()
            if stripped and stripped not in ("", "No matches found", "No results", "No files found"):
                return False
        elif isinstance(output, dict):
            content = output.get("content", output.get("text", ""))
            if isinstance(content, str) and content.strip():
                return False
            # Content-block lists (e.g. [{"type": "text", "text": ...}]) are
            # non-empty results too — mirror the bare-list handling below.
            if isinstance(content, list) and content:
                return False
        elif isinstance(output, list) and output:
            return False

    return True


def detect_fruitless_streaks(steps: list[dict]) -> list[dict]:
    """Detect consecutive steps with empty search/grep tool output.

    Returns list of streaks:
    ``[{"start_step": int, "end_step": int, "length": int, "tools": [str]}]``
    """
    streaks: list[dict] = []
    current_streak: dict | None = None

    for s in steps[:_MAX_STEPS]:
        if not s.get("tool_calls"):
            if current_streak is not None and current_streak["length"] >= 3:
                streaks.append(current_streak)
            current_streak = None
            continue

        if _is_fruitless_step(s):
            tool_names = [tc.get("tool_name", "?") for tc in s["tool_calls"]]
            step_idx = s.get("index", 0)
            if current_streak is None:
                current_streak = {
                    "start_step": step_idx,
                    "end_step": step_idx,
                    "length": 1,
                    "tools": tool_names,
                }
            else:
                current_streak["end_step"] = step_idx
                current_streak["length"] += 1
                current_streak["tools"].extend(tool_names)
        else:
            if current_streak is not None and current_streak["length"] >= 3:
                streaks.append(current_streak)
            current_streak = None

    if current_streak is not None and current_streak["length"] >= 3:
        streaks.append(current_streak)

    return streaks


def compute_autonomy_ratio(steps: list[dict]) -> float:
    """Compute autonomy ratio from trigger fields.

    Returns ratio of autonomous steps to total assistant steps (0.0 to 1.0).
    Falls back to 1 - (user_steps / total_steps) if trigger field is absent.
    """
    assistant_steps = [s for s in steps if s.get("role") == "assistant"]
    if not assistant_steps:
        return 0.0

    # Autonomy = share of turns not directly driven by the user.
    user_steps = sum(1 for s in steps if s.get("role") == "user")
    total = len(steps)
    return round(1.0 - (user_steps / total), 4) if total > 0 else 0.0


def detect_tool_selection_antipatterns(steps: list[dict]) -> list[dict]:
    """Detect Bash commands used for file reading when Read tool exists.

    Returns list of flagged steps.
    """
    # Only flag commands where sed/cat/head IS the primary command reading a file,
    # not when used in pipes (e.g., "grep ... | head -20" is legitimate).
    _BASH_READ_PATTERNS = [
        re.compile(r"^sed\s+-n\s+'"),          # sed as primary command
        re.compile(r"^cat\s+[\"']?[/\\a-zA-Z]"),  # cat as primary command
        re.compile(r"^head\s+-[n0-9]"),        # head as primary command
        re.compile(r"^tail\s+-[n0-9]"),        # tail as primary command
    ]

    flagged: list[dict] = []
    for s in steps[:_MAX_STEPS]:
        for tc in s.get("tool_calls", []):
            # OpenCode emits lowercase tool names — match both spellings,
            # like _is_search_call above.
            if tc.get("tool_name") not in BASH_TOOL_NAMES:
                continue
            inp = tc.get("input", {})
            cmd = inp.get("command", "") if isinstance(inp, dict) else ""
            for pattern in _BASH_READ_PATTERNS:
                if pattern.search(cmd):
                    flagged.append({
                        "step": s.get("index", 0),
                        "command": cmd[:80],
                        "pattern": pattern.pattern,
                    })
                    break
    return flagged


# ---------------------------------------------------------------------------
# 6b. Failed same-path edit retries + repeated identical search
# ---------------------------------------------------------------------------

_EDIT_THRASH_WINDOW = 15
_EDIT_THRASH_MIN = 3
_REPEATED_SEARCH_MIN = 3


def detect_edit_thrash(steps: list[dict]) -> list[dict]:
    """Detect repeated Write/Edit on one path that includes failed attempts.

    Successful iterate/fix loops on the same file are normal and are NOT
    flagged. A cluster qualifies only when ≥3 write attempts hit the same path
    within a short window **and** at least one of those attempts failed.

    Returns
    -------
    list[dict]
        ``{"path", "count", "fail_count", "steps", "start_step", "end_step"}``
    """
    # path -> list of (step_idx, failed)
    by_path: dict[str, list[tuple[int, bool]]] = {}
    for s in steps[:_MAX_STEPS]:
        idx = int(s.get("index", 0))
        for tc in s.get("tool_calls") or []:
            name = tc.get("tool_name") or ""
            if name not in _WRITE_TOOL_NAMES:
                continue
            inp = tc.get("input", {})
            path = write_target_path(inp) if isinstance(inp, dict) else ""
            if not path:
                continue
            path = path.replace("\\", "/")
            by_path.setdefault(path, []).append((idx, tool_call_failed(tc)))

    thrash: list[dict] = []
    for path, events in by_path.items():
        if len(events) < _EDIT_THRASH_MIN:
            continue
        best: list[tuple[int, bool]] | None = None
        best_fails = 0
        for i in range(len(events)):
            window = [events[i]]
            for j in range(i + 1, len(events)):
                if events[j][0] - events[i][0] > _EDIT_THRASH_WINDOW:
                    break
                window.append(events[j])
            if len(window) < _EDIT_THRASH_MIN:
                continue
            fail_count = sum(1 for _, failed in window if failed)
            if fail_count < 1:
                continue
            if best is None or len(window) > len(best):
                best = window
                best_fails = fail_count
        if best is None:
            continue
        step_list = [idx for idx, _ in best]
        thrash.append({
            "path": path,
            "count": len(best),
            "fail_count": best_fails,
            "steps": step_list,
            "start_step": step_list[0],
            "end_step": step_list[-1],
        })

    thrash.sort(key=lambda t: (-t["fail_count"], -t["count"], t["start_step"], t["path"]))
    return thrash


def _search_signature(tc: dict) -> str | None:
    """Normalize a search tool call into a comparable signature, or None."""
    if not _is_search_call(tc):
        return None
    name = tc.get("tool_name") or ""
    inp = tc.get("input", {}) if isinstance(tc.get("input"), dict) else {}
    if name in ("Grep", "grep"):
        pattern = str(inp.get("pattern") or "").strip()
        path = str(inp.get("path") or "").strip()
        if not pattern:
            return None
        return f"grep:{pattern}|{path}"
    if name in ("Glob", "glob"):
        pattern = str(inp.get("pattern") or "").strip()
        path = str(inp.get("path") or "").strip()
        if not pattern:
            return None
        return f"glob:{pattern}|{path}"
    if name in BASH_TOOL_NAMES:
        cmd = str(inp.get("command") or "")
        normalized = " ".join(cmd.split())
        if not normalized:
            return None
        return f"bash:{normalized}"
    # Other search tools — use name + primary arg
    for key in ("pattern", "query", "path"):
        if inp.get(key):
            return f"{name}:{inp[key]}"
    return f"{name}:"


def detect_repeated_searches(steps: list[dict]) -> list[dict]:
    """Detect the same search signature used ≥3 times with empty results.

    Complements :func:`detect_fruitless_streaks` (consecutive-only) by catching
    non-consecutive repeats of an identical query/pattern.

    Returns
    -------
    list[dict]
        ``{"signature": str, "count": int, "steps": [int], "display": str}``
    """
    # signature -> step indices where that search was fruitless on the step
    by_sig: dict[str, list[int]] = {}
    for s in steps[:_MAX_STEPS]:
        if not _is_fruitless_step(s):
            continue
        idx = int(s.get("index", 0))
        seen_this_step: set[str] = set()
        for tc in s.get("tool_calls") or []:
            sig = _search_signature(tc)
            if not sig or sig in seen_this_step:
                continue
            seen_this_step.add(sig)
            by_sig.setdefault(sig, []).append(idx)

    out: list[dict] = []
    for sig, step_list in by_sig.items():
        unique_steps = sorted(set(step_list))
        if len(unique_steps) < _REPEATED_SEARCH_MIN:
            continue
        # Prefer non-consecutive repeats (fruitless streaks already cover runs).
        consecutive = all(
            unique_steps[i] + 1 == unique_steps[i + 1]
            for i in range(len(unique_steps) - 1)
        )
        if consecutive:
            continue
        display = sig
        if sig.startswith("grep:"):
            display = sig[len("grep:"):]
        elif sig.startswith("glob:"):
            display = sig[len("glob:"):]
        elif sig.startswith("bash:"):
            display = sig[len("bash:"):]
        out.append({
            "signature": sig,
            "count": len(unique_steps),
            "steps": unique_steps,
            "display": display[:80],
        })

    out.sort(key=lambda r: (-r["count"], r["steps"][0], r["signature"]))
    return out
