"""Trajectory diagnostics: file interaction, failure chains, root-cause attribution, performance bottlenecks."""

from __future__ import annotations

from trajviz.tool_vocab import (
    BASH_TOOL_NAMES,
    WRITE_TOOL_NAMES as _WRITE_TOOL_SET,
    parse_skill_name as _parse_skill_name,
    write_target_path as _write_target_path,
)

import os
import re
import statistics

from .shell_cmd import tool_chart_name
from .tool_failure import tool_call_error_kind, tool_call_failed


# ---------------------------------------------------------------------------
# 1. File Interaction Analysis
# ---------------------------------------------------------------------------

# Tools whose input dict has a structured file path field
_TOOL_FILE_FIELDS: dict[str, tuple[tuple[str, ...], str]] = {
    # tool_name -> (candidate_input_keys, interaction_type)
    # Multiple keys per tool let us handle scaffolds that use different field
    # spellings — e.g. Claude Code emits ``file_path`` (snake) while OpenCode
    # emits ``filePath`` (camel). The lookup picks whichever the call carries.
    "Read":         (("file_path", "filePath", "path"), "read"),
    "read":         (("file_path", "filePath", "path"), "read"),
    "Write":        (("file_path", "filePath", "path"), "write"),
    "write":        (("file_path", "filePath", "path"), "write"),
    "Edit":         (("file_path", "filePath", "path"), "write"),
    "edit":         (("file_path", "filePath", "path"), "write"),
    "NotebookEdit": (("notebook_path", "notebookPath"), "write"),
}

# Tools that search (return multiple file references in pattern/path fields)
_TOOL_SEARCH_FIELDS: dict[str, tuple[str, str]] = {
    "Glob": ("pattern", "search"),
    "glob": ("pattern", "search"),
    "Grep": ("pattern", "search"),
    "grep": ("pattern", "search"),
}

# Regex for path-like tokens in bash commands
# Matches: /unix/path, ./relative, ../parent, d:/windows/path
# Handles both unquoted and quoted paths ("d:/foo", 'd:/foo')
_PATH_RE = re.compile(
    r"""(?:^|[\s=;|&(])"""                       # preceded by whitespace or delimiter
    r"""(?:"""
    r"""["']((?:/|\.\.?/|[a-zA-Z]:[/\\])[^"']*?)["']"""   # quoted path
    r"""|"""
    r"""((?:/|\.\.?/|[a-zA-Z]:[/\\])[^\s;|&)'"]+)"""      # unquoted path
    r""")"""
)


def _extract_bash_paths(command: str) -> list[str]:
    """Extract file-path-like tokens from a bash command string."""
    paths = []
    for m in _PATH_RE.finditer(command):
        p = m.group(1) or m.group(2)  # group 1 = quoted, group 2 = unquoted
        if not p:
            continue
        p = p.rstrip(",:")
        # Skip very short or unlikely paths
        if len(p) < 2 or p.rstrip("/\\") in ("", "/", "./", ".."):
            continue
        # Must contain at least one path component with alphanumeric chars
        parts = [seg for seg in p.replace("\\", "/").split("/") if seg]
        if parts and any(c.isalnum() for c in parts[-1]):
            paths.append(p)
    return paths


def _is_skill_definition_path(path: str) -> bool:
    """True when *path* is a skill definition file (``SKILL.md``)."""
    name = path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return bool(name) and name.lower() == "skill.md"


def extract_file_interactions(steps: list[dict]) -> list[dict]:
    """Extract file path references from every tool call in parsed steps.

    Returns a list of records: {step, tool, path, type, tokens}
    where type is one of: read, write, search, skill.
    """
    interactions: list[dict] = []

    for step in steps:
        step_idx = step["index"]
        step_tokens = step["tokens"]["total"]

        for tc in step.get("tool_calls", []):
            tool_name = tc.get("tool_name", "")
            inp = tc.get("input", {})
            if not isinstance(inp, dict):
                inp = {}

            found_paths: list[tuple[str, str]] = []  # (path, interaction_type)

            # Structured file tools — try every candidate field name so the
            # extractor works across schema spellings (snake_case / camelCase).
            if tool_name in _TOOL_FILE_FIELDS:
                fields, itype = _TOOL_FILE_FIELDS[tool_name]
                for field in fields:
                    path = inp.get(field, "")
                    if path:
                        found_paths.append((str(path), itype))
                        break

            # Search tools — extract path (directory) and pattern (for Glob only)
            elif tool_name in _TOOL_SEARCH_FIELDS:
                _, itype = _TOOL_SEARCH_FIELDS[tool_name]
                path = inp.get("path", "")
                if path:
                    found_paths.append((str(path), itype))
                # Glob patterns are file paths; Grep patterns are text regexes
                if tool_name in ("Glob", "glob"):
                    pattern = inp.get("pattern", "")
                    if pattern:
                        found_paths.append((str(pattern), itype))

            # Bash — heuristic path extraction
            elif tool_name in BASH_TOOL_NAMES:
                command = inp.get("command", "")
                if command:
                    for p in _extract_bash_paths(command):
                        # Classify: if command looks like a write operation
                        write_cmds = ("mv ", "cp ", "rm ", "mkdir ", "touch ",
                                      "> ", ">> ", "tee ")
                        itype = "write" if any(command.strip().startswith(c) or f" {c}" in command for c in write_cmds) else "read"
                        found_paths.append((p, itype))

            skill = _parse_skill_name(tool_name, inp)
            if skill:
                found_paths.append((f"skill:{skill}", "skill"))

            for path, itype in found_paths:
                # Normalize backslashes to forward slashes for consistency
                path = path.replace("\\", "/")
                if itype == "read" and _is_skill_definition_path(path):
                    itype = "skill"
                interactions.append({
                    "step": step_idx,
                    "tool": tool_name,
                    "path": path,
                    "type": itype,
                    "tokens": step_tokens,
                })

    return interactions


def identify_target_files(steps: list[dict]) -> set[str]:
    """Identify target files: files in patch parts + successful Edit/Write calls."""
    targets: set[str] = set()

    for step in steps:
        for part in step.get("parts", []):
            if part.get("type") == "patch":
                for f in part.get("files", []):
                    if f:
                        targets.add(str(f).replace("\\", "/"))

        for tc in step.get("tool_calls", []):
            tool_name = tc.get("tool_name", "")
            status = tc.get("status", "")
            if tool_name in _WRITE_TOOL_SET and status not in ("error", "failed", "failure"):
                inp = tc.get("input", {})
                if isinstance(inp, dict):
                    # OpenCode uses ``filePath``; Claude Code uses ``file_path``.
                    path = _write_target_path(inp)
                    if path:
                        # Match extract_file_interactions, which normalizes
                        # backslashes; _paths_match/normpath won't on POSIX.
                        targets.add(str(path).replace("\\", "/"))

    return targets


def _normalize_path(path: str) -> str:
    """Normalize a path for comparison (resolve . and .., strip trailing /)."""
    return os.path.normpath(path) if path else path


def _paths_match(a: str, b: str) -> bool:
    """Check if two paths refer to the same file (suffix match for relative vs absolute)."""
    na, nb = _normalize_path(a), _normalize_path(b)
    if na == nb:
        return True
    # One might be absolute, the other relative — check suffix match
    return na.endswith("/" + nb) or nb.endswith("/" + na)


def compute_file_targeting_metrics(
    interactions: list[dict],
    target_files: set[str],
    total_steps: int,
) -> dict:
    """Compute file-targeting efficiency metrics.

    Returns dict with:
    - steps_to_first_touch: dict[file, {absolute, relative}]
    - avg_steps_to_first_touch: float
    - exploration_ratio: float
    - per_file_token_cost: dict[file, int]
    """
    if not interactions or not target_files or total_steps == 0:
        return {
            "steps_to_first_touch": {},
            "avg_steps_to_first_touch": None,
            "exploration_ratio": None,
            "per_file_token_cost": {},
        }

    # First touch per target file
    first_touch: dict[str, int] = {}
    for target in target_files:
        for inter in interactions:
            if _paths_match(inter["path"], target):
                if target not in first_touch or inter["step"] < first_touch[target]:
                    first_touch[target] = inter["step"]

    steps_to_first_touch = {}
    for f, step_idx in first_touch.items():
        steps_to_first_touch[f] = {
            "absolute": step_idx,
            "relative": round(step_idx / total_steps, 4) if total_steps > 0 else 0,
        }

    avg_sft = None
    if first_touch:
        avg_sft = round(sum(first_touch.values()) / len(first_touch), 2)

    # Exploration ratio: unique files touched / target file count
    all_files = {_normalize_path(i["path"]) for i in interactions if i["type"] in ("read", "search")}
    exploration_ratio = round(len(all_files) / len(target_files), 2) if target_files else None

    # Per-file token cost: divide step tokens equally among files in that step
    file_token_cost: dict[str, float] = {}
    # Group interactions by step
    step_files: dict[int, list[str]] = {}
    step_tokens: dict[int, int] = {}
    for inter in interactions:
        sid = inter["step"]
        step_files.setdefault(sid, []).append(inter["path"])
        step_tokens[sid] = inter["tokens"]

    for sid, files in step_files.items():
        tok = step_tokens.get(sid, 0)
        if files and tok > 0:
            share = tok / len(files)
            for f in files:
                nf = _normalize_path(f)
                file_token_cost[nf] = file_token_cost.get(nf, 0) + share

    per_file_token_cost = {f: round(v) for f, v in sorted(file_token_cost.items(), key=lambda x: -x[1])}

    return {
        "steps_to_first_touch": steps_to_first_touch,
        "avg_steps_to_first_touch": avg_sft,
        "exploration_ratio": exploration_ratio,
        "per_file_token_cost": per_file_token_cost,
    }


# ---------------------------------------------------------------------------
# 2. Failure Chain Analysis
# ---------------------------------------------------------------------------

def _step_has_error(step: dict) -> bool:
    """Check if a step has at least one failed tool call."""
    if step.get("error_count", 0) > 0:
        return True
    return any(tool_call_failed(tc) for tc in step.get("tool_calls", []))


def _error_tool_target(tc: dict) -> tuple[str, str]:
    """Extract (tool_name, primary_target) from a tool call for comparison."""
    tool_name = tc.get("tool_name", "")
    inp = tc.get("input", {})
    if not isinstance(inp, dict):
        return (tool_name, "")
    for k in ("file_path", "command", "pattern", "path", "query"):
        if inp.get(k):
            return (tool_name, str(inp[k])[:80])
    return (tool_name, "")


def detect_failure_chains(steps: list[dict]) -> list[dict]:
    """Find maximal sequences of consecutive assistant steps with errors.

    Returns list of chain dicts: {start, end, steps: [indices]}
    """
    chains: list[dict] = []
    current_chain: list[int] = []

    for step in steps:
        if step.get("role") != "assistant":
            continue  # user steps don't break chains
        if _step_has_error(step):
            current_chain.append(step["index"])
        else:
            if current_chain:
                chains.append({
                    "start": current_chain[0],
                    "end": current_chain[-1],
                    "steps": list(current_chain),
                })
                current_chain = []

    if current_chain:
        chains.append({
            "start": current_chain[0],
            "end": current_chain[-1],
            "steps": list(current_chain),
        })

    return chains


def classify_chain_steps(chain: dict, steps: list[dict]) -> list[dict]:
    """Classify each step in a failure chain as first_error, recovery_attempt, or cascade.

    Returns list of {step_idx, classification} dicts.
    """
    step_map = {s["index"]: s for s in steps}
    chain_steps = chain["steps"]

    if not chain_steps:
        return []

    # Get first error's tool+target signature
    first_step = step_map.get(chain_steps[0], {})
    first_error_sigs = set()
    for tc in first_step.get("tool_calls", []):
        # Same predicate as _step_has_error: a chain opened by a cancelled or
        # timed-out call must still yield first-error signatures, or identical
        # retries would all be classified "cascade".
        if tool_call_failed(tc):
            first_error_sigs.add(_error_tool_target(tc))

    result = [{"step_idx": chain_steps[0], "classification": "first_error"}]

    for idx in chain_steps[1:]:
        step = step_map.get(idx, {})
        step_sigs = set()
        for tc in step.get("tool_calls", []):
            step_sigs.add(_error_tool_target(tc))

        # Recovery if same tool+target as first error
        if step_sigs & first_error_sigs:
            result.append({"step_idx": idx, "classification": "recovery_attempt"})
        else:
            result.append({"step_idx": idx, "classification": "cascade"})

    return result


def link_chains_to_agents(
    chains: list[dict],
    steps: list[dict],
    agent_summaries: list[dict],
) -> list[dict]:
    """Annotate failure chains with parent agent spawning info.

    Returns chains with added spawning_agent and spawning_step fields where applicable.
    """
    if not agent_summaries:
        return chains

    from .metrics import effective_agent

    # Build agent_id -> spawned_by_step lookup
    spawn_map: dict[str, int] = {}
    for a in agent_summaries:
        if a.get("spawned_by_step") is not None:
            spawn_map[a["agent_id"]] = a["spawned_by_step"]

    step_map = {s["index"]: s for s in steps}

    for chain in chains:
        first_step = step_map.get(chain["start"], {})
        agent_id = effective_agent(first_step)
        if agent_id and agent_id in spawn_map:
            chain["spawning_step"] = spawn_map[agent_id]
            chain["spawning_agent"] = "main"

    return chains


def compute_failure_chain_metrics(chains: list[dict], total_assistant_steps: int) -> dict:
    """Aggregate failure chain metrics."""
    if not chains:
        return {
            "total_chains": 0,
            "total_chain_steps": 0,
            "longest_chain": 0,
            "chain_step_pct": 0.0,
        }

    total_chain_steps = sum(len(c["steps"]) for c in chains)
    longest = max(len(c["steps"]) for c in chains)

    return {
        "total_chains": len(chains),
        "total_chain_steps": total_chain_steps,
        "longest_chain": longest,
        "chain_step_pct": round(total_chain_steps / total_assistant_steps * 100, 1) if total_assistant_steps > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 3. Root-Cause Attribution
# ---------------------------------------------------------------------------

def _error_pattern(tc: dict) -> str:
    """Extract a short error pattern from a tool call."""
    # Check explicit error field
    err = tc.get("error") or ""
    if err:
        return str(err).strip().split("\n")[0][:120]

    # Check exit code
    meta = tc.get("metadata", {})
    if isinstance(meta, dict) and meta.get("exit") not in (None, 0):
        return f"exit code {meta['exit']}"

    # Check status
    status = tc.get("status", "")
    if status in ("error", "failed", "failure"):
        output = tc.get("output", "")
        if output:
            return str(output).strip().split("\n")[0][:120]
        return f"status: {status}"

    return "unknown error"


def cluster_errors(steps: list[dict]) -> list[dict]:
    """Group error tool calls by (display tool, error_pattern).

    Bash/shell calls are labeled with the peeled base command (via
    :func:`trajviz.insight.shell_cmd.tool_chart_name`) so ``npm test`` failures
    do not collapse into a generic ``Bash: exit code`` cluster.

    Returns sorted list of cluster dicts:
    ``{tool, error_class, pattern, count, steps, first_step, last_step}``
    where ``error_class`` is ``"system"`` (scaffold) or ``"tool"`` (agentic).
    """
    clusters: dict[tuple[str, str], dict] = {}

    for step in steps:
        for tc in step.get("tool_calls", []):
            if not tool_call_failed(tc):
                continue

            display = tool_chart_name(tc)
            pattern = _error_pattern(tc)
            key = (display, pattern)

            if key not in clusters:
                clusters[key] = {
                    "tool": display,
                    "error_class": tool_call_error_kind(tc),
                    "pattern": pattern,
                    "count": 0,
                    "steps": [],
                    "first_step": step["index"],
                    "last_step": step["index"],
                }
            c = clusters[key]
            c["count"] += 1
            if step["index"] not in c["steps"]:
                c["steps"].append(step["index"])
            c["last_step"] = max(c["last_step"], step["index"])

    return sorted(clusters.values(), key=lambda c: (-c["count"], c["first_step"]))


def annotate_clusters_with_agents(
    clusters: list[dict],
    steps: list[dict],
    agent_summaries: list[dict],
) -> list[dict]:
    """Annotate error clusters with parent agent info when errors occur in sub-agents."""
    if not agent_summaries:
        return clusters

    from .metrics import effective_agent

    spawn_map: dict[str, int] = {}
    for a in agent_summaries:
        if a.get("spawned_by_step") is not None:
            spawn_map[a["agent_id"]] = a["spawned_by_step"]

    step_map = {s["index"]: s for s in steps}

    for cluster in clusters:
        first_step = step_map.get(cluster["first_step"], {})
        agent_id = effective_agent(first_step)
        if agent_id and agent_id in spawn_map:
            cluster["parent_agent"] = "main"
            cluster["parent_step"] = spawn_map[agent_id]

    return clusters


def format_root_cause_summary(clusters: list[dict]) -> list[str]:
    """Generate human-readable summary text per root-cause cluster."""
    summaries: list[str] = []
    for c in clusters:
        steps_range = f"steps {c['first_step']}..{c['last_step']}" if c["first_step"] != c["last_step"] else f"step {c['first_step']}"
        text = f"{c['count']}x {c['tool']} failures: '{c['pattern']}' ({steps_range})"
        if c.get("parent_agent"):
            text += f" \u2014 traced to {c['parent_agent']} step {c['parent_step']}"
        summaries.append(text)
    return summaries


# ---------------------------------------------------------------------------
# 4. Bottleneck Explanation
# ---------------------------------------------------------------------------

def decompose_hotspot_duration(
    step: dict,
    analytics_row: dict | None = None,
    idle_gap: float | None = None,
) -> dict:
    """Decompose a step's duration into tool execution, LLM inference, and idle components.

    Returns {tool_s, inference_s, idle_s, tool_pct, inference_pct, idle_pct,
             timing_incomplete, dominant_tool}
    """
    duration = step.get("duration") or 0
    if duration <= 0:
        return {
            "tool_s": 0, "inference_s": 0, "idle_s": 0,
            "tool_pct": 0, "inference_pct": 0, "idle_pct": 0,
            "timing_incomplete": True, "dominant_tool": None,
        }

    idle_s = max(0, idle_gap or 0)

    # Sum tool call durations
    tool_time_ms = 0
    timing_complete = True
    dominant_tool_name = ""
    dominant_tool_target = ""
    dominant_tool_dur = 0

    from .metrics import tool_call_duration_ms

    for tc in step.get("tool_calls", []):
        v = tool_call_duration_ms(tc)
        dur_ms = v if v is not None else 0
        if v is None:
            timing_complete = False

        tool_time_ms += dur_ms
        if dur_ms > dominant_tool_dur:
            dominant_tool_dur = dur_ms
            dominant_tool_name = tc.get("tool_name", "")
            inp = tc.get("input", {})
            if isinstance(inp, dict):
                for k in ("file_path", "command", "pattern", "path"):
                    if inp.get(k):
                        dominant_tool_target = str(inp[k])[:50]
                        break

    tool_s = min(tool_time_ms / 1000.0, duration)  # cap at step duration
    # inference = the step's own duration minus tool time. idle_s is the gap
    # BEFORE the step (prev completion -> this start) and is disjoint from the
    # step's duration, so it must not be subtracted here.
    inference_s = max(0, duration - tool_s)

    dominant_tool = None
    if dominant_tool_name:
        dominant_tool = {
            "name": dominant_tool_name,
            "target": dominant_tool_target,
            "duration_s": round(dominant_tool_dur / 1000.0, 2),
        }

    # All three percentages share one base — the full wall-clock span the step
    # accounts for (its own duration plus the disjoint pre-step idle gap) — so
    # they sum to ~100% and idle_pct can never exceed 100.
    total = duration + idle_s

    return {
        "tool_s": round(tool_s, 2),
        "inference_s": round(inference_s, 2),
        "idle_s": round(idle_s, 2),
        "tool_pct": round(tool_s / total * 100, 1) if total > 0 else 0,
        "inference_pct": round(inference_s / total * 100, 1) if total > 0 else 0,
        "idle_pct": round(idle_s / total * 100, 1) if total > 0 else 0,
        "timing_incomplete": not timing_complete,
        "dominant_tool": dominant_tool,
    }


def explain_hotspot(step: dict, decomposition: dict) -> str:
    """Generate a one-line explanation for a hotspot step."""
    idx = step["index"]
    dur = step.get("duration") or 0
    d = decomposition

    parts = []

    # Dominant component first
    if d["tool_pct"] >= d["inference_pct"] and d["tool_pct"] >= d["idle_pct"]:
        dt = d.get("dominant_tool")
        if dt:
            target = f": {dt['target']}" if dt["target"] else ""
            parts.append(f"{d['tool_s']}s executing tools ({dt['name']}{target} {dt['duration_s']}s)")
        else:
            parts.append(f"{d['tool_s']}s executing tools")
        if d["inference_s"] > 0:
            tok = step["tokens"]["total"]
            parts.append(f"{d['inference_s']}s LLM inference ({tok:,} tokens)")
    elif d["idle_pct"] >= d["inference_pct"]:
        parts.append(f"{d['idle_s']}s idle gap before step (queuing or rate limiting)")
        if d["tool_s"] > 0:
            parts.append(f"{d['tool_s']}s tool execution")
        # Mirror the other branches: never drop a component of the step's own
        # duration just because the pre-step idle gap dominates.
        if d["inference_s"] > 0:
            tok = step["tokens"]["total"]
            parts.append(f"{d['inference_s']}s LLM inference ({tok:,} tokens)")
    else:
        tok = step["tokens"]["total"]
        parts.append(f"{d['inference_s']}s LLM inference ({tok:,} tokens)")
        if d["tool_s"] > 0:
            parts.append(f"{d['tool_s']}s tool execution")

    if d["idle_s"] > 0 and d["idle_pct"] < d["tool_pct"] and d["idle_pct"] < d["inference_pct"]:
        parts.append(f"{d['idle_s']}s idle")

    incomplete = " [timing incomplete]" if d["timing_incomplete"] else ""
    return f"Step {idx}: {dur:.1f}s \u2014 {', '.join(parts)}{incomplete}"


# ---------------------------------------------------------------------------
# 4b. Performance bottlenecks (outlier + actionable cause)
# ---------------------------------------------------------------------------

_BN_MIN_ABS_S = 8.0
_BN_MIN_IDLE_S = 10.0
_BN_MIN_TOOL_S = 5.0
_BN_MIN_INFERENCE_S = 15.0
_BN_MAX_ISSUES = 3


def _duration_outlier_threshold(durations: list[float]) -> float:
    """Session-relative floor: absolute minimum, or median×2 / mean+1.5σ when possible."""
    if not durations:
        return _BN_MIN_ABS_S
    from .metrics import _percentile

    median = float(statistics.median(durations))
    p90 = float(_percentile(durations, 0.90))
    floor = max(_BN_MIN_ABS_S, median * 2.0, p90)

    if len(durations) >= 8:
        mean = statistics.mean(durations)
        try:
            stdev = statistics.stdev(durations)
        except statistics.StatisticsError:
            stdev = 0.0
        if stdev > 0:
            sigma = mean + 1.5 * stdev
            # Prefer the stricter of relative floors so uniform slow runs stay quiet.
            floor = max(_BN_MIN_ABS_S, min(floor, max(median * 2.0, sigma)))
    return floor


def _classify_performance_cause(
    decomp: dict,
    step: dict,
    analytics_row: dict | None,
) -> str | None:
    """Return cause key if the dominant component is an actionable bottleneck."""
    idle_pct = float(decomp.get("idle_pct") or 0)
    tool_pct = float(decomp.get("tool_pct") or 0)
    inf_pct = float(decomp.get("inference_pct") or 0)
    idle_s = float(decomp.get("idle_s") or 0)
    tool_s = float(decomp.get("tool_s") or 0)
    inference_s = float(decomp.get("inference_s") or 0)

    # Idle/queue first — often the true wall-clock bottleneck and distinct from step work.
    if idle_pct >= 40 and idle_s >= _BN_MIN_IDLE_S:
        return "idle"

    if tool_pct >= 50 and tool_s >= _BN_MIN_TOOL_S:
        return "tool"

    if inf_pct >= 50 and inference_s >= _BN_MIN_INFERENCE_S:
        tokens = step.get("tokens") or {}
        tok_total = int(tokens.get("total") or 0) if isinstance(tokens, dict) else 0
        cache_ratio = None
        if analytics_row is not None:
            cache_ratio = analytics_row.get("cache_ratio")
        # Context waste: large prompt and/or weak cache hit rate.
        if tok_total >= 40_000 or (
            cache_ratio is not None and tok_total > 0 and float(cache_ratio) < 0.35
        ):
            return "context"
        # Extremely long model-side turn even without cache signal.
        if inference_s >= 30:
            return "inference"

    return None


def detect_performance_bottlenecks(
    steps: list[dict],
    step_analytics: list[dict],
    *,
    max_issues: int = _BN_MAX_ISSUES,
) -> list[dict]:
    """Detect real performance bottlenecks (outlier + clear cause).

    Requires the step to exceed a session-relative duration floor **and** have
    a dominant actionable cause: idle/queue, tool wait, or context/inference.
    Prefer this over duration top-N rankings for Issues, KPI, and analysis.

    Returns
    -------
    list[dict]
        ``step_idx``, ``duration``, ``cause``, ``title``, ``detail``, ``why``,
        ``decomposition``, ``explanation``.
    """
    asst = [
        s for s in steps
        if s.get("role") == "assistant" and isinstance(s.get("duration"), (int, float))
        and float(s["duration"]) > 0
    ]
    if not asst:
        return []

    durations = [float(s["duration"]) for s in asst]
    threshold = _duration_outlier_threshold(durations)
    analytics_map = {a["index"]: a for a in step_analytics}

    candidates: list[dict] = []
    for step in asst:
        duration = float(step["duration"])
        idx = int(step["index"])
        analytics_row = analytics_map.get(idx)
        idle_gap = analytics_row.get("idle_before_s") if analytics_row else None
        idle_for_impact = max(0.0, float(idle_gap or 0))
        # Idle gaps sit outside step.duration; include them so queue stalls qualify.
        if duration + idle_for_impact < threshold:
            continue
        decomp = decompose_hotspot_duration(step, analytics_row, idle_gap)
        cause = _classify_performance_cause(decomp, step, analytics_row)
        if cause is None:
            continue

        explanation = explain_hotspot(step, decomp)
        dt = decomp.get("dominant_tool") or {}
        tool_name = str(dt.get("name") or "tool")
        tool_target = str(dt.get("target") or "")
        tool_s = float(decomp.get("tool_s") or 0)
        idle_s = float(decomp.get("idle_s") or 0)

        if cause == "idle":
            title = f"Idle/queue bottleneck before #{idx} ({idle_s:.0f}s gap)"
            detail = explanation[:200]
            why = (
                "Large pre-step idle usually means queuing or rate limiting — "
                "reduce parallel pressure or wait policy rather than changing prompts."
            )
        elif cause == "tool":
            label = f"{tool_name}: {tool_target}" if tool_target else tool_name
            short = label if len(label) <= 40 else (label[:37] + "…")
            title = f"Tool bottleneck: {short} ({tool_s:.1f}s)"
            detail = explanation[:200]
            why = (
                "Most of this outlier step was spent in a tool — prefer cheaper tools, "
                "narrower scopes, or caching results instead of repeating heavy calls."
            )
        elif cause == "context":
            title = f"Context/cache bottleneck at #{idx} ({duration:.1f}s)"
            detail = explanation[:200]
            why = (
                "Long inference with heavy tokens or weak cache hits — trim context, "
                "improve cache locality, or avoid reloading large files each turn."
            )
        else:  # inference
            title = f"Inference bottleneck at #{idx} ({duration:.1f}s)"
            detail = explanation[:200]
            why = (
                "Outlier model-side turn with little tool wait — often oversized "
                "reasoning or prompt; tighten instructions or split the task."
            )

        candidates.append({
            "step_idx": idx,
            "duration": duration,
            "cause": cause,
            "title": title,
            "detail": detail,
            "why": why,
            "decomposition": decomp,
            "explanation": explanation,
        })

    # Prefer largest wall impact; for idle, use idle_s as secondary sort key via duration+idle.
    def _impact(item: dict) -> float:
        d = item["decomposition"]
        return float(item["duration"]) + float(d.get("idle_s") or 0)

    candidates.sort(key=_impact, reverse=True)
    return candidates[:max_issues]
