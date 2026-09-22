"""Context-window occupancy, compaction, and Cursor-style composition.

Trajectories rarely record the hidden system prompt or tool JSON schemas, so
the billed occupancy from token metrics is the source of truth for *how full*
the window is. Logged text (messages, skill bodies, spawn prompts, tool
outputs) is tokenized at ≈4 characters/token and attributed to buckets.
Whatever billed occupancy remains is **not in the log** rather than guessed.
"""

from __future__ import annotations

import html
import json
from typing import Any

from trajviz.tool_vocab import SPAWN_TOOL_NAMES, is_mcp_tool, parse_skill_name

from .metrics import effective_agent, tagged_subagent_display_label
from .parser import infer_non_cache_input, spawned_child_session_id

# Display order follows Cursor's tray, then TrajViz-only tool_outputs
# (Cursor folds tool results into Conversation).
USAGE_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("system", "System prompt", "#6b7280"),
    ("tools", "Tool definitions", "#7c3aed"),
    ("rules", "Rules", "#e11d48"),
    ("skills", "Skills", "#ca8a04"),
    ("mcp", "MCP", "#0d9488"),
    ("subagents", "Subagent definitions", "#2563eb"),
    ("summarized", "Summarized conversation", "#8b5cf6"),
    ("conversation", "Conversations", "#ea580c"),
    ("tool_outputs", "Tool outputs", "#059669"),
    ("unattributed", "unattributed", "#94a3b8"),
)

_SYSTEM_ROLES = frozenset({"system", "developer"})
_TOOL_DEF_KEYS = ("tools", "tool_definitions", "toolDefinitions", "tool_schemas")
_RULE_KEYS = ("rules", "user_rules", "userRules", "always_apply_rules", "alwaysApplyRules")
_MCP_KEYS = ("mcp", "mcp_servers", "mcpServers", "mcp_tools", "mcpTools")
_SPAWN_NAMES_LOWER = frozenset(name.lower() for name in SPAWN_TOOL_NAMES)
_ACCOUNTABLE_KEYS = tuple(key for key, _label, _color in USAGE_CATEGORIES if key != "unattributed")
# Occupancy-drop / tool-prune mark a smaller window but do not store a summary.
_EXPLICIT_COMPACTION_KINDS = frozenset({
    "compaction_part",
    "compaction_message",
    "summary",
    "compress_step",
})
SNAPSHOT_CURRENT = "current"


# ---------------------------------------------------------------------------
# Occupancy, compaction, and context-window pressure
# ---------------------------------------------------------------------------

PRESSURE_ALL_AGENTS = "__all__"
PRESSURE_MAIN_AGENT = "__main__"

# Occupancy is a compaction candidate when it falls below this fraction of the
# previous non-zero same-agent occupancy.
_OCCUPANCY_DROP_RATIO = 0.7
# Cursor-style ``tokens.context_window`` snapshots are piecewise-constant
# (copied onto every assistant until the next user turn). A 30% drop is still
# a real summarization; billed per-turn occupancy stays on 0.7 to ignore jitter.
_SNAPSHOT_DROP_RATIO = 0.95
_SNAPSHOT_DROP_MIN_TOKENS = 1000
# Tools pruned in one OpenCode pass share timestamps within this window.
_PRUNE_WAVE_GAP_MS = 30_000

# High-confidence model-id prefixes. Unknown models fall back to
# ``DEFAULT_CONTEXT_WINDOW_LIMIT`` (editable in the UI).
_MODEL_CONTEXT_LIMITS: tuple[tuple[str, int], ...] = (
    ("claude", 200_000),
    ("gpt-4o", 128_000),
)
DEFAULT_CONTEXT_WINDOW_LIMIT = 128_000


def pressure_agent_key(agent_id: str) -> str:
    """Dropdown value for an ``effective_agent`` / pressure-agent id."""
    return PRESSURE_MAIN_AGENT if not agent_id else str(agent_id)


def pressure_agent_id(dropdown_key: str | None) -> str | None:
    """Map a dropdown value to an agent id. ``None`` means all agents."""
    if dropdown_key in (None, "", PRESSURE_ALL_AGENTS):
        return None
    if dropdown_key == PRESSURE_MAIN_AGENT:
        return ""
    return dropdown_key


def _pressure_agent(step: dict) -> str:
    """Context-window identity for a step.

    Each OpenCode/CodeArts session has its own window, even when child
    sessions are not tagged ``isSubAgent`` / ``(subagent)``. Prefer
    ``session_id`` when present; otherwise fall back to ``effective_agent``.
    """
    session_id = step.get("session_id") or ""
    if isinstance(session_id, str) and session_id:
        return session_id
    role = step.get("role", "")
    if role not in ("user", "compaction"):
        return effective_agent(step)
    agent = step.get("agent", "") or ""
    suffix_sub = isinstance(agent, str) and agent.endswith("(subagent)")
    if step.get("is_sub_agent") or suffix_sub:
        return agent or ""
    return ""


def _is_occupancy_step(step: dict) -> bool:
    """True when the step contributes a live context-window occupancy point."""
    if step.get("is_compaction_checkpoint") or step.get("role") == "compaction":
        return False
    return step.get("role") == "assistant"


def _context_window_tokens(step: dict) -> int | None:
    """Explicit per-step window occupancy, or None when the step has none."""
    tokens = step.get("tokens") if isinstance(step.get("tokens"), dict) else {}
    if "context_window" not in tokens:
        return None
    window = tokens.get("context_window")
    if isinstance(window, (int, float)) and not isinstance(window, bool) and window >= 0:
        return int(window)
    return None


def _snapshot_agents(steps: list[dict]) -> set[str]:
    return {
        _pressure_agent(step)
        for step in steps
        if _is_occupancy_step(step) and _context_window_tokens(step) is not None
    }


def _snapshot_drop_host(steps: list[dict], prev_idx: int, new_idx: int, new_step: dict) -> dict:
    """Prefer the user turn between occupancy plateaus as the compaction host."""
    users: list[dict] = []
    for step in steps:
        idx = int(step.get("index", -1))
        if idx <= prev_idx:
            continue
        if idx >= new_idx:
            break
        if step.get("role") == "user":
            users.append(step)
    if len(users) == 1:
        return users[0]
    return new_step


def _agent_pressure_label(agent_id: str, steps: list[dict]) -> str:
    if not agent_id:
        return "main"
    tagged = tagged_subagent_display_label(agent_id, steps)
    if tagged:
        return tagged
    for step in steps:
        if _pressure_agent(step) != agent_id:
            continue
        name = step.get("agent") or ""
        title = step.get("session_title") or ""
        if name:
            return name if len(name) <= 40 else name[:39] + "\u2026"
        if title:
            return title if len(title) <= 40 else title[:39] + "\u2026"
        break
    return agent_id[:8] + "\u2026" if len(agent_id) > 8 else agent_id


def _disambiguate_pressure_labels(agent_ids: list[str], steps: list[dict]) -> dict[str, str]:
    """Unique dropdown/legend labels; suffix a short session id on collisions."""
    from collections import Counter

    raw = {agent_id: _agent_pressure_label(agent_id, steps) for agent_id in agent_ids}
    counts = Counter(raw.values())
    labels: dict[str, str] = {}
    for agent_id, label in raw.items():
        if counts[label] > 1 and agent_id:
            suffix = agent_id[-6:] if len(agent_id) > 6 else agent_id
            labels[agent_id] = f"{label} ({suffix})"
        else:
            labels[agent_id] = label
    return labels


def step_context_occupancy(step: dict) -> dict:
    """Schema-normalized prompt occupancy for one step.

    ``occupancy = fresh input + cache read``. Fresh input is inferred by
    :func:`trajviz.insight.parser.infer_non_cache_input` so Claude Code
    (cache-exclusive input) and OpenCode (cache-excluded input) agree.

    Formats whose per-step totals are the step's own tokens rather than the
    loaded prompt (ICode) carry a resolved ``tokens.context_window``; it is
    the source of truth for occupancy. Its content is prompt (cache-hit)
    tokens, so fresh is zero.
    """
    tokens = step.get("tokens") if isinstance(step.get("tokens"), dict) else {}
    window = _context_window_tokens(step)
    if window is not None:
        return {"fresh": 0, "cache_read": window, "occupancy": window}
    tok_total = tokens.get("total", 0) or 0
    tok_input = tokens.get("input", 0) or 0
    tok_output = tokens.get("output", 0) or 0
    tok_reasoning = tokens.get("reasoning", 0) or 0
    cache_read = tokens.get("cache_read", 0) or 0
    fresh = infer_non_cache_input(
        total_tokens=tok_total,
        input_tokens=tok_input,
        output_tokens=tok_output,
        reasoning_tokens=tok_reasoning,
        cache_read_tokens=cache_read,
    )
    occupancy = fresh + (cache_read or 0)
    return {
        "fresh": int(fresh),
        "cache_read": int(cache_read or 0),
        "occupancy": int(occupancy),
    }


def coerce_window_limit(value: object) -> int | None:
    """Positive token limit from a UI or raw value, else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().lower().replace(",", "").replace("_", "")
        if not text:
            return None
        multiplier = 1
        if text.endswith("m"):
            multiplier = 1_000_000
            text = text[:-1]
        elif text.endswith("k"):
            multiplier = 1_000
            text = text[:-1]
        try:
            value = float(text) * multiplier
        except ValueError:
            return None
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    return None


def infer_context_window_limit(
    steps: list[dict],
    raw: dict | None = None,
) -> int | None:
    """Return a context-window token limit, or None when it cannot be known.

    Prefers an explicit CodeArts ``context_tokens`` / ``contextToken`` field,
    then a small model-id prefix table. Never guesses from peak occupancy.
    """
    if isinstance(raw, dict):
        md = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        for key in ("context_tokens", "contextToken"):
            coerced = coerce_window_limit(md.get(key))
            if coerced:
                return coerced
        snap = md.get("context_snapshot") if isinstance(md.get("context_snapshot"), dict) else {}
        coerced = coerce_window_limit(snap.get("max_tokens"))
        if coerced:
            return coerced
        base = raw.get("chat_base_info")
        if isinstance(base, dict):
            coerced = coerce_window_limit(base.get("contextToken"))
            if coerced:
                return coerced
    for step in steps:
        model_id = (step.get("model_id") or "").lower()
        if not model_id:
            continue
        for prefix, limit in _MODEL_CONTEXT_LIMITS:
            if model_id.startswith(prefix):
                return limit
    return None


def resolve_context_window_limit(
    steps: list[dict],
    raw: dict | None = None,
    *,
    override: object = None,
) -> int:
    """Window size for occupancy %: user override, else inferred, else 128k."""
    coerced = coerce_window_limit(override)
    if coerced:
        return coerced
    inferred = infer_context_window_limit(steps, raw)
    if inferred:
        return inferred
    return DEFAULT_CONTEXT_WINDOW_LIMIT


def _previous_agent_occupancy(
    steps: list[dict],
    index: int,
    agent_id: str,
) -> int | None:
    for step in reversed(steps[:index]):
        if not _is_occupancy_step(step):
            continue
        if _pressure_agent(step) != agent_id:
            continue
        occ = step_context_occupancy(step)["occupancy"]
        if occ > 0:
            return occ
    return None


def _next_agent_occupancy(
    steps: list[dict],
    index: int,
    agent_id: str,
) -> int | None:
    for step in steps[index + 1:]:
        if not _is_occupancy_step(step):
            continue
        if _pressure_agent(step) != agent_id:
            continue
        occ = step_context_occupancy(step)["occupancy"]
        if occ > 0:
            return occ
    return None


def coalesce_compaction_events(events: list[dict]) -> list[dict]:
    """Merge adjacent same-session compaction signals into one event."""
    if not events:
        return []
    ordered = sorted(events, key=lambda e: int(e.get("step") or 0))
    merged: list[dict] = [dict(ordered[0])]
    for event in ordered[1:]:
        prev = merged[-1]
        step = int(event.get("step") or 0)
        if step - int(prev.get("step") or 0) <= 1:
            after = event.get("occupancy_after")
            if isinstance(after, (int, float)) and after > 0:
                prev["occupancy_after"] = int(after)
            dropped = event.get("dropped")
            if isinstance(dropped, (int, float)) and dropped > (prev.get("dropped") or 0):
                prev["dropped"] = int(dropped)
            if event.get("kind") == "summary":
                prev["kind"] = "summary"
            continue
        merged.append(dict(event))
    for event in merged:
        before = event.get("occupancy_before")
        after = event.get("occupancy_after")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            event["dropped"] = max(0, int(before) - int(after))
    return merged


def _splice_compaction_into_points(
    points: list[dict],
    events: list[dict],
) -> list[dict]:
    """Insert a vertical occupancy cliff at each compaction step.

    Live occupancy is recorded on assistant turns, so without this splice the
    line slopes between the last pre-compaction turn and the first summary
    turn, and the compaction marker floats above the line.
    """
    if not points or not events:
        return [dict(p) for p in points]
    out = [dict(p) for p in points]
    for event in coalesce_compaction_events(events):
        step = int(event.get("step") or 0)
        before = event.get("occupancy_before")
        after = event.get("occupancy_after")
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        if not (0 < after < before):
            continue
        prev = nxt = None
        at_step: list[dict] = []
        for point in out:
            if point["step"] < step:
                prev = point
            elif point["step"] == step:
                at_step.append(point)
            elif nxt is None:
                nxt = point
        src_before = prev or (at_step[0] if at_step else None)
        src_after = nxt or (at_step[-1] if at_step else src_before)
        if src_before is None:
            continue
        if src_after is None:
            src_after = src_before
        out.append({
            "step": step,
            "local_turn": src_before.get("local_turn", 0),
            "fresh": src_before.get("fresh", 0),
            "cache_read": src_before.get("cache_read", 0),
            "occupancy": int(before),
        })
        out.append({
            "step": step,
            "local_turn": src_after.get("local_turn", 0),
            "fresh": src_after.get("fresh", 0),
            "cache_read": src_after.get("cache_read", 0),
            "occupancy": int(after),
        })
    out.sort(key=lambda p: (p["step"], -int(p["occupancy"])))
    deduped: list[dict] = []
    for point in out:
        if (
            deduped
            and deduped[-1]["step"] == point["step"]
            and deduped[-1]["occupancy"] == point["occupancy"]
        ):
            continue
        deduped.append(point)
    return deduped


def _compacted_timestamp_ms(value: object) -> int | None:
    """Epoch-ms from a tool ``time.compacted`` stamp, or None if missing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    ts = int(value)
    if ts <= 0:
        return None
    # Unix seconds (2001–2286) vs already-ms OpenCode ``Date.now()``.
    if 1_000_000_000 <= ts < 10_000_000_000:
        ts *= 1000
    return ts


def _cluster_timestamps(timestamps: list[int], gap_ms: int = _PRUNE_WAVE_GAP_MS) -> list[int]:
    """Earliest timestamp of each prune wave (one OpenCode prune pass)."""
    if not timestamps:
        return []
    ordered = sorted(set(timestamps))
    clusters: list[list[int]] = [[ordered[0]]]
    for ts in ordered[1:]:
        if ts - clusters[-1][-1] <= gap_ms:
            clusters[-1].append(ts)
        else:
            clusters.append([ts])
    return [cluster[0] for cluster in clusters]


def _occupancy_step_at_or_after(
    steps: list[dict],
    agent_id: str,
    ts_ms: int,
    fallback: dict,
) -> dict:
    """First same-window occupancy turn at or after *ts_ms*.

    OpenCode stamps ``time.compacted`` when old tool outputs are pruned, not
    when those tools originally ran. The smaller window is first billed on the
    next occupancy turn after that stamp.
    """
    last_before: dict | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        if _pressure_agent(step) != agent_id or not _is_occupancy_step(step):
            continue
        created = step.get("time_created_ms")
        if not isinstance(created, (int, float)):
            continue
        if int(created) >= ts_ms:
            return step
        last_before = step
    return last_before or fallback


def detect_compaction_events(steps: list[dict]) -> list[dict]:
    """Detect compaction / prune / occupancy-drop events per agent.

    Explicit log signals are preferred. Occupancy-drop is a fallback only
    when the drop is *per session* and *persists* on the next turn — a
    one-step dip that recovers is cache jitter, not compaction.
    """
    from collections import defaultdict

    events: list[dict] = []
    explicit_steps: set[int] = set()
    prune_stamps: dict[str, list[int]] = defaultdict(list)
    prune_fallback: dict[tuple[str, int], dict] = {}

    def _event(step: dict, kind: str, agent_id: str, *, position: int | None = None, **extra) -> dict:
        idx = int(step.get("index", 0))
        pos = position if position is not None else idx
        before = extra.get("occupancy_before")
        if before is None:
            before = _previous_agent_occupancy(steps, pos, agent_id)
        after = extra.get("occupancy_after")
        if after is None:
            after = step_context_occupancy(step)["occupancy"]
        # Compaction parts live on user turns with no token totals — look
        # ahead to the next live occupancy of this window.
        if not after:
            after = _next_agent_occupancy(steps, pos, agent_id)
        dropped = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            dropped = max(0, int(before) - int(after))
        return {
            "step": idx,
            "agent": agent_id,
            "kind": kind,
            "occupancy_before": int(before) if isinstance(before, (int, float)) else None,
            "occupancy_after": int(after) if isinstance(after, (int, float)) else None,
            "dropped": dropped,
        }

    for i, step in enumerate(steps):
        idx = int(step.get("index", i))
        agent_id = _pressure_agent(step)

        if step.get("is_compaction_checkpoint") or step.get("role") == "compaction":
            events.append(_event(step, "compaction_message", agent_id, position=i))
            explicit_steps.add(idx)
            continue

        for part in step.get("parts") or []:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "compaction":
                events.append(_event(step, "compaction_part", agent_id, position=i))
                explicit_steps.add(idx)
            elif ptype in ("step_start", "step_finish"):
                name_l = str(part.get("name", "")).lower()
                if "compress" in name_l or "compact" in name_l:
                    events.append(_event(step, "compress_step", agent_id, position=i))
                    explicit_steps.add(idx)

        if _is_occupancy_step(step) and step.get("summary") is True:
            events.append(_event(step, "summary", agent_id, position=i))
            explicit_steps.add(idx)

        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            ts = _compacted_timestamp_ms(tc.get("time_compacted"))
            if ts is None:
                continue
            prune_stamps[agent_id].append(ts)
            prune_fallback.setdefault((agent_id, ts), step)

    for agent_id, stamps in prune_stamps.items():
        for wave_ts in _cluster_timestamps(stamps):
            fallback = None
            for (aid, ts), step in prune_fallback.items():
                if aid != agent_id:
                    continue
                if abs(ts - wave_ts) <= _PRUNE_WAVE_GAP_MS:
                    fallback = step
                    break
            if fallback is None:
                continue
            host = _occupancy_step_at_or_after(steps, agent_id, wave_ts, fallback)
            host_pos = next(
                (i for i, step in enumerate(steps) if step is host),
                int(host.get("index", 0)),
            )
            host_idx = int(host.get("index", host_pos))
            prune_event = _event(host, "tool_prune", agent_id, position=host_pos)
            # A prune that does not reduce occupancy (dropped == 0) means
            # either the prune happened after the last logged step so we
            # cannot observe its effect, or the next turn's input grew
            # enough to mask it.  In both cases the event is noise.
            # dropped is None when before/after are unavailable — keep those.
            if prune_event.get("dropped") == 0:
                continue
            events.append(prune_event)
            explicit_steps.add(host_idx)

    snapshot_agents = _snapshot_agents(steps)
    occ_seq: dict[str, list[tuple[int, int, dict]]] = defaultdict(list)
    for step in steps:
        if not _is_occupancy_step(step):
            continue
        agent_id = _pressure_agent(step)
        if agent_id in snapshot_agents:
            occ = _context_window_tokens(step)
            if occ is None:
                continue
        else:
            occ = step_context_occupancy(step)["occupancy"]
            if occ <= 0:
                continue
        occ_seq[agent_id].append(
            (int(step.get("index", 0)), occ, step),
        )

    explicit_at = {(e["agent"], e["step"]) for e in events}
    for agent_id, points in occ_seq.items():
        uses_snapshot = agent_id in snapshot_agents
        # Without cache_read, occupancy equals per-turn fresh input, not
        # cumulative context-window size.  Input naturally swings between
        # turns (a large tool output followed by a short reply), so an
        # occupancy drop is just normal variance, not compaction.
        if not uses_snapshot and not any(
            step_context_occupancy(step)["cache_read"] > 0
            for _idx, _occ, step in points
        ):
            continue
        drop_ratio = _SNAPSHOT_DROP_RATIO if uses_snapshot else _OCCUPANCY_DROP_RATIO
        for i in range(1, len(points)):
            prev_idx, prev_occ, prev_step = points[i - 1]
            idx, occ, step = points[i]
            dropped = prev_occ - occ
            if occ >= prev_occ * drop_ratio:
                continue
            if uses_snapshot and dropped < _SNAPSHOT_DROP_MIN_TOKENS:
                continue
            if idx in explicit_steps or (agent_id, idx) in explicit_at:
                continue
            # A stored compaction already reset this window; the next occupancy
            # turns are the new baseline, not a second prune/compaction.
            if any(
                eagent == agent_id and prev_idx < estep < idx
                for eagent, estep in explicit_at
            ):
                continue
            if _is_summary_turn(prev_step):
                continue
            # Adjacent explicit compaction (part on the previous user turn, etc.)
            if any(abs(idx - estep) <= 1 and eagent == agent_id
                   for eagent, estep in explicit_at):
                continue
            if not uses_snapshot:
                if i + 1 < len(points):
                    next_occ = points[i + 1][1]
                    # Recovered on the next turn → cache/prefix jitter, not compaction.
                    if next_occ >= prev_occ * 0.8:
                        continue
                elif occ >= prev_occ * 0.4:
                    # Last point: only count a severe drop we cannot confirm.
                    continue
            host = _snapshot_drop_host(steps, prev_idx, idx, step) if uses_snapshot else step
            host_idx = int(host.get("index", idx))
            events.append(_event(
                host, "occupancy_drop", agent_id,
                occupancy_before=prev_occ, occupancy_after=occ,
            ))
            explicit_steps.add(host_idx)

    events.sort(key=lambda e: (e["step"], e["kind"]))
    return events


# A compaction below this fraction of the window limit ran well before the
# window was full — usually agent-initiated, worth flagging as a possible
# premature loss of context.
PREMATURE_COMPACTION_RATIO = 0.5


def detect_premature_compactions(
    steps: list[dict],
    raw: dict | None = None,
    *,
    window_limit: object = None,
) -> list[dict]:
    """Compactions that ran before the window was full or grew it.

    Only explicit compaction events are considered (stored summaries,
    compaction messages/parts, compress steps) — not inferred occupancy
    drops. A compaction is flagged when it left the window at least as
    large as before (``grew``: the summary cost more than the compacted
    messages freed) or when the live window before it was below
    ``PREMATURE_COMPACTION_RATIO`` of the window limit (``low_occupancy``).
    """
    flagged: list[dict] = []
    limit = resolve_context_window_limit(steps, raw, override=window_limit)
    for event in detect_compaction_events(steps):
        if event.get("kind") not in _EXPLICIT_COMPACTION_KINDS:
            continue
        before = event.get("occupancy_before")
        after = event.get("occupancy_after")
        grew = isinstance(before, (int, float)) and isinstance(after, (int, float)) and int(after) >= int(before)
        premature = isinstance(before, (int, float)) and int(before) < limit * PREMATURE_COMPACTION_RATIO
        if not (grew or premature):
            continue
        flagged.append(
            {
                "step": int(event.get("step", 0)),
                "agent": event.get("agent", ""),
                "kind": event.get("kind", ""),
                "occupancy_before": int(before) if isinstance(before, (int, float)) else None,
                "occupancy_after": int(after) if isinstance(after, (int, float)) else None,
                "before_pct": (
                    round(100.0 * int(before) / limit, 1)
                    if isinstance(before, (int, float)) and limit
                    else None
                ),
                "window_limit": limit,
                "grew": bool(grew),
                "reason": "window_grew" if grew else "low_occupancy",
            }
        )
    return flagged


def pressure_agent_choices(steps: list[dict]) -> list[tuple[str, str]]:
    """``(label, value)`` pairs for the Context Utilization agent dropdown."""
    order: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if not _is_occupancy_step(step):
            continue
        agent_id = _pressure_agent(step)
        if agent_id not in seen:
            seen.add(agent_id)
            order.append(agent_id)
    labels = _disambiguate_pressure_labels(order, steps)
    choices = [("All agents", PRESSURE_ALL_AGENTS)]
    for agent_id in order:
        choices.append((labels[agent_id], pressure_agent_key(agent_id)))
    return choices


def context_pressure_series(
    steps: list[dict],
    *,
    agent_key: str | None = None,
    raw: dict | None = None,
    window_limit: int | float | str | None = None,
) -> dict:
    """Build per-agent occupancy series and compaction events for charting.

    Occupancy points skip compaction-only checkpoints so the line shows the
    live window (the sawtooth). ``agent_key`` of ``__all__`` / ``None``
    overlays every agent; ``__main__`` is the empty effective-agent id.
    """
    target = pressure_agent_id(agent_key)
    events = detect_compaction_events(steps)
    if target is not None:
        events = [e for e in events if e["agent"] == target]

    snapshot_agents = _snapshot_agents(steps)

    agents_order: list[str] = []
    seen: set[str] = set()
    points_by_agent: dict[str, list[dict]] = {}
    local_turn: dict[str, int] = {}
    for step in steps:
        if not _is_occupancy_step(step):
            continue
        agent_id = _pressure_agent(step)
        if target is not None and agent_id != target:
            continue
        if agent_id in snapshot_agents and _context_window_tokens(step) is None:
            continue
        if agent_id not in seen:
            seen.add(agent_id)
            agents_order.append(agent_id)
            points_by_agent[agent_id] = []
            local_turn[agent_id] = 0
        local_turn[agent_id] += 1
        occ = step_context_occupancy(step)
        points_by_agent[agent_id].append({
            "step": int(step.get("index", 0)),
            "local_turn": local_turn[agent_id],
            "fresh": occ["fresh"],
            "cache_read": occ["cache_read"],
            "occupancy": occ["occupancy"],
        })

    window_limit = resolve_context_window_limit(steps, raw, override=window_limit)
    labels = _disambiguate_pressure_labels(agents_order, steps)
    agents = []
    for agent_id in agents_order:
        agent_events = [e for e in events if e.get("agent") == agent_id]
        agents.append({
            "agent_id": agent_id,
            "key": pressure_agent_key(agent_id),
            "label": labels[agent_id],
            "points": _splice_compaction_into_points(
                points_by_agent[agent_id], agent_events,
            ),
        })
    return {
        "agents": agents,
        "events": events,
        "window_limit": window_limit,
        "agent_key": agent_key or PRESSURE_ALL_AGENTS,
    }


def context_pressure_stats(series: dict) -> dict:
    """Peak occupancy, optional peak %, compaction count, and largest drop."""
    peak = 0
    for agent in series.get("agents") or []:
        for point in agent.get("points") or []:
            peak = max(peak, int(point.get("occupancy") or 0))
    events = series.get("events") or []
    largest_drop = 0
    for event in events:
        dropped = event.get("dropped")
        if isinstance(dropped, (int, float)):
            largest_drop = max(largest_drop, int(dropped))
    window_limit = series.get("window_limit")
    peak_pct = None
    if isinstance(window_limit, (int, float)) and window_limit > 0 and peak > 0:
        peak_pct = round(100.0 * peak / window_limit, 1)
    return {
        "peak_occupancy": peak,
        "peak_pct": peak_pct,
        "compaction_count": len(events),
        "largest_drop": largest_drop,
        "window_limit": window_limit if isinstance(window_limit, (int, float)) else None,
    }


def estimate_tokens(text: str) -> int:
    """Approximate token count without a tokenizer (≈4 characters / token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def format_token_count(n: int) -> str:
    """Compact token count for the usage header (76.8k, 1.2M)."""
    n = int(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000:
        text = f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    elif n >= 1_000:
        text = f"{n / 1_000:.1f}k".replace(".0k", "k")
    else:
        text = str(n)
    return sign + text


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        if not value:
            return ""
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _is_spawn_call(tool_call: dict) -> bool:
    name = str(tool_call.get("tool_name") or "")
    if name.lower() in _SPAWN_NAMES_LOWER:
        return True
    inp = tool_call.get("input")
    if isinstance(inp, dict) and inp.get("subagent_type"):
        return True
    return bool(spawned_child_session_id(tool_call.get("metadata") or {}))


def _raw_blob_tokens(raw: dict | None, keys: tuple[str, ...]) -> int:
    """Tokenize named blobs (tool schemas, rules, MCP catalogs) when present."""
    if not isinstance(raw, dict):
        return 0
    blocks: list[Any] = [raw]
    for key in ("metadata", "info", "agent"):
        block = raw.get(key)
        if isinstance(block, dict):
            blocks.append(block)
    candidates: list[Any] = []
    seen_ids: set[int] = set()
    for block in blocks:
        for key in keys:
            if key not in block:
                continue
            item = block.get(key)
            ident = id(item)
            if ident in seen_ids:
                continue
            seen_ids.add(ident)
            candidates.append(item)
    total = 0
    for item in candidates:
        if not item:
            continue
        total += estimate_tokens(_stringify(item))
    return total


def _last_stored_compaction_step(
    events: list[dict], agent_id: str, anchor_step: int,
) -> int | None:
    """Latest stored compaction strictly before *anchor_step*, if any."""
    last: int | None = None
    for event in events:
        if event.get("agent") != agent_id:
            continue
        if event.get("kind") not in _EXPLICIT_COMPACTION_KINDS:
            continue
        step = int(event.get("step") or 0)
        if step < anchor_step:
            last = step if last is None else max(last, step)
    return last


def _compaction_part_text(part: dict) -> str:
    chunks: list[str] = []
    seen: set[str] = set()
    for key in ("summary", "text", "recent"):
        value = part.get(key)
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            chunks.append(value)
    return "\n".join(chunks)


def _is_summary_turn(step: dict) -> bool:
    """True when this step is a stored compaction summary, not a live turn.

    OpenCode writes the summary as an assistant message with ``summary: True``
    and ``agent`` / ``mode`` of ``compaction``. User prompts may carry a
    ``summary: {diffs: []}`` dict — that is not a summary turn.
    """
    if step.get("summary") is True:
        return True
    if str(step.get("role") or "") == "compaction":
        return True
    if bool(step.get("is_compaction_checkpoint")):
        return True
    if str(step.get("mode") or "").lower() == "compaction":
        return True
    return str(step.get("agent") or "").lower() == "compaction"


def parse_usage_snapshot(value: object) -> int | None:
    """Occupancy step for a snapshot dropdown value, or None for the current window."""
    if value in (None, "", SNAPSHOT_CURRENT) or isinstance(value, bool):
        return None
    try:
        step = int(str(value).strip())
    except ValueError:
        return None
    return step if step >= 0 else None


def _occupancy_point_before(
    steps: list[dict], agent_id: str, before_step: int,
) -> tuple[int, int]:
    """Last occupancy ``(step, tokens)`` for *agent_id* strictly before *before_step*."""
    last_step, last_occ = -1, 0
    for step in steps:
        if not isinstance(step, dict) or not _is_occupancy_step(step):
            continue
        if _pressure_agent(step) != agent_id:
            continue
        idx = int(step.get("index", 0))
        if idx >= before_step:
            continue
        last_step = idx
        last_occ = step_context_occupancy(step)["occupancy"]
    return last_step, last_occ


def _occupancy_at(
    steps: list[dict], step_index: int, target: str | None,
) -> tuple[str, int, int] | None:
    """``(agent_id, step, occupancy)`` at an occupancy step, else None."""
    for step in steps:
        if not isinstance(step, dict) or not _is_occupancy_step(step):
            continue
        if int(step.get("index", 0)) != step_index:
            continue
        agent_id = _pressure_agent(step)
        if target is not None and agent_id != target:
            continue
        return agent_id, step_index, step_context_occupancy(step)["occupancy"]
    return None


def usage_snapshot_choices(
    steps: list[dict],
    *,
    agent_key: str | None = None,
) -> list[tuple[str, str]]:
    """``(label, value)`` pairs for one agent's pre-compaction occupancy turns.

    All-agents view has no snapshot list — each compaction belongs to one window.
    """
    choices: list[tuple[str, str]] = [("Current window", SNAPSHOT_CURRENT)]
    target = pressure_agent_id(agent_key)
    if target is None or not steps:
        return choices
    events = [
        e for e in detect_compaction_events(steps) if e.get("agent") == target
    ]
    seen: set[int] = set()
    for event in coalesce_compaction_events(events):
        pre_step, pre_occ = _occupancy_point_before(
            steps, target, int(event.get("step") or 0),
        )
        if pre_step < 0 or pre_step in seen:
            continue
        seen.add(pre_step)
        occ_txt = format_token_count(pre_occ)
        choices.append((f"Before compaction · step {pre_step} ({occ_txt})", str(pre_step)))
    return choices


def _window_occupancy(
    steps: list[dict], target: str | None,
) -> tuple[str, int, int, int, int]:
    """Peak and latest occupancy for the busiest matching agent.

    Returns ``(agent_id, peak_step, peak_occ, latest_step, latest_occ)``.
    """
    peak_agent, peak_step, peak_occ = "", -1, 0
    latest_by_agent: dict[str, tuple[int, int]] = {}
    for step in steps:
        if not isinstance(step, dict) or not _is_occupancy_step(step):
            continue
        agent_id = _pressure_agent(step)
        if target is not None and agent_id != target:
            continue
        occ = step_context_occupancy(step)["occupancy"]
        idx = int(step.get("index", 0))
        latest_by_agent[agent_id] = (idx, occ)
        if occ > peak_occ:
            peak_agent, peak_step, peak_occ = agent_id, idx, occ
    latest_step, latest_occ = latest_by_agent.get(peak_agent, (-1, 0))
    return peak_agent, peak_step, peak_occ, latest_step, latest_occ


def _accumulate_step(step: dict, buckets: dict[str, int], *, summarized_only: bool = False) -> None:
    role = str(step.get("role") or "")
    if role in _SYSTEM_ROLES:
        if summarized_only:
            return
        for part in step.get("parts") or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text") or part.get("summary") or ""
            if isinstance(text, str) and text:
                buckets["system"] += estimate_tokens(text)
        return

    summary_turn = _is_summary_turn(step)
    for part in step.get("parts") or []:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "compaction":
            text = _compaction_part_text(part)
            if text:
                buckets["summarized"] += estimate_tokens(text)
            continue
        if summarized_only and not summary_turn:
            continue
        if ptype == "text":
            text = part.get("text") or ""
            if not isinstance(text, str) or not text:
                continue
            if part.get("synthetic"):
                if not summarized_only:
                    buckets["system"] += estimate_tokens(text)
            elif summary_turn:
                buckets["summarized"] += estimate_tokens(text)
            else:
                buckets["conversation"] += estimate_tokens(text)
        elif ptype == "reasoning":
            text = part.get("text") or ""
            if isinstance(text, str) and text:
                if summary_turn:
                    buckets["summarized"] += estimate_tokens(text)
                elif not summarized_only:
                    buckets["conversation"] += estimate_tokens(text)

    if summarized_only:
        return

    for tool_call in step.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        inp = _stringify(tool_call.get("input"))
        out = _stringify(tool_call.get("output"))
        name = str(tool_call.get("tool_name") or "")
        if parse_skill_name(name, tool_call.get("input")):
            buckets["skills"] += estimate_tokens(inp) + estimate_tokens(out)
        elif _is_spawn_call(tool_call):
            buckets["subagents"] += estimate_tokens(inp)
            buckets["tool_outputs"] += estimate_tokens(out)
        elif is_mcp_tool(name):
            buckets["mcp"] += estimate_tokens(inp) + estimate_tokens(out)
        else:
            buckets["conversation"] += estimate_tokens(inp)
            buckets["tool_outputs"] += estimate_tokens(out)


def context_usage_breakdown(
    steps: list[dict],
    *,
    raw: dict | None = None,
    agent_key: str | None = None,
    window_limit: int | float | str | None = None,
    snapshot_step: int | None = None,
) -> dict:
    """Bucket logged context in the agent's *current* window, or at *snapshot_step*.

    Peak occupancy can precede compaction, so composition uses the latest
    occupancy point unless *snapshot_step* selects a live occupancy turn
    (typically the point right before a compaction). Percentages of *window*
    use the resolved window limit (user override, inferred model/metadata, or
    128k).
    """
    empty_buckets = {key: 0 for key, _label, _color in USAGE_CATEGORIES}
    empty = {
        "buckets": empty_buckets,
        "occupancy": 0,
        "peak_occupancy": 0,
        "window_limit": None,
        "loaded_pct": None,
        "agent_id": "",
        "peak_step": None,
        "step": None,
        "scaled": False,
        "agent_key": agent_key or PRESSURE_ALL_AGENTS,
    }
    if not steps:
        return empty

    target = pressure_agent_id(agent_key)
    agent_id, peak_step, peak_occupancy, anchor_step, occupancy = _window_occupancy(
        steps, target,
    )
    if snapshot_step is not None:
        hit = _occupancy_at(steps, snapshot_step, target)
        if hit is not None:
            agent_id, anchor_step, occupancy = hit
    window_limit = resolve_context_window_limit(steps, raw, override=window_limit)
    buckets = dict(empty_buckets)
    raw_dict = raw if isinstance(raw, dict) else None
    buckets["tools"] = _raw_blob_tokens(raw_dict, _TOOL_DEF_KEYS)
    buckets["rules"] = _raw_blob_tokens(raw_dict, _RULE_KEYS)
    buckets["mcp"] = _raw_blob_tokens(raw_dict, _MCP_KEYS)

    if anchor_step >= 0:
        last_comp = _last_stored_compaction_step(
            detect_compaction_events(steps), agent_id, anchor_step,
        )
        start = 0 if last_comp is None else last_comp + 1
        lo = max(0, last_comp - 1) if last_comp is not None else start
        for step in steps:
            if not isinstance(step, dict):
                continue
            idx = int(step.get("index", 0))
            if _pressure_agent(step) != agent_id:
                continue
            if last_comp is not None and lo <= idx < start:
                _accumulate_step(step, buckets, summarized_only=True)
            elif start <= idx <= anchor_step:
                _accumulate_step(step, buckets)

    accounted = sum(buckets[key] for key in _ACCOUNTABLE_KEYS)
    scaled = False
    if occupancy > accounted:
        buckets["unattributed"] = occupancy - accounted
    elif occupancy > 0 and accounted > occupancy:
        scale = occupancy / accounted
        scaled_vals = {key: int(buckets[key] * scale) for key in _ACCOUNTABLE_KEYS}
        if sum(scaled_vals.values()) == 0:
            largest = max(_ACCOUNTABLE_KEYS, key=lambda key: buckets[key])
            scaled_vals[largest] = occupancy
        else:
            drift = occupancy - sum(scaled_vals.values())
            adjust_key = max(_ACCOUNTABLE_KEYS, key=lambda key: (scaled_vals[key], buckets[key]))
            scaled_vals[adjust_key] = max(0, scaled_vals[adjust_key] + drift)
        buckets.update(scaled_vals)
        buckets["unattributed"] = 0
        scaled = True

    loaded_pct = None
    if isinstance(window_limit, (int, float)) and window_limit > 0 and occupancy > 0:
        loaded_pct = round(100.0 * occupancy / window_limit, 1)

    return {
        "buckets": buckets,
        "occupancy": occupancy,
        "peak_occupancy": peak_occupancy,
        "window_limit": window_limit,
        "loaded_pct": loaded_pct,
        "agent_id": agent_id,
        "peak_step": peak_step if peak_step >= 0 else None,
        "step": anchor_step if anchor_step >= 0 else None,
        "scaled": scaled,
        "agent_key": agent_key or PRESSURE_ALL_AGENTS,
    }


def residual_display_label(breakdown: dict) -> str:
    """Legend name for billed occupancy that is not in the trajectory text.

    When leftover dwarfs logged system/tool text, treat it as hidden harness
    definitions; otherwise it is tokenizer drift or other unlogged prefix.
    """
    buckets = breakdown.get("buckets") or {}
    leftover = int(buckets.get("unattributed") or 0)
    logged_prefix = int(buckets.get("system") or 0) + int(buckets.get("tools") or 0)
    if leftover > logged_prefix:
        return "Harness system definitions (not included in log)"
    return "Other (not included in log)"


def usage_segments(breakdown: dict) -> list[dict]:
    """Legend rows for buckets with a non-zero token estimate."""
    buckets: dict[str, int] = breakdown.get("buckets") or {}
    occupancy = int(breakdown.get("occupancy") or 0)
    window = breakdown.get("window_limit")
    denom_window = int(window) if isinstance(window, (int, float)) and window > 0 else occupancy
    rows: list[dict] = []
    for key, label, color in USAGE_CATEGORIES:
        tokens = int(buckets.get(key) or 0)
        if tokens <= 0:
            continue
        window_pct = round(100.0 * tokens / denom_window, 1) if denom_window else 0.0
        loaded_pct = round(100.0 * tokens / occupancy, 1) if occupancy else 0.0
        if key == "unattributed":
            label = residual_display_label(breakdown)
        rows.append({
            "key": key,
            "label": label,
            "color": color,
            "tokens": tokens,
            "window_pct": window_pct,
            "loaded_pct": loaded_pct,
        })
    return rows


def format_context_usage_html(breakdown: dict) -> str:
    """Stacked bar + legend for a :func:`context_usage_breakdown` result."""
    occupancy = int(breakdown.get("occupancy") or 0)
    window = breakdown.get("window_limit")
    has_window = isinstance(window, (int, float)) and window > 0
    loaded_pct = breakdown.get("loaded_pct")
    rows = usage_segments(breakdown)
    if occupancy <= 0 and not any(int(r.get("tokens") or 0) for r in rows):
        return ""

    if has_window and isinstance(loaded_pct, (int, float)):
        head_left = f"{html.escape(f'{loaded_pct:g}')}% full"
        head_right = (
            f"{html.escape(format_token_count(occupancy))} / "
            f"{html.escape(format_token_count(int(window)))}"
        )
    elif has_window:
        head_left = "Context loaded"
        head_right = (
            f"{html.escape(format_token_count(occupancy))} / "
            f"{html.escape(format_token_count(int(window)))}"
        )
    else:
        head_left = "Context loaded"
        head_right = (
            f"{html.escape(format_token_count(occupancy))} tokens"
            " · window limit unknown"
        )

    bar_total = int(window) if has_window else occupancy
    if occupancy > bar_total:
        bar_total = occupancy
    bar_parts: list[str] = []
    filled = 0
    if bar_total > 0:
        for row in rows:
            tokens = int(row.get("tokens") or 0)
            if tokens <= 0:
                continue
            filled += tokens
            width = 100.0 * tokens / bar_total
            label = html.escape(str(row["label"]))
            color = html.escape(str(row["color"]))
            bar_parts.append(
                f'<div class="ctx-usage-seg" style="width:{width:.3f}%;'
                f'background:{color}" title="{label}: {tokens:,} tokens"></div>'
            )
        free = max(0, bar_total - min(filled, bar_total))
        if free:
            width = 100.0 * free / bar_total
            bar_parts.append(
                f'<div class="ctx-usage-seg ctx-usage-free" style="width:{width:.3f}%"'
                ' title="Free"></div>'
            )
    bar_html = f'<div class="ctx-usage-bar">{"".join(bar_parts)}</div>'

    show_window_col = has_window
    legend_rows: list[str] = []
    for row in rows:
        tokens = int(row.get("tokens") or 0)
        name = (
            f'<span class="ctx-usage-swatch" style="background:{html.escape(str(row["color"]))}"></span>'
            f'{html.escape(str(row["label"]))}'
        )
        token_cell = html.escape(format_token_count(tokens))
        loaded_cell = html.escape(f'{row["loaded_pct"]:g}%')
        if show_window_col:
            window_cell = html.escape(f'{row["window_pct"]:g}%')
            legend_rows.append(
                f'<div class="ctx-usage-name">{name}</div>'
                f'<div class="ctx-usage-tokens">{token_cell}</div>'
                f'<div class="ctx-usage-pct-cell">{window_cell}</div>'
                f'<div class="ctx-usage-pct-cell">{loaded_cell}</div>'
            )
        else:
            legend_rows.append(
                f'<div class="ctx-usage-name">{name}</div>'
                f'<div class="ctx-usage-tokens">{token_cell}</div>'
                f'<div class="ctx-usage-pct-cell">{loaded_cell}</div>'
            )

    if show_window_col:
        legend_head = (
            '<div class="ctx-usage-name ctx-usage-head-cell">Category</div>'
            '<div class="ctx-usage-tokens ctx-usage-head-cell">Tokens</div>'
            '<div class="ctx-usage-pct-cell ctx-usage-head-cell">% window</div>'
            '<div class="ctx-usage-pct-cell ctx-usage-head-cell">% loaded</div>'
        )
        legend_class = "ctx-usage-legend ctx-usage-legend-window"
    else:
        legend_head = (
            '<div class="ctx-usage-name ctx-usage-head-cell">Category</div>'
            '<div class="ctx-usage-tokens ctx-usage-head-cell">Tokens</div>'
            '<div class="ctx-usage-pct-cell ctx-usage-head-cell">% loaded</div>'
        )
        legend_class = "ctx-usage-legend ctx-usage-legend-loaded"

    return (
        '<div class="ctx-usage">'
        '<div class="ctx-usage-head">'
        f'<span class="ctx-usage-pct">{head_left}</span>'
        f'<span class="ctx-usage-counts">{head_right}</span>'
        "</div>"
        f"{bar_html}"
        f'<div class="{legend_class}">{legend_head}{"".join(legend_rows)}</div>'
        "</div>"
    )
