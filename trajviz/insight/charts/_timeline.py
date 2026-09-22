"""Timeline agent identity shared by charts and workflow cards."""

from collections import Counter, defaultdict
from collections.abc import Callable

from ..metrics import effective_agent, tagged_subagent_display_label


def _session_primary_agents(steps: list[dict]) -> dict[str, str]:
    """Most common non-compaction agent name per session_id."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for s in steps:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("session_id") or "")
        name = str(s.get("agent") or "").strip()
        if not sid or not name:
            continue
        if name == "compaction" or s.get("role") == "compaction" or s.get("is_compaction_checkpoint"):
            continue
        counts[sid][name] += 1
    return {sid: c.most_common(1)[0][0] for sid, c in counts.items() if c}


def _timeline_context(steps: list[dict]) -> tuple[bool, bool, dict[str, str]]:
    """Detect how to split agents on swimlane / run-group timelines.

    Returns ``(multi_session, use_agent_names, session_primary_agents)``:
    - multi_session: several ``session_id`` values (OpenCode/CodeArts subagents
      often lack ``isSubAgent`` tags — group by session instead)
    - use_agent_names: ``effective_agent`` is empty for everyone but ``agent``
      field varies (e.g. OpenCode plan/build, or Claude agent ids with a
      defaulted ``is_sub_agent=False``)
    - session_primary_agents: dominant mode per session (folds compaction)
    """
    primary = _session_primary_agents(steps)
    sessions = {str(s.get("session_id") or "") for s in steps if isinstance(s, dict) and s.get("session_id")}
    if len(sessions) > 1:
        return True, False, primary

    eas: set[str] = set()
    names: set[str] = set()
    for s in steps:
        if not isinstance(s, dict):
            continue
        if s.get("role") not in ("assistant", "user", "compaction"):
            continue
        eas.add(effective_agent(s))
        name = str(s.get("agent") or "").strip()
        if name:
            names.add(name)
    use_names = (not (eas - {""})) and len(names) > 1
    return False, use_names, primary


def _timeline_agent_id(
    step: dict,
    *,
    multi_session: bool = False,
    use_agent_names: bool = False,
    primary_agents: dict[str, str] | None = None,
) -> str:
    """Fine-grained identity for segment boundaries (session / subagent / mode).

    Multi-session ids are ``session_id`` or ``session_id::agent`` when the
    agent field is set — so plan/build modes on the same OpenCode root
    session stay distinct, and parallel explore sessions do not merge.
    Compaction steps inherit the session's primary agent name.
    """
    primary_agents = primary_agents or {}
    if multi_session:
        sid = step.get("session_id") or ""
        if isinstance(sid, str) and sid:
            name = str(step.get("agent") or "").strip()
            if name == "compaction" or step.get("role") == "compaction" or step.get("is_compaction_checkpoint"):
                name = primary_agents.get(sid, "")
            return f"{sid}::{name}" if name else sid
    ea = effective_agent(step)
    if ea:
        return ea
    if use_agent_names:
        return str(step.get("agent") or "").strip()
    return ""


def _timeline_id_session(agent_id: str) -> str:
    """Session portion of a multi-session timeline id (before ``::``)."""
    if "::" in agent_id:
        return agent_id.split("::", 1)[0]
    return agent_id


def _trunc_timeline_label(name: str) -> str:
    return name if len(name) <= 20 else name[:19] + "…"


def _timeline_display_label(agent_id: str, steps: list[dict]) -> str:
    """Legend / color bucket for a timeline agent id."""
    if not agent_id:
        return "main"
    tagged = tagged_subagent_display_label(agent_id, steps)
    if tagged:
        return tagged
    sid = _timeline_id_session(agent_id)
    mode = ""
    if "::" in agent_id:
        mode = agent_id.split("::", 1)[1].strip()

    # Composite multi-session id: prefer the embedded agent mode name
    if mode:
        return _trunc_timeline_label(mode)
    for s in steps:
        if not isinstance(s, dict):
            continue
        if (s.get("session_id") or "") == sid and sid == agent_id:
            name = str(s.get("agent") or "").strip()
            if name:
                return _trunc_timeline_label(name)
            title = str(s.get("session_title") or "").strip()
            if title:
                return _trunc_timeline_label(title)
            break
        if effective_agent(s) == agent_id or str(s.get("agent") or "").strip() == agent_id:
            name = str(s.get("agent") or "").strip()
            if name and name == agent_id:
                return _trunc_timeline_label(name)
    short = agent_id[:12] if len(agent_id) > 12 else agent_id
    return f"sub {short}"


def _disambiguate_timeline_labels(ids: list[str], steps: list[dict]) -> dict[str, str]:
    """Unique legend labels; suffix short session id when names collide."""
    raw = {aid: _timeline_display_label(aid, steps) for aid in ids}
    counts = Counter(raw.values())
    out: dict[str, str] = {}
    for aid, label in raw.items():
        if counts[label] > 1 and aid:
            sid = _timeline_id_session(aid)
            suffix = sid[-6:] if len(sid) > 6 else sid
            out[aid] = f"{label} ({suffix})"
        else:
            out[aid] = label
    return out


def _legend_label(agent_id: str, labels: dict[str, str]) -> str:
    """Display name for a timeline agent id; empty id is main."""
    if agent_id in labels:
        return labels[agent_id]
    return agent_id or "main"


def timeline_agent_id_of(steps: list[dict]) -> Callable[[dict], str]:
    """Per-step timeline identity without building color maps or legend labels."""
    multi, use_names, primary = _timeline_context(steps)

    def agent_id(step: dict) -> str:
        return _timeline_agent_id(
            step,
            multi_session=multi,
            use_agent_names=use_names,
            primary_agents=primary,
        )

    return agent_id


def bind_timeline_agents(
    steps: list[dict],
) -> tuple[dict[str, int], dict[str, str], Callable[[dict], str]]:
    """Color map, legend labels, and per-step timeline identity.

    Charts and workflow cards must use this together so OpenCode
    ``session_id::mode`` keys stay consistent with the palette.
    """
    agent_id = timeline_agent_id_of(steps)
    order: list[str] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        agent = agent_id(s)
        if agent not in order:
            order.append(agent)
    if "" in order and order[0] != "":
        order.remove("")
        order.insert(0, "")
    if not order:
        order = [""]
    color_map = {aid: i for i, aid in enumerate(order)}
    labels = _disambiguate_timeline_labels(list(color_map.keys()), steps)
    return color_map, labels, agent_id
