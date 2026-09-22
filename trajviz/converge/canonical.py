"""Canonical action model: canonicalization, effect labeling, semantic equivalence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from trajviz.tool_vocab import BASH_TOOL_NAMES as _BASH_TOOLS, WRITE_TOOL_NAMES as _WRITE_TOOLS
# Shared single-source helpers (C8/C9): validation-command detection lives in
# insight/patterns.py and bash path extraction in insight/diagnostics.py (the
# quoted + Windows-drive-aware implementation). Both modules import only
# stdlib/tool_vocab at module level, so there is no import cycle.
from trajviz.insight.diagnostics import _extract_bash_paths
from trajviz.insight.patterns import _is_validation_command


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ActionCost:
    tokens: int = 0            # original step-level total tokens (for reference)
    latency_ms: int = 0
    token_share: int = 0       # this action's share of step tokens (step_tokens / N)


@dataclass
class CanonicalAction:
    """A single canonical action from an agent trajectory.

    Effect labels are **observable** (what happened within this run), not **evaluative**
    (whether the action was correct). The distinction:
    - `survived`: FILE_WRITE was not reverted within this run (does NOT mean "correct")
    - `failed`: tool call errored or exit code != 0
    - `reverted`: subsequent write to same file overwrites before final state
    - `justified`: FILE_READ of a relevant file or validation COMMAND
    - `unknown`: default — not wasteful; agents legitimately read files outside the patch

    Evaluative judgments (was this action correct?) require anchor/oracle grounding.
    """
    step_index: int = 0
    action_type: str = ""       # FILE_READ | FILE_WRITE | SEARCH | COMMAND | AGENT_SPAWN | REASON
    target: str = ""            # normalized path, command, or pattern
    tool: str = ""              # original tool name
    tool_call_id: str = ""      # stable ID from parsed tool call (for 1:1 failure attribution)
    args: dict = field(default_factory=dict)
    start_time: int | float | None = None  # epoch ms (from tool call or step timing)
    end_time: int | float | None = None    # epoch ms
    status: str = ""            # raw tool status
    cost: ActionCost = field(default_factory=ActionCost)
    effect_label: str = "unknown"  # survived | failed | reverted | justified | unknown
    effect_detail: dict = field(default_factory=dict)
    phase_label: str | None = None   # LLM-derived: understand | plan | implement | debug | validate | report
    action_label: str | None = None  # LLM-derived fine-grained action (e.g., code_reading, implement_runtime_logic)


# ---------------------------------------------------------------------------
# Action type mapping
# ---------------------------------------------------------------------------

_READ_TOOLS = {"Read", "read"}
_SEARCH_TOOLS = {"Glob", "glob", "Grep", "grep", "find", "ToolSearch"}
# Native spawn names plus DSH leftovers if the loader did not map them to Agent.
# Do not include Claude Code ``Task`` — that would recategorize non-DSH runs.
_SPAWN_TOOLS = {"Agent", "agent", "subagent", "subagent_fork"}
_PLANNING_TOOLS = {"todowrite", "TodoWrite", "TaskCreate", "TaskUpdate", "TaskList"}
# Navigation/utility commands excluded from alignment (like REASON)
_NAVIGATION_COMMANDS = {"cd", "pwd", "ls", "echo", "export", "set", "source"}

# sed/awk read by default (B22): `sed -n '1,50p' file` is a view, not a write.
# Only explicit in-place flags (see _is_in_place_edit) make them writes,
# mirroring the Codex mapping in insight/formats/codex.py.
_FUZZY_KEYWORDS: dict[str, str] = {
    "grep": "SEARCH", "rg": "SEARCH", "ag": "SEARCH", "find": "SEARCH",
    "cat": "FILE_READ", "head": "FILE_READ", "tail": "FILE_READ", "less": "FILE_READ",
    "sed": "FILE_READ", "awk": "FILE_READ", "tee": "FILE_WRITE",
}

_SED_IN_PLACE_RE = re.compile(r"(?:^|\s)(?:-i\S*|--in-place\b)")
_AWK_IN_PLACE_RE = re.compile(r"(?:^|\s)(?:-i\s+inplace\b|--in-place\b)")


def _is_in_place_edit(base_cmd: str, command: str) -> bool:
    """True when a sed/awk invocation edits files in place (B22).

    sed: ``-i`` / ``-i.bak`` / ``--in-place``; awk: only gawk's
    ``-i inplace`` / ``--in-place`` forms (plain ``awk -i`` loads a library).
    """
    if base_cmd == "sed":
        return bool(_SED_IN_PLACE_RE.search(command))
    if base_cmd in ("awk", "gawk"):
        return bool(_AWK_IN_PLACE_RE.search(command))
    return False


_WSL_UNC_RE = re.compile(
    r"(?i)^(?://*)(?:wsl\.localhost|wsl\$)/[^/]+(/home/.*)$"
)


def _normalize_target(path: str) -> str:
    """Normalize a file path for comparison.

    Handles Windows backslashes, drive letters, and WSL UNC shares
    (``\\\\wsl.localhost\\Ubuntu-26.04\\home\\...`` → ``/home/...``).
    """
    if not path:
        return path
    import posixpath
    p = path.strip().replace("\\", "/")
    match = _WSL_UNC_RE.match(p)
    if match:
        p = match.group(1)
    if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
        p = p[2:]
    return posixpath.normpath(p)


def _extract_base_command(command: str) -> str:
    """Extract the base command (first non-assignment token) from a shell command."""
    tokens = command.strip().split()
    for tok in tokens:
        # Skip any number of leading VAR=value environment assignments.
        if "=" in tok and not tok.startswith("="):
            continue
        return tok
    return tokens[0] if tokens else command.strip()


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def canonicalize_steps(
    steps: list[dict],
    step_labels: dict[int, dict[str, str]] | None = None,
) -> list[CanonicalAction]:
    """Map parsed steps into a list of CanonicalAction.

    Args:
        step_labels: Optional mapping from step index to
            ``{"phase": ..., "action": ...}`` from the step labeler.
            When provided, each CanonicalAction gets ``phase_label`` and
            ``action_label`` set accordingly.

    Token cost is distributed equally across non-REASON actions within each step
    to prevent step-token duplication. REASON actions get token_share=0.
    """
    actions: list[CanonicalAction] = []

    for step in steps:
        step_idx = step["index"]
        step_tokens = step["tokens"]["total"]
        step_start = len(actions)  # track where this step's actions begin
        label = (step_labels or {}).get(step_idx, {})

        # Reasoning/text parts → REASON actions
        for part in step.get("parts", []):
            ptype = part.get("type", "")
            if ptype in ("text", "reasoning"):
                t_created = step.get("time_created_ms")
                t_completed = step.get("time_completed_ms")
                latency = 0
                if isinstance(t_created, (int, float)) and isinstance(t_completed, (int, float)):
                    latency = max(0, int(t_completed - t_created))
                actions.append(CanonicalAction(
                    step_index=step_idx,
                    action_type="REASON",
                    target="",
                    tool=ptype,
                    args={},
                    status="",
                    cost=ActionCost(tokens=step_tokens, latency_ms=latency, token_share=0),
                    effect_label="unknown",
                    effect_detail={},
                    phase_label=label.get("phase"),
                    action_label=label.get("action"),
                ))
                break  # One REASON per step max

        # Tool calls → typed actions
        for tc in step.get("tool_calls", []):
            tool_name = tc.get("tool_name", "")
            tc_id = tc.get("tool_id", "") or tc.get("id", "") or ""
            inp = tc.get("input", {})
            if not isinstance(inp, dict):
                inp = {}
            status = tc.get("status", "")

            # Determine action_type and target
            action_type = "COMMAND"
            target = ""

            if tool_name in _READ_TOOLS:
                action_type = "FILE_READ"
                target = _normalize_target(
                    inp.get("file_path", "") or inp.get("filePath", ""))
            elif tool_name in _WRITE_TOOLS:
                action_type = "FILE_WRITE"
                from trajviz.tool_vocab import write_target_path
                target = _normalize_target(write_target_path(inp))
            elif tool_name in _SEARCH_TOOLS:
                action_type = "SEARCH"
                pattern = inp.get("pattern", "")
                scope = _normalize_target(inp.get("path", ""))
                target = f"{pattern}@{scope}" if scope else pattern
            elif tool_name in _BASH_TOOLS:
                command = inp.get("command", "")
                base_cmd = _extract_base_command(command)
                if base_cmd in _NAVIGATION_COMMANDS:
                    action_type = "REASON"
                    target = ""
                elif "|" not in command and base_cmd in _FUZZY_KEYWORDS:
                    action_type = _FUZZY_KEYWORDS[base_cmd]
                    if action_type == "FILE_READ" and _is_in_place_edit(base_cmd, command):
                        action_type = "FILE_WRITE"  # sed -i / gawk -i inplace (B22)
                    paths = _extract_bash_paths(command)
                    if action_type in ("FILE_READ", "FILE_WRITE") and paths:
                        target = _normalize_target(paths[0])
                    elif action_type == "SEARCH":
                        parts = command.split()
                        non_flag_args = [p for p in parts[1:] if not p.startswith("-")]
                        pattern = non_flag_args[0] if non_flag_args else ""
                        scope = _normalize_target(paths[0]) if paths else ""
                        target = f"{pattern}@{scope}" if scope else pattern
                    else:
                        target = base_cmd
                else:
                    action_type = "COMMAND"
                    target = base_cmd
            elif tool_name in _SPAWN_TOOLS:
                action_type = "AGENT_SPAWN"
                target = inp.get("description", inp.get("prompt", ""))[:80]
            elif tool_name in _PLANNING_TOOLS:
                action_type = "REASON"
                target = ""
            else:
                action_type = "COMMAND"
                target = tool_name

            # Compute latency from tool timing
            ts = tc.get("time_start")
            te = tc.get("time_end")
            latency_ms = 0
            if isinstance(ts, (int, float)) and isinstance(te, (int, float)) and te >= ts:
                latency_ms = int(te - ts)
            else:
                dm = tc.get("duration_ms")
                if dm is None:
                    dm = (tc.get("metadata") or {}).get("totalDurationMs")
                if isinstance(dm, (int, float)) and dm > 0:
                    latency_ms = int(dm)

            actions.append(CanonicalAction(
                step_index=step_idx,
                action_type=action_type,
                target=target,
                tool=tool_name,
                tool_call_id=tc_id,
                args=inp,
                start_time=ts if isinstance(ts, (int, float)) else None,
                end_time=te if isinstance(te, (int, float)) else None,
                status=status,
                cost=ActionCost(tokens=step_tokens, latency_ms=latency_ms, token_share=0),
                effect_label="unknown",
                effect_detail={},
                phase_label=label.get("phase"),
                action_label=label.get("action"),
            ))

        # Distribute token_share equally across non-REASON actions in this step.
        # If a step has only REASON actions (no tool calls), the REASON action
        # carries the full step tokens as token_share for conservation.
        step_actions = actions[step_start:]
        non_reason = [a for a in step_actions if a.action_type != "REASON"]
        if non_reason:
            share = step_tokens // len(non_reason)
            remainder = step_tokens - share * len(non_reason)
            for i, a in enumerate(non_reason):
                a.cost.token_share = share + (remainder if i == 0 else 0)
        elif step_actions:
            # REASON-only step: give tokens to the REASON action for conservation
            step_actions[0].cost.token_share = step_tokens

    return actions


# ---------------------------------------------------------------------------
# Effect labeling
# ---------------------------------------------------------------------------

def _find_own_tool_call(
    a: CanonicalAction,
    step: dict,
    actions: list[CanonicalAction],
    action_index: int,
) -> dict | None:
    """Locate the tool call that produced action ``a`` within its step.

    Prefers exact ``tool_call_id`` matching. When ids are unavailable, falls
    back to POSITIONAL pairing (B23): canonicalize_steps emits exactly one
    action per tool call, in order, so the k-th action of this step carrying
    tool name T corresponds to the k-th tool call named T. This prevents a
    failing sibling call from being cross-attributed to a different same-named
    action (the old fallback name-scanned until it found any failure).

    Note: navigation-command Bash calls canonicalize to REASON actions but
    still occupy a tool_call slot, so the ordinal counts every prior action
    with the same tool name regardless of action_type (part-derived REASONs
    have tool 'text'/'reasoning' and never collide with real tool names).
    """
    tool_calls = step.get("tool_calls", [])
    if a.tool_call_id:
        for tc in tool_calls:
            tc_id = tc.get("tool_id", "") or tc.get("id", "") or ""
            if tc_id == a.tool_call_id:
                return tc
    ordinal = sum(
        1 for prev in actions[:action_index]
        if prev.step_index == a.step_index and prev.tool == a.tool
    )
    seen = 0
    for tc in tool_calls:
        if tc.get("tool_name") != a.tool:
            continue
        if seen == ordinal:
            return tc
        seen += 1
    return None


def assign_effect_labels(
    actions: list[CanonicalAction],
    steps: list[dict],
    anchor_files: set[str] | None = None,
) -> None:
    """Assign effect_label to each action based on full trajectory context. Mutates in place."""
    # Determine target files (patch footprint). Truthiness check (B25): an
    # EMPTY anchor set (e.g. a patch whose headers matched no `a/ b/` prefixes)
    # must not suppress the identify_target_files fallback — downstream
    # consumers already treat an empty set as "un-anchored".
    target_files: set[str] = set()
    if anchor_files:
        target_files = {_normalize_target(f) for f in anchor_files}
    else:
        from trajviz.insight.diagnostics import identify_target_files
        target_files = {_normalize_target(f) for f in identify_target_files(steps)}

    # Collect test/interface files for justification
    test_patterns = {"test_", "_test.", ".test.", "spec_", "_spec.", ".spec.", "tests/", "__tests__/"}
    import_files: set[str] = set()  # files imported by target files
    error_files: set[str] = set()   # files referenced in errors

    # Scan tool call outputs for import/require references to build import_files
    _IMPORT_RE = re.compile(
        r"""(?:import\s+.*?from\s+['"]([^'"]+)['"]"""  # JS/TS import from
        r"""|require\s*\(\s*['"]([^'"]+)['"]"""          # JS require
        r"""|from\s+([\w.]+)\s+import"""                 # Python from X import
        r"""|import\s+([\w.]+))""",                      # Python import X
        re.MULTILINE,
    )
    for step in steps:
        for tc in step.get("tool_calls", []):
            # Check if this tool call reads a target file — scan its output for imports
            inp = tc.get("input", {})
            if isinstance(inp, dict):
                fp = _normalize_target(inp.get("file_path", ""))
                if fp in target_files:
                    output = tc.get("output", "")
                    if isinstance(output, str):
                        for m in _IMPORT_RE.finditer(output):
                            imported = m.group(1) or m.group(2) or m.group(3) or m.group(4)
                            if imported:
                                # Normalize relative imports to potential file paths
                                norm = _normalize_target(imported.replace(".", "/"))
                                import_files.add(norm)
            error_text = tc.get("error", "")
            status = tc.get("status", "")
            if not error_text and status in ("error", "failed", "failure"):
                error_text = tc.get("output", "")
            if isinstance(error_text, str) and error_text:
                for path in _extract_bash_paths(error_text):
                    error_files.add(_normalize_target(path))

    # Build write sequence per target for reverted detection
    writes_by_target: dict[str, list[int]] = {}
    for i, a in enumerate(actions):
        if a.action_type == "FILE_WRITE":
            writes_by_target.setdefault(a.target, []).append(i)

    # Find the last write per target (surviving write)
    last_write_per_target: dict[str, int] = {}
    for target, indices in writes_by_target.items():
        last_write_per_target[target] = indices[-1]

    # Build step lookup for O(1) access by step_index
    step_by_index: dict[int, dict] = {s["index"]: s for s in steps}

    for i, a in enumerate(actions):
        if a.action_type == "REASON":
            a.effect_label = "unknown"
            continue

        # Failed: tool errored or bad exit code
        if a.status in ("error", "failed", "failure", "cancelled", "timeout"):
            a.effect_label = "failed"
            a.effect_detail["reason"] = "tool_status_error"
            continue
        # Check exit code from the action's OWN tool call only (B23): id match
        # when available, positional pairing otherwise — never a name-scan that
        # picks up a failing sibling call.
        step = step_by_index.get(a.step_index)
        own_tc = _find_own_tool_call(a, step, actions, i) if step is not None else None
        if own_tc is not None:
            m = own_tc.get("metadata", {})
            if isinstance(m, dict) and m.get("exit") not in (None, 0):
                a.effect_label = "failed"
                a.effect_detail["reason"] = f"exit_code_{m['exit']}"
                continue

        if a.action_type == "FILE_WRITE":
            if a.target in last_write_per_target and last_write_per_target[a.target] != i:
                a.effect_label = "reverted"
                a.effect_detail["reason"] = "overwritten_by_later_write"
                a.effect_detail["survives_to_final_patch"] = False
            else:
                a.effect_label = "survived"
                a.effect_detail["survives_to_final_patch"] = True
                a.effect_detail["reason"] = "survived_within_run"

        elif a.action_type == "COMMAND":
            # Read the command text from the action's OWN call (B23), not the
            # first same-named sibling.
            command = ""
            if own_tc is not None:
                inp = own_tc.get("input", {})
                if isinstance(inp, dict):
                    command = inp.get("command", "")
            if command and _is_validation_command(command):
                a.effect_label = "justified"
                a.effect_detail["reason"] = "validation_command"
            else:
                a.effect_label = "survived"
                a.effect_detail["reason"] = "command_completed"

        elif a.action_type == "FILE_READ":
            nt = a.target
            if nt in target_files:
                a.effect_label = "justified"
                a.effect_detail["reason"] = "target_file"
            elif any(p in nt for p in test_patterns):
                a.effect_label = "justified"
                a.effect_detail["reason"] = "test_file"
            elif nt in error_files:
                a.effect_label = "justified"
                a.effect_detail["reason"] = "error_reference"
            elif any(
                nt.endswith(imp)
                or (nt.rsplit("/", 1)[-1].split(".")[0]
                    and imp.endswith(nt.rsplit("/", 1)[-1].split(".")[0]))
                for imp in import_files if imp
            ):
                a.effect_label = "justified"
                a.effect_detail["reason"] = "imported_by_target"
            else:
                a.effect_label = "unknown"

        elif a.action_type == "SEARCH" or a.action_type == "AGENT_SPAWN":
            a.effect_label = "unknown"


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_RATE = 50.0


def compute_action_cost(action: CanonicalAction, token_rate: float = DEFAULT_TOKEN_RATE) -> float:
    """Compute scalar cost: token_share + (latency_ms / 1000 * token_rate).

    Uses token_share (the action's proportional share of step tokens) instead of
    the full step token count to prevent cost inflation.
    """
    return action.cost.token_share + (action.cost.latency_ms / 1000.0 * token_rate)


# ---------------------------------------------------------------------------
# Semantic equivalence
# ---------------------------------------------------------------------------

# Effect compatibility for semantic equivalence matching.
# Asymmetric by design: "survived" and "justified" are interchangeable
# (both represent "useful" work), while "failed" and "reverted" only match
# themselves (same observable outcome). "unknown" is a wildcard that matches
# anything, since we lack evidence to distinguish it.
_EFFECT_COMPAT: dict[str, set[str]] = {
    "survived": {"survived", "justified"},
    "justified": {"survived", "justified"},
    "failed": {"failed"},
    "reverted": {"reverted"},
    "unknown": {"survived", "failed", "reverted", "justified", "unknown"},
}


def _targets_match(a: str, b: str) -> bool:
    """Check if two targets match.

    Handles different repo roots by comparing path suffixes.
    '/home/user/repo/src/app.ts' and 'D:/other/repo/src/app.ts'
    both end with 'src/app.ts' and should match.
    """
    if not a or not b:
        return a == b
    na, nb = _normalize_target(a), _normalize_target(b)
    if na == nb:
        return True
    # Suffix match: check if one path ends with a significant suffix of the other
    if na.endswith("/" + nb) or nb.endswith("/" + na):
        return True
    # Cross-root match: find longest common suffix of path segments
    pa = na.split("/")
    pb = nb.split("/")
    common = 0
    for sa, sb in zip(reversed(pa), reversed(pb), strict=False):
        if sa == sb:
            common += 1
        else:
            break
    # Match if at least 2 common path segments (e.g., "core/v1/types.go" = 3)
    # and the common suffix includes the filename
    return common >= 2


def _search_targets_match(a_target: str, b_target: str) -> bool:
    """Compare SEARCH targets ('pattern@scope' or bare 'pattern') (B27).

    Patterns must be equal; scopes get the same cross-root suffix matching as
    FILE_READ/FILE_WRITE targets (_targets_match), so identical searches run
    from different workspace roots still align. A missing scope on either side
    is treated as a wildcard.
    """
    pa, sep_a, sa = a_target.rpartition("@")
    if not sep_a:
        pa, sa = sa, ""
    pb, sep_b, sb = b_target.rpartition("@")
    if not sep_b:
        pb, sb = sb, ""
    return pa == pb and (not sa or not sb or _targets_match(sa, sb))


def semantic_equivalent(
    a: CanonicalAction,
    b: CanonicalAction,
    fuzzy_commands: bool = False,
) -> bool:
    """Check if two canonical actions are semantically equivalent."""
    # Effect compatibility check
    a_compat = _EFFECT_COMPAT.get(a.effect_label, {"unknown"})
    b_compat = _EFFECT_COMPAT.get(b.effect_label, {"unknown"})
    if b.effect_label not in a_compat and a.effect_label not in b_compat:
        return False

    # Direct type+target match
    if a.action_type == b.action_type:
        if a.action_type == "COMMAND":
            return _extract_base_command(a.target) == _extract_base_command(b.target)
        if a.action_type == "SEARCH":
            return _search_targets_match(a.target, b.target)
        if a.action_type in ("FILE_READ", "FILE_WRITE"):
            return _targets_match(a.target, b.target)
        if a.action_type == "AGENT_SPAWN":
            return True  # any spawn matches any spawn
        if a.action_type == "REASON":
            return True
        return a.target == b.target

    # Fuzzy command matching
    if fuzzy_commands and (a.action_type == "COMMAND" or b.action_type == "COMMAND"):
        cmd_action = a if a.action_type == "COMMAND" else b
        other = b if a.action_type == "COMMAND" else a
        reduced = reduce_composite_command(cmd_action)
        if reduced and reduced.action_type == other.action_type:
            if other.action_type in ("FILE_READ", "FILE_WRITE"):
                return _targets_match(reduced.target, other.target)
            if other.action_type == "SEARCH":
                return _search_targets_match(reduced.target, other.target)
    return False


# ---------------------------------------------------------------------------
# Composite command reduction
# ---------------------------------------------------------------------------

def reduce_composite_command(action: CanonicalAction) -> CanonicalAction | None:
    """Attempt to reduce a COMMAND action to a finer-grained action type.

    Uses last-pipe-stage rule: in 'cat file | grep pattern', grep is the last stage.
    Returns a new CanonicalAction with the reduced type, or None if ambiguous.
    """
    if action.action_type != "COMMAND":
        return None

    command = action.args.get("command", "")
    if not command:
        return None

    # Split by pipe, take last stage
    stages = [s.strip() for s in command.split("|")]
    last_stage = stages[-1]
    first_token = last_stage.split()[0] if last_stage.split() else ""

    reduced_type = _FUZZY_KEYWORDS.get(first_token)
    if not reduced_type:
        return None
    if reduced_type == "FILE_READ" and _is_in_place_edit(first_token, last_stage):
        reduced_type = "FILE_WRITE"  # keep classification consistent with B22

    # Build the target from the LAST pipe stage (the stage that determines the
    # reduced type), not the whole command, so an earlier stage's path cannot
    # leak into the target.
    if reduced_type == "SEARCH":
        # Native SEARCH targets are 'pattern@scope' (or bare 'pattern');
        # emit the same format so strict equality in semantic_equivalent's
        # fuzzy branch can actually match (B26).
        parts = last_stage.split()
        non_flag_args = [p for p in parts[1:] if not p.startswith("-")]
        pattern = non_flag_args[0] if non_flag_args else ""
        if not pattern:
            return None
        stage_paths = _extract_bash_paths(last_stage)
        scope = _normalize_target(stage_paths[0]) if stage_paths else ""
        target = f"{pattern}@{scope}" if scope else pattern
    else:
        paths = _extract_bash_paths(last_stage)
        if not paths:
            return None
        target = _normalize_target(paths[0])

    return CanonicalAction(
        step_index=action.step_index,
        action_type=reduced_type,
        target=target,
        tool=action.tool,
        args=action.args,
        status=action.status,
        cost=action.cost,
        effect_label=action.effect_label,
        effect_detail=action.effect_detail,
    )
