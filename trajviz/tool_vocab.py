"""Shared tool-name vocabulary — the single source of truth.

Three modules previously carried hand-maintained copies of "which tools write
files" (insight/patterns.py, insight/metrics.py, converge/canonical.py) plus a
fourth partial check in insight/diagnostics.py — and all four had drifted
pairwise (MultiEdit invisible to converge, NotebookEdit/patch missing from the
edit-precision metric, notebook edits absent from the target-file footprint).
Every consumer now imports from here; add new scaffold tool names HERE only.
"""

# Tools whose successful invocation writes file content. Mixed casings are the
# literal spellings emitted by the supported scaffolds (Claude Code, Cursor, OpenCode,
# CodeArts, ICode, Codex CLI, Pi, DeepSeek Harness adapters).
WRITE_TOOL_NAMES: frozenset[str] = frozenset({
    "Edit", "edit", "Write", "write",
    "NotebookEdit", "patch",
    "MultiEdit", "multiedit",
    "str_replace_editor", "create_file",
    "StrReplace",
})

# Path-bearing input keys used by write tools across scaffolds, in precedence
# order (Claude Code file_path/notebook_path; OpenCode filePath; text-editor
# tools use bare path).
WRITE_PATH_KEYS: tuple[str, ...] = (
    "file_path", "filePath", "notebook_path", "notebookPath", "path",
)


def write_target_path(tool_input: dict) -> str:
    """The file path a write-tool call targets ('' when absent/malformed)."""
    if not isinstance(tool_input, dict):
        return ""
    for key in WRITE_PATH_KEYS:
        value = tool_input.get(key)
        if value:
            return str(value)
    return ""


# Tool names whose input names a Skill (Claude Code Skill, Cursor Skill, etc.).
SKILL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "skill",
        "skills",
        "invoke_skill",
        "run_skill",
    }
)

# Input keys that carry the skill id, in precedence order.
SKILL_NAME_KEYS: tuple[str, ...] = (
    "skill",
    "name",
    "skill_name",
    "skillName",
    "id",
    "skill_id",
)


def parse_skill_name(tool_name: str, tool_input: object) -> str | None:
    """Return skill id when this call invokes a Skill tool; else None."""
    name_l = (tool_name or "").lower()
    if name_l not in SKILL_TOOL_NAMES:
        return None
    if isinstance(tool_input, dict):
        for key in SKILL_NAME_KEYS:
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    if isinstance(tool_input, str) and tool_input.strip():
        return tool_input.strip()
    return "(unnamed skill)"


# Tools that spawn a child agent / sub-session. Mixed casings are the literal
# names emitted by Claude Code (Task), OpenCode (task / Agent), DSH
# (subagent / subagent_fork), and Codex (spawn_agent).
SPAWN_TOOL_NAMES: frozenset[str] = frozenset({
    "Agent", "agent",
    "Task", "task",
    "subagent", "subagent_fork",
    "spawn_agent",
})

# Shell tool spellings across Claude Code, Cursor, OpenCode/Pi/DSH, and adapters.
BASH_TOOL_NAMES: frozenset[str] = frozenset({
    "Bash", "bash", "BashCommand",
    "Shell",
})

# Scaffold/harness primitives (search, file I/O). Failures of these are
# "system" errors on the Step Duration chart. Bash is intentionally excluded —
# it usually runs user-defined scripts, so its failures count as tool errors.
# Skill/Task/MCP/custom tools are also not system.
SYSTEM_TOOL_NAMES: frozenset[str] = frozenset({
    "Grep", "grep", "Glob", "glob", "find",
    "Read", "read",
    "WebFetch", "WebSearch", "ToolSearch",
    "LS", "ls", "ListDir", "list_dir",
}) | WRITE_TOOL_NAMES


def is_mcp_tool(tool_name: str) -> bool:
    """Return True when *tool_name* follows an MCP naming convention.

    Two conventions are recognised:

    * **Claude Code** - ``mcp__<server>__<tool>`` (double-underscore prefix
      and separator).
    * **OpenCode / CodeArts** - ``<hyphenated-server>_<tool>`` (the server
      name contains a hyphen, e.g. ``kernel-test-runner_run_test``).

    No built-in scaffold tool (Read, Bash, Edit, Glob) contains a hyphen
    or a double underscore, so both heuristics are false-positive free.
    """
    name = tool_name or ""
    if name.startswith("mcp__"):
        return True
    if "__" in name:
        return True
    return "-" in name
