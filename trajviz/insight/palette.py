"""Unified color palette constants for charts and UI components."""

# Token category colors — used consistently across all token-related charts.
TOKEN_COLORS = {
    "fresh_input": "#3b82f6",   # blue
    "cache_read": "#34d399",    # emerald (stronger than previous mint)
    "cache_write": "#14b8a6",   # teal — cache-family, distinct from cache_read
    "output": "#f59e0b",        # amber
    "reasoning": "#8b5cf6",     # violet
}

# Session/lane color cycle shared by the subagent swimlane and timeline charts.
SESSION_COLORS = [
    "#3b82f6", "#8b5cf6", "#059669", "#d97706", "#e11d48", "#0891b2",
]

# Human-user lane on the agent swimlane (light Workflow --wf-border-user).
USER_SWIMLANE_COLOR = "#1e40af"

# Agent color palette — first entry is "main", rest cycle for sub-agents.
AGENT_COLORS = [
    "#6b7280",  # main (grey)
    "#2563eb",  # sub-agent 1 (blue)
    "#d946ef",  # sub-agent 2 (fuchsia)
    "#059669",  # sub-agent 3 (emerald)
    "#ea580c",  # sub-agent 4 (orange)
    "#8b5cf6",  # sub-agent 5 (violet)
]

# CSS-variable equivalents for workflow cards (bg, border pairs).
AGENT_CSS_COLORS = [
    ("var(--ov-card)", "var(--ov-muted)"),   # main
    ("#dbeafe", "#2563eb"),                    # blue
    ("#fae8ff", "#d946ef"),                    # fuchsia
    ("#d1fae5", "#059669"),                    # emerald
    ("#ffedd5", "#ea580c"),                    # orange
    ("#ede9fe", "#8b5cf6"),                    # violet
]

# Step Duration chart: scaffold primitives vs agentic/custom tool failures.
DURATION_ERROR_COLORS = {
    "system": "#d97706",  # amber — Grep / Read / Write / …
    "tool": "#dc2626",    # red — Bash / Skill / Task / MCP / workflow tools
}

# Tool outcome colors.
TOOL_OUTCOME_COLORS = {
    "success": "#059669",     # green
    "failure": "#dc2626",     # red
}

# Label taxonomy phase colors — maps 6 labeling phases to colors.
LABEL_PHASE_COLORS: dict[str, str] = {
    "understand": "#3b82f6",   # blue
    "plan":       "#8b5cf6",   # violet
    "implement":  "#059669",   # emerald
    "debug":      "#dc2626",   # red
    "validate":   "#f59e0b",   # amber
    "report":     "#ec4899",   # pink
}

# General chart accent color.
CHART_ACCENT = "#6366f1"      # indigo (for single-series bar charts)

# Plotly dark-mode layout template — transparent bg with light text/gridlines.
# Apply via fig.update_layout(**PLOTLY_DARK_TEMPLATE) when dark mode is active.
PLOTLY_DARK_TEMPLATE: dict = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#e2e8f0"},
    "xaxis": {
        "gridcolor": "rgba(255,255,255,0.08)",
        "zerolinecolor": "rgba(255,255,255,0.12)",
        "tickfont": {"color": "#cbd5e1"},
        "title_font": {"color": "#e2e8f0"},
    },
    "yaxis": {
        "gridcolor": "rgba(255,255,255,0.08)",
        "zerolinecolor": "rgba(255,255,255,0.12)",
        "tickfont": {"color": "#cbd5e1"},
        "title_font": {"color": "#e2e8f0"},
    },
    "legend": {"font": {"color": "#cbd5e1"}},
    "coloraxis": {"colorbar": {"tickfont": {"color": "#cbd5e1"}, "title_font": {"color": "#e2e8f0"}}},
}
