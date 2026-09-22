"""HTML and Plotly presenters for a LoadedSession. No Gradio imports."""

from .issues import build_overview_issues_html
from .label_ui import build_label_ui_payload
from .overview import (
    build_chart_outputs,
    build_diagnostics_outputs,
    build_overview_kpi_html,
    build_overview_outputs,
    build_summary_outputs,
    empty_plotly_fig,
    load_warnings_html,
    trajectory_format_label,
)
from .patterns import (
    build_antipattern_html,
    render_failure_patterns_html,
    render_tool_sequences_html,
)
from .raw import raw_json_text
from .workflow import (
    ALL_FEATURE_FILTER,
    DETAIL_PLACEHOLDER,
    FEATURE_FILTERS,
    FILTER_CHIPS_DEFAULT,
    ROLE_FILTERS,
    build_filtered_workflow_outputs,
    build_workflow_outputs,
    filter_workflow_steps,
)

__all__ = [
    "ALL_FEATURE_FILTER",
    "DETAIL_PLACEHOLDER",
    "FEATURE_FILTERS",
    "FILTER_CHIPS_DEFAULT",
    "ROLE_FILTERS",
    "build_antipattern_html",
    "build_chart_outputs",
    "build_diagnostics_outputs",
    "build_filtered_workflow_outputs",
    "build_label_ui_payload",
    "build_overview_issues_html",
    "build_overview_kpi_html",
    "build_overview_outputs",
    "build_summary_outputs",
    "build_workflow_outputs",
    "empty_plotly_fig",
    "filter_workflow_steps",
    "load_warnings_html",
    "raw_json_text",
    "render_failure_patterns_html",
    "render_tool_sequences_html",
    "trajectory_format_label",
]
