"""Plotly chart builders for trajectory visualization."""

from . import _layout  # noqa: F401  — pandas before plotly; see _layout.py
from ._timeline import bind_timeline_agents, timeline_agent_id_of
from .activity import (
    build_context_pressure_chart,
    build_file_interaction_chart,
    build_plan_timeline_chart,
)
from .label_charts import (
    build_label_action_count_chart,
    build_label_action_duration_chart,
    build_label_phase_count_chart,
    build_label_phase_duration_chart,
    build_label_timeline_chart,
    build_phase_count_comparison_chart,
    build_phase_duration_comparison_chart,
)
from .swimlanes import (
    build_agent_swimlane_chart,
    build_run_group_agent_timeline,
    build_tool_outcome_timeline,
)
from .usage import (
    build_agent_token_chart,
    build_duration_chart,
    build_skill_agent_chart,
    build_token_chart,
    build_tool_chart,
    build_tool_duration_chart,
)

__all__ = [
    "bind_timeline_agents",
    "build_agent_swimlane_chart",
    "build_agent_token_chart",
    "build_context_pressure_chart",
    "build_duration_chart",
    "build_file_interaction_chart",
    "build_label_action_count_chart",
    "build_label_action_duration_chart",
    "build_label_phase_count_chart",
    "build_label_phase_duration_chart",
    "build_label_timeline_chart",
    "build_phase_count_comparison_chart",
    "build_phase_duration_comparison_chart",
    "build_plan_timeline_chart",
    "build_run_group_agent_timeline",
    "build_skill_agent_chart",
    "build_token_chart",
    "build_tool_chart",
    "build_tool_duration_chart",
    "build_tool_outcome_timeline",
    "timeline_agent_id_of",
]
