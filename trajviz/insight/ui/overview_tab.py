"""Overview tab: performance through labels, plus pressure-agent and label callbacks."""

from __future__ import annotations

import html
import os
from dataclasses import dataclass

import gradio as gr
import plotly.graph_objects as go

from ..charts import build_context_pressure_chart
from ..context_usage import (
    DEFAULT_CONTEXT_WINDOW_LIMIT,
    PRESSURE_ALL_AGENTS,
    SNAPSHOT_CURRENT,
    context_pressure_series,
    parse_usage_snapshot,
    usage_snapshot_choices,
)
from ..formatting import format_context_pressure_html
from ..help import HELP_TEXT
from ..presenters.label_ui import build_label_ui_payload
from ..presenters.issues import (
    build_overview_issues_html,
    collect_overview_issues,
    rank_issues,
)
from ..presenters.overview import (
    build_chart_outputs,
    build_diagnostics_outputs,
    build_overview_outputs,
    empty_plotly_fig,
)
from ..issue_judge import JUDGE_ISSUE_CAP, iter_judge_overview_issues
from ..llm_config import resolve_analysis_config
from ..session import LoadedSession, build_loaded_session
from .shared import SharedState
from .upload import UploadRefs


@dataclass
class OverviewRefs:
    session_detail_html: gr.HTML
    overview_kpi_html: gr.HTML
    overview_section: gr.Radio
    performance_section: gr.Column
    efficiency_section: gr.Column
    tools_section: gr.Column
    agents_section: gr.Column
    diagnostics_section: gr.Column
    deep_dive_section: gr.Column
    labels_section: gr.Column
    issues_html: gr.HTML
    suggest_fixes_btn: gr.Button
    metrics_md: gr.Markdown
    token_chart: gr.Plot
    duration_chart: gr.Plot
    behavior_md: gr.Markdown
    tool_chart: gr.Plot
    tool_duration_chart: gr.Plot
    skill_chart: gr.Plot
    tool_outcome_chart: gr.Plot
    agent_summary_html: gr.HTML
    agent_token_chart: gr.Plot
    agent_swimlane_chart: gr.Plot
    diag_summary_html: gr.HTML
    diag_file_chart: gr.Plot
    diag_pressure_html: gr.HTML
    diag_pressure_agent: gr.Dropdown
    diag_window_limit: gr.Number
    diag_usage_snapshot: gr.Dropdown
    diag_pressure_chart: gr.Plot
    diag_rootcause_html: gr.HTML
    plan_timeline_chart: gr.Plot
    hotspots_md: gr.Markdown
    per_message_md: gr.Markdown
    label_status_html: gr.HTML
    label_charts_row1: gr.Row
    label_phase_count_chart: gr.Plot
    label_action_count_chart: gr.Plot
    label_charts_row2: gr.Row
    label_phase_dur_chart: gr.Plot
    label_action_dur_chart: gr.Plot
    label_timeline_row: gr.Row
    label_timeline_chart: gr.Plot


OVERVIEW_SECTION_NAMES = [
    "Summary",
    "Tools",
    "Agents",
    "Context Utilization",
    "Diagnostics",
    "Deep Dive",
    "Labels",
]


def layout(overview_kpi_html: gr.HTML) -> OverviewRefs:
    with gr.TabItem("Overview"):
        # Debug hero: Issues first; KPI strip lives above the main Tabs.
        issues_html = gr.HTML("")
        # Opt-in only: judging sends trajectory excerpts (prompts, commands,
        # file paths, tool output) to the configured LLM endpoint, so it must
        # never run without an explicit click.
        suggest_fixes_btn = gr.Button(
            "Suggest fixes with LLM (sends trajectory excerpts to the configured provider)",
            size="sm",
            variant="secondary",
        )
        session_detail_html = gr.HTML("")

        overview_section_names = OVERVIEW_SECTION_NAMES
        with gr.Row(elem_classes=["overview-content-layout"]):
            with gr.Column(scale=0, min_width=160, elem_classes=["overview-section-nav"]):
                gr.HTML("<div class='overview-nav-title'>Contents</div>")
                overview_section = gr.Radio(
                    choices=overview_section_names,
                    value="Summary",
                    show_label=False,
                    container=False,
                    elem_classes=["overview-section-radio"],
                )

            with gr.Column(scale=1, min_width=0, elem_classes=["overview-section-content"]):
                with gr.Column(visible=True) as performance_section:
                    gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_summary'])}</div>")
                    with gr.Row(equal_height=True):
                        token_chart = gr.Plot(show_label=False, label="Token Usage")
                        duration_chart = gr.Plot(
                            show_label=False,
                            label="Step Duration",
                            elem_id="duration-chart",
                        )

                with gr.Column(visible=False) as efficiency_section:
                    gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_context_utilization'])}</div>")
                    with gr.Row():
                        diag_pressure_agent = gr.Dropdown(
                            label="Agent",
                            choices=[("All agents", PRESSURE_ALL_AGENTS)],
                            value=PRESSURE_ALL_AGENTS,
                            visible=False,
                            interactive=True,
                            scale=2,
                        )
                        diag_usage_snapshot = gr.Dropdown(
                            label="Window snapshot",
                            choices=[("Current window", SNAPSHOT_CURRENT)],
                            value=SNAPSHOT_CURRENT,
                            visible=False,
                            interactive=True,
                            scale=2,
                        )
                        diag_window_limit = gr.Number(
                            label="Window limit (tokens)",
                            value=DEFAULT_CONTEXT_WINDOW_LIMIT,
                            precision=0,
                            minimum=1,
                            step=1000,
                            interactive=True,
                            scale=1,
                        )
                    diag_pressure_html = gr.HTML("")
                    diag_pressure_chart = gr.Plot(
                        show_label=False,
                        label="Context Window Pressure",
                    )

                with gr.Column(visible=False) as tools_section:
                    gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_tools'])}</div>")
                    behavior_md = gr.Markdown("")
                    with gr.Row(equal_height=True):
                        tool_chart = gr.Plot(show_label=False, label="Tool Call Frequency")
                        tool_duration_chart = gr.Plot(
                            show_label=False,
                            label="Tool Call Duration",
                            elem_id="tool-duration-chart",
                        )
                    with gr.Row(equal_height=True):
                        tool_outcome_chart = gr.Plot(
                            show_label=False,
                            label="Tool Outcome Timeline",
                            elem_id="tool-outcome-chart",
                        )
                    with gr.Row(equal_height=True):
                        skill_chart = gr.Plot(show_label=False, label="Skill Calls by Agent")
                        gr.Column(scale=1)

                with gr.Column(visible=False) as agents_section:
                    gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_agents'])}</div>")
                    agent_summary_html = gr.HTML("")
                    with gr.Row(equal_height=True):
                        agent_swimlane_chart = gr.Plot(
                            show_label=False,
                            label="Agent Swimlane",
                            elem_id="agent-swimlane-chart",
                        )
                    with gr.Row(equal_height=True):
                        agent_token_chart = gr.Plot(show_label=False, label="Token Breakdown by Agent")
                        gr.Column(scale=1)

                with gr.Column(visible=False) as diagnostics_section:
                    gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_diagnostics'])}</div>")
                    diag_summary_html = gr.HTML("")
                    diag_file_chart = gr.Plot(
                        show_label=False,
                        label="File Interaction Timeline",
                        elem_id="diag-file-chart",
                        elem_classes=["resizable-chart"],
                    )
                    diag_rootcause_html = gr.HTML("")
                    plan_timeline_chart = gr.Plot(show_label=False, label="Plan Progress Timeline")

                with gr.Column(visible=False) as deep_dive_section:
                    gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_deep_dive'])}</div>")
                    metrics_md = gr.Markdown("")
                    hotspots_md = gr.Markdown("")
                    per_message_md = gr.Markdown("")

                with gr.Column(visible=False) as labels_section:
                    gr.HTML("<div class='section-subtitle'>Phase and action classification from labeled JSON</div>")
                    label_status_html = gr.HTML(
                        "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                        "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>"
                    )
                    with gr.Row(equal_height=True, visible=False) as label_charts_row1:
                        label_phase_count_chart = gr.Plot(show_label=False, label="Phase Count Distribution")
                        label_action_count_chart = gr.Plot(show_label=False, label="Action Count Distribution")
                    with gr.Row(equal_height=True, visible=False) as label_charts_row2:
                        label_phase_dur_chart = gr.Plot(show_label=False, label="Phase Duration Distribution")
                        label_action_dur_chart = gr.Plot(show_label=False, label="Action Duration Distribution")
                    with gr.Row(equal_height=True, visible=False) as label_timeline_row:
                        label_timeline_chart = gr.Plot(show_label=False, label="Step Timeline")

    return OverviewRefs(
        session_detail_html=session_detail_html,
        overview_kpi_html=overview_kpi_html,
        overview_section=overview_section,
        performance_section=performance_section,
        efficiency_section=efficiency_section,
        tools_section=tools_section,
        agents_section=agents_section,
        diagnostics_section=diagnostics_section,
        deep_dive_section=deep_dive_section,
        labels_section=labels_section,
        issues_html=issues_html,
        suggest_fixes_btn=suggest_fixes_btn,
        metrics_md=metrics_md,
        token_chart=token_chart,
        duration_chart=duration_chart,
        behavior_md=behavior_md,
        tool_chart=tool_chart,
        tool_duration_chart=tool_duration_chart,
        skill_chart=skill_chart,
        tool_outcome_chart=tool_outcome_chart,
        agent_summary_html=agent_summary_html,
        agent_token_chart=agent_token_chart,
        agent_swimlane_chart=agent_swimlane_chart,
        diag_summary_html=diag_summary_html,
        diag_file_chart=diag_file_chart,
        diag_pressure_html=diag_pressure_html,
        diag_pressure_agent=diag_pressure_agent,
        diag_window_limit=diag_window_limit,
        diag_usage_snapshot=diag_usage_snapshot,
        diag_pressure_chart=diag_pressure_chart,
        diag_rootcause_html=diag_rootcause_html,
        plan_timeline_chart=plan_timeline_chart,
        hotspots_md=hotspots_md,
        per_message_md=per_message_md,
        label_status_html=label_status_html,
        label_charts_row1=label_charts_row1,
        label_phase_count_chart=label_phase_count_chart,
        label_action_count_chart=label_action_count_chart,
        label_charts_row2=label_charts_row2,
        label_phase_dur_chart=label_phase_dur_chart,
        label_action_dur_chart=label_action_dur_chart,
        label_timeline_row=label_timeline_row,
        label_timeline_chart=label_timeline_chart,
    )


def load_slots(refs: OverviewRefs) -> dict:
    """Named Gradio components filled by the load packer."""
    return {
        "overview_kpi_html": refs.overview_kpi_html,
        "session_detail_html": refs.session_detail_html,
        "issues_html": refs.issues_html,
        "metrics_md": refs.metrics_md,
        "token_chart": refs.token_chart,
        "duration_chart": refs.duration_chart,
        "behavior_md": refs.behavior_md,
        "tool_chart": refs.tool_chart,
        "tool_duration_chart": refs.tool_duration_chart,
        "skill_chart": refs.skill_chart,
        "tool_outcome_chart": refs.tool_outcome_chart,
        "agent_summary_html": refs.agent_summary_html,
        "agent_token_chart": refs.agent_token_chart,
        "agent_swimlane_chart": refs.agent_swimlane_chart,
        "diag_summary_html": refs.diag_summary_html,
        "diag_pressure_html": refs.diag_pressure_html,
        "diag_pressure_agent": refs.diag_pressure_agent,
        "diag_window_limit": refs.diag_window_limit,
        "diag_usage_snapshot": refs.diag_usage_snapshot,
        "diag_pressure_chart": refs.diag_pressure_chart,
        "diag_file_chart": refs.diag_file_chart,
        "diag_rootcause_html": refs.diag_rootcause_html,
        "plan_timeline_chart": refs.plan_timeline_chart,
        "hotspots_md": refs.hotspots_md,
        "per_message_md": refs.per_message_md,
    }


def pack_load(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict:
    """Named Overview values for a load (empty when *session* is None)."""
    del banner
    fig = empty_plotly_fig()
    if session is None:
        return {
            "overview_kpi_html": gr.update(value="", visible=False),
            "session_detail_html": "",
            "issues_html": "",
            "metrics_md": "",
            "token_chart": fig,
            "duration_chart": fig,
            "behavior_md": "",
            "tool_chart": fig,
            "tool_duration_chart": fig,
            "skill_chart": fig,
            "tool_outcome_chart": fig,
            "agent_summary_html": "",
            "agent_token_chart": fig,
            "agent_swimlane_chart": fig,
            "diag_summary_html": "",
            "diag_pressure_html": "",
            "diag_pressure_agent": gr.update(
                choices=[("All agents", PRESSURE_ALL_AGENTS)],
                value=PRESSURE_ALL_AGENTS,
                visible=False,
            ),
            "diag_window_limit": DEFAULT_CONTEXT_WINDOW_LIMIT,
            "diag_usage_snapshot": gr.update(
                choices=[("Current window", SNAPSHOT_CURRENT)],
                value=SNAPSHOT_CURRENT,
                visible=False,
            ),
            "diag_pressure_chart": fig,
            "diag_file_chart": fig,
            "diag_rootcause_html": "",
            "plan_timeline_chart": fig,
            "hotspots_md": "",
            "per_message_md": "",
        }

    ov = build_overview_outputs(session)
    ch = build_chart_outputs(session, dark=dark)
    dg = build_diagnostics_outputs(session, dark=dark)
    return {
        "overview_kpi_html": gr.update(value=ov["kpi_html"], visible=True),
        "session_detail_html": ov["session_detail"],
        "issues_html": ov["issues_html"],
        "metrics_md": ov["metrics_text"],
        "token_chart": ch["tok_fig"],
        "duration_chart": ch["dur_fig"],
        "behavior_md": ov["behavior_text"],
        "tool_chart": ch["tl_fig"],
        "tool_duration_chart": ch["tool_dur_fig"],
        "skill_chart": ch["skill_fig"],
        "tool_outcome_chart": ch["tool_outcome_fig"],
        "agent_summary_html": ch["agent_cards_html"],
        "agent_token_chart": ch["agent_tok_fig"],
        "agent_swimlane_chart": ch["swimlane_fig"],
        "diag_summary_html": dg["diag_summary_html"],
        "diag_pressure_html": dg["diag_pressure_html"],
        "diag_pressure_agent": gr.update(
            choices=dg["diag_pressure_dropdown"]["choices"],
            value=dg["diag_pressure_dropdown"]["value"],
            visible=dg["diag_pressure_dropdown"]["visible"],
        ),
        "diag_window_limit": session.pressure_series.get("window_limit") or DEFAULT_CONTEXT_WINDOW_LIMIT,
        "diag_usage_snapshot": _snapshot_dropdown_update(session.steps),
        "diag_pressure_chart": dg["diag_pressure_chart"],
        "diag_file_chart": dg["diag_file_chart"],
        "diag_rootcause_html": dg["diag_rootcause_html"],
        "plan_timeline_chart": ch["plan_timeline_fig"],
        "hotspots_md": ov["hotspots_text"],
        "per_message_md": ov["per_message_text"],
    }


def _snapshot_dropdown_update(
    steps: list[dict],
    *,
    agent_key: str | None = None,
    value: str = SNAPSHOT_CURRENT,
) -> dict:
    choices = usage_snapshot_choices(steps, agent_key=agent_key)
    values = {choice_value for _label, choice_value in choices}
    selected = value if value in values else SNAPSHOT_CURRENT
    return gr.update(
        choices=choices,
        value=selected,
        visible=len(choices) > 1,
    )


def bind(
    refs: OverviewRefs,
    shared: SharedState,
    upload: UploadRefs,
    load_events: tuple = (),  # kept for the uniform tab-bind signature
) -> None:
    overview_section_names = OVERVIEW_SECTION_NAMES
    overview_sections = (
        refs.performance_section,
        refs.tools_section,
        refs.agents_section,
        refs.efficiency_section,
        refs.diagnostics_section,
        refs.deep_dive_section,
        refs.labels_section,
    )

    def show_overview_section(selected):
        return tuple(gr.update(visible=name == selected) for name in overview_section_names)

    refs.overview_section.change(
        fn=show_overview_section,
        inputs=[refs.overview_section],
        outputs=list(overview_sections),
    ).then(
        fn=None,
        js="() => { if (window.tvExpandFileTimeline) { window.tvExpandFileTimeline(); setTimeout(window.tvExpandFileTimeline, 200); } }",
    )

    def on_suggest_fixes(steps, raw):
        """Run the LLM Issues judge on request; progress shows inside the panel."""
        from ..presenters.issues import render_overview_issues_html

        if not steps:
            yield render_overview_issues_html([])
            return

        cfg = resolve_analysis_config()
        raw_dict = raw if isinstance(raw, dict) else {}
        session = build_loaded_session("", raw_dict)

        if not cfg.ready:
            missing = ", ".join(cfg.missing)
            banner = f"Configure {missing} in .env to auto-suggest fixes"
            yield build_overview_issues_html(session, banner=banner)
            return

        ranked = rank_issues(collect_overview_issues(session))
        if not ranked:
            yield build_overview_issues_html(session, issues=ranked)
            return

        judged = ranked
        errors: list[str] = []
        for progress in iter_judge_overview_issues(
            session, ranked, config=cfg, limit=JUDGE_ISSUE_CAP,
        ):
            judged = progress.issues
            errors = progress.errors
            if not progress.finished:
                short = progress.current_title
                if len(short) > 48:
                    short = short[:45] + "…"
                progress_line = (
                    f"Suggesting fixes {progress.current}/{progress.total} — {short}"
                )
                yield build_overview_issues_html(
                    session, issues=judged, progress=progress_line,
                )
                continue

            ok = sum(1 for i in judged if i.judgment is not None)
            if errors and ok == 0:
                banner = f"Judge failed ({len(errors)}). First: {errors[0][:160]}"
                yield build_overview_issues_html(
                    session, issues=ranked, banner=banner,
                )
                return

            banner = ""
            if errors:
                banner = (
                    f"{len(errors)} issue(s) could not be judged"
                    f" ({ok}/{min(len(ranked), JUDGE_ISSUE_CAP)} ok)."
                )
            elif len(ranked) > JUDGE_ISSUE_CAP:
                banner = f"Judged {ok}/{JUDGE_ISSUE_CAP} (capped)."
            yield build_overview_issues_html(
                session, issues=judged, banner=banner,
            )

    judge_event = dict(
        fn=on_suggest_fixes,
        inputs=[shared.state_steps, shared.state_raw],
        outputs=[refs.issues_html],
        show_progress="minimal",
        concurrency_id="issues_judge",
    )
    # Explicit click ONLY. Chaining this onto load_events would ship trajectory
    # excerpts off-box for every file a user opens, with no consent and no way
    # to decline; the heuristic (offline) Issues panel is what load renders.
    refs.suggest_fixes_btn.click(**judge_event)

    def _rebuild_utilization(agent_key, window_limit, snapshot_key, steps, raw, dark):
        if not steps:
            return empty_plotly_fig(), ""
        key = agent_key or PRESSURE_ALL_AGENTS
        raw_dict = raw if isinstance(raw, dict) else None
        snapshot_step = (
            None if key == PRESSURE_ALL_AGENTS else parse_usage_snapshot(snapshot_key)
        )
        series = context_pressure_series(
            steps,
            agent_key=key,
            raw=raw_dict,
            window_limit=window_limit,
        )
        fig = build_context_pressure_chart(
            steps,
            dark=bool(dark),
            series=series,
            highlight_step=snapshot_step,
        )
        html_strip = format_context_pressure_html(
            series,
            steps=steps,
            raw=raw_dict,
            agent_key=key,
            snapshot_step=snapshot_step,
        )
        return fig, html_strip

    def on_agent_change(agent_key, window_limit, steps, raw, dark):
        fig, html_strip = _rebuild_utilization(
            agent_key, window_limit, SNAPSHOT_CURRENT, steps, raw, dark,
        )
        return fig, html_strip, _snapshot_dropdown_update(steps or [], agent_key=agent_key)

    util_state = [shared.state_steps, shared.state_raw, shared.state_dark]
    util_outputs = [refs.diag_pressure_chart, refs.diag_pressure_html]
    refs.diag_pressure_agent.change(
        fn=on_agent_change,
        inputs=[refs.diag_pressure_agent, refs.diag_window_limit, *util_state],
        outputs=[*util_outputs, refs.diag_usage_snapshot],
    )
    refs.diag_window_limit.change(
        fn=_rebuild_utilization,
        inputs=[
            refs.diag_pressure_agent,
            refs.diag_window_limit,
            refs.diag_usage_snapshot,
            *util_state,
        ],
        outputs=util_outputs,
    )
    refs.diag_usage_snapshot.change(
        fn=_rebuild_utilization,
        inputs=[
            refs.diag_pressure_agent,
            refs.diag_window_limit,
            refs.diag_usage_snapshot,
            *util_state,
        ],
        outputs=util_outputs,
    )

    _empty_label_fig = go.Figure()
    _empty_label_fig.update_layout(
        template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

    def do_load_labels(upload_obj, dark=False):
        """Load labeled JSON and update all label UI components."""
        file_path = None
        if upload_obj is not None:
            file_path = upload_obj if isinstance(upload_obj, str) else upload_obj.name

        empty = _empty_label_fig

        if not file_path or not os.path.isfile(file_path):
            return (
                "",
                "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>",
                gr.update(visible=False),
                empty,
                empty,
                gr.update(visible=False),
                empty,
                empty,
                gr.update(visible=False),
                empty,
            )

        try:
            payload = build_label_ui_payload(file_path, dark=bool(dark))
        except Exception as exc:
            return (
                "",
                f"<div style='padding:1em;color:#dc2626;text-align:center;'>Error: {html.escape(str(exc))}</div>",
                gr.update(visible=False),
                empty,
                empty,
                gr.update(visible=False),
                empty,
                empty,
                gr.update(visible=False),
                empty,
            )

        return (
            payload["badge_html"],
            payload["status_html"],
            gr.update(visible=True),
            payload["phase_count_fig"],
            payload["action_count_fig"],
            gr.update(visible=True),
            payload["phase_duration_fig"],
            payload["action_duration_fig"],
            gr.update(visible=True),
            payload["timeline_fig"],
        )

    label_outputs = [
        upload.label_badge_html,
        refs.label_status_html,
        refs.label_charts_row1,
        refs.label_phase_count_chart,
        refs.label_action_count_chart,
        refs.label_charts_row2,
        refs.label_phase_dur_chart,
        refs.label_action_dur_chart,
        refs.label_timeline_row,
        refs.label_timeline_chart,
    ]
    label_inputs = [upload.label_file_upload, shared.state_dark]
    upload.label_load_btn.click(
        fn=do_load_labels,
        inputs=label_inputs,
        outputs=label_outputs,
    )
    upload.label_file_upload.change(
        fn=do_load_labels,
        inputs=label_inputs,
        outputs=label_outputs,
    )

    def _reset_labels():
        empty = _empty_label_fig
        return (
            "",
            (
                "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>"
            ),
            gr.update(visible=False),
            empty,
            empty,
            gr.update(visible=False),
            empty,
            empty,
            gr.update(visible=False),
            empty,
            gr.update(value=None),
        )

    _reset_outputs = label_outputs + [upload.label_file_upload]
    upload.load_btn.click(fn=_reset_labels, inputs=None, outputs=_reset_outputs)
    upload.file_upload.change(fn=_reset_labels, inputs=None, outputs=_reset_outputs)
