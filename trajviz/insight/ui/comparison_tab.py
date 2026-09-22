"""Comparison tab: N-run scorecard and pairwise Converge report."""

from __future__ import annotations

import os
from dataclasses import dataclass

import gradio as gr
import plotly.graph_objects as go

from ..charts import build_run_group_agent_timeline
from ..comparison import run_comparison
from ..loaders import FORMAT_DROPDOWN_CHOICES
from ..run_group import (
    build_run_group_behavior_html,
    build_run_group_scorecard,
    build_run_group_scorecard_html,
    normalize_run_paths,
)
from .shared import SharedState
from .upload import UploadRefs


@dataclass
class ComparisonRefs:
    cmp_status_html: gr.HTML
    rg_format_selector: gr.Dropdown
    rg_file_upload: gr.File
    rg_run_btn: gr.Button
    rg_scorecard_html: gr.HTML
    rg_agent_timeline_chart: gr.Plot
    rg_behavior_html: gr.HTML
    cmp_format_selector: gr.Dropdown
    cmp_file_upload: gr.File
    cmp_anchor_upload: gr.File
    cmp_ref_labels_upload: gr.File
    cmp_run_btn: gr.Button
    cmp_report_html: gr.HTML
    cmp_phase_count_chart: gr.Plot
    cmp_phase_duration_chart: gr.Plot


def layout() -> ComparisonRefs:
    with gr.TabItem("Comparison"):
        _cmp_placeholder = (
            "<div style='padding:2em;color:var(--ov-muted);text-align:center;font-size:14px;'>"
            "Load a trajectory in the Overview tab first &mdash; it becomes the "
            "<b>baseline</b> for <b>Run group</b> and the <b>compared</b> trajectory "
            "for pairwise comparison."
            "<br><span style='font-size:12px;'>"
            "In Run group, upload one or more additional runs to scorecard against Overview."
            "</span></div>"
        )
        cmp_status_html = gr.HTML(_cmp_placeholder)
        with gr.Accordion("Run group (N trajectories)", open=True, elem_classes=["per-message-acc"]):
            gr.Markdown(
                "_The trajectory loaded in **Overview** is included as the "
                "**baseline**. Upload one or more additional runs of the same "
                "task (different models, harnesses, or prompts). Builds a "
                "metrics scorecard, agent timeline, behavioral similarity, "
                "tool/skill coverage, action/file matrices, and waste patterns "
                "vs the Overview run. For a full pairwise report, use "
                "**Pairwise comparison** below._",
            )
            with gr.Row(equal_height=True):
                rg_format_selector = gr.Dropdown(
                    label="Format hint",
                    choices=FORMAT_DROPDOWN_CHOICES,
                    value="",
                    interactive=True,
                    scale=1,
                    min_width=140,
                )
                rg_file_upload = gr.File(
                    label="Comparison runs (.json / .jsonl / .zip) — select one or more",
                    file_types=[".json", ".jsonl", ".zip"],
                    file_count="multiple",
                    scale=3,
                )
            with gr.Row(equal_height=False):
                rg_run_btn = gr.Button(
                    "Build scorecard",
                    variant="primary",
                    size="sm",
                    scale=0,
                    min_width=140,
                )
            rg_scorecard_html = gr.HTML(
                "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                "Load a trajectory in <b>Overview</b>, upload one or more "
                "comparison runs, then click <b>Build scorecard</b>.</div>"
            )
            rg_agent_timeline_chart = gr.Plot(
                show_label=False,
                label="Agent timeline (by run)",
                visible=False,
            )
            rg_behavior_html = gr.HTML("")
        with gr.Accordion("Pairwise comparison", open=False, elem_classes=["per-message-acc"]):
            with gr.Row(equal_height=True):
                cmp_format_selector = gr.Dropdown(
                    label="Format",
                    choices=FORMAT_DROPDOWN_CHOICES,
                    value="",
                    interactive=True,
                    scale=1,
                    min_width=140,
                )
                cmp_file_upload = gr.File(
                    label="Reference Trajectory (.json / .jsonl / .zip)",
                    file_types=[".json", ".jsonl", ".zip"],
                    scale=2,
                )
                cmp_anchor_upload = gr.File(
                    label="Anchor Patch (optional, .patch/.diff)",
                    file_types=[".patch", ".diff"],
                    scale=1,
                )
            with gr.Row(equal_height=True):
                cmp_ref_labels_upload = gr.File(
                    label="Reference Labels (optional, *_labeled.json)",
                    file_types=[".json"],
                    scale=4,
                )
            with gr.Row(equal_height=False):
                gr.Markdown(
                    "_The uploaded trajectory is treated as the "
                    "**reference/baseline**; the trajectory loaded on the "
                    "Overview tab is the **compared** one._",
                )
                cmp_run_btn = gr.Button(
                    "Run Comparison",
                    variant="primary",
                    size="sm",
                    scale=1,
                    min_width=140,
                )
            cmp_report_html = gr.HTML("")
            with gr.Row(equal_height=True):
                cmp_phase_count_chart = gr.Plot(
                    show_label=False, label="Step Count by Phase — Reference vs Compared"
                )
                cmp_phase_duration_chart = gr.Plot(
                    show_label=False, label="Duration by Phase — Reference vs Compared"
                )
    return ComparisonRefs(
        cmp_status_html=cmp_status_html,
        rg_format_selector=rg_format_selector,
        rg_file_upload=rg_file_upload,
        rg_run_btn=rg_run_btn,
        rg_scorecard_html=rg_scorecard_html,
        rg_agent_timeline_chart=rg_agent_timeline_chart,
        rg_behavior_html=rg_behavior_html,
        cmp_format_selector=cmp_format_selector,
        cmp_file_upload=cmp_file_upload,
        cmp_anchor_upload=cmp_anchor_upload,
        cmp_ref_labels_upload=cmp_ref_labels_upload,
        cmp_run_btn=cmp_run_btn,
        cmp_report_html=cmp_report_html,
        cmp_phase_count_chart=cmp_phase_count_chart,
        cmp_phase_duration_chart=cmp_phase_duration_chart,
    )


def bind(refs: ComparisonRefs, shared: SharedState, upload: UploadRefs) -> None:
    def on_run_group_scorecard(files, format_hint, dark, overview_raw):
        """Build an N-run scorecard; Overview trajectory is the baseline."""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_white",
            height=200,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        paths = normalize_run_paths(files)
        hint = format_hint or None
        if hint == "":
            hint = None
        baseline = overview_raw if isinstance(overview_raw, dict) and overview_raw else None
        result = build_run_group_scorecard(
            paths,
            format_hint=hint,
            baseline_raw=baseline,
        )
        scorecard = build_run_group_scorecard_html(
            result,
            include_behavior=False,
        )
        timeline_runs = result.get("timeline_runs") or []
        if result.get("ok") and timeline_runs:
            fig = build_run_group_agent_timeline(
                timeline_runs,
                dark=bool(dark),
            )
            chart_update = gr.update(value=fig, visible=True)
        else:
            chart_update = gr.update(value=empty_fig, visible=False)
        behavior = build_run_group_behavior_html(result) if result.get("ok") else ""
        return scorecard, chart_update, behavior

    refs.rg_run_btn.click(
        fn=on_run_group_scorecard,
        inputs=[refs.rg_file_upload, refs.rg_format_selector, shared.state_dark, shared.state_raw],
        outputs=[refs.rg_scorecard_html, refs.rg_agent_timeline_chart, refs.rg_behavior_html],
    )

    def on_run_comparison(ref_file, anchor_file, ref_format, ref_labels_file, cmp_labels_file, overview_raw, dark):
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )

        if not overview_raw:
            return (
                "<div style='color:var(--ov-warn);padding:1em;text-align:center;'>"
                "Load a trajectory in the Overview tab first — it will be the compared trajectory.</div>",
                empty_fig,
                empty_fig,
                "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>"
                "Load a trajectory in the Overview tab first.</div>",
            )

        if ref_file is None:
            return (
                "<div style='color:var(--ov-warn);padding:1em;text-align:center;'>"
                "Upload a reference trajectory first.</div>",
                empty_fig,
                empty_fig,
                "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>"
                "Upload a reference trajectory.</div>",
            )

        ref_path = ref_file if isinstance(ref_file, str) else ref_file.name
        anchor_path = None
        if anchor_file is not None:
            anchor_path = anchor_file if isinstance(anchor_file, str) else anchor_file.name

        from ..loaders import load_trajectory as _load_traj

        ref_raw = _load_traj(ref_path, format_hint=ref_format or None)

        result = run_comparison(
            ref_raw=ref_raw,
            cmp_raw=overview_raw,
            anchor_path=anchor_path,
            token_rate=50.0,
            fuzzy=False,
            dark=bool(dark),
        )

        phase_count_fig = empty_fig
        phase_duration_fig = empty_fig
        ref_labels_path = (
            ref_labels_file
            if isinstance(ref_labels_file, str)
            else (ref_labels_file.name if ref_labels_file else None)
        )
        cmp_labels_path = (
            cmp_labels_file
            if isinstance(cmp_labels_file, str)
            else (cmp_labels_file.name if cmp_labels_file else None)
        )
        if ref_labels_path and cmp_labels_path:
            try:
                from ..charts import (
                    build_phase_count_comparison_chart,
                    build_phase_duration_comparison_chart,
                )
                from ..labels import aggregate_labels, load_labeled_json

                ref_agg = aggregate_labels(load_labeled_json(ref_labels_path))
                cmp_agg = aggregate_labels(load_labeled_json(cmp_labels_path))
                ref_label_name = os.path.basename(ref_path) if ref_path else "reference"
                cmp_label_name = (
                    os.path.basename(overview_raw.get("_source_path", ""))
                    if isinstance(overview_raw, dict)
                    else "compared"
                ) or "compared"
                phase_count_fig = build_phase_count_comparison_chart(
                    ref_agg["phase_counts"],
                    cmp_agg["phase_counts"],
                    ref_label=ref_label_name,
                    cmp_label=cmp_label_name,
                    dark=bool(dark),
                )
                phase_duration_fig = build_phase_duration_comparison_chart(
                    ref_agg["phase_durations"],
                    cmp_agg["phase_durations"],
                    ref_label=ref_label_name,
                    cmp_label=cmp_label_name,
                    dark=bool(dark),
                )
            except Exception as exc:
                import logging

                logging.getLogger(__name__).debug("Phase comparison chart build failed: %s", exc)

        if result.get("ok", True):
            status = "<div style='color:var(--ov-success);padding:0.5em;font-size:13px;'>Comparison complete.</div>"
        else:
            status = (
                "<div style='color:var(--ov-warn);padding:0.5em;font-size:13px;'>"
                "Comparison failed &mdash; see the report panel for details.</div>"
            )

        return (result["report_html"], phase_count_fig, phase_duration_fig, status)

    refs.cmp_run_btn.click(
        fn=on_run_comparison,
        inputs=[
            refs.cmp_file_upload,
            refs.cmp_anchor_upload,
            refs.cmp_format_selector,
            refs.cmp_ref_labels_upload,
            upload.label_file_upload,
            shared.state_raw,
            shared.state_dark,
        ],
        outputs=[refs.cmp_report_html, refs.cmp_phase_count_chart, refs.cmp_phase_duration_chart, refs.cmp_status_html],
    )
