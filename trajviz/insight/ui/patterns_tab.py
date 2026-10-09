"""Patterns tab layout."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ..presenters.patterns import (
    build_antipattern_html,
    render_failure_patterns_html,
    render_tool_sequences_html,
)
from ..session import LoadedSession


@dataclass
class PatternsRefs:
    patterns_tool_html: gr.HTML
    patterns_failure_html: gr.HTML
    antipattern_summary_html: gr.HTML


def layout() -> PatternsRefs:
    with gr.TabItem("Patterns"):
        gr.HTML(
            "<div class='section-subtitle'>Recurring patterns detected in the trajectory — tool sequences, failure clusters, and phase transition anomalies.</div>"
        )
        with gr.Accordion("Tool Sequence Patterns", open=True, elem_classes=["per-message-acc"]):
            patterns_tool_html = gr.HTML(
                "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>Load a trajectory to detect tool sequence patterns.</div>"
            )
        with gr.Accordion("Failure Patterns", open=True, elem_classes=["per-message-acc"]):
            patterns_failure_html = gr.HTML(
                "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>Load a trajectory to detect failure patterns.</div>"
            )
        with gr.Accordion("Anti-Pattern Summary", open=True, elem_classes=["per-message-acc"]):
            antipattern_summary_html = gr.HTML(
                "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>Load a trajectory to detect anti-patterns.</div>"
            )
    return PatternsRefs(
        patterns_tool_html=patterns_tool_html,
        patterns_failure_html=patterns_failure_html,
        antipattern_summary_html=antipattern_summary_html,
    )


def load_slots(refs: PatternsRefs) -> dict:
    """Named Gradio components filled by the load packer."""
    return {
        "patterns_tool_html": refs.patterns_tool_html,
        "patterns_failure_html": refs.patterns_failure_html,
        "antipattern_summary_html": refs.antipattern_summary_html,
    }


def pack_load(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict:
    """Named Patterns values for a load (empty when *session* is None)."""
    del dark, banner
    if session is None:
        return {
            "patterns_tool_html": "",
            "patterns_failure_html": "",
            "antipattern_summary_html": "",
        }
    return {
        "patterns_tool_html": render_tool_sequences_html(session.tool_sequences),
        "patterns_failure_html": render_failure_patterns_html(session.failure_patterns),
        "antipattern_summary_html": build_antipattern_html(session),
    }
