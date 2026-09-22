"""Shared Gradio state passed into tab layout/bind helpers."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr


@dataclass
class SharedState:
    state_steps: gr.State
    state_raw: gr.State
    state_dark: gr.State
    state_analysis_brief: gr.State
