"""Raw Data tab."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ..presenters.raw import raw_json_text
from ..session import LoadedSession


@dataclass
class RawRefs:
    raw_json: gr.Code


def layout() -> RawRefs:
    with gr.TabItem("Raw Data"):
        raw_json = gr.Code(
            label="Full trajectory JSON",
            language="json",
            value="",
            max_lines=50,
        )
    return RawRefs(raw_json=raw_json)


def load_slots(refs: RawRefs) -> dict:
    """Named Gradio components filled by the load packer."""
    return {"raw_json": refs.raw_json}


def pack_load(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict:
    """Named Raw Data values for a load (empty when *session* is None)."""
    del dark, banner
    if session is None:
        return {"raw_json": ""}
    return {"raw_json": raw_json_text(session.raw)}
