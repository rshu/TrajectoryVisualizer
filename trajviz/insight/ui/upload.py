"""Upload row, title, and load warnings."""

from __future__ import annotations

import os
import shutil
import traceback
from dataclasses import dataclass

import gradio as gr

from ..loaders import FORMAT_DROPDOWN_CHOICES
from ..presenters.overview import load_warnings_html
from ..report import ReportError, write_report_file
from ..session import LoadedSession
from .shared import SharedState


@dataclass
class UploadRefs:
    upload_accordion: gr.Column
    format_selector: gr.Dropdown
    file_upload: gr.File
    load_btn: gr.Button
    export_btn: gr.DownloadButton
    label_file_upload: gr.File
    label_load_btn: gr.Button
    summary_area: gr.Column
    summary_banner: gr.HTML
    label_badge_html: gr.HTML


def layout() -> UploadRefs:
    gr.HTML(
        "<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:4px;'>"
        "<span style='font-size:20px;font-weight:700;letter-spacing:-0.02em;'>TrajViz</span>"
        "<span style='font-size:12px;color:var(--ov-muted);'>Coding Agent Trajectory Analysis Dashboard</span>"
        "</div>"
    )

    with gr.Column(elem_classes=["upload-row"]) as upload_accordion, gr.Row(equal_height=True):
        format_selector = gr.Dropdown(
            label="Format",
            choices=FORMAT_DROPDOWN_CHOICES,
            value="",
            interactive=True,
            scale=1,
            min_width=140,
        )
        with gr.Column(scale=2, min_width=200):
            file_upload = gr.File(
                label="Trajectory (.json / .jsonl / .zip)",
                file_types=[".json", ".jsonl", ".zip"],
                height=110,
            )
            with gr.Row():
                load_btn = gr.Button("Load Trajectory", variant="primary", size="sm", min_width=120)
                export_btn = gr.DownloadButton(
                    label="Export HTML",
                    variant="secondary",
                    size="sm",
                    min_width=120,
                    interactive=False,
                )
        with gr.Column(scale=2, min_width=200):
            label_file_upload = gr.File(
                label="Labels (optional)",
                file_types=[".json"],
                height=110,
            )
            label_load_btn = gr.Button("Load Labels", variant="secondary", size="sm", min_width=120)

    with gr.Column(visible=False) as summary_area:
        summary_banner = gr.HTML("", elem_classes=["summary-banner"])
        label_badge_html = gr.HTML("")

    return UploadRefs(
        upload_accordion=upload_accordion,
        format_selector=format_selector,
        file_upload=file_upload,
        load_btn=load_btn,
        export_btn=export_btn,
        label_file_upload=label_file_upload,
        label_load_btn=label_load_btn,
        summary_area=summary_area,
        summary_banner=summary_banner,
        label_badge_html=label_badge_html,
    )


def load_slots(refs: UploadRefs) -> dict:
    """Named Gradio components filled by the load packer."""
    return {
        "summary_area": refs.summary_area,
        "upload_accordion": refs.upload_accordion,
        "summary_banner": refs.summary_banner,
        "export_btn": refs.export_btn,
    }


_TEMP_EXPORT_PREFIX = "trajviz-report-"
_last_temp_export_dir: str | None = None


def _export_off():
    return gr.update(value=None, interactive=False)


def _replace_temp_export_dir(new_path: str) -> None:
    """Drop the previous mkdtemp export dir after a replacement file is written."""
    global _last_temp_export_dir
    parent = os.path.dirname(os.path.abspath(new_path))
    previous = _last_temp_export_dir
    if os.path.basename(parent).startswith(_TEMP_EXPORT_PREFIX):
        _last_temp_export_dir = parent
    if previous and previous != parent and os.path.basename(previous).startswith(_TEMP_EXPORT_PREFIX):
        shutil.rmtree(previous, ignore_errors=True)


def prepare_html_export(raw, _steps=None, dark=False):
    """Build the HTML snapshot after a trajectory loads.

    Gradio's DownloadButton only saves in the same click that already has a
    file URL. Generating on click (then JS-clicking a hidden button) is
    blocked as a non-gesture download, and ``value=callable`` toasts on page
    load. So the report is prepared here and the button just downloads.
    """
    payload = raw if isinstance(raw, dict) else {}
    if not payload or payload.get("_error"):
        return _export_off()
    try:
        path = write_report_file(payload, dark=bool(dark))
    except ReportError:
        return _export_off()
    except Exception:
        traceback.print_exc()
        return _export_off()
    _replace_temp_export_dir(path)
    return gr.update(value=path, interactive=True)


def bind_export(refs: UploadRefs, shared: SharedState, load_events: tuple) -> None:
    """Enable Export HTML after a successful load (not as DownloadButton value=)."""
    for event in load_events:
        event.success(
            fn=prepare_html_export,
            inputs=[shared.state_raw, shared.state_steps, shared.state_dark],
            outputs=refs.export_btn,
            show_progress="minimal",
            show_progress_on=refs.export_btn,
        )


def pack_load(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict:
    """Named upload-row values for a load (error banner when *session* is None).

    Export is always disarmed here so a reload cannot download the previous
    file; ``prepare_html_export`` re-enables the button after a successful load.
    """
    del dark
    export_off = _export_off()
    if session is None:
        has_banner = bool(banner)
        return {
            "summary_area": gr.update(visible=has_banner),
            "upload_accordion": gr.update(),
            "summary_banner": gr.update(value=banner, visible=has_banner),
            "export_btn": export_off,
        }
    warnings = load_warnings_html(session)
    return {
        # Keep area visible so label badges can appear after Load Labels.
        "summary_area": gr.update(visible=True),
        "upload_accordion": gr.update(),
        "summary_banner": gr.update(value=warnings, visible=bool(warnings.strip())),
        "export_btn": export_off,
    }
