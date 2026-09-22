"""Attribution tab: DECAF diagnosis callbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass

import gradio as gr

from ..loaders import detect_format
from .shared import SharedState
from .upload import UploadRefs


@dataclass
class AttributionRefs:
    attr_status_html: gr.HTML
    attr_run_btn: gr.Button
    attr_agent_override: gr.Textbox
    attr_inst_override: gr.Textbox
    attr_root_override: gr.Textbox
    attr_result_html: gr.HTML


def layout() -> AttributionRefs:
    with gr.TabItem("Attribution"):
        gr.HTML(
            "<div class='section-subtitle'>DECAF capability failure attribution "
            "&mdash; which of the seven workflow capabilities broke, with tiered "
            "evidence (deductive / associational / model-inferred). Gold-grounded: "
            "needs the reference patch + test outcome for this task.</div>"
        )
        _attr_placeholder = (
            "<div style='padding:2em;color:var(--ov-muted);text-align:center;font-size:14px;'>"
            "Load a trajectory in the Overview tab &mdash; diagnosis runs automatically on load (<b>Diagnose failure</b> re-runs it with the overrides below). "
            "For a corpus trajectory (…/trajectory/&lt;agent&gt;/&lt;instance&gt;.json) the agent "
            "and instance are auto-detected from the path; for an uploaded file, set them below."
            "</div>"
        )
        attr_status_html = gr.HTML(_attr_placeholder)
        with gr.Row(equal_height=True):
            attr_run_btn = gr.Button("Diagnose failure", variant="primary", size="sm", scale=1, min_width=140)
        with (
            gr.Accordion(
                "Override agent / instance / corpus (for uploaded files)",
                open=False,
                elem_classes=["per-message-acc"],
            ),
            gr.Row(equal_height=True),
        ):
            attr_agent_override = gr.Textbox(label="Agent", placeholder="auto-detected from path", scale=1)
            attr_inst_override = gr.Textbox(label="Instance id", placeholder="auto-detected from path", scale=2)
            attr_root_override = gr.Textbox(
                label="ARGUS corpus root", placeholder="default: sibling TraceProbe checkout", scale=2
            )
        attr_result_html = gr.HTML("")
    return AttributionRefs(
        attr_status_html=attr_status_html,
        attr_run_btn=attr_run_btn,
        attr_agent_override=attr_agent_override,
        attr_inst_override=attr_inst_override,
        attr_root_override=attr_root_override,
        attr_result_html=attr_result_html,
    )


def bind(refs: AttributionRefs, shared: SharedState, upload: UploadRefs, load_events) -> None:
    def on_diagnose(overview_raw, agent_override, inst_override, root_override):
        from dataclasses import asdict

        from .. import attribution as _attr
        from ..rendering import build_attribution_html

        if not overview_raw:
            return (
                build_attribution_html(
                    {"available": False, "reason": "Load a trajectory in the Overview tab first."}
                ),
                "<div style='color:var(--ov-warn);padding:0.5em;font-size:13px;'>No trajectory loaded.</div>",
            )

        src = overview_raw.get("_source_path", "") if isinstance(overview_raw, dict) else ""
        src_sha = overview_raw.get("_source_sha256") if isinstance(overview_raw, dict) else None
        agent = (agent_override or "").strip()
        inst = (inst_override or "").strip()
        root = (root_override or "").strip()
        if src:
            if not inst:
                inst = os.path.splitext(os.path.basename(src))[0]
            if not agent:
                agent = os.path.basename(os.path.dirname(src))

        fmt = None
        if isinstance(overview_raw, dict):
            detected = detect_format(overview_raw)
            fmt = None if detected == "unknown" else detected

        result = _attr.diagnose(
            agent=agent or None,
            instance_id=inst or None,
            source_path=src or None,
            fmt=fmt or None,
            expected_sha=src_sha,
            argus_root=root or None,
        )
        html_out = build_attribution_html(asdict(result))
        import html as _html

        ident = _html.escape(f"{result.agent}/{result.instance_id}")
        status = (
            "<div style='color:var(--ov-success);padding:0.5em;font-size:13px;'>"
            f"Diagnosis complete &mdash; {ident}.</div>"
            if result.available
            else "<div style='color:var(--ov-muted);padding:0.5em;font-size:13px;'>"
            "No gold-grounded attribution &mdash; see the note below.</div>"
        )
        return html_out, status

    refs.attr_run_btn.click(
        fn=on_diagnose,
        inputs=[shared.state_raw, refs.attr_agent_override, refs.attr_inst_override, refs.attr_root_override],
        outputs=[refs.attr_result_html, refs.attr_status_html],
        concurrency_id="attribution",
    )

    def _clear_attribution():
        return (
            "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
            "Diagnosing the loaded trajectory&hellip;</div>",
            "",
        )

    upload.load_btn.click(
        fn=_clear_attribution, outputs=[refs.attr_result_html, refs.attr_status_html], concurrency_id="attribution"
    )
    upload.file_upload.change(
        fn=_clear_attribution, outputs=[refs.attr_result_html, refs.attr_status_html], concurrency_id="attribution"
    )

    def on_diagnose_autoload(overview_raw, root_override):
        html_out, status = on_diagnose(overview_raw, "", "", root_override)
        return html_out, status, "", ""

    for _ev in load_events:
        _ev.then(
            fn=on_diagnose_autoload,
            inputs=[shared.state_raw, refs.attr_root_override],
            outputs=[refs.attr_result_html, refs.attr_status_html, refs.attr_agent_override, refs.attr_inst_override],
            concurrency_id="attribution",
        )
