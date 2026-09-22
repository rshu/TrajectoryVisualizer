"""Labels section payload: phase/action charts and the upload badge."""

from __future__ import annotations

import html

from ..charts import (
    build_label_action_count_chart,
    build_label_action_duration_chart,
    build_label_phase_count_chart,
    build_label_phase_duration_chart,
    build_label_timeline_chart,
)
from ..labels import aggregate_labels, load_labeled_json
from ..palette import LABEL_PHASE_COLORS


def build_label_ui_payload(file_path: str, dark: bool = False) -> dict:
    """Build UI-facing label payload for a *_labeled.json label file."""
    data = load_labeled_json(file_path)
    agg = aggregate_labels(data)
    pc_fig = build_label_phase_count_chart(agg["phase_counts"], dark=dark)
    ac_fig = build_label_action_count_chart(agg["action_counts"], agg["action_to_phase"], dark=dark)
    pd_fig = build_label_phase_duration_chart(agg["phase_durations"], dark=dark)
    ad_fig = build_label_action_duration_chart(agg["action_durations"], agg["action_to_phase"], dark=dark)
    tl_fig = build_label_timeline_chart(agg["steps"], dark=dark)

    n_steps = len(agg.get("steps", []))
    phase_counts = agg.get("phase_counts", {})
    n_phases = len(phase_counts)

    bar_segments = "".join(
        f"<span style='flex:{count};background:{LABEL_PHASE_COLORS.get(phase, '#6b7280')};height:8px;'"
        f" title='{html.escape(str(phase))}: {count}'></span>"
        for phase, count in phase_counts.items()
        if count > 0
    )
    phase_bar = (
        (
            "<div style='display:flex;border-radius:4px;overflow:hidden;width:200px;margin-top:4px;'>"
            f"{bar_segments}</div>"
        )
        if bar_segments
        else ""
    )

    phase_chips = "".join(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;"
        f"background:{LABEL_PHASE_COLORS.get(phase, '#6b7280')};color:white;'>"
        f"{html.escape(str(phase))}: {count}</span>"
        for phase, count in phase_counts.items()
        if count > 0
    )

    badge = (
        "<div style='display:flex;flex-direction:column;gap:6px;margin:6px 0;'>"
        "<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
        "<span style='background:#059669;color:white;"
        "padding:4px 12px;border-radius:12px;font-size:13px;'>"
        f"Labels loaded — {n_steps} steps, {n_phases} phases</span>"
        "<a href='#' onclick=\"var t=document.querySelectorAll('button[role=tab]');"
        "for(var i=0;i<t.length;i++){if(t[i].textContent.trim()==='Overview'){t[i].click();break;}}"
        "var r=document.querySelector(&quot;.overview-section-radio input[value='Labels']&quot;);"
        "if(r){r.click();r.scrollIntoView({behavior:'smooth',block:'center'});}"
        "return false;\" style='font-size:12px;color:#059669;text-decoration:underline;cursor:pointer;'>"
        "Jump to Labels</a>"
        "</div>"
        f"<div style='display:flex;gap:4px;flex-wrap:wrap;'>{phase_chips}</div>"
        f"{phase_bar}"
        "</div>"
    )
    return {
        "badge_html": badge,
        "status_html": "",
        "phase_count_fig": pc_fig,
        "action_count_fig": ac_fig,
        "phase_duration_fig": pd_fig,
        "action_duration_fig": ad_fig,
        "timeline_fig": tl_fig,
    }
