"""Bridge: run the catalog-bound DETECTOR_REGISTRY over a loaded trajectory.

The dashboard historically surfaced a handful of ad-hoc heuristics from
``patterns.py``. This module instead runs the *same* deterministic ``[S]``
anti-pattern detectors that back the research catalog (``core/catalog.py``) — the
ones covered by the ``tests/test_det_*`` golden tests — so what the UI shows and
what the catalog defines cannot drift apart.

Catalog detectors consume canonical actions (``converge.canonical``), so we
canonicalize the parsed steps, assign effect labels, build a
``DetectorContext`` (tool exposure / workspace files / optional semantic
labels), and run each detector *under its gating preconditions* — recording
"gated" (could not fire) distinctly from "clear" (did not fire).
"""

from __future__ import annotations

import html
from typing import Any

from trajectory_visualizer.core.catalog import by_band
from trajectory_visualizer.core.detection import DetectorContext

from ..converge.canonical import assign_effect_labels, canonicalize_steps
from .detectors import DETECTOR_REGISTRY

# Bands runnable without semantic labels. "[H]" detectors need the labeler and
# are added once label-threading through canonicalization is wired.
DEFAULT_BANDS: tuple[str, ...] = ("[S]",)


def _looks_like_path(target: str) -> bool:
    return bool(target) and ("/" in target or target.endswith(
        (".py", ".js", ".ts", ".md", ".txt", ".json", ".toml", ".cfg", ".yaml", ".yml")
    ))


def labels_from_labeled_json(data: dict) -> dict[int, dict[str, str]]:
    """Build a ``{step_index: {phase, action}}`` map from a loaded ``*_labeled.json``.

    The ``[H]`` semantic detectors read this directly off ``DetectorContext.labels``
    (keyed by step index), so the dashboard can pass it straight through.
    """
    out: dict[int, dict[str, str]] = {}
    for i, s in enumerate(data.get("steps", []) or []):
        if isinstance(s, dict):
            out[i] = {"phase": str(s.get("phase", "")), "action": str(s.get("action", ""))}
    return out


def _enrich_action_outputs(actions: list, steps: list[dict]) -> None:
    """Attach each action's tool *output* into ``args['output']``.

    ``canonicalize_steps`` carries only the tool *input* (query/pattern/path),
    so output-dependent detectors — notably ``empty-result-churn``, which checks
    whether a SEARCH returned nothing — cannot fire on real canonicalized data.
    We match by ``tool_call_id`` and inject the output here (in the dashboard
    bridge, leaving the shared/frozen canonicalization untouched).
    """
    out_by_id: dict[str, Any] = {}
    for s in steps:
        for tc in s.get("tool_calls", []):
            tid = tc.get("tool_id") or tc.get("tool_call_id") or tc.get("id")
            if tid:
                out_by_id[tid] = tc.get("output", "")
    for a in actions:
        tid = getattr(a, "tool_call_id", "")
        if tid and tid in out_by_id and isinstance(a.args, dict) and "output" not in a.args:
            a.args["output"] = out_by_id[tid]


def build_detector_context(
    steps: list[dict],
    labels: dict[int, dict[str, str]] | None = None,
) -> DetectorContext:
    """Build a DetectorContext from parsed steps.

    - tool_exposure: every tool name the agent invoked (drives tool/capability
      gating).
    - workspace_files: best-effort set of path-like targets referenced anywhere
      in the run (drives config-gating; we cannot see the real workspace, so a
      memory file is only "present" if the trajectory touched it).
    """
    tools: set[str] = set()
    files: set[str] = set()
    for s in steps:
        for tc in s.get("tool_calls", []):
            name = tc.get("tool_name") or tc.get("name")
            if name:
                tools.add(str(name))
    for s in steps:
        for tc in s.get("tool_calls", []):
            inp = tc.get("input") if isinstance(tc.get("input"), dict) else {}
            for key in ("file_path", "path", "filename", "file"):
                v = inp.get(key)
                if isinstance(v, str) and _looks_like_path(v):
                    files.add(v)
    return DetectorContext(
        tool_exposure=frozenset(tools),
        workspace_files=frozenset(files),
        labels=labels or {},
    )


def run_catalog_detectors(
    steps: list[dict],
    labels: dict[int, dict[str, str]] | None = None,
    bands: tuple[str, ...] = DEFAULT_BANDS,
) -> list[dict[str, Any]]:
    """Run every catalog detector in ``bands`` over ``steps``.

    Returns one record per detector:
      {id, name, band, phase, status: 'fired'|'clear'|'gated',
       reason: <gating reason or None>, detections: [{span, evidence}, ...]}
    """
    actions = canonicalize_steps(steps, labels or None)
    assign_effect_labels(actions, steps)
    _enrich_action_outputs(actions, steps)
    ctx = build_detector_context(steps, labels)

    out: list[dict[str, Any]] = []
    for band in bands:
        for rec in by_band(band):
            if rec.id not in DETECTOR_REGISTRY:
                continue
            ok, reason = ctx.gating_satisfied(rec.id)
            if not ok:
                out.append({
                    "id": rec.id, "name": rec.name, "band": rec.band,
                    "phase": rec.phase, "status": "gated", "reason": reason,
                    "detections": [],
                })
                continue
            dets = DETECTOR_REGISTRY[rec.id](actions, ctx)
            out.append({
                "id": rec.id, "name": rec.name, "band": rec.band,
                "phase": rec.phase,
                "status": "fired" if dets else "clear", "reason": None,
                "detections": [
                    {"span": list(d.span), "evidence": dict(d.evidence)} for d in dets
                ],
            })
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    """Return {fired, clear, gated, total_detections} counts."""
    fired = sum(1 for r in results if r["status"] == "fired")
    clear = sum(1 for r in results if r["status"] == "clear")
    gated = sum(1 for r in results if r["status"] == "gated")
    total = sum(len(r["detections"]) for r in results)
    return {"fired": fired, "clear": clear, "gated": gated, "total_detections": total}


def render_catalog_detectors_html(results: list[dict[str, Any]]) -> str:
    """Render catalog-detector results as a dashboard panel (fired + gated)."""
    if not results:
        return ("<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                "Load a trajectory to run the catalog detectors.</div>")
    s = summarize(results)
    head = (
        f"<div style='font-size:12px;color:var(--ov-muted);margin-bottom:8px;'>"
        f"{s['fired']} detector(s) fired &middot; {s['total_detections']} detection(s) &middot; "
        f"{s['clear']} clear &middot; {s['gated']} could not fire (gated)</div>"
    )

    fired = [r for r in results if r["status"] == "fired"]
    rows = []
    for r in fired:
        spans = ", ".join(
            f"#{d['span'][0]}–{d['span'][1]}" if d['span'][0] != d['span'][1]
            else f"#{d['span'][0]}"
            for d in r["detections"][:8]
        )
        if len(r["detections"]) > 8:
            spans += f" …(+{len(r['detections']) - 8})"
        rows.append(
            f"<tr style='border-bottom:1px solid var(--ov-border);'>"
            f"<td style='padding:6px 10px;font-weight:600;'>{html.escape(r['name'])}</td>"
            f"<td style='padding:6px 10px;font-family:monospace;font-size:11px;color:var(--ov-muted);'>{html.escape(r['id'])}</td>"
            f"<td style='padding:6px 10px;font-size:11px;'>{html.escape(str(r['phase'] or '—'))}</td>"
            f"<td style='padding:6px 10px;text-align:center;font-weight:700;'>{len(r['detections'])}</td>"
            f"<td style='padding:6px 10px;font-size:11px;color:var(--ov-muted);'>{spans}</td>"
            f"</tr>"
        )
    table = ""
    if rows:
        table = (
            "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<tr style='background:var(--ov-table-header-bg);'>"
            "<th style='text-align:left;padding:6px 10px;'>Anti-pattern</th>"
            "<th style='text-align:left;padding:6px 10px;'>Detector id</th>"
            "<th style='text-align:left;padding:6px 10px;'>Phase</th>"
            "<th style='text-align:center;padding:6px 10px;'>Count</th>"
            "<th style='text-align:left;padding:6px 10px;'>Spans</th></tr>"
            + "".join(rows) + "</table>"
        )
    else:
        table = ("<div style='padding:0.5em 1em;color:var(--ov-muted);'>"
                 "No catalog anti-patterns fired.</div>")

    gated = [r for r in results if r["status"] == "gated"]
    gated_html = ""
    if gated:
        items = "".join(
            f"<li><code>{html.escape(g['id'])}</code> &mdash; {html.escape(g['reason'] or '')}</li>"
            for g in gated
        )
        gated_html = (
            f"<details style='margin-top:8px;font-size:12px;color:var(--ov-muted);'>"
            f"<summary>{len(gated)} detector(s) could not fire (gating preconditions unmet)</summary>"
            f"<ul style='margin:6px 0 0 1em;'>{items}</ul></details>"
        )
    return head + table + gated_html
