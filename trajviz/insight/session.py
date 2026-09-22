"""Gradio-free load pipeline: ingest a trajectory into a LoadedSession DTO."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from .analytics import compute_step_analytics
from .context_usage import (
    PRESSURE_ALL_AGENTS,
    context_pressure_series,
    detect_premature_compactions,
    pressure_agent_choices,
)
from .diagnostics import (
    annotate_clusters_with_agents,
    cluster_errors,
    compute_failure_chain_metrics,
    detect_failure_chains,
    detect_performance_bottlenecks,
    extract_file_interactions,
    identify_target_files,
    link_chains_to_agents,
)
from .loaders import check_format_selection, detect_format, load_trajectory
from .formatting import wall_clock_fmt
from .metrics import (
    build_message_metrics,
    compute_agent_summary,
    compute_diagnostic_metrics,
    compute_health_verdict,
    compute_metrics,
)
from .parser import parse_steps
from .patterns import (
    build_structural_phase_segments,
    compute_plan_metrics,
    detect_edit_thrash,
    detect_failure_patterns,
    detect_fruitless_streaks,
    detect_phase_anomalies,
    detect_repeated_searches,
    detect_tool_selection_antipatterns,
    detect_tool_sequences,
    extract_plan_history,
)

# Maximum steps to process — keeps rendering, metrics, and charts bounded.
MAX_STEPS = 2000

LoadErrorCode = Literal["not_found", "mismatch", "unknown", "read_error"]

_NOISY_ROOT_CAUSE_FORMATS = frozenset({"opencode", "codearts", "codex"})


@dataclass(frozen=True)
class LoadError:
    """Failed ingest. Messages are plain text — the UI packer wraps HTML."""

    code: LoadErrorCode
    message: str
    selected: str | None = None
    detected: str | None = None


@dataclass
class LoadedSession:
    """Domain snapshot of a successfully loaded trajectory (no HTML / Plotly / Gradio)."""

    path: str
    raw: dict
    steps: list[dict]
    steps_total: int
    format: str
    message_rows: list[dict]
    metrics: dict
    verdicts: list[dict]
    agent_summaries: list[dict]
    step_analytics: list[dict]
    wall_clock: str
    tool_sequences: list[dict]
    failure_patterns: list[dict]
    file_interactions: list[dict]
    target_files: set[str]
    failure_chains: list
    chain_metrics: dict
    clusters: list
    performance_bottlenecks: list
    diagnostic_metrics: dict
    pressure_series: dict
    pressure_choices: list
    show_root_cause: bool
    plan_history: list
    plan_metrics: dict
    fruitless_streaks: list
    tool_selection: list
    edit_thrash: list
    repeated_searches: list
    phase_regressions: list
    premature_compactions: list
    truncated: bool = False


def _path_exists(file_path: str) -> bool:
    return os.path.isfile(file_path) or (
        os.path.isdir(file_path) and os.path.isfile(os.path.join(file_path, "session.jsonl"))
    )


def load_session(path: str, format_hint: str = "") -> LoadedSession | LoadError:
    """Load and analyze a trajectory file. Does not build HTML, Plotly, or Gradio updates."""
    if not path or not _path_exists(path):
        return LoadError(code="not_found", message="No file selected or file not found.")

    hint = format_hint or None
    raw = load_trajectory(path, format_hint=hint)
    if raw.get("_error_code") == "mismatch":
        return LoadError(
            code="mismatch",
            message=str(raw.get("_error") or "Format mismatch."),
            selected=raw.get("_selected") or format_hint or None,
            detected=raw.get("_detected") or None,
        )
    if "_error" in raw:
        return LoadError(code="read_error", message=str(raw["_error"]))

    detected = detect_format(raw)
    gate = check_format_selection(detected, format_hint)
    if gate == "unknown":
        return LoadError(
            code="unknown",
            message=(
                "Could not detect trajectory format. "
                "Select a format from the dropdown and try again."
            ),
        )
    if gate == "mismatch":
        return LoadError(
            code="mismatch",
            message=(
                f"Format mismatch: selected {format_hint} but file detected as {detected}."
            ),
            selected=format_hint or None,
            detected=detected,
        )

    return build_loaded_session(path, raw, detected=detected)


def build_loaded_session(path: str, raw: dict, *, detected: str | None = None) -> LoadedSession:
    """Analyze already-loaded *raw* into a ``LoadedSession`` (no file I/O)."""
    detected = detected if detected is not None else detect_format(raw)
    steps = parse_steps(raw)
    steps_total = len(steps)
    truncated = False
    if steps_total > MAX_STEPS:
        steps = steps[:MAX_STEPS]
        truncated = True

    message_rows = build_message_metrics(steps)
    metrics = compute_metrics(steps, raw, message_rows=message_rows)
    _, wfmt = wall_clock_fmt(metrics)

    step_analytics = compute_step_analytics(steps)
    verdicts = compute_health_verdict(metrics, step_analytics if steps else [])
    agent_summaries = compute_agent_summary(steps, raw)

    interactions = extract_file_interactions(steps)
    target_files = identify_target_files(steps)
    chains = detect_failure_chains(steps)
    chains = link_chains_to_agents(chains, steps, agent_summaries)
    chain_metrics = compute_failure_chain_metrics(
        chains, sum(1 for s in steps if s.get("role") == "assistant")
    )
    clusters = cluster_errors(steps)
    clusters = annotate_clusters_with_agents(clusters, steps, agent_summaries)
    performance_bottlenecks = detect_performance_bottlenecks(steps, step_analytics)
    traj = raw.get("trajectory") or raw.get("messages") or []
    diagnostic_metrics = compute_diagnostic_metrics(
        steps,
        traj if isinstance(traj, list) else [],
        tool_fail=metrics.get("tool_fail"),
    )
    pressure_series = context_pressure_series(
        steps,
        agent_key=PRESSURE_ALL_AGENTS,
        raw=raw,
    )
    pressure_choices = pressure_agent_choices(steps)
    plan_history = extract_plan_history(steps)
    plan_metrics = compute_plan_metrics(plan_history)
    structural_phases = build_structural_phase_segments(steps)
    phase_regressions = [
        a for a in detect_phase_anomalies(steps, structural_phases)
        if a.get("category") == "unintentional_drift"
    ]

    return LoadedSession(
        path=path,
        raw=raw,
        steps=steps,
        steps_total=steps_total,
        format=detected,
        message_rows=message_rows,
        metrics=metrics,
        verdicts=verdicts,
        agent_summaries=agent_summaries,
        step_analytics=step_analytics,
        wall_clock=wfmt,
        tool_sequences=detect_tool_sequences(steps),
        failure_patterns=detect_failure_patterns(steps),
        file_interactions=interactions,
        target_files=target_files,
        failure_chains=chains,
        chain_metrics=chain_metrics,
        clusters=clusters,
        performance_bottlenecks=performance_bottlenecks,
        diagnostic_metrics=diagnostic_metrics,
        pressure_series=pressure_series,
        pressure_choices=pressure_choices,
        show_root_cause=detected not in _NOISY_ROOT_CAUSE_FORMATS,
        plan_history=plan_history,
        plan_metrics=plan_metrics,
        fruitless_streaks=detect_fruitless_streaks(steps),
        tool_selection=detect_tool_selection_antipatterns(steps),
        edit_thrash=detect_edit_thrash(steps),
        repeated_searches=detect_repeated_searches(steps),
        phase_regressions=phase_regressions,
        premature_compactions=detect_premature_compactions(steps, raw),
        truncated=truncated,
    )
