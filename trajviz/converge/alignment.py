"""Trajectory alignment: greedy forward-match, P/R/F1, harmful divergence."""

from __future__ import annotations

import os
import re

from .canonical import (
    CanonicalAction, compute_action_cost, semantic_equivalent,
    canonicalize_steps, assign_effect_labels, DEFAULT_TOKEN_RATE,
)


def _describe_format(raw: dict) -> str:
    """Human-readable format name for a loaded trajectory (best-effort)."""
    from trajviz.insight.loaders import detect_format, FORMAT_LABELS
    fmt = detect_format(raw) if isinstance(raw, dict) else "unknown"
    return FORMAT_LABELS.get(fmt, fmt or "unknown")


def _session_duration_s(raw: dict) -> float | None:
    """Session wall-clock duration in seconds from timing.started_at / finished_at."""
    if not isinstance(raw, dict):
        return None
    timing = raw.get("timing") if isinstance(raw.get("timing"), dict) else {}
    started, finished = timing.get("started_at"), timing.get("finished_at")
    if not (isinstance(started, str) and isinstance(finished, str) and started and finished):
        return None
    try:
        from datetime import datetime
        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
        f = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        delta = (f - s).total_seconds()
        return delta if delta >= 0 else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Greedy forward-match alignment
# ---------------------------------------------------------------------------

def align_trajectories(
    reference: list[CanonicalAction],
    compared: list[CanonicalAction],
    fuzzy_commands: bool = False,
) -> dict:
    """Longest common subsequence alignment with effect_label compatibility.

    Uses dynamic programming to find the maximum-weight monotonic matching:
    matched pairs (i1,j1), (i2,j2), ... satisfy i1<i2 and j1<j2, preserving
    trajectory order in both sequences. This measures trajectory convergence
    (ordered behavioral similarity), not unordered behavioral overlap.

    Returns {matched_pairs, unrecovered, extra} where:
    - matched_pairs: list of (ref_idx, cmp_idx) tuples (monotonically ordered)
    - unrecovered: list of ref indices with no match
    - extra: list of cmp indices not matched
    """
    # Filter out REASON actions for alignment
    ref_non_reason = [(i, a) for i, a in enumerate(reference) if a.action_type != "REASON"]
    cmp_non_reason = [(j, a) for j, a in enumerate(compared) if a.action_type != "REASON"]

    n = len(ref_non_reason)
    m = len(cmp_non_reason)

    if n == 0 or m == 0:
        return {
            "matched_pairs": [],
            "unrecovered": [i for i, _ in ref_non_reason],
            "extra": [j for j, _ in cmp_non_reason],
        }

    # Precompute match matrix to avoid redundant semantic_equivalent calls
    match = [[False] * m for _ in range(n)]
    for i in range(n):
        ref_action = ref_non_reason[i][1]
        for j in range(m):
            cmp_action = cmp_non_reason[j][1]
            match[i][j] = semantic_equivalent(ref_action, cmp_action, fuzzy_commands)

    # DP table: dp[i][j] = max matches using ref_non_reason[:i] and cmp_non_reason[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if match[i - 1][j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to recover matched pairs (reuses precomputed match matrix)
    matched_pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if match[i - 1][j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            ref_idx = ref_non_reason[i - 1][0]
            cmp_idx = cmp_non_reason[j - 1][0]
            matched_pairs.append((ref_idx, cmp_idx))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    matched_pairs.reverse()  # backtrack produces reverse order

    matched_ref = {p[0] for p in matched_pairs}
    matched_cmp = {p[1] for p in matched_pairs}
    unrecovered = [i for i, _ in ref_non_reason if i not in matched_ref]
    extra = [j for j, _ in cmp_non_reason if j not in matched_cmp]

    return {
        "matched_pairs": matched_pairs,
        "unrecovered": unrecovered,
        "extra": extra,
    }


# ---------------------------------------------------------------------------
# Alignment metrics
# ---------------------------------------------------------------------------

def compute_alignment_metrics(
    alignment: dict,
    reference: list[CanonicalAction],
    compared: list[CanonicalAction],
    token_rate: float = DEFAULT_TOKEN_RATE,
) -> dict:
    """Compute reference_recall, behavioral_precision, F1, overhead_ratio."""
    ref_non_reason = [a for a in reference if a.action_type != "REASON"]
    cmp_non_reason = [a for a in compared if a.action_type != "REASON"]

    reference_weight = sum(compute_action_cost(a, token_rate) for a in ref_non_reason)
    compared_weight = sum(compute_action_cost(a, token_rate) for a in cmp_non_reason)

    matched_ref_weight = sum(
        compute_action_cost(reference[i], token_rate)
        for i, _ in alignment["matched_pairs"]
    )
    matched_cmp_weight = sum(
        compute_action_cost(compared[j], token_rate)
        for _, j in alignment["matched_pairs"]
    )

    # Fall back to count-based metrics when token weights are 0
    # (e.g., Codex trajectories which lack per-action token data).
    # The fallback is JOINT (B24): if either side lacks cost data, BOTH sides
    # switch to counts, so overhead_ratio and P/R/F1 never divide an action
    # count by a token weight (mixed units made them meaningless).
    if ((reference_weight == 0 and ref_non_reason)
            or (compared_weight == 0 and cmp_non_reason)):
        reference_weight = len(ref_non_reason)
        compared_weight = len(cmp_non_reason)
        matched_ref_weight = sum(1 for i, _ in alignment["matched_pairs"]
                                 if reference[i].action_type != "REASON")
        matched_cmp_weight = sum(1 for _, j in alignment["matched_pairs"]
                                 if compared[j].action_type != "REASON")

    recall = matched_ref_weight / reference_weight if reference_weight > 0 else 0.0
    precision = matched_cmp_weight / compared_weight if compared_weight > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    overhead = compared_weight / reference_weight if reference_weight > 0 else 0.0

    return {
        "reference_recall": round(recall, 4),
        "behavioral_precision": round(precision, 4),
        "alignment_f1": round(f1, 4),
        "overhead_ratio": round(overhead, 4),
    }


# ---------------------------------------------------------------------------
# Harmful divergence
# ---------------------------------------------------------------------------

def compute_harmful_divergence(
    extra_indices: list[int],
    compared: list[CanonicalAction],
    token_rate: float = DEFAULT_TOKEN_RATE,
    dead_end_steps: set[int] | None = None,
) -> dict:
    """Compute harmful_cost and harmful_ratio from extra actions.

    Includes failed, reverted, and dead_end_branch actions per the spec.
    """
    cmp_non_reason = [a for a in compared if a.action_type != "REASON"]
    compared_weight = sum(compute_action_cost(a, token_rate) for a in cmp_non_reason)
    dead_end_steps = dead_end_steps or set()

    harmful_cost_tokens = 0
    harmful_cost_latency = 0
    for idx in extra_indices:
        if idx < len(compared):
            a = compared[idx]
            if (a.effect_label in ("failed", "reverted")
                    or a.step_index in dead_end_steps):
                harmful_cost_tokens += a.cost.token_share
                harmful_cost_latency += a.cost.latency_ms

    harmful_scalar = harmful_cost_tokens + (harmful_cost_latency / 1000.0 * token_rate)
    harmful_ratio = harmful_scalar / compared_weight if compared_weight > 0 else 0.0

    return {
        "harmful_ratio": round(min(harmful_ratio, 1.0), 4),
        "harmful_cost": {
            "tokens": harmful_cost_tokens,
            "latency_ms": harmful_cost_latency,
        },
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def _parse_anchor_files(patch_path: str | None) -> set[str] | None:
    """Extract anchor file paths from a unified diff's ``a/``/``b/`` headers.

    Returns ``None`` (un-anchored) when the patch is missing/unreadable — and
    also when it contains no recognizable ``a/ b/`` headers (B25): an empty
    set would silently empty the target-file grounding in assign_effect_labels
    while every downstream ``if anchor_files:`` check still reported
    anchor_mode='self'. Shared by build_comparison_report and the Insight UI's
    run_comparison so the two entry points cannot drift.
    """
    if not patch_path:
        return None
    try:
        with open(patch_path) as f:
            content = f.read()
    except Exception:
        return None
    anchor_files = set(re.findall(r'^[+-]{3}\s+[ab]/(.+)$', content, re.MULTILINE))
    return anchor_files or None


def build_comparison_report(
    ref_file: str,
    cmp_file: str,
    token_rate: float = DEFAULT_TOKEN_RATE,
    fuzzy_commands: bool = False,
    anchor_patch: str | None = None,
    task_id: str = "",
    ref_labels: dict[int, dict[str, str]] | None = None,
    cmp_labels: dict[int, dict[str, str]] | None = None,
) -> dict:
    """Load, canonicalize, align, and produce the full comparison report.

    Thin file-path entry point: loads/parses both trajectories and the anchor
    patch, then delegates to :func:`build_comparison_report_from_steps` (which
    the Insight UI's run_comparison also calls with pre-loaded data).

    Args:
        ref_labels, cmp_labels: Optional step-label mappings from the step
            labeler. When provided, CanonicalActions carry phase/action labels
            and divergence confidence scoring is phase-aware.

    Raises:
        ValueError: if either trajectory fails to load *or yields no steps*. A
            file that is missing, unreadable, unparseable or simply not a
            trajectory must never become a zero-valued report: ``run_batch``
            would record it as a success and fold its fabricated zeros into
            cross-task aggregate statistics. Note that an unrecognised JSON
            *object* is returned by ``load_trajectory`` with no ``_error`` at
            all, so checking ``_error`` alone is not enough — it is the
            empty-steps check that closes that door. The Insight UI's
            ``run_comparison`` already refuses these (``ok: False``); this is
            the same guard for the file-path entry point.
    """
    from trajviz.insight.loaders import load_trajectory
    from trajviz.insight.parser import parse_steps

    ref_raw = load_trajectory(ref_file)
    cmp_raw = load_trajectory(cmp_file)
    ref_steps = parse_steps(ref_raw)
    cmp_steps = parse_steps(cmp_raw)
    for label, raw, path, steps in (
        ("reference", ref_raw, ref_file, ref_steps),
        ("compared", cmp_raw, cmp_file, cmp_steps),
    ):
        if isinstance(raw, dict) and raw.get("_error"):
            raise ValueError(f"Could not load {label} trajectory {path!r}: {raw['_error']}")
        if not steps:
            fmt = (raw.get("_format") or raw.get("_detected") or "unrecognised") if isinstance(raw, dict) else "unrecognised"
            raise ValueError(
                f"Could not load {label} trajectory {path!r}: no steps parsed "
                f"(detected format: {fmt}) — refusing to report a zero-valued comparison."
            )

    return build_comparison_report_from_steps(
        ref_raw, cmp_raw, ref_steps, cmp_steps,
        token_rate=token_rate,
        fuzzy_commands=fuzzy_commands,
        anchor_files=_parse_anchor_files(anchor_patch),
        ref_path=ref_file,
        cmp_path=cmp_file,
        task_id=task_id,
        ref_labels=ref_labels,
        cmp_labels=cmp_labels,
    )


def build_comparison_report_from_steps(
    ref_raw: dict,
    cmp_raw: dict,
    ref_steps: list[dict],
    cmp_steps: list[dict],
    *,
    token_rate: float = DEFAULT_TOKEN_RATE,
    fuzzy_commands: bool = False,
    anchor_files: set[str] | None = None,
    ref_path: str = "",
    cmp_path: str = "",
    task_id: str = "",
    ref_labels: dict[int, dict[str, str]] | None = None,
    cmp_labels: dict[int, dict[str, str]] | None = None,
) -> dict:
    """Core comparison pipeline over already-loaded trajectories (R21).

    Single implementation shared by build_comparison_report (file paths) and
    trajviz.insight.comparison.run_comparison (pre-loaded dicts).
    """
    from .milestones import (
        extract_milestones, compute_milestone_deltas,
        segment_by_milestones, compare_segments,
    )
    from .divergence import classify_divergences, compute_pattern_costs

    # Canonicalize (attach phase/action labels when available)
    ref_actions = canonicalize_steps(ref_steps, step_labels=ref_labels)
    cmp_actions = canonicalize_steps(cmp_steps, step_labels=cmp_labels)

    # Effect labeling
    assign_effect_labels(ref_actions, ref_steps, anchor_files)
    assign_effect_labels(cmp_actions, cmp_steps, anchor_files)

    # Layer 1: Outcome
    ref_success = _detect_success(ref_steps)
    cmp_success = _detect_success(cmp_steps)
    ref_tokens = sum(s["tokens"]["total"] for s in ref_steps)
    cmp_tokens = sum(s["tokens"]["total"] for s in cmp_steps)

    outcome = {
        "reference_success": ref_success,
        "compared_success": cmp_success,
        "reference_steps": len(ref_steps),
        "compared_steps": len(cmp_steps),
        "reference_tokens": ref_tokens,
        "compared_tokens": cmp_tokens,
        "reference_filename": os.path.basename(ref_path) if ref_path else "",
        "compared_filename": os.path.basename(cmp_path) if cmp_path else "",
        "reference_format": _describe_format(ref_raw),
        "compared_format": _describe_format(cmp_raw),
        "reference_duration_s": _session_duration_s(ref_raw),
        "compared_duration_s": _session_duration_s(cmp_raw),
        "reference_tool_calls": sum(len(s.get("tool_calls", [])) for s in ref_steps),
        "compared_tool_calls": sum(len(s.get("tool_calls", [])) for s in cmp_steps),
        "success_detection": "heuristic (finish marker, not task correctness)",
    }

    # Layer 2: Alignment
    alignment = align_trajectories(ref_actions, cmp_actions, fuzzy_commands)
    metrics = compute_alignment_metrics(alignment, ref_actions, cmp_actions, token_rate)

    # Determine target files for milestone grounding
    from trajviz.insight.diagnostics import identify_target_files
    def _norm(p):
        return os.path.normpath(p) if p else p
    if anchor_files:
        milestone_targets = {_norm(f) for f in anchor_files}
    else:
        ref_targets = {_norm(f) for f in identify_target_files(ref_steps)}
        cmp_targets = {_norm(f) for f in identify_target_files(cmp_steps)}
        milestone_targets = ref_targets | cmp_targets if (ref_targets or cmp_targets) else None

    # Milestones
    ref_milestones = extract_milestones(ref_actions, target_files=milestone_targets)
    cmp_milestones = extract_milestones(cmp_actions, target_files=milestone_targets)
    milestone_deltas = compute_milestone_deltas(ref_milestones, cmp_milestones)

    ref_segments = segment_by_milestones(ref_actions, ref_milestones)
    cmp_segments = segment_by_milestones(cmp_actions, cmp_milestones)
    segment_result = compare_segments(
        ref_segments, cmp_segments, ref_milestones, cmp_milestones,
        ref_actions, cmp_actions, token_rate,
    )

    # Layer 3: Divergence
    extra_actions = [cmp_actions[j] for j in alignment["extra"] if j < len(cmp_actions)]
    matched_actions = [cmp_actions[j] for _, j in alignment["matched_pairs"] if j < len(cmp_actions)]
    patterns = classify_divergences(extra_actions, matched_actions, cmp_actions,
                                     matched_pairs=alignment["matched_pairs"],
                                     anchor_files=anchor_files,
                                     reference_actions=ref_actions)
    compute_pattern_costs(patterns, token_rate)

    # Collect dead_end_branch steps for harmful divergence
    dead_end_steps: set[int] = set()
    for p in patterns:
        if p.get("type") == "dead_end_branch":
            dead_end_steps.update(p.get("steps", []))
    harmful = compute_harmful_divergence(
        alignment["extra"], cmp_actions, token_rate, dead_end_steps)

    # Anchor mode and notes
    anchor_mode = "external" if anchor_files else "self"
    notes = []

    # Check outcome divergence
    if anchor_mode == "self" and (ref_success != cmp_success):
        notes.append(
            "Warning: runs produced different outcomes. Unanchored Layer 2 metrics are "
            "informational (behavioral similarity), not evaluative (quality). Consider "
            "using --anchor-patch for grounded comparison."
        )
    # Check patch-content divergence (same outcome but different files modified)
    if anchor_mode == "self" and ref_success == cmp_success:
        ref_write_targets = {a.target for a in ref_actions
                            if a.action_type == "FILE_WRITE" and a.effect_label == "survived"}
        cmp_write_targets = {a.target for a in cmp_actions
                            if a.action_type == "FILE_WRITE" and a.effect_label == "survived"}
        if ref_write_targets != cmp_write_targets and (ref_write_targets or cmp_write_targets):
            notes.append(
                "Warning: runs produced different patches (different files modified). "
                "Unanchored Layer 2 metrics are informational. Consider using --anchor-patch."
            )
    notes.append("This comparison is observational for one task pair.")
    notes.append("Patterns are not promoted to general knowledge until confirmed across tasks.")

    # Determine agent names from metadata
    ref_agent = ref_raw.get("metadata", {}).get("generator_name", "reference") if isinstance(ref_raw.get("metadata"), dict) else "reference"
    cmp_agent = cmp_raw.get("metadata", {}).get("generator_name", "compared") if isinstance(cmp_raw.get("metadata"), dict) else "compared"

    # Anchor analysis (only when externally anchored)
    anchor_analysis = None
    if anchor_mode == "external" and anchor_files:
        from .anchor import compute_anchor_analysis
        anchor_analysis = compute_anchor_analysis(
            ref_actions, cmp_actions, anchor_files)

    # Evaluation layers (R25: compute_eval_layers reads only patterns and
    # anchor_analysis; the old throwaway alignment dict was never consumed)
    from .eval_layers import compute_eval_layers
    eval_layers = compute_eval_layers(patterns, anchor_analysis)

    # Confidence badges
    confidence = {
        "alignment": "informational" if (anchor_mode == "self" and ref_success != cmp_success) else "heuristic",
        "milestones": "anchored" if anchor_files else "heuristic",
        "segments": "heuristic",
        "divergence": "heuristic",
        "outcome": "heuristic",
    }

    return {
        "task_id": task_id,
        "reference_agent": ref_agent,
        "compared_agent": cmp_agent,
        "outcome": outcome,
        "alignment": {**metrics, **harmful},
        "milestones": milestone_deltas,
        "ref_milestones": ref_milestones,
        "cmp_milestones": cmp_milestones,
        "segments": segment_result,
        "patterns": patterns,
        "anchor_mode": anchor_mode,
        "anchor_analysis": anchor_analysis,
        "eval_layers": eval_layers,
        "confidence": confidence,
        "evidence_level": "single_pair_hypothesis",
        "notes": notes,
    }


def _detect_success(steps: list[dict]) -> bool:
    """Heuristic: run succeeded if the last assistant step has finish=stop/end_turn.

    This is a parser-level heuristic, not grounded in tests, patch correctness,
    or task completion. It detects whether the agent reached a normal stopping
    point (as opposed to crashing, timing out, or being interrupted). A run
    can return True here and still have produced an incorrect patch. The outcome
    field in the report should be interpreted as "agent completed normally," not
    "agent solved the task correctly."
    """
    for s in reversed(steps):
        if s.get("role") == "assistant":
            return s.get("finish") in ("stop", "end_turn")
    return False
