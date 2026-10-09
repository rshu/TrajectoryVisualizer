"""Milestone extraction, segmentation, and per-segment comparison."""

from __future__ import annotations

from .canonical import CanonicalAction, compute_action_cost, DEFAULT_TOKEN_RATE


# ---------------------------------------------------------------------------
# Default validation patterns
# ---------------------------------------------------------------------------

DEFAULT_VALIDATION_PATTERNS = {"test", "build", "lint", "check", "verify"}


# ---------------------------------------------------------------------------
# Milestone identity
# ---------------------------------------------------------------------------

# Defined here, next to extract_milestones, as the single source of truth (C5):
# charts.py and rendering.py each used to carry their own copy. The list also
# fixes the display ORDER of the milestone timeline, which is why it is a list
# and not just the label map's keys.
MILESTONE_NAMES = [
    "first_relevant_file",
    "first_edit",
    "first_surviving_edit",
    "first_passing_validation",
    "final_patch",
]

MILESTONE_LABELS = {
    "first_relevant_file": "First Relevant File",
    "first_edit": "First Edit",
    "first_surviving_edit": "First Surviving Edit",
    "first_passing_validation": "First Passing Validation",
    "final_patch": "Final Patch",
}


# ---------------------------------------------------------------------------
# Milestone extraction
# ---------------------------------------------------------------------------

def extract_milestones(
    actions: list[CanonicalAction],
    validation_patterns: set[str] | None = None,
    target_files: set[str] | None = None,
) -> dict[str, int | None]:
    """Extract first-occurrence milestones from a canonical action sequence.

    Args:
        target_files: Set of normalized file paths from the patch footprint or anchor.
            When provided, first_relevant_file and first_edit are grounded to these files.

    Returns dict with milestone name → step_index or None.
    """
    if validation_patterns is None:
        validation_patterns = DEFAULT_VALIDATION_PATTERNS

    import os
    def _norm(p: str) -> str:
        return os.path.normpath(p) if p else p

    norm_targets = {_norm(f) for f in target_files} if target_files else None

    def _is_target(target: str) -> bool:
        """Normalize-and-suffix-match against the target set (R26: hoisted
        from three verbatim copies below; behavior unchanged)."""
        nt = _norm(target)
        return any(
            nt == t or nt.endswith("/" + t) or t.endswith("/" + nt)
            for t in norm_targets
        )

    milestones: dict[str, int | None] = {
        "first_relevant_file": None,
        "first_edit": None,
        "first_surviving_edit": None,
        "first_passing_validation": None,
        "final_patch": None,
    }

    last_successful_write_step = None

    for a in actions:
        if a.action_type == "REASON":
            continue

        # first_relevant_file: earliest FILE_READ or SEARCH referencing a target file
        if milestones["first_relevant_file"] is None:
            if a.action_type in ("FILE_READ", "SEARCH"):
                if norm_targets is not None:
                    if _is_target(a.target):
                        milestones["first_relevant_file"] = a.step_index
                elif a.effect_label in ("justified", "survived"):
                    # Fallback: use effect label when no target_files provided
                    milestones["first_relevant_file"] = a.step_index

        # first_edit: earliest FILE_WRITE to a target file (or any file if no targets)
        if milestones["first_edit"] is None and a.action_type == "FILE_WRITE":
            if norm_targets is None or _is_target(a.target):
                milestones["first_edit"] = a.step_index

        # first_surviving_edit: earliest FILE_WRITE with effect_label=survived
        # Grounded to target files when available
        if milestones["first_surviving_edit"] is None:
            if a.action_type == "FILE_WRITE" and a.effect_label == "survived":
                if norm_targets is None or _is_target(a.target):
                    milestones["first_surviving_edit"] = a.step_index

        # first_passing_validation: earliest recognized validation COMMAND that
        # completed. assign_effect_labels marks validation commands as
        # "justified" (reason=validation_command), NOT "survived", so match on
        # that; fall back to a pattern match for commands labeled "survived".
        if milestones["first_passing_validation"] is None and a.action_type == "COMMAND":
            reason = a.effect_detail.get("reason") if isinstance(a.effect_detail, dict) else None
            base_cmd = a.target.split("/")[-1] if "/" in a.target else a.target
            if reason == "validation_command" or (
                a.effect_label in ("justified", "survived")
                and any(vp in base_cmd for vp in validation_patterns)
            ):
                milestones["first_passing_validation"] = a.step_index

        # Track last successful write for final_patch
        if a.action_type == "FILE_WRITE" and a.effect_label == "survived":
            last_successful_write_step = a.step_index

    milestones["final_patch"] = last_successful_write_step
    return milestones


# ---------------------------------------------------------------------------
# Milestone deltas
# ---------------------------------------------------------------------------

def compute_milestone_deltas(
    ref_milestones: dict[str, int | None],
    cmp_milestones: dict[str, int | None],
) -> dict[str, int | None]:
    """Compute step-index deltas for milestones present in both trajectories."""
    deltas: dict[str, int | None] = {}
    for name in ref_milestones:
        ref_val = ref_milestones.get(name)
        cmp_val = cmp_milestones.get(name)
        if ref_val is not None and cmp_val is not None:
            deltas[f"{name}_delta"] = cmp_val - ref_val
        else:
            deltas[f"{name}_delta"] = None
    return deltas


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _milestone_order(milestones: dict[str, int | None]) -> list[str]:
    """Return milestone names in order of occurrence, excluding None."""
    present = [(name, step) for name, step in milestones.items() if step is not None]
    present.sort(key=lambda x: x[1])
    return [name for name, _ in present]


def segment_by_milestones(
    actions: list[CanonicalAction],
    milestones: dict[str, int | None],
) -> list[dict]:
    """Partition actions into segments between consecutive milestones.

    Returns list of {label, actions, start_step, end_step}.
    """
    order = _milestone_order(milestones)
    if not order:
        non_reason = [a for a in actions if a.action_type != "REASON"]
        return [{
            "label": "start → end",
            "actions": non_reason,
            "start_step": non_reason[0].step_index if non_reason else 0,
            "end_step": non_reason[-1].step_index if non_reason else 0,
        }]

    non_reason = [a for a in actions if a.action_type != "REASON"]
    boundaries = [("start", 0)]
    for name in order:
        boundaries.append((name, milestones[name]))
    boundaries.append(("end", non_reason[-1].step_index + 1 if non_reason else 0))

    segments = []
    for i in range(len(boundaries) - 1):
        label = f"{boundaries[i][0]} → {boundaries[i+1][0]}"
        start = boundaries[i][1]
        end = boundaries[i + 1][1]
        seg_actions = [a for a in non_reason if start <= a.step_index < end]
        segments.append({
            "label": label,
            "actions": seg_actions,
            "start_step": start,
            "end_step": end,
        })

    return segments


# ---------------------------------------------------------------------------
# Segment comparison
# ---------------------------------------------------------------------------

def _unique_step_count(actions: list[CanonicalAction]) -> int:
    """Count unique step indices (original trajectory steps, not action count)."""
    return len({a.step_index for a in actions})


def compare_segments(
    ref_segments: list[dict],
    cmp_segments: list[dict],
    ref_milestones: dict[str, int | None],
    cmp_milestones: dict[str, int | None],
    ref_actions: list[CanonicalAction],
    cmp_actions: list[CanonicalAction],
    token_rate: float = DEFAULT_TOKEN_RATE,
) -> dict:
    """Compare segments. Paired when milestone order matches, per-trajectory when not.

    Paired segments use real within-segment alignment to compute recall and precision.
    """
    from .alignment import align_trajectories
    from .alignment import compute_alignment_metrics

    ref_order = _milestone_order(ref_milestones)
    cmp_order = _milestone_order(cmp_milestones)

    if ref_order == cmp_order:
        # Paired comparison with real within-segment alignment
        paired = []
        for ref_seg, cmp_seg in zip(ref_segments, cmp_segments, strict=False):
            ref_seg_actions = ref_seg["actions"]
            cmp_seg_actions = cmp_seg["actions"]

            if ref_seg_actions and cmp_seg_actions:
                seg_alignment = align_trajectories(ref_seg_actions, cmp_seg_actions)
                seg_metrics = compute_alignment_metrics(
                    seg_alignment, ref_seg_actions, cmp_seg_actions, token_rate)
                recall = seg_metrics["reference_recall"]
                precision = seg_metrics["behavioral_precision"]
                overhead = seg_metrics["overhead_ratio"]
            else:
                ref_cost = sum(compute_action_cost(a, token_rate) for a in ref_seg_actions)
                cmp_cost = sum(compute_action_cost(a, token_rate) for a in cmp_seg_actions)
                recall = 1.0 if not ref_seg_actions else 0.0
                precision = 1.0 if not cmp_seg_actions else 0.0
                overhead = cmp_cost / ref_cost if ref_cost > 0 else 0.0

            paired.append({
                "segment": ref_seg["label"],
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "overhead": round(overhead, 2),
            })
        return {
            "milestone_order_matches": True,
            "segment_comparison": paired,
        }
    else:
        # Per-trajectory segments — report step counts and tokens
        ref_segs = []
        for seg in ref_segments:
            ref_segs.append({
                "segment": seg["label"],
                "steps": _unique_step_count(seg["actions"]),
                "tokens": sum(a.cost.token_share for a in seg["actions"]),
            })
        cmp_segs = []
        for seg in cmp_segments:
            cmp_segs.append({
                "segment": seg["label"],
                "steps": _unique_step_count(seg["actions"]),
                "tokens": sum(a.cost.token_share for a in seg["actions"]),
            })
        return {
            "milestone_order_matches": False,
            "reference_segments": ref_segs,
            "compared_segments": cmp_segs,
        }
