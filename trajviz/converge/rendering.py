"""HTML rendering for Converge comparison reports."""

from __future__ import annotations

import html

from .milestones import MILESTONE_LABELS
from .styles import CONVERGE_CSS


def _esc(value: object) -> str:
    """HTML-escape any value converted to string."""
    return html.escape(str(value))


def build_anchor_analysis_html(anchor_analysis: dict) -> str:
    """Render anchor analysis data as an HTML section.

    The anchor_analysis dict is from anchor.compute_anchor_analysis() with keys:
    total_anchor_files, file_classes, reference{write_precision, write_recall,
    write_recall_by_class, off_patch_write_ratio, files_written, anchor_files_written,
    time_to_first_anchor_read, time_to_first_anchor_write}, compared{...}.
    """
    parts: list[str] = []
    parts.append('<div class="cvg-anchor-section">')
    parts.append("<h2>Anchor Analysis</h2>")

    total_anchor = anchor_analysis.get("total_anchor_files", 0)

    # File class summary
    file_classes = anchor_analysis.get("file_classes", {})
    if file_classes:
        sorted_classes = sorted(file_classes.items(), key=lambda x: x[1], reverse=True)
        total = sum(v for _, v in sorted_classes)
        summary = ", ".join(f"{count} {_esc(name)}" for name, count in sorted_classes)
        parts.append(f'<p class="cvg-anchor-metric">File classes: {summary} ({total} total)</p>')

    # Side-by-side metrics table
    ref_data = anchor_analysis.get("reference", {})
    cmp_data = anchor_analysis.get("compared", {})

    parts.append('<table class="cvg-anchor-table">')
    parts.append("<thead><tr><th>Metric</th><th>Reference</th><th>Compared</th></tr></thead>")
    parts.append("<tbody>")

    # Write precision
    ref_wp = ref_data.get("write_precision") or 0
    cmp_wp = cmp_data.get("write_precision") or 0
    ref_aw = ref_data.get("anchor_files_written", 0)
    ref_tw = ref_data.get("files_written", 0)
    cmp_aw = cmp_data.get("anchor_files_written", 0)
    cmp_tw = cmp_data.get("files_written", 0)
    parts.append(f"<tr><td>Write Precision</td>"
                 f"<td>{ref_wp * 100:.1f}% ({ref_aw}/{ref_tw})</td>"
                 f"<td>{cmp_wp * 100:.1f}% ({cmp_aw}/{cmp_tw})</td></tr>")

    # Write recall
    ref_wr = ref_data.get("write_recall") or 0
    cmp_wr = cmp_data.get("write_recall") or 0
    parts.append(f"<tr><td>Write Recall</td>"
                 f"<td>{ref_wr * 100:.1f}% ({ref_aw}/{total_anchor})</td>"
                 f"<td>{cmp_wr * 100:.1f}% ({cmp_aw}/{total_anchor})</td></tr>")

    # Off-patch ratio
    ref_opr = ref_data.get("off_patch_write_ratio") or 0
    cmp_opr = cmp_data.get("off_patch_write_ratio") or 0
    parts.append(f"<tr><td>Off-Patch Ratio</td>"
                 f"<td>{ref_opr * 100:.1f}%</td>"
                 f"<td>{cmp_opr * 100:.1f}%</td></tr>")

    # First anchor read
    ref_fr = ref_data.get("time_to_first_anchor_read")
    cmp_fr = cmp_data.get("time_to_first_anchor_read")
    ref_fr_str = f"Step {ref_fr}" if ref_fr is not None else "N/A"
    cmp_fr_str = f"Step {cmp_fr}" if cmp_fr is not None else "N/A"
    parts.append(f"<tr><td>First Anchor Read</td>"
                 f"<td>{_esc(ref_fr_str)}</td>"
                 f"<td>{_esc(cmp_fr_str)}</td></tr>")

    # First anchor write
    ref_fw = ref_data.get("time_to_first_anchor_write")
    cmp_fw = cmp_data.get("time_to_first_anchor_write")
    ref_fw_str = f"Step {ref_fw}" if ref_fw is not None else "N/A"
    cmp_fw_str = f"Step {cmp_fw}" if cmp_fw is not None else "N/A"
    parts.append(f"<tr><td>First Anchor Write</td>"
                 f"<td>{_esc(ref_fw_str)}</td>"
                 f"<td>{_esc(cmp_fw_str)}</td></tr>")

    parts.append("</tbody></table>")

    # Per-class recall breakdown (from write_recall_by_class in each agent's data)
    ref_by_class = ref_data.get("write_recall_by_class", {})
    cmp_by_class = cmp_data.get("write_recall_by_class", {})
    all_classes = sorted(set(ref_by_class.keys()) | set(cmp_by_class.keys()) | set(file_classes.keys()))
    if all_classes:
        parts.append("<h3>Per-Class Anchor Write Recall</h3>")
        parts.append('<table class="cvg-anchor-table">')
        parts.append("<thead><tr><th>File Class</th><th>Anchor Files</th>"
                     "<th>Reference Recall</th><th>Compared Recall</th></tr></thead>")
        parts.append("<tbody>")
        for cls_name in all_classes:
            anchor_count = file_classes.get(cls_name, 0)
            ref_recall = ref_by_class.get(cls_name) or 0
            cmp_recall = cmp_by_class.get(cls_name) or 0
            parts.append(f"<tr><td>{_esc(cls_name)}</td>"
                         f"<td>{anchor_count}</td>"
                         f"<td>{ref_recall * 100:.1f}%</td>"
                         f"<td>{cmp_recall * 100:.1f}%</td></tr>")
        parts.append("</tbody></table>")

    parts.append("</div>")
    return "\n".join(parts)


def build_comparison_report_html(report: dict) -> str:
    """Render the full comparison report as styled HTML.

    The report dict is the output of alignment.build_comparison_report().
    """
    outcome = report.get("outcome", {})
    patterns = report.get("patterns", [])
    anchor_analysis = report.get("anchor_analysis")
    ref_agent = _esc(report.get("reference_agent", "reference"))
    cmp_agent = _esc(report.get("compared_agent", "compared"))
    task_id = _esc(report.get("task_id", ""))

    parts: list[str] = []

    # Wrapper open
    parts.append(f"<style>{CONVERGE_CSS}</style>")
    parts.append('<div class="cvg-report">')

    # Title
    title = "Converge Comparison Report"
    if task_id:
        title += f" &mdash; {task_id}"
    parts.append(f"<h2>{title}</h2>")

    # ── Outcome table ────────────────────────────────────────
    parts.append("<h2>Outcome</h2>")
    parts.append('<table class="cvg-outcome-table">')

    def _fmt_duration(secs):
        if secs is None:
            return "—"
        try:
            secs = float(secs)
        except (TypeError, ValueError):
            return "—"
        if secs < 60:
            return f"{secs:.1f}s"
        m, s = divmod(secs, 60)
        if m < 60:
            return f"{int(m)}m {s:.0f}s"
        h, m = divmod(m, 60)
        return f"{int(h)}h {int(m)}m"

    ref_filename = outcome.get("reference_filename") or ref_agent
    cmp_filename = outcome.get("compared_filename") or cmp_agent
    parts.append("<thead><tr><th>Metric</th>"
                 f"<th>{_esc(ref_filename)}</th>"
                 f"<th>{_esc(cmp_filename)}</th></tr></thead>")
    parts.append("<tbody>")
    parts.append(f"<tr><td>Format</td>"
                 f"<td>{_esc(outcome.get('reference_format', '—'))}</td>"
                 f"<td>{_esc(outcome.get('compared_format', '—'))}</td></tr>")
    parts.append(f"<tr><td>Steps</td>"
                 f"<td>{_esc(outcome.get('reference_steps', 0))}</td>"
                 f"<td>{_esc(outcome.get('compared_steps', 0))}</td></tr>")
    parts.append(f"<tr><td>Total Tokens</td>"
                 f"<td>{outcome.get('reference_tokens', 0):,}</td>"
                 f"<td>{outcome.get('compared_tokens', 0):,}</td></tr>")
    parts.append(f"<tr><td>Total Tool Calls</td>"
                 f"<td>{_esc(outcome.get('reference_tool_calls', 0))}</td>"
                 f"<td>{_esc(outcome.get('compared_tool_calls', 0))}</td></tr>")
    parts.append(f"<tr><td>Duration</td>"
                 f"<td>{_esc(_fmt_duration(outcome.get('reference_duration_s')))}</td>"
                 f"<td>{_esc(_fmt_duration(outcome.get('compared_duration_s')))}</td></tr>")
    parts.append("</tbody></table>")

    # ── Milestones table — per-trajectory step index ─────────
    ref_ms = report.get("ref_milestones", {}) or {}
    cmp_ms = report.get("cmp_milestones", {}) or {}
    if ref_ms or cmp_ms:
        # Preserve the milestone order from the reference side; append any
        # keys that only appear in the compared side.
        ordered_keys = list(ref_ms.keys())
        for k in cmp_ms:
            if k not in ordered_keys:
                ordered_keys.append(k)

        def _fmt_step(v):
            return f"#{int(v)}" if isinstance(v, (int, float)) else "N/A"

        parts.append("<h2>Milestones</h2>")
        parts.append('<table class="cvg-milestone-table">')
        parts.append("<thead><tr><th>Milestone</th>"
                     f"<th>{_esc(ref_filename)}</th>"
                     f"<th>{_esc(cmp_filename)}</th></tr></thead>")
        parts.append("<tbody>")
        for name in ordered_keys:
            label = MILESTONE_LABELS.get(name, name.replace("_", " ").title())
            parts.append(f"<tr><td>{_esc(label)}</td>"
                         f"<td>{_esc(_fmt_step(ref_ms.get(name)))}</td>"
                         f"<td>{_esc(_fmt_step(cmp_ms.get(name)))}</td></tr>")
        parts.append("</tbody></table>")

    # ── Top divergence patterns ──────────────────────────────
    if patterns:
        parts.append("<h2>Divergence Patterns</h2>")
        _PATTERN_GLOSSARY = [
            ("dead_end_branch",
             "Agent started a line of work and later abandoned it without any surviving effect."),
            ("reverted_and_rewritten",
             "Write that was overwritten or undone by a later write to the same file — instability."),
            ("iterative_refinement",
             "Repeated writes to the same file that gradually improved the result — possibly intentional."),
            ("broad_exploration",
             "Reads or searches that ranged far outside the files that actually needed changing."),
            ("error_recovery_overhead",
             "Extra steps the agent spent handling or working around tool errors."),
            ("redundant_search",
             "Back-to-back searches that duplicated or barely refined an earlier query."),
            ("ordering_inefficiency",
             "Correct actions performed in an order that cost extra steps (e.g., edit before reading)."),
            ("premature_validation",
             "Ran tests/build/lint before the necessary edits were in place."),
        ]
        parts.append(
            '<p style="font-size:12px;color:var(--ov-muted,#6b7280);margin:4px 0 10px;">'
            'Each row is a cluster of unmatched compared-agent actions attributed to one pattern. '
            'Pattern meanings:</p>'
        )
        parts.append('<ul style="font-size:12px;color:var(--ov-muted,#6b7280);margin:0 0 12px 18px;padding:0;">')
        for name, desc in _PATTERN_GLOSSARY:
            parts.append(
                f'<li style="margin:2px 0;"><code>{_esc(name)}</code> &mdash; {_esc(desc)}</li>'
            )
        parts.append('</ul>')
        # Sort by token cost descending
        sorted_patterns = sorted(
            patterns,
            key=lambda p: p.get("estimated_extra_cost", {}).get("tokens", 0),
            reverse=True,
        )
        parts.append('<table class="cvg-outcome-table">')
        parts.append("<thead><tr><th>Pattern</th><th>Steps</th>"
                     "<th>Extra Tokens</th><th>Evidence</th></tr></thead>")
        parts.append("<tbody>")
        for p in sorted_patterns:
            ptype = _esc(p.get("type", "unknown"))
            steps = len(p.get("steps", []))
            tokens = p.get("estimated_extra_cost", {}).get("tokens", 0)
            evidence = "; ".join(p.get("evidence", [])[:3])
            parts.append(
                f"<tr><td>{ptype}</td><td>{steps}</td>"
                f"<td>{tokens:,}</td><td>{_esc(evidence)}</td></tr>"
            )
        parts.append("</tbody></table>")

    # ── Anchor analysis (external anchor mode only) ──────────
    # R22: brings the HTML report to parity with the CLI's anchor section.
    if anchor_analysis:
        parts.append(build_anchor_analysis_html(anchor_analysis))

    # Wrapper close
    parts.append("</div>")

    return "\n".join(parts)
