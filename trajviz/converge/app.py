"""Standalone Gradio app for Converge trajectory comparison."""

from __future__ import annotations

import html as html_stdlib
import traceback

import gradio as gr

from .styles import CONVERGE_CSS


def build_ui() -> gr.Blocks:
    """Build and return the Converge Gradio Blocks app."""

    with gr.Blocks(
        title="Converge",
        css=CONVERGE_CSS,
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# Converge\nCross-agent trajectory comparison and alignment analysis.")

        with gr.Row():
            with gr.Column(scale=1):
                ref_file = gr.File(label="Reference Trajectory", file_types=[".json", ".jsonl", ".zip", ".log"])
            with gr.Column(scale=1):
                cmp_file = gr.File(label="Compared Trajectory", file_types=[".json", ".jsonl", ".zip", ".log"])

        with gr.Row():
            anchor_patch = gr.File(label="Anchor Patch (optional)", file_types=[".patch", ".diff", ".txt"])
            token_rate = gr.Number(label="Token Rate", value=50.0, precision=1)
            fuzzy_commands = gr.Checkbox(label="Fuzzy Commands", value=False)

        compare_btn = gr.Button("Compare", variant="primary")

        # Results
        report_html = gr.HTML(label="Comparison Report")

        with gr.Row():
            milestone_plot = gr.Plot(show_label=False, label="Milestone Timeline")

        with gr.Row():
            segment_plot = gr.Plot(show_label=False, label="Segment Costs")

        with gr.Row():
            waterfall_plot = gr.Plot(show_label=False, label="Divergence Waterfall")

        with gr.Row():
            anchor_class_plot = gr.Plot(show_label=False, label="Anchor Write Recall by Class", visible=False)

        def do_compare(ref_upload, cmp_upload, anchor_upload, rate, fuzzy):
            """Callback: load files, run comparison, build charts and HTML."""
            from .alignment import build_comparison_report
            from .rendering import build_comparison_report_html
            from .charts import (
                build_milestone_timeline_chart,
                build_segment_cost_chart,
                build_divergence_waterfall_chart,
                build_anchor_class_chart,
                _empty_figure,
            )
            # Milestones and segments are now read from report (single source of truth)

            if ref_upload is None or cmp_upload is None:
                empty = _empty_figure(message="Upload both trajectory files to compare")
                return (
                    "<div class='cvg-report'><p>Please upload both a reference and compared trajectory.</p></div>",
                    empty, empty, empty,
                    gr.update(value=empty, visible=False),
                )

            try:
                ref_path = ref_upload.name if hasattr(ref_upload, "name") else str(ref_upload)
                cmp_path = cmp_upload.name if hasattr(cmp_upload, "name") else str(cmp_upload)
                anchor_path = None
                if anchor_upload is not None:
                    anchor_path = anchor_upload.name if hasattr(anchor_upload, "name") else str(anchor_upload)

                report = build_comparison_report(
                    ref_file=ref_path,
                    cmp_file=cmp_path,
                    token_rate=float(rate),
                    fuzzy_commands=bool(fuzzy),
                    anchor_patch=anchor_path,
                    task_id="",
                )

                # Build report HTML
                html_out = build_comparison_report_html(report)

                # Build milestone timeline chart from report data (single source of truth)
                ref_ms = report.get("ref_milestones", {})
                cmp_ms = report.get("cmp_milestones", {})
                milestone_fig = build_milestone_timeline_chart(ref_ms, cmp_ms)

                # Build segment cost chart
                seg_data = report.get("segments", {})
                milestone_order_matches = seg_data.get("milestone_order_matches", False)
                segment_fig = build_segment_cost_chart(seg_data, milestone_order_matches)

                # Build divergence waterfall
                waterfall_fig = build_divergence_waterfall_chart(report.get("patterns", []))

                # Build anchor class chart
                anchor_analysis = report.get("anchor_analysis")
                anchor_fig = build_anchor_class_chart(anchor_analysis)
                anchor_visible = anchor_analysis is not None

                return (
                    html_out, milestone_fig, segment_fig, waterfall_fig,
                    gr.update(value=anchor_fig, visible=anchor_visible),
                )

            except Exception as exc:
                tb = traceback.format_exc()
                err_html = (
                    f"<div class='cvg-report'><div class='cvg-warning'>"
                    f"<b>Error:</b> {html_stdlib.escape(str(exc))}"
                    f"<pre style='font-size:0.75rem;margin-top:0.5rem;'>{html_stdlib.escape(tb)}</pre>"
                    f"</div></div>"
                )
                empty = _empty_figure(message="Comparison failed")
                return err_html, empty, empty, empty, gr.update(value=empty, visible=False)

        compare_btn.click(
            fn=do_compare,
            inputs=[ref_file, cmp_file, anchor_patch, token_rate, fuzzy_commands],
            outputs=[report_html, milestone_plot, segment_plot, waterfall_plot, anchor_class_plot],
        )

        # ── Batch tab ──────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("## Batch Comparison")
        gr.Markdown("Upload a manifest JSON file to compare multiple trajectory pairs.")

        with gr.Row():
            batch_file = gr.File(label="Batch Manifest (JSON)", file_types=[".json"])
            batch_token_rate = gr.Number(label="Token Rate", value=50.0, precision=1)
            batch_fuzzy = gr.Checkbox(label="Fuzzy Commands", value=False)

        batch_btn = gr.Button("Run Batch", variant="primary")
        batch_report_html = gr.HTML(label="Batch Report")

        def do_batch(manifest_upload, rate, fuzzy):
            """Callback: parse manifest, run batch, aggregate, display."""
            from .batch import (
                parse_manifest, run_batch, aggregate_reports,
                compute_pattern_frequency, promote_patterns,
            )

            if manifest_upload is None:
                return "<div class='cvg-report'><p>Please upload a batch manifest JSON file.</p></div>"

            try:
                manifest_path = manifest_upload.name if hasattr(manifest_upload, "name") else str(manifest_upload)
                entries = parse_manifest(manifest_path)
                results = run_batch(entries, token_rate=float(rate), fuzzy_commands=bool(fuzzy))
                aggregate = aggregate_reports(results)
                frequency = compute_pattern_frequency(results)
                promoted = promote_patterns(frequency)

                # Render as HTML
                import html as html_mod
                parts = ['<div class="cvg-report">']
                parts.append("<h2>Batch Report</h2>")
                parts.append(f"<p>{aggregate.get('success_count', 0)} of {aggregate.get('task_count', 0)} tasks succeeded</p>")

                # Metrics table
                metrics = aggregate.get("metrics", {})
                if metrics:
                    parts.append("<h3>Aggregate Metrics</h3>")
                    parts.append('<table class="cvg-outcome-table"><thead><tr>'
                                 '<th>Metric</th><th>Mean</th><th>Median</th><th>P95</th></tr></thead><tbody>')
                    for name, stats in metrics.items():
                        parts.append(f"<tr><td>{html_mod.escape(name)}</td>"
                                     f"<td>{stats.get('mean', 0):.4f}</td>"
                                     f"<td>{stats.get('median', 0):.4f}</td>"
                                     f"<td>{stats.get('p95', 0):.4f}</td></tr>")
                    parts.append("</tbody></table>")

                # Pattern frequency
                if frequency:
                    parts.append("<h3>Pattern Frequency</h3>")
                    parts.append('<table class="cvg-outcome-table"><thead><tr>'
                                 '<th>Pattern</th><th>Tasks</th><th>Prevalence</th><th>Level</th></tr></thead><tbody>')
                    for ptype, data in sorted(frequency.items(), key=lambda x: -x[1].get("count", 0)):
                        level = promoted.get(ptype, "hypothesis")
                        badge = "supported_finding" if level == "supported_finding" else "hypothesis"
                        parts.append(f"<tr><td>{html_mod.escape(ptype)}</td>"
                                     f"<td>{data.get('count', 0)}</td>"
                                     f"<td>{data.get('prevalence', 0)*100:.0f}%</td>"
                                     f"<td><span class='cvg-badge'>{html_mod.escape(badge)}</span></td></tr>")
                    parts.append("</tbody></table>")

                parts.append("</div>")
                return "".join(parts)

            except Exception as exc:
                tb = traceback.format_exc()
                return (
                    f"<div class='cvg-report'><div class='cvg-warning'>"
                    f"<b>Error:</b> {html_stdlib.escape(str(exc))}"
                    f"<pre style='font-size:0.75rem;'>{html_stdlib.escape(tb)}</pre>"
                    f"</div></div>"
                )

        batch_btn.click(
            fn=do_batch,
            inputs=[batch_file, batch_token_rate, batch_fuzzy],
            outputs=[batch_report_html],
        )

        # ── Before/After section ──────────────────────────
        gr.Markdown("---")
        gr.Markdown("## Before/After Comparison")
        gr.Markdown("Upload two manifest files to compare pre- and post-intervention batches.")

        with gr.Row():
            before_file = gr.File(label="Before Manifest (JSON)", file_types=[".json"])
            after_file = gr.File(label="After Manifest (JSON)", file_types=[".json"])
        with gr.Row():
            ba_token_rate = gr.Number(label="Token Rate", value=50.0, precision=1)
            ba_fuzzy = gr.Checkbox(label="Fuzzy Commands", value=False)

        ba_btn = gr.Button("Compare Before/After", variant="primary")
        ba_report_html = gr.HTML(label="Intervention Report")

        def do_before_after(before_upload, after_upload, rate, fuzzy):
            """Callback: run before/after comparison."""
            from .batch import parse_manifest, run_batch
            from .intervention import (
                pair_tasks, compute_metric_deltas, compute_pattern_deltas,
                detect_guardrail_regressions, build_intervention_report,
            )

            if before_upload is None or after_upload is None:
                return "<div class='cvg-report'><p>Please upload both before and after manifests.</p></div>"

            try:
                before_path = before_upload.name if hasattr(before_upload, "name") else str(before_upload)
                after_path = after_upload.name if hasattr(after_upload, "name") else str(after_upload)

                before_entries = parse_manifest(before_path)
                after_entries = parse_manifest(after_path)
                before_results = run_batch(before_entries, token_rate=float(rate), fuzzy_commands=bool(fuzzy))
                after_results = run_batch(after_entries, token_rate=float(rate), fuzzy_commands=bool(fuzzy))

                paired, before_only, after_only = pair_tasks(before_results, after_results)
                metric_deltas = compute_metric_deltas(paired)
                pattern_deltas = compute_pattern_deltas(paired)
                guardrails = detect_guardrail_regressions(metric_deltas)
                report = build_intervention_report(
                    before_path, after_path, paired, before_only, after_only,
                    metric_deltas, pattern_deltas, guardrails)

                import html as html_mod
                parts = ['<div class="cvg-report">']
                parts.append("<h2>Intervention Report</h2>")
                parts.append(f"<p>Paired tasks: {report['paired_tasks']}</p>")

                # Recommendation
                rec = report.get("recommendation", "")
                if rec:
                    parts.append(f'<div class="cvg-note"><b>Recommendation:</b> {html_mod.escape(rec)}</div>')

                # Guardrails
                for g in report.get("guardrail_warnings", []):
                    parts.append(f'<div class="cvg-warning">! {html_mod.escape(g["warning"])}</div>')

                # Delta table
                deltas = report.get("intervention_effect", {})
                if deltas:
                    parts.append("<h3>Metric Deltas</h3>")
                    parts.append('<table class="cvg-outcome-table"><thead><tr>'
                                 '<th>Metric</th><th>Before</th><th>After</th>'
                                 '<th>Delta</th><th>Direction</th></tr></thead><tbody>')
                    for name, d in deltas.items():
                        color = "#059669" if d["direction"] == "improved" else (
                            "#dc2626" if d["direction"] == "regressed" else "#6b7280")
                        parts.append(
                            f'<tr><td>{html_mod.escape(name)}</td>'
                            f'<td>{d["before_mean"]:.4f}</td>'
                            f'<td>{d["after_mean"]:.4f}</td>'
                            f'<td>{d["delta"]:+.4f}</td>'
                            f'<td style="color:{color};">{d["direction"]}</td></tr>')
                    parts.append("</tbody></table>")

                parts.append("</div>")
                return "".join(parts)

            except Exception as exc:
                tb = traceback.format_exc()
                return (
                    f"<div class='cvg-report'><div class='cvg-warning'>"
                    f"<b>Error:</b> {html_stdlib.escape(str(exc))}"
                    f"<pre style='font-size:0.75rem;'>{html_stdlib.escape(tb)}</pre>"
                    f"</div></div>"
                )

        ba_btn.click(
            fn=do_before_after,
            inputs=[before_file, after_file, ba_token_rate, ba_fuzzy],
            outputs=[ba_report_html],
        )

    return app
