"""CLI entry point for Converge trajectory comparison."""

from __future__ import annotations

import argparse
import json
import sys

from .alignment import build_comparison_report


def _print_summary(report: dict) -> None:
    """Print a human-readable summary of the comparison report."""
    outcome = report.get("outcome", {})
    alignment = report.get("alignment", {})
    patterns = report.get("patterns", [])

    print("=" * 60)
    print("Converge Comparison Summary")
    if report.get("task_id"):
        print(f"Task: {report['task_id']}")
    print("=" * 60)

    # Outcome
    ref_agent = report.get("reference_agent", "reference")
    cmp_agent = report.get("compared_agent", "compared")
    print(f"\n  {ref_agent}: {'Success' if outcome.get('reference_success') else 'Failure'}"
          f"  ({outcome.get('reference_steps', 0)} steps, {outcome.get('reference_tokens', 0):,} tokens)")
    print(f"  {cmp_agent}: {'Success' if outcome.get('compared_success') else 'Failure'}"
          f"  ({outcome.get('compared_steps', 0)} steps, {outcome.get('compared_tokens', 0):,} tokens)")

    # Alignment metrics with confidence badges
    conf = report.get("confidence", {})
    align_badge = f" [{conf.get('alignment', 'heuristic')}]"
    print(f"\n  Recall:     {alignment.get('reference_recall', 0) * 100:.1f}%{align_badge}")
    print(f"  Precision:  {alignment.get('behavioral_precision', 0) * 100:.1f}%{align_badge}")
    print(f"  F1:         {alignment.get('alignment_f1', 0) * 100:.1f}%{align_badge}")
    print(f"  Overhead:   {alignment.get('overhead_ratio', 0):.2f}x{align_badge}")
    print(f"  Harmful:    {alignment.get('harmful_ratio', 0) * 100:.1f}%{align_badge}")

    # Anchor mode
    print(f"\n  Anchor mode: {report.get('anchor_mode', 'self')}")
    print(f"  Confidence: alignment={conf.get('alignment', '?')}, "
          f"milestones={conf.get('milestones', '?')}, "
          f"outcome={conf.get('outcome', '?')}")

    # Top 3 patterns
    if patterns:
        sorted_patterns = sorted(
            patterns,
            key=lambda p: p.get("estimated_extra_cost", {}).get("tokens", 0),
            reverse=True,
        )
        print("\n  Top divergence patterns:")
        for p in sorted_patterns[:3]:
            ptype = p.get("type", "unknown")
            tokens = p.get("estimated_extra_cost", {}).get("tokens", 0)
            print(f"    - {ptype}: {tokens:,} extra tokens")

    # Evaluation layers
    eval_layers = report.get("eval_layers")
    if eval_layers:
        print("\n  Evaluation Layers:")
        _verdict_symbols = {"strong": "OK", "moderate": "~~", "weak": "!!", "n/a": "--"}
        for layer_name, layer in eval_layers.items():
            verdict = layer.get("verdict", "n/a")
            symbol = _verdict_symbols.get(verdict, "??")
            desc = layer.get("description", "")
            print(f"    [{symbol}] {layer_name.replace('_', ' ').title()}: {verdict} — {desc}")
            for mname, mval in layer.get("metrics", {}).items():
                if mval is not None:
                    if isinstance(mval, float):
                        print(f"        {mname}: {mval:.4f}")
                    else:
                        print(f"        {mname}: {mval}")

    # Anchor Analysis
    anchor_analysis = report.get("anchor_analysis")
    if anchor_analysis:
        print("\n  Anchor Analysis:")
        file_classes = anchor_analysis.get("file_classes", {})
        class_parts = []
        total_files = 0
        for cls_name, cls_count in sorted(file_classes.items(), key=lambda x: x[1], reverse=True):
            class_parts.append(f"{cls_count} {cls_name}")
            total_files += cls_count
        print(f"    File classes: {', '.join(class_parts)} ({total_files} total)")

        for agent_key, agent_label in [("reference", "Reference"), ("compared", "Compared")]:
            agent_data = anchor_analysis.get(agent_key, {})
            wp = agent_data.get("write_precision") or 0
            wr = agent_data.get("write_recall") or 0
            opr = agent_data.get("off_patch_write_ratio") or 0
            anchor_written = agent_data.get("anchor_files_written", 0)
            total_written = agent_data.get("files_written", 0)
            total_anchor = anchor_analysis.get("total_anchor_files", 0)
            first_read = agent_data.get("time_to_first_anchor_read")
            first_write = agent_data.get("time_to_first_anchor_write")
            print(f"\n    {agent_label}:")
            print(f"      Write precision: {wp * 100:.1f}%  ({anchor_written}/{total_written} files in anchor)")
            print(f"      Write recall:    {wr * 100:.1f}%  ({anchor_written}/{total_anchor} anchor files)")
            print(f"      Off-patch ratio: {opr * 100:.1f}%")
            fr_str = f"step {first_read}" if first_read is not None else "N/A"
            fw_str = f"step {first_write}" if first_write is not None else "N/A"
            print(f"      First anchor read:  {fr_str}")
            print(f"      First anchor write: {fw_str}")

    # Notes
    notes = report.get("notes", [])
    if notes:
        print()
        for note in notes:
            print(f"  * {note}")

    print()


def _print_batch_summary(batch_report: dict) -> None:
    """Print a human-readable summary of the batch aggregate report."""
    summary = batch_report.get("summary", {})
    frequency = batch_report.get("pattern_frequency", {})
    consistency = batch_report.get("consistency", {})

    print("=" * 60)
    print("Converge Batch Comparison Summary")
    print(f"Manifest: {batch_report.get('manifest', '?')}")
    print("=" * 60)

    print(f"\n  Tasks: {summary.get('task_count', 0)} total, "
          f"{summary.get('success_count', 0)} succeeded, "
          f"{summary.get('failure_count', 0)} failed")
    if summary.get("anchored_count", 0) > 0:
        print(f"  Anchored: {summary['anchored_count']} of {summary['success_count']}")

    # Aggregate metrics
    metrics = summary.get("metrics", {})
    if metrics:
        print("\n  Aggregate Metrics (across tasks):")
        print(f"  {'Metric':<35s} {'Mean':>8s} {'Median':>8s} {'P5':>8s} {'P95':>8s} {'Stdev':>8s}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for name, stats in metrics.items():
            mean = stats.get("mean", 0)
            median = stats.get("median", 0)
            p5 = stats.get("p5", 0)
            p95 = stats.get("p95", 0)
            stdev = stats.get("stdev", 0)
            note = f" ({stats['note']})" if "note" in stats else ""
            print(f"  {name:<35s} {mean:>8.4f} {median:>8.4f} {p5:>8.4f} {p95:>8.4f} {stdev:>8.4f}{note}")

    # Pattern frequency with promotion badges
    if frequency:
        print("\n  Pattern Frequency:")
        for ptype, data in sorted(frequency.items(), key=lambda x: -x[1].get("count", 0)):
            count = data.get("count", 0)
            prevalence = data.get("prevalence", 0)
            level = data.get("evidence_level", "hypothesis")
            badge = " [FINDING]" if level == "supported_finding" else ""
            print(f"    {ptype}: {count} tasks ({prevalence*100:.0f}%){badge}")

    # Consistency
    if consistency:
        high_cv = [(k, v) for k, v in consistency.items() if v is not None and v > 0.5]
        if high_cv:
            print("\n  Consistency warnings (CV > 0.5 — high variance across tasks):")
            for name, cv in sorted(high_cv, key=lambda x: -x[1]):
                print(f"    {name}: CV={cv:.2f}")

    # Per-task summary
    per_task = batch_report.get("per_task", [])
    if per_task:
        print("\n  Per-task results:")
        for t in per_task:
            status = t.get("status", "?")
            tid = t.get("task_id", "?")
            if status == "failed":
                print(f"    {tid}: FAILED — {t.get('error', '?')[:60]}")
            else:
                a = t.get("alignment_summary", {})
                print(f"    {tid}: R={a.get('recall', 0):.3f} P={a.get('precision', 0):.3f} "
                      f"F1={a.get('f1', 0):.3f} H={a.get('harmful_ratio', 0):.3f}")

    print()


def _print_intervention_summary(report: dict) -> None:
    """Print a human-readable summary of the before/after intervention report."""
    print("=" * 60)
    print("Converge Intervention Report")
    print("=" * 60)

    print(f"\n  Before: {report.get('before_manifest', '?')}")
    print(f"  After:  {report.get('after_manifest', '?')}")
    print(f"  Paired tasks: {report.get('paired_tasks', 0)}")
    if report.get("unpaired_before", 0) or report.get("unpaired_after", 0):
        print(f"  Unpaired: {report.get('unpaired_before', 0)} before-only, "
              f"{report.get('unpaired_after', 0)} after-only")

    # Metric deltas
    deltas = report.get("intervention_effect", {})
    if deltas:
        print("\n  Metric Deltas:")
        print(f"  {'Metric':<35s} {'Before':>8s} {'After':>8s} {'Delta':>8s} {'Dir':>10s} {'Sig':>5s}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*5}")
        for name, d in deltas.items():
            sig = "*" if d.get("significant") else ""
            arrow = {"improved": "UP", "regressed": "DOWN", "unchanged": "—"}.get(d["direction"], "?")
            print(f"  {name:<35s} {d['before_mean']:>8.4f} {d['after_mean']:>8.4f} "
                  f"{d['delta']:>+8.4f} {arrow:>10s} {sig:>5s}")

    # Guardrail warnings
    guardrails = report.get("guardrail_warnings", [])
    if guardrails:
        print("\n  GUARDRAIL WARNINGS:")
        for g in guardrails:
            print(f"    ! {g['warning']}")

    # Pattern deltas
    pattern_deltas = report.get("pattern_deltas", {})
    changed = {k: v for k, v in pattern_deltas.items() if v.get("direction") != "unchanged"}
    if changed:
        print("\n  Pattern Changes:")
        for ptype, d in sorted(changed.items(), key=lambda x: x[1].get("delta", 0)):
            arrow = {"improved": "DOWN", "regressed": "UP", "unchanged": "—"}.get(d["direction"], "?")
            print(f"    {ptype}: {d['before_frequency']*100:.0f}% → {d['after_frequency']*100:.0f}% ({arrow})")

    # Recommendation
    rec = report.get("recommendation", "")
    if rec:
        print(f"\n  Recommendation: {rec}")

    print()


def main() -> None:
    """Parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(
        prog="trajectory-converge",
        description="Compare two agent trajectories and produce alignment metrics.",
    )

    # Pairwise mode
    parser.add_argument("ref_file", nargs="?", default=None,
                        help="Path to the reference trajectory file")
    parser.add_argument("cmp_file", nargs="?", default=None,
                        help="Path to the compared trajectory file")

    # Batch mode
    parser.add_argument("--batch", default=None,
                        help="Path to a batch manifest JSON file (mutually exclusive with ref/cmp)")

    # Before/after mode
    parser.add_argument("--before", default=None,
                        help="Path to pre-intervention manifest (use with --after)")
    parser.add_argument("--after", default=None,
                        help="Path to post-intervention manifest (use with --before)")

    # Common options
    parser.add_argument("--anchor-patch", default=None,
                        help="Path to an external anchor patch file for grounded comparison")
    parser.add_argument("--token-rate", type=float, default=50.0,
                        help="Token-equivalent rate for latency normalization (default: 50)")
    parser.add_argument("--fuzzy-commands", action="store_true",
                        help="Enable fuzzy matching of composite bash commands to tool actions")
    parser.add_argument("--output", choices=["json", "summary"], default="json",
                        help="Output format (default: json)")
    parser.add_argument("--task-id", default="",
                        help="Optional task identifier included in the report")

    args = parser.parse_args()

    # Validate mutual exclusivity
    modes = sum([
        bool(args.batch),
        bool(args.before or args.after),
        bool(args.ref_file and args.cmp_file),
    ])
    if modes > 1:
        parser.error("Use one mode: positional ref/cmp, --batch, or --before/--after")
    if modes == 0:
        parser.error("Provide ref_file and cmp_file, --batch <manifest>, or --before/--after <manifests>")
    if bool(args.before) != bool(args.after):
        parser.error("--before and --after must be used together")

    if args.before and args.after:
        # Before/after mode
        from .batch import parse_manifest, run_batch
        from .intervention import (
            pair_tasks, compute_metric_deltas, compute_pattern_deltas,
            detect_guardrail_regressions, build_intervention_report,
        )
        try:
            before_entries = parse_manifest(args.before)
            after_entries = parse_manifest(args.after)
            before_results = run_batch(before_entries, token_rate=args.token_rate,
                                       fuzzy_commands=args.fuzzy_commands)
            after_results = run_batch(after_entries, token_rate=args.token_rate,
                                      fuzzy_commands=args.fuzzy_commands)
            paired, before_only, after_only = pair_tasks(before_results, after_results)
            metric_deltas = compute_metric_deltas(paired)
            pattern_deltas = compute_pattern_deltas(paired)
            guardrails = detect_guardrail_regressions(metric_deltas)
            report = build_intervention_report(
                args.before, args.after, paired, before_only, after_only,
                metric_deltas, pattern_deltas, guardrails)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.output == "summary":
            _print_intervention_summary(report)
        else:
            print(json.dumps(report, indent=2, default=str))

    elif args.batch:
        # Batch mode
        from .batch import (
            parse_manifest, run_batch, aggregate_reports,
            compute_pattern_frequency, promote_patterns,
            compute_consistency, build_batch_report,
        )
        try:
            entries = parse_manifest(args.batch)
            results = run_batch(
                entries,
                token_rate=args.token_rate,
                fuzzy_commands=args.fuzzy_commands,
                progress_callback=lambda cur, tot: print(
                    f"  [{cur}/{tot}] complete", file=sys.stderr) if args.output != "json" else None,
            )
            aggregate = aggregate_reports(results)
            frequency = compute_pattern_frequency(results)
            promoted = promote_patterns(frequency)
            consistency = compute_consistency(aggregate)
            batch_report = build_batch_report(
                args.batch, results, aggregate, frequency, promoted, consistency)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.output == "summary":
            _print_batch_summary(batch_report)
        else:
            print(json.dumps(batch_report, indent=2, default=str))
    else:
        # Pairwise mode
        try:
            report = build_comparison_report(
                ref_file=args.ref_file,
                cmp_file=args.cmp_file,
                token_rate=args.token_rate,
                fuzzy_commands=args.fuzzy_commands,
                anchor_patch=args.anchor_patch,
                task_id=args.task_id,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.output == "summary":
            _print_summary(report)
        else:
            print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
