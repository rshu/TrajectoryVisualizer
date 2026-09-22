"""Regression tests for the 2026-08 converge/comparison review fixes.

Covers:
- B22: read-only sed/awk no longer classified FILE_WRITE (and no longer flip
       real edits to 'reverted'); in-place forms still classify as writes.
- B23: exit-code failure attribution is positional for id-less tool calls.
- B24: count-based fallback in compute_alignment_metrics is joint, never
       mixing token weights with action counts.
- B25: an anchor patch with no recognizable ``a/ b/`` headers degrades
       explicitly to un-anchored (None), and an empty anchor set can no
       longer suppress the target-file fallback in assign_effect_labels.
- B26/B27: fuzzy-command SEARCH reduction emits 'pattern@scope' targets and
       SEARCH equivalence gets cross-root scope matching.
- B28: batch _percentile uses true nearest-rank (ceil), not a truncated index.
- R22: the HTML report includes the anchor-analysis section when present.
- B15 (producer): run_comparison returns an explicit "ok" field.
"""

import os
import tempfile
import unittest

from trajviz.converge.alignment import (
    _parse_anchor_files,
    align_trajectories,
    compute_alignment_metrics,
)
from trajviz.converge.batch import _percentile
from trajviz.converge.canonical import (
    ActionCost,
    CanonicalAction,
    assign_effect_labels,
    canonicalize_steps,
    semantic_equivalent,
)
from trajviz.converge.rendering import build_comparison_report_html
from trajviz.insight.comparison import run_comparison


def _step(idx: int, tool_calls: list[dict], tokens: int = 100) -> dict:
    return {
        "index": idx,
        "tokens": {"total": tokens},
        "parts": [],
        "tool_calls": tool_calls,
    }


def _bash(command: str, tool_id: str = "", **extra) -> dict:
    tc = {"tool_name": "Bash", "input": {"command": command}, "status": "success"}
    if tool_id:
        tc["tool_id"] = tool_id
    tc.update(extra)
    return tc


class SedAwkClassificationTests(unittest.TestCase):
    """B22: sed/awk read by default; only in-place flags are writes."""

    def test_read_only_sed_is_file_read(self):
        steps = [_step(0, [_bash("sed -n '1,50p' /repo/src/app.py", "t1")])]
        actions = canonicalize_steps(steps)
        self.assertEqual(actions[0].action_type, "FILE_READ")

    def test_read_only_awk_is_file_read(self):
        steps = [_step(0, [_bash("awk '{print $1}' /repo/data.csv", "t1")])]
        actions = canonicalize_steps(steps)
        self.assertEqual(actions[0].action_type, "FILE_READ")

    def test_in_place_sed_is_file_write(self):
        for cmd in ("sed -i 's/a/b/' /repo/f.py",
                    "sed -i.bak 's/a/b/' /repo/f.py",
                    "sed --in-place 's/a/b/' /repo/f.py"):
            actions = canonicalize_steps([_step(0, [_bash(cmd, "t1")])])
            self.assertEqual(actions[0].action_type, "FILE_WRITE", cmd)

    def test_gawk_inplace_is_file_write(self):
        steps = [_step(0, [_bash("awk -i inplace '{print}' /repo/f.py", "t1")])]
        actions = canonicalize_steps(steps)
        self.assertEqual(actions[0].action_type, "FILE_WRITE")

    def test_tee_stays_file_write(self):
        steps = [_step(0, [_bash("tee /repo/out.log", "t1")])]
        actions = canonicalize_steps(steps)
        self.assertEqual(actions[0].action_type, "FILE_WRITE")

    def test_sed_view_does_not_revert_real_edit(self):
        """The proof scenario: a later `sed -n` view of an edited file must
        not flip the real Edit to 'reverted'."""
        steps = [
            _step(0, [{"tool_name": "Edit", "tool_id": "e1",
                       "input": {"file_path": "/repo/src/app.py"},
                       "status": "success"}]),
            _step(1, [_bash("sed -n '1,50p' /repo/src/app.py", "t2")]),
        ]
        actions = canonicalize_steps(steps)
        assign_effect_labels(actions, steps)
        edit = actions[0]
        self.assertEqual(edit.action_type, "FILE_WRITE")
        self.assertEqual(edit.effect_label, "survived")
        self.assertTrue(edit.effect_detail.get("survives_to_final_patch"))


class PositionalFailureAttributionTests(unittest.TestCase):
    """B23: a failing id-less call must not mark same-tool siblings failed."""

    def test_only_failing_sibling_is_marked_failed(self):
        steps = [_step(0, [
            _bash("make build", metadata={"exit": 0}),
            _bash("make test", metadata={"exit": 1}),
        ])]
        actions = canonicalize_steps(steps)
        assign_effect_labels(actions, steps)
        self.assertNotEqual(actions[0].effect_label, "failed")
        self.assertEqual(actions[1].effect_label, "failed")
        self.assertEqual(actions[1].effect_detail.get("reason"), "exit_code_1")


class JointCountFallbackTests(unittest.TestCase):
    """B24: identical behavior must yield overhead ~1.0 even when only one
    side carries token data."""

    @staticmethod
    def _pair():
        ref = [
            CanonicalAction(step_index=0, action_type="FILE_READ",
                            target="/r/a.py", cost=ActionCost(token_share=5000)),
            CanonicalAction(step_index=1, action_type="FILE_WRITE",
                            target="/r/a.py", cost=ActionCost(token_share=5000)),
        ]
        cmp_ = [
            CanonicalAction(step_index=0, action_type="FILE_READ", target="/r/a.py"),
            CanonicalAction(step_index=1, action_type="FILE_WRITE", target="/r/a.py"),
        ]
        return ref, cmp_

    def test_one_sided_zero_weight_falls_back_jointly(self):
        ref, cmp_ = self._pair()
        alignment = align_trajectories(ref, cmp_)
        metrics = compute_alignment_metrics(alignment, ref, cmp_)
        self.assertEqual(metrics["reference_recall"], 1.0)
        self.assertEqual(metrics["behavioral_precision"], 1.0)
        self.assertEqual(metrics["alignment_f1"], 1.0)
        self.assertEqual(metrics["overhead_ratio"], 1.0)

    def test_reverse_direction_also_consistent(self):
        ref, cmp_ = self._pair()
        alignment = align_trajectories(cmp_, ref)
        metrics = compute_alignment_metrics(alignment, cmp_, ref)
        self.assertEqual(metrics["overhead_ratio"], 1.0)
        self.assertEqual(metrics["alignment_f1"], 1.0)


class AnchorPatchParsingTests(unittest.TestCase):
    """B25: patches without a/ b/ prefixes degrade to un-anchored (None)."""

    def _write_patch(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".patch")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        self.addCleanup(os.unlink, path)
        return path

    def test_git_style_patch_extracts_files(self):
        path = self._write_patch(
            "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x\n+y\n")
        self.assertEqual(_parse_anchor_files(path), {"src/app.py"})

    def test_prefixless_patch_returns_none_not_empty_set(self):
        path = self._write_patch(
            "--- src/app.py\n+++ src/app.py\n@@ -1 +1 @@\n-x\n+y\n")
        self.assertIsNone(_parse_anchor_files(path))

    def test_missing_patch_returns_none(self):
        self.assertIsNone(_parse_anchor_files(None))
        self.assertIsNone(_parse_anchor_files("/nonexistent/anchor.patch"))

    def test_empty_anchor_set_does_not_suppress_target_fallback(self):
        """Defense in depth: assign_effect_labels(anchor_files=set()) must
        behave like anchor_files=None (fall back to identify_target_files),
        so a target-file read still labels 'justified'."""
        steps = [
            _step(0, [{"tool_name": "Edit", "tool_id": "e1",
                       "input": {"file_path": "/repo/src/app.py"},
                       "status": "success"}]),
            _step(1, [{"tool_name": "Read", "tool_id": "r1",
                       "input": {"file_path": "/repo/src/app.py"},
                       "status": "success"}]),
        ]
        actions_none = canonicalize_steps(steps)
        assign_effect_labels(actions_none, steps, anchor_files=None)
        actions_empty = canonicalize_steps(steps)
        assign_effect_labels(actions_empty, steps, anchor_files=set())
        self.assertEqual(
            [a.effect_label for a in actions_empty],
            [a.effect_label for a in actions_none],
        )
        read_action = actions_empty[1]
        self.assertEqual(read_action.action_type, "FILE_READ")
        self.assertEqual(read_action.effect_label, "justified")


class DshSpawnToolTests(unittest.TestCase):
    def test_unmapped_subagent_name_is_agent_spawn(self):
        steps = [_step(0, [{"tool_name": "subagent", "tool_id": "c1",
                            "input": {"description": "explore"},
                            "status": "success"}])]
        actions = canonicalize_steps(steps)
        self.assertEqual(actions[0].action_type, "AGENT_SPAWN")

    def test_claude_task_is_not_agent_spawn(self):
        steps = [_step(0, [{"tool_name": "Task", "tool_id": "t1",
                            "input": {"description": "explore"},
                            "status": "success"}])]
        actions = canonicalize_steps(steps)
        self.assertEqual(actions[0].action_type, "COMMAND")


class FuzzySearchTargetTests(unittest.TestCase):
    """B26/B27: reduced SEARCH targets are comparable with native ones and
    scopes match across workspace roots."""

    def test_reduced_grep_matches_native_search(self):
        cmd = CanonicalAction(
            step_index=0, action_type="COMMAND", target="grep",
            args={"command": "grep -n TODO /repo/src/app.py"})
        native = CanonicalAction(
            step_index=0, action_type="SEARCH", target="TODO@/repo/src/app.py")
        self.assertTrue(semantic_equivalent(cmd, native, fuzzy_commands=True))

    def test_search_scope_matches_across_roots(self):
        a = CanonicalAction(step_index=0, action_type="SEARCH",
                            target="TODO@/home/u1/repo/src")
        b = CanonicalAction(step_index=0, action_type="SEARCH",
                            target="TODO@/tmp/work/repo/src")
        self.assertTrue(semantic_equivalent(a, b))

    def test_different_patterns_never_match(self):
        a = CanonicalAction(step_index=0, action_type="SEARCH",
                            target="TODO@/home/u1/repo/src")
        b = CanonicalAction(step_index=0, action_type="SEARCH",
                            target="FIXME@/home/u1/repo/src")
        self.assertFalse(semantic_equivalent(a, b))


class BatchPercentileTests(unittest.TestCase):
    """B28: nearest-rank percentiles (ceil), matching the docstring."""

    def test_p95_of_ten_values_is_max(self):
        vals = [float(v) for v in range(1, 11)]
        self.assertEqual(_percentile(vals, 0.95), 10.0)

    def test_p5_of_ten_values_is_first_rank(self):
        vals = [float(v) for v in range(1, 11)]
        self.assertEqual(_percentile(vals, 0.05), 1.0)

    def test_p99_separates_from_p95_for_n_30(self):
        vals = [float(v) for v in range(1, 31)]
        self.assertEqual(_percentile(vals, 0.95), 29.0)
        self.assertEqual(_percentile(vals, 0.99), 30.0)

    def test_boundaries_and_empty(self):
        vals = [1.0, 2.0, 3.0]
        self.assertEqual(_percentile(vals, 0.0), 1.0)
        self.assertEqual(_percentile(vals, 1.0), 3.0)
        self.assertEqual(_percentile([], 0.95), 0.0)

    def test_harmful_ratio_batch_scenario(self):
        """Spec scenario: batch of 10 harmful_ratio values 0.0..0.9 must
        report p95 = 0.9 (true nearest-rank), not 0.8."""
        vals = [round(0.1 * i, 1) for i in range(10)]
        self.assertEqual(_percentile(vals, 0.95), 0.9)


class AnchorSectionRenderingTests(unittest.TestCase):
    """R22: HTML report gains the anchor-analysis section when present."""

    def test_anchor_section_rendered_when_present(self):
        report = {
            "outcome": {}, "patterns": [],
            "anchor_analysis": {
                "total_anchor_files": 1,
                "file_classes": {"source": 1},
                "reference": {"write_precision": 1.0, "write_recall": 1.0,
                              "anchor_files_written": 1, "files_written": 1},
                "compared": {"write_precision": 0.5, "write_recall": 1.0,
                             "anchor_files_written": 1, "files_written": 2},
            },
        }
        html = build_comparison_report_html(report)
        self.assertIn("Anchor Analysis", html)
        self.assertIn("Write Precision", html)

    def test_no_anchor_section_when_absent(self):
        html = build_comparison_report_html(
            {"outcome": {}, "patterns": [], "anchor_analysis": None})
        self.assertNotIn("Anchor Analysis", html)


class RunComparisonOkContractTests(unittest.TestCase):
    """B15 producer side: run_comparison reports success explicitly."""

    def test_success_returns_ok_true(self):
        result = run_comparison({"trajectory": []}, {"trajectory": []})
        self.assertIs(result["ok"], True)
        self.assertTrue(result["report_html"])

    def test_reference_load_error_returns_ok_false(self):
        result = run_comparison({"_error": "not json"}, {"trajectory": []})
        self.assertIs(result["ok"], False)
        self.assertIn("Error loading reference trajectory", result["report_html"])

    def test_compared_load_error_returns_ok_false(self):
        result = run_comparison({"trajectory": []}, {"_error": "truncated"})
        self.assertIs(result["ok"], False)
        self.assertIn("Error loading compared trajectory", result["report_html"])

    def test_pipeline_exception_returns_ok_false(self):
        # cmp_raw=None raises inside the try block ("_error" in None)
        result = run_comparison({"trajectory": []}, None)
        self.assertIs(result["ok"], False)
        self.assertIn("Comparison failed", result["report_html"])


if __name__ == "__main__":
    unittest.main()
