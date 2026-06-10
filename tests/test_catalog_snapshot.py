"""Snapshot of the pattern catalog (the 'single source of truth').

This catalog has diverged from the sibling argus/ copy before (different [S]
count and detector ids). This test fails on any add/remove/rename so the change
is deliberate and the cross-repo divergence stays visible. If you intend the
change, update the expected sets here in the same commit.
"""

import unittest

from trajectory_visualizer.core.catalog import ALL_PATTERNS, by_band

EXPECTED_S = {
    "memory-bypass", "premature-code-action",
    "empty-result-churn", "search-loop", "re-read-churn",
    "plan-stall", "plan-thrash", "plan-less-execution",
    "edit-without-inspection", "edit-thrash",
    "late-validation", "validation-avoidance", "test-retry-loop",
    "error-spiral", "recovery-free-retry",
    "verification-skip", "unsupported-completion-claim",
    "redundant-search", "shell-over-tool", "tool-oscillation",
}
EXPECTED_H = {
    "phase-oscillation", "premature-implementation",
    "semantic-fruitless-exploration", "semantic-plan-stall",
    "debug-wo-hypothesis", "prompt-skim", "memory-contamination",
}
EXPECTED_DIVERGENCE = {
    "rapid-rewrite", "scope-drift", "off-anchor-exploration",
    "dead-end-exploration", "ordering-inefficiency", "iterative-refinement",
}


class CatalogSnapshotTests(unittest.TestCase):
    def test_structural_band(self):
        self.assertEqual({p.id for p in by_band("[S]")}, EXPECTED_S)

    def test_hypothesis_band(self):
        self.assertEqual({p.id for p in by_band("[H]")}, EXPECTED_H)
        self.assertTrue(all(p.requires_semantic_labels for p in by_band("[H]")))

    def test_divergence_band(self):
        self.assertEqual({p.id for p in by_band("divergence")}, EXPECTED_DIVERGENCE)

    def test_total_count_and_unique_ids(self):
        self.assertEqual(len(ALL_PATTERNS), 33)  # 20 + 7 + 6
        ids = [p.id for p in ALL_PATTERNS]
        self.assertEqual(len(ids), len(set(ids)), "duplicate detector ids in catalog")


if __name__ == "__main__":
    unittest.main()
