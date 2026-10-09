"""Regression tests for Converge ordering-inefficiency detection."""

import unittest

from trajviz.converge.alignment import align_trajectories
from trajviz.converge.canonical import CanonicalAction
from trajviz.converge.divergence import classify_divergences


def _actions(*targets: str) -> list[CanonicalAction]:
    return [
        CanonicalAction(
            step_index=index,
            action_type="COMMAND",
            target=target,
            effect_label="survived",
        )
        for index, target in enumerate(targets)
    ]


def _classify(
    reference: list[CanonicalAction],
    compared: list[CanonicalAction],
) -> list[dict]:
    alignment = align_trajectories(reference, compared)
    extra = [compared[index] for index in alignment["extra"]]
    matched = [compared[index] for _, index in alignment["matched_pairs"]]
    return classify_divergences(
        extra,
        matched,
        compared,
        matched_pairs=alignment["matched_pairs"],
        reference_actions=reference,
    )


class OrderingInefficiencyTests(unittest.TestCase):
    def test_substantial_reordering_is_detected_after_monotonic_alignment(self):
        reference = _actions("a", "b", "c", "d", "e")
        compared = _actions("e", "d", "c", "b", "a")

        patterns = _classify(reference, compared)
        ordering = [p for p in patterns if p["type"] == "ordering_inefficiency"]

        self.assertEqual(len(ordering), 1)
        self.assertIn("10 inversions", ordering[0]["evidence"][0])
        self.assertIn("5 unique common actions", ordering[0]["evidence"][0])

    def test_minor_reordering_stays_below_threshold(self):
        reference = _actions("a", "b", "c", "d", "e")
        compared = _actions("a", "c", "b", "d", "e")

        patterns = _classify(reference, compared)

        self.assertNotIn("ordering_inefficiency", {p["type"] for p in patterns})

    def test_reordering_at_thirty_percent_threshold_is_detected(self):
        reference = _actions("a", "b", "c", "d", "e")
        compared = _actions("b", "c", "d", "a", "e")

        patterns = _classify(reference, compared)
        ordering = next(p for p in patterns if p["type"] == "ordering_inefficiency")

        self.assertIn("3 inversions", ordering["evidence"][0])
        self.assertIn("threshold 3", ordering["evidence"][0])

    def test_duplicate_actions_count_once(self):
        reference = _actions("a", "b", "c", "d")
        compared = _actions("d", "d", "c", "b", "a", "a")

        patterns = _classify(reference, compared)
        ordering = next(p for p in patterns if p["type"] == "ordering_inefficiency")

        self.assertIn("6 inversions", ordering["evidence"][0])
        self.assertIn("4 unique common actions", ordering["evidence"][0])
        self.assertEqual(len(ordering["steps"]), 4)

    def test_reference_actions_remains_optional_for_legacy_callers(self):
        compared = _actions("c", "b", "a")

        patterns = classify_divergences([], [], compared, matched_pairs=[])

        self.assertNotIn("ordering_inefficiency", {p["type"] for p in patterns})


if __name__ == "__main__":
    unittest.main()
