"""End-to-end smoke test for the converge comparison pipeline:
canonicalize -> effect labels -> align -> metrics -> milestones -> divergence.
"""

import unittest

from trajectory_visualizer.converge import canonical, alignment, milestones, divergence


def _tc(name, inp, status="completed", output="ok"):
    return {"type": "tool_call", "tool_name": name, "input": inp,
            "output": output, "status": status, "tool_id": f"t-{name}"}


def _step(i, parts):
    return {"index": i, "role": "assistant",
            "tokens": {"total": 100, "input": 50, "output": 50},
            "duration": 1.0, "tool_calls": parts, "parts": parts}


class ConvergePipelineTests(unittest.TestCase):
    def setUp(self):
        # Reference: read then edit a.py. Compared: same, plus an extra rewrite.
        ref = [_step(0, [_tc("Read", {"file_path": "a.py"})]),
               _step(1, [_tc("Edit", {"file_path": "a.py"})])]
        cmp = [_step(0, [_tc("Read", {"file_path": "a.py"})]),
               _step(1, [_tc("Edit", {"file_path": "a.py"})]),
               _step(2, [_tc("Edit", {"file_path": "a.py"})])]
        self.ra = canonical.canonicalize_steps(ref)
        self.ca = canonical.canonicalize_steps(cmp)
        canonical.assign_effect_labels(self.ra, ref)
        canonical.assign_effect_labels(self.ca, cmp)
        self.al = alignment.align_trajectories(self.ra, self.ca)

    def test_canonicalization(self):
        self.assertEqual([a.action_type for a in self.ra], ["FILE_READ", "FILE_WRITE"])
        self.assertEqual([a.action_type for a in self.ca],
                         ["FILE_READ", "FILE_WRITE", "FILE_WRITE"])

    def test_alignment_matches_reference_and_flags_extra(self):
        self.assertEqual(self.al["unrecovered"], [])      # all reference matched
        self.assertEqual(self.al["extra"], [1])           # the middle extra write
        m = alignment.compute_alignment_metrics(self.al, self.ra, self.ca)
        self.assertEqual(m["reference_recall"], 1.0)
        self.assertLess(m["behavioral_precision"], 1.0)   # compared has overhead

    def test_milestones(self):
        ms = milestones.extract_milestones(self.ra)
        self.assertEqual(ms["first_relevant_file"], 0)
        self.assertEqual(ms["first_edit"], 1)

    def test_divergence_flags_rewrite(self):
        extra = [self.ca[i] for i in self.al["extra"]]
        matched = [self.ca[j] for _, j in self.al["matched_pairs"]]
        divs = divergence.classify_divergences(extra, matched, self.ca, self.al["matched_pairs"])
        self.assertIn("reverted_and_rewritten", [d.get("type") for d in divs])


if __name__ == "__main__":
    unittest.main()
