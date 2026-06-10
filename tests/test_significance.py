"""scipy is a declared dependency, so the intervention test must use Wilcoxon
(not silently degrade to the weaker sign test)."""

import unittest

# Import the module (not the function) so pytest does not collect the
# production helper `test_significance` as if it were a test case.
from trajectory_visualizer.converge import intervention


class SignificanceTests(unittest.TestCase):
    def test_uses_wilcoxon_when_scipy_present(self):
        before = [10, 11, 9, 12, 10, 11]
        after = [7, 8, 6, 9, 7, 8]
        result = intervention.test_significance(before, after)
        self.assertEqual(result["test_method"], "wilcoxon")
        self.assertIsNotNone(result["p_value"])
        self.assertTrue(result["significant"])

    def test_too_few_observations_reported(self):
        result = intervention.test_significance([1.0], [2.0])
        self.assertEqual(result["test_method"], "none")
        self.assertFalse(result["significant"])


if __name__ == "__main__":
    unittest.main()
