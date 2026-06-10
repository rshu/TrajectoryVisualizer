"""infer_non_cache_input must not double-discount cache on formats lacking a total."""

import unittest

from trajectory_visualizer.insight.parser import infer_non_cache_input


class NonCacheInputTests(unittest.TestCase):
    def test_no_total_treats_input_as_fresh(self):
        # OpenCode shape: no per-step total -> total=0. input(100) is already
        # fresh; must NOT subtract cache_read(80) again.
        self.assertEqual(infer_non_cache_input(0, 100, 0, 0, 80), 100)

    def test_fresh_schema_total_includes_cache_read(self):
        # total = input + output + reasoning + cache_read -> input is fresh.
        self.assertEqual(infer_non_cache_input(650, 100, 50, 0, 200), 100)

    def test_cached_schema_input_includes_cache(self):
        # total = input + output (+reasoning); input includes cache_read.
        self.assertEqual(infer_non_cache_input(150, 100, 50, 0, 80), 20)

    def test_total_only_no_breakdown(self):
        self.assertEqual(infer_non_cache_input(500, 0, 0, 0, 0), 500)


if __name__ == "__main__":
    unittest.main()
