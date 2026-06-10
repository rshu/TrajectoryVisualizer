"""Claude Code token totals must include cache-creation tokens.

Anthropic reports input_tokens / cache_read_input_tokens /
cache_creation_input_tokens as disjoint, additively-billed categories.
"""

import unittest

from trajectory_visualizer.insight.loaders import _cc_extract_usage


class CCTokenUsageTests(unittest.TestCase):
    def test_total_includes_cache_creation(self):
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 300,
        }
        r = _cc_extract_usage(usage)
        self.assertEqual(r["total"], 650)  # 100 + 50 + 200 + 300
        self.assertEqual(r["cache"], {"read": 200, "write": 300})

    def test_no_cache_is_plain_sum(self):
        r = _cc_extract_usage({"input_tokens": 10, "output_tokens": 5})
        self.assertEqual(r["total"], 15)

    def test_empty_usage_is_zeroed(self):
        r = _cc_extract_usage(None)
        self.assertEqual(r["total"], 0)


if __name__ == "__main__":
    unittest.main()
