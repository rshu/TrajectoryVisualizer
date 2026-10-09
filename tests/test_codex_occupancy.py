"""Context occupancy of a Codex step is its largest single request, not their sum.

A displayed Codex step can span several model responses — commentary, then a
tool call, then more commentary — and the converter rightly SUMS their
``last_token_usage`` deltas, because that is what the run consumed. Occupancy is
a different question: the window holds one prompt at a time. Reading the summed
input as one prompt made rollouts report a peak well above 100% of their
declared window (a step of four responses of about 100k tokens each read as
nearly 400k). The converter now also records the largest single request, and
occupancy uses it.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from trajviz.insight.context_usage import context_pressure_series, context_pressure_stats, step_context_occupancy
from trajviz.insight.session import LoadedSession, load_session

_WINDOW = 258_400


def _ev(ts: int, etype: str, payload: dict) -> dict:
    return {"timestamp": f"2026-01-01T00:{ts // 60:02d}:{ts % 60:02d}Z", "type": etype, "payload": payload}


def _tokens(ts: int, prompt: int, cached: int, cumulative: int) -> dict:
    usage = {"input_tokens": prompt, "cached_input_tokens": cached, "output_tokens": 200,
             "reasoning_output_tokens": 0, "total_tokens": prompt + 200}
    total = dict(usage, input_tokens=cumulative, total_tokens=cumulative + 200)
    return _ev(ts, "event_msg", {"type": "token_count", "info": {
        "last_token_usage": usage, "total_token_usage": total, "model_context_window": _WINDOW}})


def _rollout() -> list[dict]:
    rows = [
        _ev(0, "session_meta", {"id": "s", "cwd": "/w"}),
        _ev(1, "event_msg", {"type": "task_started", "model_context_window": _WINDOW}),
        _ev(2, "response_item", {"type": "message", "role": "user",
                                 "content": [{"type": "input_text", "text": "go"}]}),
    ]
    # One displayed assistant step made of four model responses of ~98k each.
    cumulative = 0
    for k, prompt in enumerate((97_000, 98_000, 98_500, 97_900)):
        ts = 10 + 10 * k
        rows.append(_ev(ts, "response_item", {"type": "message", "role": "assistant",
                                              "content": [{"type": "output_text", "text": f"part {k}"}]}))
        cumulative += prompt
        rows.append(_tokens(ts + 1, prompt, prompt - 1_000, cumulative))
    rows.append(_ev(60, "event_msg", {"type": "task_complete"}))
    return rows


class CodexMultiResponseOccupancyTests(unittest.TestCase):
    def setUp(self):
        path = os.path.join(tempfile.mkdtemp(), "rollout.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(json.dumps(r) for r in _rollout()))
        result = load_session(path)
        self.assertIsInstance(result, LoadedSession)
        self.session = result
        self.step = next(s for s in result.steps if s.get("role") == "assistant")

    def test_tokens_still_sum_every_response(self):
        self.assertEqual(self.step["tokens"]["input"], 97_000 + 98_000 + 98_500 + 97_900)

    def test_occupancy_is_the_largest_single_request(self):
        occ = step_context_occupancy(self.step)
        self.assertEqual(occ["occupancy"], 98_500)
        self.assertEqual(occ["cache_read"], 97_500)
        self.assertEqual(occ["fresh"], 1_000)

    def test_peak_pressure_stays_within_the_declared_window(self):
        stats = context_pressure_stats(context_pressure_series(self.session.steps, raw=self.session.raw))
        self.assertEqual(stats["window_limit"], _WINDOW)
        self.assertLessEqual(stats["peak_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
