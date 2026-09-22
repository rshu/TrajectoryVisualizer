"""Per-step analytics aligned with the trajectory step model."""

from .metrics import tool_call_stats_duration_ms
from .parser import infer_non_cache_input


def compute_step_analytics(steps: list[dict]) -> list[dict]:
    """Compute derived per-message metrics aligned 1:1 with steps."""
    analytics: list[dict] = []
    for i, step in enumerate(steps):
        duration_s = step["duration"]

        # Non-spawn tool time (may still overcount parallel calls within a step).
        tool_time_ms = 0
        for tc in step["tool_calls"]:
            v = tool_call_stats_duration_ms(tc)
            if v is not None:
                tool_time_ms += v

        tool_time_share = None
        if duration_s is not None and duration_s > 0:
            tool_time_share = round(tool_time_ms / (duration_s * 1000), 4)

        tok_total = step["tokens"]["total"]
        cache_read = step["tokens"]["cache_read"]

        tok_per_s = None
        if duration_s is not None and duration_s > 0:
            tok_per_s = round(tok_total / duration_s, 1)

        cache_ratio = round(cache_read / tok_total, 4) if tok_total > 0 else 0.0
        input_tok = step["tokens"]["input"]
        output_tok = step["tokens"]["output"]
        reasoning_tok = step["tokens"].get("reasoning", 0)
        non_cache_tok = infer_non_cache_input(
            total_tokens=tok_total,
            input_tokens=input_tok,
            output_tokens=output_tok,
            reasoning_tokens=reasoning_tok,
            cache_read_tokens=cache_read,
        )
        out_in_ratio = round(output_tok / input_tok, 4) if input_tok > 0 else None

        # Sorted unique part types
        part_types = sorted({p.get("type", "") for p in step["parts"]} - {""})
        part_mix = ",".join(part_types)

        # Idle gap from previous step
        idle_before_s = None
        if i > 0:
            prev_completed = steps[i - 1].get("time_completed_ms")
            this_created = step.get("time_created_ms")
            if (isinstance(prev_completed, (int, float))
                    and isinstance(this_created, (int, float))):
                idle_before_s = round((this_created - prev_completed) / 1000, 2)

        analytics.append({
            "index": step["index"],
            "role": step["role"],
            "agent": step.get("agent", ""),
            "model_id": step.get("model_id", ""),
            "duration_s": duration_s,
            "tool_time_ms": tool_time_ms,
            "tool_time_share": tool_time_share,
            "tok_total": tok_total,
            "tok_per_s": tok_per_s,
            "cache_ratio": cache_ratio,
            "non_cache_tok": non_cache_tok,
            "out_in_ratio": out_in_ratio,
            "tool_calls": step["tool_call_count"],
            "finish": step["finish"],
            "part_mix": part_mix,
            "idle_before_s": idle_before_s,
        })

    return analytics
