"""AI Trajectory Analysis: pack dashboard stats and ask an LLM about them."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import requests

from .analytics import compute_step_analytics
from .diagnostics import (
    compute_failure_chain_metrics,
    detect_failure_chains,
    detect_performance_bottlenecks,
    extract_file_interactions,
)
from .formatting import wall_clock_fmt
from .llm_config import AnalysisLLMConfig, resolve_analysis_config, setup_help_text
from .loaders import FORMAT_LABELS, detect_format
from .metrics import (
    build_message_metrics,
    compute_agent_summary,
    compute_health_verdict,
    compute_metrics,
    extract_agent_info,
    generate_agent_insights,
)
from .patterns import (
    detect_failure_patterns,
    detect_fruitless_streaks,
    detect_tool_selection_antipatterns,
    detect_tool_sequences,
)
from .tool_failure import tool_call_failed

if TYPE_CHECKING:
    from .session import LoadedSession

_BRIEF_CHAR_LIMIT = 28_000
_OUTPUT_CLIP = 180
_ARG_CLIP = 100
_HISTORY_TURNS = 12  # user+assistant messages kept

AUTO_ANALYSIS_QUESTION = (
    "请根据仪表盘统计分析这次轨迹：标出主要性能瓶颈（步骤编号、阶段、耗时与原因）、"
    "错误模式、根因，以及可落地的优化建议。"
)

SYSTEM_PROMPT = """You are TrajViz's trajectory analyst. You see a structured brief of
dashboard statistics for one coding-agent run. The user cannot see the brief —
they see the dashboard.

Your job:
- Locate where the run went wrong and which steps to inspect (use step indices).
- Explain performance bottlenecks using duration, tokens, cache, and tool timing.
- Point at wasted work (retries, empty searches, tool-selection antipatterns).
- Identify pipeline stages actually present in this run (planning, code generation,
  transformation, optimization, compile/build, testing, repair, verification, etc.).
- If the brief lacks evidence for a claim, say so. Do not invent tool outputs.
- Do not ask the user to paste the trajectory; you already have the stats.

When the user asks for an overall analysis (including the first pass after load),
structure the reply in Markdown as:

## 主要性能瓶颈
1. 步骤 X — [阶段] ([严重程度])
   耗时: …
   问题: …
   结果: …
   分析: …（为何慢或失败；引用 tool_time_share、tool_wait_pct、p95_tool_duration、
   max_tool_duration、tokens_per_second、tokens_per_tool 等 brief 中的指标）
2. …

## 错误模式
按类型归纳主要错误，并标出出现的步骤。

## 根因
基于指标与步骤证据解释根因，不要只复述症状。

## 优化建议
1. [方向]: [具体建议]
2. …

Each bottleneck must cite an exact step number, its duration, and why the approach
is inefficient. Follow-up questions should be answered directly and conversationally
in the same language — do not repeat the full report unless asked.

Language:
- Write the entire reply in Simplified Chinese (简体中文).
- English only for: code, tool names, file paths, JSON keys, metric keys,
  model names, and step numbers, so they still match the dashboard.
"""

ChatFn = Callable[[AnalysisLLMConfig, str, list[dict[str, str]]], str]


def _clip(text: str, limit: int) -> str:
    text = text.replace("\r\n", "\n").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _tool_arg_preview(tool_call: dict) -> str:
    inp = tool_call.get("input")
    if isinstance(inp, str) and inp.strip():
        return _clip(inp.replace("\n", " "), _ARG_CLIP)
    if not isinstance(inp, dict):
        return ""
    for key in ("command", "file_path", "path", "pattern", "query", "glob_pattern", "glob"):
        value = inp.get(key)
        if value:
            return f"{key}={_clip(str(value).replace(chr(10), ' '), _ARG_CLIP)}"
    return _clip(json.dumps(inp, ensure_ascii=False), _ARG_CLIP)


def _first_user_task(steps: list[dict]) -> str:
    for step in steps:
        if step.get("role") != "user":
            continue
        preview = step.get("text_preview") or ""
        if isinstance(preview, str) and preview.strip():
            return _clip(preview, 600)
    return ""


def _session_header(raw: dict, steps: list[dict], metrics: dict, wall_fmt: str) -> list[str]:
    fmt = detect_format(raw) if isinstance(raw, dict) else "unknown"
    label = FORMAT_LABELS.get(fmt, fmt or "unknown")
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    model_id, provider_id, _agent_id = extract_agent_info(steps)
    title = metadata.get("title") if isinstance(metadata.get("title"), str) else ""
    info = raw.get("info")
    if not title and isinstance(info, dict) and isinstance(info.get("title"), str):
        title = info["title"]
    lines = [
        f"format: {label} ({fmt})",
        f"steps: {metrics.get('total_steps', 0)}",
        f"wall_clock: {wall_fmt}",
        f"tool_calls: {metrics.get('tool_call_count', 0)} "
        f"({metrics.get('tool_success_rate', 0)}% success, "
        f"{metrics.get('tool_fail', 0)} failed)",
    ]
    if model_id:
        lines.append(f"model: {model_id}" + (f" / {provider_id}" if provider_id else ""))
    if title:
        lines.append(f"title: {_clip(title, 160)}")
    task = _first_user_task(steps)
    if task:
        lines.append(f"user_task: {task}")
    return lines


def _pct_from_fraction(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{round(float(value) * 100, 1)}%"
    except (TypeError, ValueError):
        return "n/a"


def _metrics_lines(metrics: dict, steps: list[dict]) -> list[str]:
    tok = metrics.get("tokens") if isinstance(metrics.get("tokens"), dict) else {}
    time_share = metrics.get("tool_time_fraction")
    fail_rate = metrics.get("tool_system_failure_rate")
    lines = [
        f"avg_duration_s: {metrics.get('avg_duration', 0)}",
        f"median_duration_s: {metrics.get('median_duration', 0)}",
        f"p95_duration_s: {metrics.get('p95_duration', 0)}",
        f"max_duration_s: {metrics.get('max_duration', 0)}",
        f"assistant_steps: {metrics.get('assistant_steps', 0)}",
        f"user_steps: {metrics.get('user_steps', 0)}",
        f"tokens_total: {tok.get('total', 0)}",
        f"tokens_input: {tok.get('input', 0)}",
        f"tokens_output: {tok.get('output', 0)}",
        f"tokens_reasoning: {tok.get('reasoning', 0)}",
        f"tokens_cache_read: {tok.get('cache_read', 0)}",
        f"tokens_cache_write: {tok.get('cache_write', 0)}",
        f"fresh_input_tokens: {metrics.get('non_cache_tokens', 0)} "
        f"({metrics.get('non_cache_ratio', 0)}%)",
        f"avg_tokens_per_step: {metrics.get('avg_tokens_per_step', 0)}",
        f"tokens_per_second: {metrics.get('tokens_per_second', 0)}",
        f"tokens_per_tool: {metrics.get('tokens_per_tool', 0)}",
        f"output_tokens_per_sec: {metrics.get('output_tokens_per_sec')}",
        f"p95_tool_duration: {metrics.get('p95_tool_duration', 0)}s",
        f"max_tool_duration: {metrics.get('max_tool_duration', 0)}s",
        f"avg_tool_duration: {metrics.get('avg_tool_duration', 0)}s",
        f"tool_wait_pct: {metrics.get('tool_wait_share', 0)}%",
        f"tool_time_share: {_pct_from_fraction(time_share)}",
        f"tool_system_failure_rate: {_pct_from_fraction(fail_rate)}",
        f"tool_breakdown: {metrics.get('tool_breakdown') or {}}",
    ]
    if metrics.get("time_to_first_token") is not None:
        lines.append(f"time_to_first_token: {metrics.get('time_to_first_token')}s")
    totals = [
        (step.get("tokens") or {}).get("total", 0)
        for step in steps
        if isinstance(step.get("tokens"), dict)
    ]
    if totals:
        lines.append(
            f"step_tokens_min/max/avg: {min(totals)} / {max(totals)} / "
            f"{round(sum(totals) / len(totals))}"
        )
    if metrics.get("output_throughput_incomplete"):
        lines.append(
            "output_throughput_coverage: "
            f"{metrics.get('output_throughput_timed_steps')}/"
            f"{metrics.get('output_throughput_total_steps')} timed assistant steps"
        )
    return lines


def _file_lines_from_interactions(interactions: list[dict]) -> list[str]:
    if not interactions:
        return ["(none)"]
    unique = {item.get("path") for item in interactions if item.get("path")}
    reads = sum(1 for item in interactions if item.get("type") == "read")
    writes = sum(1 for item in interactions if item.get("type") == "write")
    searches = sum(1 for item in interactions if item.get("type") == "search")
    return [
        f"unique_files: {len(unique)}",
        f"read_count: {reads}",
        f"write_count: {writes}",
        f"search_count: {searches}",
    ]


def _tool_sequence_lines(sequences: list[dict]) -> list[str]:
    if not sequences:
        return ["(none)"]
    lines = []
    for item in sequences[:5]:
        seq = " → ".join(str(name) for name in (item.get("sequence") or []))
        steps = item.get("step_indices") or []
        lines.append(
            f"- {seq} ×{item.get('frequency', 0)} at steps {steps[:8]}"
        )
    return lines


def _verdict_lines(verdicts: list[dict]) -> list[str]:
    lines = []
    for item in verdicts:
        lines.append(
            f"- {item.get('metric')}: {item.get('status')} "
            f"({item.get('label')}) — {item.get('detail')}"
        )
    return lines


def _agent_lines(summaries: list[dict]) -> list[str]:
    if not summaries:
        return ["(single-agent or unlabeled)"]
    lines = []
    for agent in summaries:
        lines.append(
            f"- {agent.get('label')}: {agent.get('step_count')} steps, "
            f"{agent.get('total_tokens')} tok, {agent.get('total_duration_s')}s, "
            f"{agent.get('tool_call_count')} tools, {agent.get('error_count')} errors, "
            f"cache {agent.get('cache_efficiency_pct')}%"
        )
    for insight in generate_agent_insights(summaries):
        lines.append(f"- insight: {insight}")
    return lines


def _bottleneck_lines(bottlenecks: list[dict]) -> list[str]:
    if not bottlenecks:
        return ["(none)"]
    lines = []
    for item in bottlenecks:
        lines.append(
            f"- step {item.get('step_idx')}: {item.get('duration')}s — "
            f"{_clip(str(item.get('title') or ''), 120)}"
        )
        detail = item.get("detail")
        if detail:
            lines.append(f"  {_clip(str(detail), 200)}")
        why = item.get("why")
        if why:
            lines.append(f"  why: {_clip(str(why), 160)}")
    return lines


def _failure_lines(
    patterns: list[dict],
    chains: list[dict],
    chain_metrics: dict,
) -> list[str]:
    lines = [
        f"failure_chains: {chain_metrics.get('total_chains', 0)} "
        f"(longest {chain_metrics.get('longest_chain', 0)} steps, "
        f"{chain_metrics.get('chain_step_pct', 0)}% of assistant steps)"
    ]
    for chain in chains[:8]:
        lines.append(
            f"- chain steps {chain.get('start')}–{chain.get('end')}: "
            f"{chain.get('steps')}"
        )
    if not patterns:
        lines.append("error_clusters: (none)")
        return lines
    lines.append("error_clusters:")
    for pat in patterns[:8]:
        recovery = pat.get("recovery_path") or []
        rec = " → ".join(str(name) for name in recovery) if recovery else "unrecovered"
        lines.append(
            f"- {pat.get('cluster_label')} ×{pat.get('count')} "
            f"at steps {pat.get('steps', [])[:12]} recovery: {rec}"
        )
    return lines


def _waste_lines(streaks: list[dict], flags: list[dict]) -> list[str]:
    lines = []
    if streaks:
        lines.append("fruitless_search_streaks:")
        for streak in streaks[:8]:
            lines.append(
                f"- steps {streak.get('start_step')}–{streak.get('end_step')} "
                f"length {streak.get('length')} tools={streak.get('tools', [])[:8]}"
            )
    else:
        lines.append("fruitless_search_streaks: (none)")
    if flags:
        lines.append("bash_used_as_read:")
        for flag in flags[:8]:
            lines.append(f"- step {flag.get('step')}: {_clip(str(flag.get('command') or ''), 80)}")
    else:
        lines.append("bash_used_as_read: (none)")
    return lines


def _error_step_lines(steps: list[dict], limit: int = 16) -> list[str]:
    lines: list[str] = []
    for step in steps:
        if not step.get("error_count"):
            continue
        for tool_call in step.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            status = str(tool_call.get("status") or "")
            if not tool_call_failed(tool_call):
                continue
            preview = _tool_arg_preview(tool_call)
            err_type = tool_call.get("error_type")
            error = tool_call.get("error")
            output = _clip(str(tool_call.get("output") or error or ""), _OUTPUT_CLIP)
            lines.append(
                f"- step {step.get('index')} {tool_call.get('tool_name', '?')} "
                f"status={status or '?'} type={err_type or ''} {preview}"
            )
            if output:
                lines.append(f"  output: {output}")
            if len(lines) >= limit * 2:
                lines.append("- …")
                return lines
    return lines or ["(no tool errors)"]


def _timeline_lines(steps: list[dict], limit: int = 80) -> list[str]:
    n = len(steps)
    if n <= limit:
        return [_timeline_row(step) for step in steps]
    head, tail = limit // 2, limit - limit // 2
    rows = [_timeline_row(step) for step in steps[:head]]
    rows.append(f"… {n - limit} steps omitted …")
    rows.extend(_timeline_row(step) for step in steps[-tail:])
    return rows


def _timeline_row(step: dict) -> str:
    role = step.get("role") or "?"
    tools = [
        str(tc.get("tool_name") or "?")
        for tc in (step.get("tool_calls") or [])
        if isinstance(tc, dict)
    ]
    dur = step.get("duration")
    dur_s = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "-"
    tok = (step.get("tokens") or {}).get("total", 0) if isinstance(step.get("tokens"), dict) else 0
    err = " ERR" if step.get("error_count") else ""
    tool_s = ",".join(tools[:6]) if tools else "-"
    preview = ""
    if role == "user":
        preview = " " + _clip(str(step.get("text_preview") or ""), 80)
    return f"{step.get('index')} {role} {dur_s} {tok}tok {tool_s}{err}{preview}"


def _compose_brief(
    *,
    raw: dict,
    steps: list[dict],
    metrics: dict,
    wall_fmt: str,
    verdicts: list[dict],
    agents: list[dict],
    bottlenecks: list[dict],
    fail_pats: list[dict],
    chains: list,
    chain_metrics: dict,
    streaks: list[dict],
    bash_flags: list[dict],
    tool_seqs: list[dict],
    file_interactions: list[dict],
) -> str:
    sections = [
        ("SESSION", _session_header(raw, steps, metrics, wall_fmt)),
        ("HEALTH", _verdict_lines(verdicts)),
        ("PERFORMANCE", _metrics_lines(metrics, steps)),
        ("AGENTS", _agent_lines(agents)),
        ("FILES", _file_lines_from_interactions(file_interactions)),
        ("TOOL_SEQUENCES", _tool_sequence_lines(tool_seqs)),
        ("BOTTLENECKS", _bottleneck_lines(bottlenecks)),
        ("FAILURES", _failure_lines(fail_pats, chains, chain_metrics)),
        ("WASTE", _waste_lines(streaks, bash_flags)),
        ("ERROR_STEPS", _error_step_lines(steps)),
        ("STEP_INDEX", _timeline_lines(steps)),
    ]
    chunks = ["# Trajectory analysis brief", ""]
    for title, lines in sections:
        chunks.append(f"## {title}")
        chunks.extend(lines)
        chunks.append("")
    brief = "\n".join(chunks).strip() + "\n"
    if len(brief) > _BRIEF_CHAR_LIMIT:
        brief = brief[: _BRIEF_CHAR_LIMIT - 20] + "\n…[truncated]\n"
    return brief


def build_analysis_brief(steps: list[dict], raw: dict | None = None) -> str:
    """Pack dashboard statistics into a compact text brief for the LLM."""
    raw = raw if isinstance(raw, dict) else {}
    if not steps:
        return ""
    message_rows = build_message_metrics(steps)
    metrics = compute_metrics(steps, raw, message_rows=message_rows)
    _, wall_fmt = wall_clock_fmt(metrics)
    analytics = compute_step_analytics(steps)
    verdicts = compute_health_verdict(metrics, analytics)
    agents = compute_agent_summary(steps, raw)
    bottlenecks = detect_performance_bottlenecks(steps, analytics)
    fail_pats = detect_failure_patterns(steps)
    chains = detect_failure_chains(steps)
    assistant_n = sum(1 for step in steps if step.get("role") == "assistant")
    chain_metrics = compute_failure_chain_metrics(chains, assistant_n)
    streaks = detect_fruitless_streaks(steps)
    bash_flags = detect_tool_selection_antipatterns(steps)
    tool_seqs = detect_tool_sequences(steps)

    return _compose_brief(
        raw=raw,
        steps=steps,
        metrics=metrics,
        wall_fmt=wall_fmt,
        verdicts=verdicts,
        agents=agents,
        bottlenecks=bottlenecks,
        fail_pats=fail_pats,
        chains=chains,
        chain_metrics=chain_metrics,
        streaks=streaks,
        bash_flags=bash_flags,
        tool_seqs=tool_seqs,
        file_interactions=extract_file_interactions(steps),
    )


def build_analysis_brief_from_session(session: LoadedSession) -> str:
    """Build the analysis brief from a ``LoadedSession`` (no re-detection)."""
    if not session.steps:
        return ""
    return _compose_brief(
        raw=session.raw,
        steps=session.steps,
        metrics=session.metrics,
        wall_fmt=session.wall_clock,
        verdicts=session.verdicts,
        agents=session.agent_summaries,
        bottlenecks=session.performance_bottlenecks,
        fail_pats=session.failure_patterns,
        chains=session.failure_chains,
        chain_metrics=session.chain_metrics,
        streaks=session.fruitless_streaks,
        bash_flags=session.tool_selection,
        tool_seqs=session.tool_sequences,
        file_interactions=session.file_interactions,
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content or "")


def _history_for_api(history: list[dict]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for message in history:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _message_text(message.get("content")).strip()
        if not text:
            continue
        cleaned.append({"role": role, "content": text})
    return cleaned[-_HISTORY_TURNS:]


def complete_chat(
    config: AnalysisLLMConfig,
    system: str,
    messages: list[dict[str, str]],
) -> str:
    """One chat completion. ``messages`` is user/assistant only."""
    if config.provider == "anthropic":
        return _call_anthropic(config, system, messages)
    return _call_openai(config, system, messages)


def _call_openai(
    config: AnalysisLLMConfig,
    system: str,
    messages: list[dict[str, str]],
) -> str:
    url = f"{config.base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "system", "content": system}, *messages],
        "max_tokens": config.max_tokens,
    }
    if config.temperature is not None:
        body["temperature"] = config.temperature
    response = requests.post(url, json=body, headers=headers, timeout=config.timeout)
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def _call_anthropic(
    config: AnalysisLLMConfig,
    system: str,
    messages: list[dict[str, str]],
) -> str:
    url = f"{config.base_url}/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
    }
    body: dict[str, Any] = {
        "model": config.model,
        "system": system,
        "messages": messages,
        "max_tokens": config.max_tokens,
    }
    if config.temperature is not None:
        body["temperature"] = config.temperature
    response = requests.post(url, json=body, headers=headers, timeout=config.timeout)
    response.raise_for_status()
    data = response.json()
    blocks = data.get("content", [])
    texts = [block["text"] for block in blocks if isinstance(block, dict) and block.get("type") == "text"]
    if not texts:
        raise ValueError("Anthropic response contained no text blocks")
    return "\n".join(texts).strip()


def _public_http_error(exc: BaseException) -> str:
    """Human error without response bodies that might echo secrets."""
    if isinstance(exc, requests.Timeout):
        return "The analysis model timed out. Retry, or raise ANALYZE_TIMEOUT."
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        if status in {401, 403}:
            return f"The analysis API rejected the key (HTTP {status}). Check ANALYZE_API_KEY."
        if status == 404:
            return f"The analysis API returned HTTP {status}. Check ANALYZE_BASE_URL and ANALYZE_MODEL."
        return f"The analysis API returned HTTP {status}."
    if isinstance(exc, requests.RequestException):
        return "Could not reach the analysis API. Check ANALYZE_BASE_URL and the network."
    return f"The analysis model request failed: {type(exc).__name__}."


def answer_question(
    question: str,
    history: list[dict] | None,
    brief: str,
    *,
    config: AnalysisLLMConfig | None = None,
    chat_fn: ChatFn | None = None,
) -> list[dict]:
    """Append the user question and an assistant reply to *history*."""
    history = [dict(message) for message in (history or []) if isinstance(message, dict)]
    question = (question or "").strip()
    if not question:
        return history
    history.append({"role": "user", "content": question})

    if not (brief or "").strip():
        history.append({
            "role": "assistant",
            "content": "请先加载一条轨迹。我会使用当前运行的仪表盘统计来分析。",
        })
        return history

    cfg = config or resolve_analysis_config()
    if not cfg.ready:
        history.append({"role": "assistant", "content": setup_help_text(cfg)})
        return history

    api_messages = _history_for_api(history)
    system = SYSTEM_PROMPT + "\n\n" + brief
    try:
        if chat_fn is not None:
            reply = chat_fn(cfg, system, api_messages)
        else:
            reply = complete_chat(cfg, system, api_messages)
    except Exception as exc:  # noqa: BLE001 — surface a safe message in the panel
        reply = _public_http_error(exc)
    if not reply:
        reply = "(模型返回了空回复。)"
    history.append({"role": "assistant", "content": reply})
    return history


def analyze_loaded_trajectory(
    steps: list[dict],
    raw: dict | None = None,
    *,
    brief: str | None = None,
    config: AnalysisLLMConfig | None = None,
    chat_fn: ChatFn | None = None,
) -> tuple[str, list[dict]]:
    """Run the first analysis pass for a loaded run.

    Pass *brief* (including an empty string) when already packed from
    ``LoadedSession`` so the load path never re-runs detectors. Omit *brief*
    only for callers that must build it from *steps*/*raw* (tests).
    """
    if not steps:
        return "", []
    if brief is None:
        brief = build_analysis_brief(steps, raw if isinstance(raw, dict) else {})
    history = answer_question(
        AUTO_ANALYSIS_QUESTION,
        [],
        brief,
        config=config,
        chat_fn=chat_fn,
    )
    return brief, history
