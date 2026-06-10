"""
Step labeler — LLM-based classification of trajectory steps.

Reads a trajectory file (Claude Code JSON, OpenCode JSON, CodeArts JSON,
or Lingxi .log), extracts assistant steps, and labels each with a phase
tag and action tag from the taxonomy defined in TAXONOMY_REFERENCE.md.

For Lingxi trajectories, each TokenUsageEvent is one step, belonging to
its executor sub-agent (Decoder_1, Planner, Solver, etc.).

LLM configuration is read from .env (LABEL_BASE_URL, LABEL_API_KEY,
LABEL_MODEL, LABEL_TEMPERATURE, LABEL_MAX_TOKENS).  CLI flags override
.env values.

Usage:
    python scripts/step_labeler.py samples/op_trajectory.json

    python scripts/step_labeler.py samples/simple_proposal_lingxi.log

    python scripts/step_labeler.py trajectory.json \
        --output labeled.json \
        --model glm-5 \
        --base-url https://api.example.com \
        --api-key sk-xxx
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ── .env loader ─────────────────────────────────────────────────────────


def _clean_env_scalar(raw: str) -> str:
    value = raw.strip()
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = _clean_env_scalar(value)
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()
# Also try project root .env (one level up from scripts/)
_load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = _clean_env_scalar(raw)
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[warn] Invalid {name}={raw!r}; using default {default}", file=sys.stderr)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = _clean_env_scalar(raw)
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[warn] Invalid {name}={raw!r}; using default {default}", file=sys.stderr)
        return default


# ── Taxonomy loader ─────────────────────────────────────────────────────


def load_taxonomy(taxonomy_path: str) -> tuple[dict[str, list[str]], str]:
    """Read TAXONOMY_REFERENCE.md and extract phase→action mapping.

    Returns (mapping, raw_text) where mapping is {phase: [action, ...]}.
    """
    with open(taxonomy_path, encoding="utf-8") as f:
        raw_text = f.read()

    mapping: dict[str, list[str]] = {}
    current_phase = None

    for line in raw_text.splitlines():
        line = line.strip()
        # Phase header: ### understand, ### plan, etc.
        if line.startswith("### ") and not line.startswith("#### "):
            phase = line[4:].strip().lower()
            if phase and not any(c in phase for c in (" ", "(")):
                current_phase = phase
                mapping.setdefault(current_phase, [])
        # Action: - `action_name`: description
        elif line.startswith("- `") and current_phase:
            match = re.match(r"- `(\w+)`", line)
            if match:
                mapping[current_phase].append(match.group(1))

    # Extract version from first heading
    version = "unknown"
    ver_match = re.search(r"\(v(\d+)\)", raw_text)
    if ver_match:
        version = f"v{ver_match.group(1)}"

    return mapping, version


def _build_valid_sets(mapping: dict[str, list[str]]) -> tuple[set[str], set[str], dict[str, str]]:
    """Build validation sets from taxonomy mapping.

    Returns (valid_phases, valid_actions, action_to_phase).
    """
    valid_phases = set(mapping.keys())
    valid_actions: set[str] = set()
    action_to_phase: dict[str, str] = {}
    for phase, actions in mapping.items():
        for action in actions:
            valid_actions.add(action)
            action_to_phase[action] = phase
    return valid_phases, valid_actions, action_to_phase


# ── Prompt builders ─────────────────────────────────────────────────────


def build_system_prompt(taxonomy_text: str) -> str:
    return f"""You are a trajectory step classifier. Your task is to label each agent step with a phase tag and an action tag from the taxonomy below.

## Taxonomy

{taxonomy_text}

## Instructions

For each step provided, determine:
1. **phase**: The coarse workflow stage (e.g., understand, plan, implement, debug, validate, report)
2. **action**: The specific behavior within that phase (e.g., code_reading, implement_runtime_logic)

Choose the dominant intent of the step. If a step could fit multiple actions, pick the primary one.

Respond with ONLY a JSON object (no markdown fences, no explanation):
{{"phase": "<phase>", "action": "<action>"}}"""


def build_step_message(step: dict, max_chars: int = 8000) -> str:
    """Build the user message describing a step for classification."""
    parts = []

    # Basic metadata
    parts.append(f"Step #{step.get('index', '?')}")
    rnd = step.get("round")
    if rnd is not None:
        parts.append(f"Round: {rnd}")
    dur = step.get("duration")
    if dur is not None:
        parts.append(f"Duration: {dur}s")
    tok = step.get("tokens", {})
    if tok.get("total"):
        parts.append(f"Tokens: {tok['total']:,}")
    finish = step.get("finish", "")
    if finish:
        parts.append(f"Finish reason: {finish}")
    agent = step.get("agent", "") or step.get("agent_id", "")
    if agent:
        parts.append(f"Agent: {agent}")
    if step.get("is_sub_agent"):
        parts.append("Context: This is a sub-agent step")

    # Question/prompt context (CodeArts)
    question = step.get("question", [])
    if question:
        for q in question:
            if isinstance(q, dict):
                q_text = q.get("content", q.get("value", {}).get("input", ""))
                if q_text:
                    parts.append(f"Prompt:\n{str(q_text)[:500]}")
                    break

    # Tool calls (include output/error for classification accuracy)
    tool_calls = step.get("tool_calls", [])
    if tool_calls:
        tool_lines = []
        for tc in tool_calls:
            name = tc.get("tool_name", "?")
            status = tc.get("status", "?")
            # Arguments: Lingxi uses "arguments", CC/OpenCode uses "input"
            inp = tc.get("arguments") or tc.get("input", {})
            if isinstance(inp, dict):
                inp_summary = ", ".join(
                    f"{k}={repr(v)[:80]}" for k, v in list(inp.items())[:5]
                )
            else:
                inp_summary = str(inp)[:200]
            line = f"  - {name} ({status}): {inp_summary}"
            # Include tool output snippet for success/failure evidence
            # Lingxi provides short_result (concise) alongside full result
            short_result = tc.get("short_result", "")
            output = tc.get("output", "") or tc.get("result", "")
            error = tc.get("error", "")
            if error:
                line += f"\n    ERROR: {str(error)[:200]}"
            elif short_result:
                line += f"\n    Result: {str(short_result)[:200]}"
            elif output:
                out_str = str(output)
                if len(out_str) > 300:
                    out_str = out_str[:300] + "..."
                line += f"\n    Output: {out_str}"
            tool_lines.append(line)
        parts.append("Tool calls:\n" + "\n".join(tool_lines))

    # For Lingxi steps, show the triple (ToolCall + TokenUsage + ToolResult)
    if agent and tool_calls:
        triple_lines = [f"Executor: {agent}"]
        for tc in tool_calls:
            fn = tc.get("tool_name", "?")
            args = tc.get("arguments", {})
            args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in list(args.items())[:5]) if isinstance(args, dict) else str(args)[:200]
            triple_lines.append(f"ToolCall: {fn}({args_str})")
            sr = tc.get("short_result", "")
            if sr:
                triple_lines.append(f"Result: {sr}")
        tok_total = step.get("tokens", {}).get("total", 0)
        tok_in = step.get("tokens", {}).get("input", 0)
        tok_out = step.get("tokens", {}).get("output", 0)
        if tok_total:
            triple_lines.append(f"Tokens: {tok_total:,} (in={tok_in:,}, out={tok_out:,})")
        parts.append("Step triple:\n" + "\n".join(triple_lines))

    # Text content
    preview = step.get("text_preview", "")
    if preview:
        parts.append(f"Text:\n{preview}")

    # Reasoning (include all reasoning blocks)
    reasoning_blocks = [
        p["text"] for p in step.get("parts", [])
        if p.get("type") == "reasoning" and p.get("text")
    ]
    if reasoning_blocks:
        parts.append("Reasoning:\n" + "\n---\n".join(reasoning_blocks))

    # Tool execution output (CodeArts operateCacheData)
    tool_output = step.get("tool_output")
    if isinstance(tool_output, dict):
        for tool_name, tool_data in tool_output.items():
            out_content = tool_data.get("content", "") if isinstance(tool_data, dict) else str(tool_data)
            if out_content:
                out_str = str(out_content)
                if len(out_str) > 500:
                    out_str = out_str[:500] + "..."
                parts.append(f"Tool output ({tool_name}):\n{out_str}")

    content = "\n".join(parts)
    if len(content) > max_chars:
        # Keep first 75% and last 25% so both early context and final
        # outcome (e.g. tool errors at the end) are visible to the LLM.
        separator = "\n\n[... truncated ...]\n\n"
        budget = max_chars - len(separator)
        if budget <= 0:
            # max_chars too small for separator; just hard-truncate
            content = content[:max_chars]
        else:
            head_chars = (budget * 3) // 4
            tail_chars = budget - head_chars
            content = (
                content[:head_chars]
                + separator
                + content[-tail_chars:]
            )
    return content


# ── LLM caller ──────────────────────────────────────────────────────────


def call_llm(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    provider: str = "openai",
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int = 120,
) -> str:
    if provider == "anthropic":
        return _call_anthropic(base_url, api_key, model, system_prompt,
                               user_message, temperature, max_tokens, timeout)
    return _call_openai(base_url, api_key, model, system_prompt,
                        user_message, temperature, max_tokens, timeout)


def _call_openai(
    base_url: str, api_key: str, model: str,
    system_prompt: str, user_message: str,
    temperature: float | None, max_tokens: int | None, timeout: int,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    resp = requests.post(url, json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_anthropic(
    base_url: str, api_key: str, model: str,
    system_prompt: str, user_message: str,
    temperature: float | None, max_tokens: int | None, timeout: int,
) -> str:
    url = f"{base_url.rstrip('/')}/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens or 1024,
    }
    if temperature is not None:
        body["temperature"] = temperature

    resp = requests.post(url, json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    blocks = data.get("content", [])
    texts = [b["text"] for b in blocks if b.get("type") == "text"]
    if not texts:
        raise ValueError("Anthropic response contained no text blocks")
    return "\n".join(texts).strip()


# ── Trajectory loading ──────────────────────────────────────────────────


def load_assistant_steps(trajectory_path: str) -> list[dict]:
    """Load trajectory and return only assistant steps."""
    # Add repo root to path so an uninstalled clone (no `pip install -e .`) can
    # still import the trajectory_visualizer package when this script is run
    # as `python scripts/step_labeler.py` from the repo root.
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from trajectory_visualizer.insight.loaders import load_trajectory
    from trajectory_visualizer.insight.parser import parse_steps

    raw = load_trajectory(trajectory_path)
    if "_error" in raw:
        raise ValueError(f"Failed to load trajectory: {raw['_error']}")

    steps = parse_steps(raw)
    return [s for s in steps if s.get("role") == "assistant"]


# ── Label parsing and validation ────────────────────────────────────────


def parse_label_response(
    text: str,
    valid_phases: set[str],
    valid_actions: set[str],
    action_to_phase: dict[str, str],
) -> dict[str, str]:
    """Parse LLM response into validated {phase, action} dict."""
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        match = re.search(r"\{[^}]+\}", text)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return {"phase": "unknown", "action": "unknown"}
        else:
            return {"phase": "unknown", "action": "unknown"}

    raw_phase = result.get("phase") or "unknown"
    raw_action = result.get("action") or "unknown"
    phase = str(raw_phase).strip().lower()
    action = str(raw_action).strip().lower()

    # Validate action — if valid action but wrong/missing phase, fix phase
    if action in valid_actions:
        phase = action_to_phase.get(action, phase)
    elif action != "unknown":
        print(f"  [warn] Unknown action: {action}", file=sys.stderr)
        action = "unknown"

    if phase not in valid_phases and phase != "unknown":
        print(f"  [warn] Unknown phase: {phase}", file=sys.stderr)
        phase = "unknown"

    return {"phase": phase, "action": action}


# ── Main pipeline ───────────────────────────────────────────────────────


def label_trajectory(
    trajectory_path: str,
    output_path: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str = "openai",
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_content_chars: int = 8000,
    delay: float = 0.0,
    taxonomy_path: str | None = None,
) -> None:
    """Label all assistant steps in a trajectory and write output JSON."""
    # Load taxonomy
    if taxonomy_path is None:
        taxonomy_path = str(Path(__file__).resolve().parent / "TAXONOMY_REFERENCE.md")
    taxonomy_mapping, taxonomy_version = load_taxonomy(taxonomy_path)
    valid_phases, valid_actions, action_to_phase = _build_valid_sets(taxonomy_mapping)

    # Read raw taxonomy text for system prompt
    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy_text = f.read()

    system_prompt = build_system_prompt(taxonomy_text)

    # Load trajectory
    print(f"Loading trajectory: {trajectory_path}", file=sys.stderr)
    assistant_steps = load_assistant_steps(trajectory_path)
    print(f"Found {len(assistant_steps)} assistant steps", file=sys.stderr)

    if not assistant_steps:
        print("No assistant steps found — nothing to label.", file=sys.stderr)
        return

    # Label each step
    labeled_steps: list[dict] = []
    unknown_count = 0

    for i, step in enumerate(assistant_steps):
        step_idx = step.get("index", i)
        print(f"Labeling step {i + 1}/{len(assistant_steps)} (idx {step_idx})...",
              file=sys.stderr, end=" ", flush=True)

        user_message = build_step_message(step, max_chars=max_content_chars)

        # Skip empty steps — no text, no tool calls, no reasoning
        has_text = bool(step.get("text_preview", "").strip())
        has_tools = bool(step.get("tool_calls"))
        has_reasoning = any(
            p.get("type") == "reasoning" and p.get("text")
            for p in step.get("parts", [])
        )
        if not has_text and not has_tools and not has_reasoning:
            print("(empty step — skipped)", file=sys.stderr)
            label = {"phase": "unknown", "action": "unknown"}
            unknown_count += 1
            tool_names = []
            tokens = step.get("tokens", {})
            labeled_steps.append({
                "index": step_idx,
                "role": "assistant",
                "phase": label["phase"],
                "action": label["action"],
                "time_created_ms": step.get("time_created_ms"),
                "time_completed_ms": step.get("time_completed_ms"),
                "duration_s": step.get("duration"),
                "tokens_total": tokens.get("total", 0),
                "tool_calls": tool_names,
                "finish": step.get("finish", ""),
                "agent": step.get("agent", ""),
                "model_id": step.get("model_id", ""),
                "text_preview": "",
            })
            continue

        # Call LLM with one retry on error
        label = None
        for attempt in range(2):
            try:
                response = call_llm(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    provider=provider,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                label = parse_label_response(response, valid_phases, valid_actions, action_to_phase)
                break
            except Exception as exc:
                if attempt == 0:
                    print(f"[retry: {exc}]", file=sys.stderr, end=" ", flush=True)
                    time.sleep(1)
                else:
                    print(f"[error: {exc}]", file=sys.stderr)
                    label = {"phase": "unknown", "action": "unknown"}

        if label is None:
            label = {"phase": "unknown", "action": "unknown"}

        if label["phase"] == "unknown" or label["action"] == "unknown":
            unknown_count += 1

        print(f"{label['phase']}/{label['action']}", file=sys.stderr)

        # Extract tool call names
        tool_names = [tc.get("tool_name", "?") for tc in step.get("tool_calls", [])]
        tokens = step.get("tokens", {})

        labeled_steps.append({
            "index": step_idx,
            "role": "assistant",
            "phase": label["phase"],
            "action": label["action"],
            "time_created_ms": step.get("time_created_ms"),
            "time_completed_ms": step.get("time_completed_ms"),
            "duration_s": step.get("duration"),
            "tokens_total": tokens.get("total", 0),
            "tool_calls": tool_names,
            "finish": step.get("finish", ""),
            "agent": step.get("agent", "") or step.get("agent_id", ""),
            "model_id": step.get("model_id", ""),
            "text_preview": (step.get("text_preview", "") or "")[:200],
            # Format-specific (absent for formats that don't have them)
            "round": step.get("round"),
            "is_sub_agent": step.get("is_sub_agent", False),
            "session_id": step.get("session_id", ""),
            "executor_id": step.get("agent", ""),
        })

        if delay > 0 and i < len(assistant_steps) - 1:
            time.sleep(delay)

    # Write output
    output = {
        "trajectory_file": os.path.abspath(trajectory_path),
        "taxonomy_version": taxonomy_version,
        "model": model,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "steps": labeled_steps,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    success_count = len(labeled_steps) - unknown_count
    print(
        f"\nDone: {len(labeled_steps)} steps labeled "
        f"({success_count} classified, {unknown_count} unknown). "
        f"Output: {output_path}",
        file=sys.stderr,
    )


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label trajectory steps with phase/action tags using an LLM.",
    )
    parser.add_argument("input", help="Path to trajectory file (JSON or Lingxi .log)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: <input>_labeled.json)")
    parser.add_argument("--model", default=None,
                        help="LLM model (overrides LABEL_MODEL env)")
    parser.add_argument("--base-url", default=None,
                        help="LLM API base URL (overrides LABEL_BASE_URL env)")
    parser.add_argument("--api-key", default=None,
                        help="LLM API key (overrides LABEL_API_KEY env)")
    parser.add_argument("--provider", default=None,
                        help="LLM provider: openai or anthropic (overrides LABEL_PROVIDER env)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (overrides LABEL_TEMPERATURE env)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max response tokens (overrides LABEL_MAX_TOKENS env)")
    parser.add_argument("--max-content-chars", type=int, default=8000,
                        help="Max chars per step sent to LLM (default: 8000)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay in seconds between LLM calls (default: 0)")
    parser.add_argument("--taxonomy", default=None,
                        help="Path to TAXONOMY_REFERENCE.md (default: auto-detect)")

    args = parser.parse_args()

    # Resolve configuration: CLI > .env > defaults
    base_url = args.base_url or os.getenv("LABEL_BASE_URL", "")
    api_key = args.api_key or os.getenv("LABEL_API_KEY", "")
    model = args.model or os.getenv("LABEL_MODEL", "")
    provider = args.provider or os.getenv("LABEL_PROVIDER", "openai")
    temperature = args.temperature
    if temperature is None:
        temperature = _env_float("LABEL_TEMPERATURE", 0.3)
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = _env_int("LABEL_MAX_TOKENS", 1024)

    if not base_url:
        print("Error: LABEL_BASE_URL not set (use --base-url or .env)", file=sys.stderr)
        sys.exit(1)
    if not api_key:
        print("Error: LABEL_API_KEY not set (use --api-key or .env)", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("Error: LABEL_MODEL not set (use --model or .env)", file=sys.stderr)
        sys.exit(1)

    # base_url must be an http(s) URL with a host (typo / SSRF guard).
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        print(f"Error: --base-url must be an http(s) URL with a host; got {base_url!r}",
              file=sys.stderr)
        sys.exit(1)

    # Secret hygiene + data-egress notice. Labeling is a NETWORK operation: the
    # labeler sends trajectory content (assistant text, reasoning, tool I/O,
    # which can contain proprietary code and secrets) to the configured endpoint.
    if args.api_key:
        print("Warning: --api-key on the command line is visible via the process "
              "list and shell history; prefer LABEL_API_KEY in .env or the "
              "environment.", file=sys.stderr)
    print(f"Note: sending trajectory content to {parsed.scheme}://{parsed.netloc} "
          f"({provider}/{model}). Do not label trajectories containing secrets you "
          f"cannot share with that endpoint.", file=sys.stderr)

    # Output path
    output_path = args.output
    if output_path is None:
        inp = Path(args.input)
        output_path = str(inp.parent / f"{inp.stem}_labeled.json")

    label_trajectory(
        trajectory_path=args.input,
        output_path=output_path,
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        max_content_chars=args.max_content_chars,
        delay=args.delay,
        taxonomy_path=args.taxonomy,
    )


if __name__ == "__main__":
    main()
