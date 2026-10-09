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

.. deprecated::
    The v1 labeling pipeline (``label_trajectory``/``load_assistant_steps``)
    has been removed; this CLI now routes through
    ``step_labeler_v2.label_trajectory(assistant_only=True)``, which adds an
    output-overwrite guard and atomic sidecar writes while keeping the
    assistant-only output and the ``<stem>_labeled.json`` default name.
    Prefer ``scripts/step_labeler_v2.py`` for new work — it emits a label
    record for every parsed step.  The prompt/taxonomy/LLM helpers defined
    here remain the shared implementation used by v2.

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


def load_env_files() -> None:
    """Load ./.env plus the repo-root .env into os.environ (fill-only).

    Called from ``main()`` (and step_labeler_v2's ``main()``) rather than at
    module import, so merely importing this module — e.g. during test
    collection — never mutates the process environment.
    """
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
        # A level-1/2 heading (`# Title`, `## Notes`) CLOSES the current phase.
        # Without this, every later bullet keeps the last `### <phase>` — and
        # `## Notes` is the last heading in TAXONOMY_REFERENCE.md, so a Notes
        # bullet written in the file's own `- `name`: ...` style would silently
        # become a valid action of the `report` phase. `#### ` is deliberately
        # excluded: the carve-out above already decided sub-subsections refine
        # the phase they sit under, so their bullets are that phase's actions.
        elif re.match(r"#{1,2} ", line):
            current_phase = None
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
    dur = step.get("duration")
    if dur is not None:
        parts.append(f"Duration: {dur}s")
    tok = step.get("tokens", {})
    if tok.get("total"):
        parts.append(f"Tokens: {tok['total']:,}")
    finish = step.get("finish", "")
    if finish:
        parts.append(f"Finish reason: {finish}")
    agent = step.get("agent", "")
    if agent:
        parts.append(f"Agent: {agent}")
    if step.get("is_sub_agent"):
        parts.append("Context: This is a sub-agent step")

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


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    # Resolve .env config at CLI entry, not at module import (B34).
    load_env_files()
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

    # Output path
    output_path = args.output
    if output_path is None:
        inp = Path(args.input)
        output_path = str(inp.parent / f"{inp.stem}_labeled.json")

    # The v1 CLI delegates to v2's labeling loop (assistant-only output).
    # v2 supplies the output-overwrite guard and the atomic sidecar write.
    # Imported lazily: step_labeler_v2 imports this module at its top level.
    try:
        from scripts import step_labeler_v2 as v2
    except ImportError:  # Direct execution from scripts/.
        import step_labeler_v2 as v2  # type: ignore[no-redef]

    try:
        v2.label_trajectory(
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
            assistant_only=True,
        )
    except v2.OutputSafetyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
