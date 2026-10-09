"""
Step labeler v2 -- label assistant steps while preserving every source index.

The assistant classification prompt and taxonomy are intentionally shared with
``step_labeler.py``.  Unlike v1, the v2 output contains one record for every
parsed trajectory step:

* assistant steps are classified by the LLM exactly as in v1;
* user steps receive the deterministic ``user/user_prompt`` label;
* any other role receives ``unknown/unknown`` rather than being dropped.

Both ``index`` (the parser's canonical index) and ``raw_index`` (the position
in the source trajectory, when available) are written to make the mapping back
to the original trajectory explicit.

Usage:
    python scripts/step_labeler_v2.py trajectory.json
    python scripts/step_labeler_v2.py trajectory.json -o trajectory_labeled_v2.json
"""

from __future__ import annotations

import argparse
import json  # noqa: F401  (re-exported: tests patch step_labeler_v2.json.dump)
import os
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

try:
    from scripts import _common
    from scripts import step_labeler as v1
except ImportError:  # Direct execution from scripts/.
    import _common  # type: ignore[no-redef]
    import step_labeler as v1  # type: ignore[no-redef]


USER_DEFAULT_LABEL = {"phase": "user", "action": "user_prompt"}
OTHER_DEFAULT_LABEL = {"phase": "unknown", "action": "unknown"}


class OutputSafetyError(ValueError):
    """Raised when label output would replace the source trajectory."""


def _ensure_output_is_not_input(
    trajectory_path: str | Path, output_path: str | Path
) -> None:
    _common.ensure_output_does_not_overwrite(
        output_path,
        [trajectory_path],
        exc=OutputSafetyError,
        message="Output path would overwrite input trajectory: {source}",
    )


def _write_json_atomic(output_path: str | Path, data: dict[str, Any]) -> None:
    """Replace a label sidecar only after its complete JSON is durable."""
    _common.write_json_atomic(output_path, data)


def load_all_steps(trajectory_path: str) -> list[dict]:
    """Load and normalize every trajectory step, preserving parser indices."""
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from trajviz.insight.loaders import load_trajectory
    from trajviz.insight.parser import parse_steps

    raw = load_trajectory(trajectory_path)
    if "_error" in raw:
        raise ValueError(f"Failed to load trajectory: {raw['_error']}")
    return parse_steps(raw)


def _is_empty_assistant_step(step: dict) -> bool:
    has_text = bool((step.get("text_preview", "") or "").strip())
    has_tools = bool(step.get("tool_calls"))
    has_reasoning = any(
        isinstance(part, dict)
        and part.get("type") == "reasoning"
        and part.get("text")
        for part in step.get("parts", [])
    )
    return not (has_text or has_tools or has_reasoning)


def _serialize_step(
    step: dict,
    *,
    fallback_index: int,
    label: dict[str, str],
    label_source: str,
) -> dict[str, Any]:
    """Create one sidecar record without changing the source index."""
    step_index = step.get("index", fallback_index)
    raw_index = step.get("raw_index", step_index)
    role = str(step.get("role", "unknown") or "unknown").lower()
    tokens = step.get("tokens", {})
    if not isinstance(tokens, dict):
        tokens = {}

    return {
        "index": step_index,
        "raw_index": raw_index,
        "role": role,
        "phase": label["phase"],
        "action": label["action"],
        "label_source": label_source,
        "time_created_ms": step.get("time_created_ms"),
        "time_completed_ms": step.get("time_completed_ms"),
        "duration_s": step.get("duration"),
        "tokens_total": tokens.get("total", 0),
        "tool_calls": [
            tc.get("tool_name", "?")
            for tc in step.get("tool_calls", [])
            if isinstance(tc, dict)
        ],
        "finish": step.get("finish", ""),
        "agent": step.get("agent", "") or step.get("agent_id", ""),
        "model_id": step.get("model_id", ""),
        "text_preview": (step.get("text_preview", "") or "")[:200],
        "round": step.get("round"),
        "is_sub_agent": step.get("is_sub_agent", False),
        "session_id": step.get("session_id", ""),
        "executor_id": step.get("agent", "") or step.get("agent_id", ""),
    }


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
    user_phase: str = USER_DEFAULT_LABEL["phase"],
    user_action: str = USER_DEFAULT_LABEL["action"],
    assistant_only: bool = False,
) -> None:
    """Label assistants and emit labels for every parsed source index.

    With ``assistant_only=True`` (the v1 CLI's mode) only assistant records
    are written to the sidecar; user/other steps still receive deterministic
    labels internally but are filtered from the output.
    """
    _ensure_output_is_not_input(trajectory_path, output_path)
    if taxonomy_path is None:
        taxonomy_path = str(Path(__file__).resolve().parent / "TAXONOMY_REFERENCE.md")
    taxonomy_mapping, taxonomy_version = v1.load_taxonomy(taxonomy_path)
    valid_phases, valid_actions, action_to_phase = v1._build_valid_sets(taxonomy_mapping)
    with open(taxonomy_path, encoding="utf-8") as f:
        system_prompt = v1.build_system_prompt(f.read())

    print(f"Loading trajectory: {trajectory_path}", file=sys.stderr)
    all_steps = load_all_steps(trajectory_path)
    assistant_count = sum(
        1 for step in all_steps if str(step.get("role", "")).lower() == "assistant"
    )
    user_count = sum(
        1 for step in all_steps if str(step.get("role", "")).lower() == "user"
    )
    print(
        f"Found {len(all_steps)} total steps "
        f"({assistant_count} assistant, {user_count} user)",
        file=sys.stderr,
    )

    labeled_steps: list[dict[str, Any]] = []
    assistant_seen = 0
    assistant_unknown = 0
    default_count = 0

    for position, step in enumerate(all_steps):
        step_index = step.get("index", position)
        role = str(step.get("role", "unknown") or "unknown").lower()

        if role == "user":
            label = {"phase": user_phase, "action": user_action}
            label_source = "default"
            default_count += 1
            print(
                f"Labeling index {step_index} (user)... "
                f"{label['phase']}/{label['action']} (default)",
                file=sys.stderr,
            )
        elif role != "assistant":
            label = dict(OTHER_DEFAULT_LABEL)
            label_source = "fallback"
            default_count += 1
            print(
                f"Labeling index {step_index} ({role})... "
                "unknown/unknown (fallback)",
                file=sys.stderr,
            )
        else:
            assistant_seen += 1
            print(
                f"Labeling assistant {assistant_seen}/{assistant_count} "
                f"(idx {step_index})...",
                file=sys.stderr,
                end=" ",
                flush=True,
            )

            if _is_empty_assistant_step(step):
                label = dict(OTHER_DEFAULT_LABEL)
                label_source = "fallback"
                assistant_unknown += 1
                print("unknown/unknown (empty step)", file=sys.stderr)
            else:
                user_message = v1.build_step_message(
                    step, max_chars=max_content_chars
                )
                label = None
                for attempt in range(2):
                    try:
                        response = v1.call_llm(
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            system_prompt=system_prompt,
                            user_message=user_message,
                            provider=provider,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        label = v1.parse_label_response(
                            response,
                            valid_phases,
                            valid_actions,
                            action_to_phase,
                        )
                        break
                    except Exception as exc:
                        if attempt == 0:
                            print(
                                f"[retry: {exc}]",
                                file=sys.stderr,
                                end=" ",
                                flush=True,
                            )
                            time.sleep(1)
                        else:
                            print(f"[error: {exc}]", file=sys.stderr)

                if label is None:
                    label = dict(OTHER_DEFAULT_LABEL)
                label_source = (
                    "llm"
                    if label["phase"] != "unknown" and label["action"] != "unknown"
                    else "fallback"
                )
                if label_source == "fallback":
                    assistant_unknown += 1
                print(f"{label['phase']}/{label['action']}", file=sys.stderr)

        labeled_steps.append(
            _serialize_step(
                step,
                fallback_index=position,
                label=label,
                label_source=label_source,
            )
        )

        if delay > 0 and role == "assistant" and assistant_seen < assistant_count:
            time.sleep(delay)

    if len(labeled_steps) != len(all_steps):
        raise RuntimeError("Internal error: not every parsed index received a label")

    if assistant_only:
        labeled_steps = [s for s in labeled_steps if s.get("role") == "assistant"]
        if len(labeled_steps) != assistant_count:
            raise RuntimeError(
                "Internal error: assistant-only output lost assistant records"
            )

    output = {
        "schema_version": "trajectory_labels.v2",
        "trajectory_file": os.path.abspath(trajectory_path),
        "taxonomy_version": taxonomy_version,
        "model": model,
        "labeled_at": datetime.now(UTC).isoformat(),
        "defaults": {
            "user": {"phase": user_phase, "action": user_action},
            "other": dict(OTHER_DEFAULT_LABEL),
        },
        "counts": {
            "total": len(labeled_steps),
            "assistant": assistant_count,
            "user": user_count,
            "default_or_fallback": default_count + assistant_unknown,
        },
        "steps": labeled_steps,
    }

    _write_json_atomic(output_path, output)

    assistant_classified = assistant_count - assistant_unknown
    print(
        f"\nDone: emitted labels for all {len(labeled_steps)} indices; "
        f"{assistant_classified}/{assistant_count} assistant steps classified, "
        f"{user_count} user steps defaulted. Output: {output_path}",
        file=sys.stderr,
    )


def main() -> None:
    # Resolve .env config at CLI entry, not at module import (see step_labeler).
    v1.load_env_files()
    parser = argparse.ArgumentParser(
        description=(
            "Label assistant trajectory steps with an LLM and emit a label "
            "record for every original index."
        )
    )
    parser.add_argument("input", help="Path to trajectory file (JSON or Lingxi .log)")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON path (default: <input>_labeled_v2.json)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--provider", default=None, choices=("openai", "anthropic"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-content-chars", type=int, default=8000)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--taxonomy", default=None)
    parser.add_argument("--user-phase", default=USER_DEFAULT_LABEL["phase"])
    parser.add_argument("--user-action", default=USER_DEFAULT_LABEL["action"])
    args = parser.parse_args()

    base_url = args.base_url or os.getenv("LABEL_BASE_URL", "")
    api_key = args.api_key or os.getenv("LABEL_API_KEY", "")
    model = args.model or os.getenv("LABEL_MODEL", "")
    provider = args.provider or os.getenv("LABEL_PROVIDER", "openai")
    temperature = args.temperature
    if temperature is None:
        temperature = v1._env_float("LABEL_TEMPERATURE", 0.3)
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = v1._env_int("LABEL_MAX_TOKENS", 1024)

    if not base_url:
        parser.error("LABEL_BASE_URL not set (use --base-url or .env)")
    if not api_key:
        parser.error("LABEL_API_KEY not set (use --api-key or .env)")
    if not model:
        parser.error("LABEL_MODEL not set (use --model or .env)")

    output_path = args.output
    if output_path is None:
        inp = Path(args.input)
        output_path = str(inp.parent / f"{inp.stem}_labeled_v2.json")

    try:
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
            user_phase=args.user_phase,
            user_action=args.user_action,
        )
    except OutputSafetyError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
