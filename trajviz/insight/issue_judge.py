"""On-demand LLM judge for Overview Issues Fix suggestions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

from .assistant import ChatFn, complete_chat, _first_user_task, _public_http_error
from .llm_config import AnalysisLLMConfig, analysis_language, is_default_language, resolve_analysis_config
from .presenters.issues import IssueJudgment, OverviewIssue
from .session import LoadedSession

JUDGE_ISSUE_CAP = 8
_STEP_RADIUS = 2
_STEP_HARD_CAP = 12
_CLIP = 160

JUDGE_SYSTEM_PROMPT = """You are TrajViz's Issues fix judge for coding-agent runs.

You receive ONE detected workflow issue plus a clipped local workflow window,
optional first-user task text, and session skill evidence. Reply with ONLY a
single JSON object (no markdown prose outside JSON). Schema:

{
  "where": "string — durable edit target",
  "fix": "string — one concrete imperative change",
  "also": "string — optional short supporting note, or empty",
  "confidence": "high" | "medium" | "low"
}

Language:
- Write the values of "where", "fix", and "also" in {LANGUAGE}.
- Keep JSON keys in English.
- Keep "confidence" as the English enum: high | medium | low.
- File/path/tool names may stay in their original Latin spelling when that is clearer.

Rules for "where":
- Permission/auth/sandbox/cwd runtime failures → 环境 (not instructions).
- If evidence names a SKILL.md path or skill id → that skill / path.
- Otherwise → durable agent instructions for the harness (e.g. CLAUDE.md, AGENTS.md,
  OpenCode agent config). Never use this run's one-shot user message as the edit target.
  user_task is goal context only — not the place to edit.

Rules for "fix":
- One actionable change the agent author can make.
- Prefer fixes that help the agent complete user_task without repeating this issue.
- Do NOT prescribe noisy recovery tool trails (e.g. glob→glob→glob→bash) as a recipe.
- Only cite step numbers that appear in the evidence.

If evidence is thin, say so in "also" and set confidence to "low".
"""


def _clip(text: str, limit: int = _CLIP) -> str:
    text = text.replace("\r\n", "\n").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _step_window_indices(steps: tuple[int, ...], radius: int = _STEP_RADIUS) -> list[int]:
    if not steps:
        return []
    wanted: set[int] = set()
    for step in steps:
        for delta in range(-radius, radius + 1):
            wanted.add(int(step) + delta)
    return sorted(wanted)


def _tool_summary(step: dict) -> list[dict]:
    out: list[dict] = []
    for tc in step.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        name = str(tc.get("tool_name") or "")
        entry: dict = {"tool": name}
        err = tc.get("error_type") or tc.get("error")
        if err:
            entry["error"] = _clip(str(err), 80)
        inp = tc.get("input")
        if isinstance(inp, dict):
            for key in ("command", "file_path", "filePath", "path", "pattern", "query"):
                if inp.get(key):
                    entry[key] = _clip(str(inp[key]), 100)
                    break
        out.append(entry)
    return out[:8]


def pack_issue_judge_context(session: LoadedSession, issue: OverviewIssue) -> str:
    """Build a compact text brief for one issue (not the full trajectory)."""
    step_map = {
        int(s.get("index", i)): s
        for i, s in enumerate(session.steps or [])
    }
    window_idxs = _step_window_indices(issue.steps)
    ordered: list[int] = []
    for idx in list(issue.steps) + window_idxs:
        if idx not in ordered and idx in step_map:
            ordered.append(idx)
        if len(ordered) >= _STEP_HARD_CAP:
            break

    workflow: list[dict] = []
    for idx in ordered:
        step = step_map[idx]
        row: dict = {
            "index": idx,
            "role": step.get("role"),
            "tools": _tool_summary(step),
        }
        preview = step.get("text_preview")
        if isinstance(preview, str) and preview.strip():
            row["text_preview"] = _clip(preview, 120)
        workflow.append(row)

    skills: list[dict] = []
    for item in session.file_interactions or []:
        if item.get("type") != "skill":
            continue
        skills.append({
            "step": item.get("step"),
            "path": item.get("path"),
            "tool": item.get("tool"),
        })
        if len(skills) >= 12:
            break

    session_meta: dict = {
        "format": session.format or "",
    }
    user_task = _first_user_task(session.steps or [])
    if user_task:
        session_meta["user_task"] = user_task

    payload = {
        "session": session_meta,
        "issue": {
            "kind": issue.kind,
            "title": issue.title,
            "detail": issue.detail,
            "why": issue.why,
            "steps": list(issue.steps),
            "source_id": issue.source_id,
        },
        "workflow_window": workflow,
        "skills_in_run": skills,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_issue_judgment(raw: str) -> IssueJudgment:
    """Parse model output into IssueJudgment; raise ValueError on bad shape."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty judge response")

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge response contained no JSON object")
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge JSON parse failed: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Judge JSON must be an object")

    where = str(data.get("where") or "").strip()
    fix = str(data.get("fix") or "").strip()
    also = str(data.get("also") or "").strip()
    confidence_raw = str(data.get("confidence") or "medium").strip().lower()
    if confidence_raw not in {"high", "medium", "low"}:
        confidence_raw = "medium"
    if not where or not fix:
        raise ValueError("Judge JSON requires non-empty where and fix")
    return IssueJudgment(
        where=where,
        fix=fix,
        also=also,
        confidence=confidence_raw,  # type: ignore[arg-type]
    )


def judge_issue_fix(
    session: LoadedSession,
    issue: OverviewIssue,
    *,
    config: AnalysisLLMConfig | None = None,
    chat_fn: ChatFn | None = None,
) -> IssueJudgment:
    """Ask the analysis LLM for a structured Fix for *issue*."""
    cfg = config or resolve_analysis_config()
    if not cfg.ready:
        missing = ", ".join(cfg.missing) or "ANALYZE_* / LABEL_*"
        raise ValueError(f"Analysis LLM not configured ({missing})")

    context = pack_issue_judge_context(session, issue)
    user_msg = (
        "Judge a fix for this TrajViz issue. Return ONLY the JSON object. "
        f"where / fix / also must be {_judge_language()}.\n\n"
        f"{context}"
    )
    fn: ChatFn = chat_fn or complete_chat
    system = JUDGE_SYSTEM_PROMPT.replace("{LANGUAGE}", _judge_language())
    raw = fn(cfg, system, [{"role": "user", "content": user_msg}])
    return parse_issue_judgment(raw)


@dataclass(frozen=True)
class JudgeProgress:
    """One progressive snapshot while judging Overview Issues."""

    issues: list[OverviewIssue]
    errors: list[str]
    # 1-based index of the issue currently being judged; 0 when finished.
    current: int
    total: int
    current_title: str
    finished: bool


def _judge_language() -> str:
    """ANALYZE_LANGUAGE, with the documented default spelled as before."""
    lang = analysis_language()
    return "Simplified Chinese (Mandarin)" if is_default_language(lang) else lang


def _judge_error_text(exc: BaseException) -> str:
    """Judge-failure text safe to render in the Issues panel.

    The chat path already scrubs this (``assistant._public_http_error``); the
    judge must share it rather than render ``str(exc)``. requests puts the full
    request URL in an HTTPError's message, and the analysis base URL is taken
    verbatim from ANALYZE_BASE_URL, so a query-string credential (Azure
    ``api-key=``, Gemini ``key=``) would otherwise reach the page.

    ValueErrors raised by this module (parse_issue_judgment's JSON complaints,
    judge_issue_fix's "not configured") carry no endpoint or key and are the
    only debuggable signal left once HTTP text is gone, so they pass through —
    but RequestException is checked first, because requests.JSONDecodeError is
    both a RequestException and a ValueError.
    """
    if isinstance(exc, ValueError) and not isinstance(exc, requests.RequestException):
        return str(exc)[:200]
    return _public_http_error(exc)


def iter_judge_overview_issues(
    session: LoadedSession,
    issues: list[OverviewIssue],
    *,
    config: AnalysisLLMConfig | None = None,
    chat_fn: ChatFn | None = None,
    limit: int = JUDGE_ISSUE_CAP,
):
    """Yield :class:`JudgeProgress` before each call and once at the end.

    Before judging issue *i*, yields ``current=i+1`` with that issue's title.
    After all attempts, yields ``finished=True`` with ``current=0``.
    """
    cfg = config or resolve_analysis_config()
    working = list(issues)
    errors: list[str] = []
    total = min(len(working), max(0, limit))

    for i in range(total):
        yield JudgeProgress(
            issues=list(working),
            errors=list(errors),
            current=i + 1,
            total=total,
            current_title=working[i].title,
            finished=False,
        )
        try:
            judgment = judge_issue_fix(
                session, working[i], config=cfg, chat_fn=chat_fn,
            )
            working[i] = working[i].with_judgment(judgment)
        except Exception as exc:  # noqa: BLE001 — keep other issues judging
            errors.append(f"{working[i].title}: {_judge_error_text(exc)}")

    yield JudgeProgress(
        issues=list(working),
        errors=list(errors),
        current=0,
        total=total,
        current_title="",
        finished=True,
    )


def judge_overview_issues(
    session: LoadedSession,
    issues: list[OverviewIssue],
    *,
    config: AnalysisLLMConfig | None = None,
    chat_fn: ChatFn | None = None,
    limit: int = JUDGE_ISSUE_CAP,
) -> tuple[list[OverviewIssue], list[str]]:
    """Judge up to *limit* issues; return updated issues and error notes."""
    out: list[OverviewIssue] = []
    errors: list[str] = []
    for progress in iter_judge_overview_issues(
        session, issues, config=config, chat_fn=chat_fn, limit=limit,
    ):
        out = progress.issues
        errors = progress.errors
    return out, errors
