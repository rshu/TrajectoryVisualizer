"""LLM settings for the dashboard analysis panel.

Reads ``ANALYZE_*`` from ``.env`` (fill-only, never overwrites a real
environment variable). Missing ``ANALYZE_*`` values fall back to the
``LABEL_*`` keys used by ``scripts/step_labeler.py`` so one file can
drive both the labeler and the analysis chat.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_env_scalar(raw: str) -> str:
    value = raw.strip()
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _load_dotenv(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = _clean_env_scalar(value)
            if key and key not in os.environ:
                os.environ[key] = value


def load_env_files() -> None:
    """Load CWD ``.env`` then the repo-root ``.env`` (fill-only).

    Call from the dashboard entry / ``build_ui``, not at import time, so
    test collection does not mutate the process environment.
    """
    _load_dotenv(".env")
    _load_dotenv(str(_REPO_ROOT / ".env"))


def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return _clean_env_scalar(raw)


def _env_float(name: str, default: float) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = _clean_env_scalar(raw)
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
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
        return default


@dataclass(frozen=True)
class AnalysisLLMConfig:
    """Resolved OpenAI-compatible or Anthropic chat settings."""

    base_url: str
    api_key: str
    model: str
    provider: str
    temperature: float | None
    max_tokens: int
    timeout: int
    source: str  # "analyze" | "label" | "mixed" | "missing"

    @property
    def ready(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    @property
    def missing(self) -> list[str]:
        names: list[str] = []
        if not self.base_url:
            names.append("ANALYZE_BASE_URL (or LABEL_BASE_URL)")
        if not self.api_key:
            names.append("ANALYZE_API_KEY (or LABEL_API_KEY)")
        if not self.model:
            names.append("ANALYZE_MODEL (or LABEL_MODEL)")
        return names


def _pick(analyze_name: str, label_name: str, default: str = "") -> tuple[str, str]:
    """Return (value, which-prefix-won) for an ANALYZE_/LABEL_ pair."""
    analyze = _env(analyze_name)
    if analyze:
        return analyze, "analyze"
    label = _env(label_name)
    if label:
        return label, "label"
    return default, "missing"


def resolve_analysis_config() -> AnalysisLLMConfig:
    """Resolve chat settings. ``ANALYZE_*`` wins; ``LABEL_*`` fills gaps."""
    base_url, url_src = _pick("ANALYZE_BASE_URL", "LABEL_BASE_URL")
    api_key, key_src = _pick("ANALYZE_API_KEY", "LABEL_API_KEY")
    model, model_src = _pick("ANALYZE_MODEL", "LABEL_MODEL")
    provider, prov_src = _pick("ANALYZE_PROVIDER", "LABEL_PROVIDER", "openai")
    if not provider:
        provider = "openai"
    provider = provider.strip().lower()
    if provider not in {"openai", "anthropic"}:
        provider = "openai"

    sources = {url_src, key_src, model_src, prov_src} - {"missing"}
    if not (base_url and api_key and model):
        source = "missing"
    elif sources == {"analyze"}:
        source = "analyze"
    elif sources == {"label"}:
        source = "label"
    else:
        source = "mixed"

    if os.getenv("ANALYZE_TEMPERATURE") is not None:
        temperature = _env_float("ANALYZE_TEMPERATURE", 0.2)
    elif os.getenv("LABEL_TEMPERATURE") is not None:
        temperature = _env_float("LABEL_TEMPERATURE", 0.2)
    else:
        temperature = 0.2
    max_tokens = _env_int("ANALYZE_MAX_TOKENS", 0) or _env_int("LABEL_MAX_TOKENS", 2048)
    if max_tokens <= 0:
        max_tokens = 2048
    timeout = _env_int("ANALYZE_TIMEOUT", 120)
    if timeout <= 0:
        timeout = 120

    return AnalysisLLMConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        source=source,
    )


def config_status_html(*, loaded_steps: int | None = None) -> str:
    """Short HTML status for the analysis sidebar (never includes the key)."""
    cfg = resolve_analysis_config()
    bits: list[str] = []
    if cfg.ready:
        bits.append(
            f"<span class='analysis-status-ok'>Ready</span> · "
            f"<code>{_escape(cfg.model)}</code> · {_escape(cfg.provider)}"
        )
        if cfg.source == "label":
            bits.append("using LABEL_* from .env")
        elif cfg.source == "mixed":
            bits.append("ANALYZE_* + LABEL_*")
        else:
            bits.append("ANALYZE_* from .env")
    elif len(cfg.missing) >= 3:
        bits.append(
            "<span class='analysis-status-warn'>Not configured</span> · "
            "copy <code>.env.example</code> to <code>.env</code> and set "
            "<code>ANALYZE_*</code> (or <code>LABEL_*</code>)"
        )
    else:
        missing = ", ".join(cfg.missing)
        bits.append(
            f"<span class='analysis-status-warn'>Not configured</span> · "
            f"set {_escape(missing)} in <code>.env</code>"
        )
    if loaded_steps:
        bits.append(f"{loaded_steps} steps in context")
    elif loaded_steps == 0:
        bits.append("load a trajectory first")
    return "<div class='analysis-status'>" + " · ".join(bits) + "</div>"


def setup_help_text(cfg: AnalysisLLMConfig | None = None) -> str:
    """User-facing message when the panel cannot call a model."""
    cfg = cfg or resolve_analysis_config()
    missing = ", ".join(cfg.missing) if cfg.missing else "ANALYZE_BASE_URL, ANALYZE_API_KEY, ANALYZE_MODEL"
    return (
        "The analysis panel is not configured.\n\n"
        "Copy `.env.example` to `.env` in the repo root (or the process CWD) "
        f"and set: {missing}.\n\n"
        "`LABEL_*` from the step labeler is used when `ANALYZE_*` is omitted."
    )


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
