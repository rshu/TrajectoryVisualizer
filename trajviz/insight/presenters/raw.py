"""Raw Data tab presenter."""

from __future__ import annotations

import json


def raw_json_text(raw: dict) -> str:
    """Pretty-print trajectory JSON, truncated at 500KB."""
    raw_str = json.dumps(raw, indent=2, ensure_ascii=False, default=str)
    if len(raw_str) > 500_000:
        raw_str = raw_str[:500_000] + "\n\n... (truncated at 500KB)"
    return raw_str
