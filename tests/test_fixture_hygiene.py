"""Fixture hygiene — runs in EVERY environment (no DECAF or real-data dependency, so
it is NOT skipped in standalone CI).

Asserts no credential-shaped strings are committed in tests/fixtures. Failures
are redacted: they report the file and match COUNT only, never the matched
text, so a real credential is not reprinted into CI logs.
"""
import re
from pathlib import Path

_FIXTURES = Path(__file__).parent / "fixtures"

_PATTERNS = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}"                     # OpenAI/OpenRouter-style
    r"|sk-or-v1-[A-Za-z0-9]{16,}"                # OpenRouter explicit
    r"|ghp_[A-Za-z0-9]{20,}"                     # GitHub classic PAT
    r"|github_pat_[A-Za-z0-9_]{20,}"             # GitHub fine-grained PAT
    r"|gho_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}"  # GitHub OAuth/app tokens
    r"|AKIA[0-9A-Z]{16}"                         # AWS access key id
    r"|xox[abporst]-[A-Za-z0-9-]{10,}"           # Slack tokens
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"       # PEM private keys
    r"|eyJhbGciOi[A-Za-z0-9_-]{20,}"             # JWTs
    r"|Bearer\s+[A-Za-z0-9._-]{25,}"             # bearer tokens
    r"|api[_-]?key\s*[\"':=]+\s*[\"']?[A-Za-z0-9_-]{16,}",
    re.IGNORECASE,
)


def test_fixture_contains_no_secrets():
    assert _FIXTURES.is_dir(), "fixtures directory missing"
    offenders = []
    for p in _FIXTURES.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".diff", ".md", ".txt"}:
            n = len(_PATTERNS.findall(p.read_text(errors="replace")))
            if n:
                offenders.append((str(p.relative_to(_FIXTURES)), n))
    # REDACTED report: file + count only — never echo the matched strings.
    assert not offenders, (
        "credential-shaped strings found in fixtures (file, match_count): "
        f"{offenders} — inspect locally, do NOT paste matches into logs")
