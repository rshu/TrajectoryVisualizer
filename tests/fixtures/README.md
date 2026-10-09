# Test fixtures

## `reference/` + `decaf_cache/`

A minimal, hermetic reference set for the DECAF failure-attribution integration
tests (wired in `tests/conftest.py`): two Claude Code runs on public open-source
tasks —

- `astropy__astropy-13033` — deductive-only diagnosis (primary
  `code_editing/incorrect_patch`)
- `django__django-11477` — arbiter-refuted case (`refuted_unattributed`), with
  its judge + arbiter verdicts vendored under `decaf_cache/` (the
  `deepseek-v4-pro` cache namespace)

`reference/data/` follows the layout DECAF reads: `requirements/<id>.json` (task
and reference patch), `patch/<agent>/<id>.diff` (the agent's edit set),
`trajectory/<agent>/<id>.json` (the run) and `eval_<agent>.json` (whether each
run resolved its task).

**Sanitization:** session/message UUIDs and request ids are deterministic
placeholders, signature/encrypted-content blobs are emptied, and absolute paths
are rewritten under `/home/user`. The golden tests verify the sanitized runs
produce the pinned diagnosis. Fixtures are additionally **checked for
credential-shaped strings** by `tests/test_fixture_hygiene.py` (redacted
reporting). Do not add fixture content from non-public runs. A new case needs the
same transformations, after which any vendored verdict for it must be
regenerated on the sanitized bytes (see `decaf_cache/README.md`): the verdict
stamps cover the exact bytes the model saw.

Fixture contents must stay byte-stable: the golden tests pin the expected
diagnosis as literals, and the identity check in `attribution.diagnose()`
compares displayed files byte-for-byte against `reference/data/trajectory/...`.
