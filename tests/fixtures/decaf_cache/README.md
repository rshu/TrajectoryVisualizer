# Vendored DECAF verdict caches

Two LLM verdicts — one judge, one arbiter — for
`claude_code / django__django-11477`, vendored so the attribution integration
tests run hermetically: no developer-local caches, no API key, no network, and
no dependence on DECAF's own (gitignored) cache directories.

`conftest.py` points DECAF's judge and arbiter cache directories here, forces the
judge-model namespace to `deepseek-v4-pro` (the namespace these files live
under), and points `config.PROJECT_DATA` at this directory so DECAF's local
inputs (`pertest/`, `overrides/`) resolve as absent instead of reaching outside
the fixture tree.

## Provenance

`attribution._verdict_verifies` trusts a cached verdict only when its full prompt
provenance matches the current inputs: `trajectory_sha256`,
`requirements_sha256`, `prompt_version` and `evidence_schema`. A verdict whose
provenance cannot be verified is ignored and its layer is disabled — that refusal
is the security property, so the stamps in these files must never be edited just
to make a test pass.

Both verdicts were produced by `deepseek-v4-pro` with DECAF's own `judge_one` and
`verify_one`, run against this fixture tree exactly as the tests configure it
(fixture `data/`, empty `PROJECT_DATA`), so every stamp is the one DECAF wrote
for the bytes the model saw. The arbiter was asked about the single claim DECAF's
blame step submits for this case (`code_editing` / `relevant_change_omitted`)
and refuted it.

If the fixture's bytes, the prompt or `SCHEMA_VERSION` change, regenerate both
verdicts the same way, or let the layers be refused and adjust the tests to
expect refusal. Re-stamping is sound only when rendering `build_messages` for
this case before and after the change gives byte-identical prompts.
