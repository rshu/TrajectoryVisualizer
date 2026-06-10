# Security

TrajectoryVisualizer is **offline analytics for your own agent trajectories**.
Its safe, intended deployment is a single analyst on `127.0.0.1`. The notes
below matter when you step beyond that.

## Threat model

A trajectory file is **untrusted input** and routinely embeds source code, shell
output, file paths, and sometimes secrets. The dashboard renders that content
and has **no per-resource access control** — anyone who can reach the server can
see everything in a loaded trajectory.

## What the tool does to protect you

- **Auth on exposed launches.** Launching with `--share` or a non-loopback
  `--host` (e.g. `0.0.0.0`) **requires** authentication: pass `--auth USER:PASS`
  or set `$GRADIO_AUTH`. Without it the server refuses to start. Override
  deliberately with `--allow-unauthenticated` (not recommended).
- **Loopback default.** The default bind is `127.0.0.1` with no public share.
- **Input guarding.** Uploads are size-capped before being read; malformed or
  wrong-shape files produce a friendly error instead of a crash; Codex `.jsonl`
  parsing recovers from individual corrupt lines.
- **No path traversal via batch manifests.** Uploaded Converge manifests are
  confined to the manifest's directory (no absolute paths, no `..` escape), so a
  hostile manifest cannot read arbitrary files.
- **No traceback disclosure.** Errors are logged server-side; clients get a
  generic message, never a stack trace.
- **Telemetry off by default.** Gradio analytics are disabled on launch.

## Sending data off the machine (labelers)

`scripts/step_labeler.py` and `scripts/training_labeler.py` are **network**
tools: they POST trajectory content to the LLM endpoint you configure
(`LABEL_BASE_URL`). Do not label trajectories containing secrets you cannot
share with that endpoint. Prefer `LABEL_API_KEY` in `.env`/the environment over
`--api-key` on the command line (which is visible via the process list).

## Reporting a vulnerability

Please open a private report via the repository's
[Issues](https://github.com/rshu/TrajectoryVisualizer/issues) (mark it security)
or contact the maintainer directly. Do not include real secrets or proprietary
trajectories in a public report.
