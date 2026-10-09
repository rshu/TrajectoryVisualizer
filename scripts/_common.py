"""Shared safety helpers for the scripts/ CLI tools.

Consolidates the previously triplicated "output must not overwrite a source
file" guard (opencode_consolidator, codearts_consolidator, step_labeler_v2)
and the duplicated atomic-JSON-write implementation (codearts_consolidator,
step_labeler_v2) into a single module.  Each caller keeps its own exception
type and message wording by passing them in, so CLI behavior is unchanged.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from collections.abc import Iterable


def ensure_output_does_not_overwrite(
    output_path: str | Path,
    source_paths: Iterable[str | Path],
    *,
    exc: type[Exception] = ValueError,
    message: str = "Output path would overwrite input source: {source}",
    allow_stdout_dash: bool = False,
) -> None:
    """Raise ``exc`` when ``output_path`` resolves to any of ``source_paths``.

    ``message`` may contain a ``{source}`` placeholder, filled with the
    offending resolved source path.  When ``allow_stdout_dash`` is true, an
    output path of ``"-"`` (stdout) is always allowed.
    """
    if allow_stdout_dash and str(output_path) == "-":
        return
    destination = Path(output_path).expanduser().resolve()
    for source_path in source_paths:
        source = Path(source_path).expanduser().resolve()
        same_file = destination == source
        if not same_file and destination.exists() and source.exists():
            try:
                same_file = os.path.samefile(destination, source)
            except OSError:
                # The resolved-path comparison above still protects the common
                # case when a locked source cannot be inspected further.
                same_file = False
        if same_file:
            raise exc(message.format(source=source))


def write_json_atomic(output_path: str | Path, data: Any) -> None:
    """Replace ``output_path`` only after its complete JSON is durable.

    Dumps to a NamedTemporaryFile in the destination directory, fsyncs, then
    atomically renames over the target; the temporary file is removed on
    failure so an interrupted write can never leave a truncated output.
    """
    path = Path(output_path).expanduser().resolve()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass
