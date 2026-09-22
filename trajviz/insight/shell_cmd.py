"""Shell command peeling for chart labels and search detection.

Leaf module (no insight.diagnostics / patterns imports) so diagnostics can
label Bash failure clusters without a patterns circular import.
"""

from __future__ import annotations

import re
import shlex

from trajviz.tool_vocab import BASH_TOOL_NAMES

_SEARCH_BASH_PREFIXES = ("grep", "rg", "ag", "find", "locate", "fgrep", "egrep", "ripgrep")
_SHELL_PUNCTUATION = ";&|()\n"
_WRAPPER_OPTIONS_WITH_VALUES = {
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "git": {
        "-C", "-c", "--git-dir", "--work-tree", "--namespace",
        "--super-prefix", "--config-env",
    },
    "sudo": {
        "-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt",
        "-C", "--close-from", "-r", "--role", "-t", "--type", "-T",
        "--command-timeout", "-D", "--chdir", "-R", "--chroot", "-U",
        "--other-user",
    },
    "time": {"-f", "--format", "-o", "--output"},
    "nice": {"-n", "--adjustment"},
    "timeout": {"-k", "--kill-after", "-s", "--signal"},
    "xargs": {
        "-a", "--arg-file", "-E", "--eof", "-I", "--replace", "-L",
        "--max-lines", "-n", "--max-args", "-P", "--max-procs", "-s",
        "--max-chars",
    },
}
# Include Windows ``.exe`` / ``pythonw`` so chart labels name the script/module.
_PYTHON_INTERPRETER_RE = re.compile(
    r"^(?:pythonw?|pypy)\d*(?:\.\d+)*(?:\.exe)?$"
)
_PYTHON_VALUE_OPTIONS = frozenset({
    "-W", "-X", "-Q", "--check-hash-based-pycs",
})
# Drive path or .exe → treat ``\`` as a path separator, not a posix escape.
_WIN_CMD_HINT_RE = re.compile(r"(?i)(?:[A-Za-z]:\\|\.exe\b)")


def _shell_segments(command: str) -> list[list[str]]:
    """Lex shell command segments without evaluating or executing anything."""
    if "\\" in command and _WIN_CMD_HINT_RE.search(command):
        command = command.replace("\\", "/")
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_SHELL_PUNCTUATION)
    # Preserve newlines as command boundaries while still honoring quoted ones.
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        # Be conservative for malformed/unclosed quoting.
        return []

    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token and all(char in _SHELL_PUNCTUATION for char in token):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _path_basename(path: str) -> str:
    """Basename of a POSIX or Windows path, preserving case."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _shell_name(token: str) -> str:
    """Return a case-insensitive executable basename."""
    return _path_basename(token).lower()


def _is_shell_assignment(token: str) -> bool:
    """Return True for a simple POSIX-style NAME=value assignment."""
    name, separator, _ = token.partition("=")
    return bool(
        separator
        and name
        and (name[0].isalpha() or name[0] == "_")
        and all(char.isalnum() or char == "_" for char in name)
    )


def _skip_wrapper_options(tokens: list[str], index: int, wrapper: str) -> int:
    """Skip known wrapper flags, including flags whose value is separate."""
    value_options = _WRAPPER_OPTIONS_WITH_VALUES.get(wrapper, set())
    while index < len(tokens):
        option = tokens[index]
        if option == "--":
            return index + 1
        if not option.startswith("-") or option == "-":
            return index
        index += 1
        if option in value_options and index < len(tokens):
            index += 1
    return index


def _is_windows_timeout_invocation(tokens: list[str], index: int) -> bool:
    """True when ``timeout`` is the Windows sleep builtin (``/t``, ``/nobreak``)."""
    for tok in tokens[index + 1:index + 4]:
        low = tok.lower()
        if low in ("/t", "/nobreak") or low.startswith("/t:"):
            return True
        if not low.startswith("/"):
            break
    return False


def _cmd_script_tokens(tokens: list[str], index: int) -> list[str]:
    """Tokens after ``cmd /c`` or ``cmd /k``; empty if that form is not present."""
    i = index + 1
    while i < len(tokens):
        low = tokens[i].lower()
        if low in ("/c", "/k"):
            return tokens[i + 1:]
        # Other cmd switches (/d, /q, /s, /u, /a, /e:on, …).
        if low.startswith("/") or (low.startswith("-") and low != "-"):
            i += 1
            continue
        return []
    return []


def _wrapper_inner_label(rest: list[str], nesting: int, fallback: str) -> str:
    """Label the command after a ``cmd /c`` / similar wrapper."""
    if not rest:
        return fallback
    if len(rest) == 1:
        return primary_shell_command(rest[0], _nesting=nesting + 1) or fallback
    return _primary_from_segment(rest, nesting + 1) or fallback


def _wrapper_inner_runs_search(rest: list[str], nesting: int) -> bool:
    """Whether the command after a shell wrapper runs a search tool."""
    if not rest:
        return False
    if len(rest) == 1:
        return any(
            _segment_runs_search(segment, nesting + 1)
            for segment in _shell_segments(rest[0])
        )
    return _segment_runs_search(rest, nesting + 1)


# powershell / pwsh: peel -Command body or -File basename (skip other switches).
_PS_INTERPRETERS = frozenset({
    "powershell", "powershell.exe", "pwsh", "pwsh.exe",
})
_PS_OPTIONS = {
    "c": "command", "command": "command",
    "f": "file", "file": "file",
    "e": "encodedcommand", "ec": "encodedcommand", "encodedcommand": "encodedcommand",
    "wd": "workingdirectory", "workingdirectory": "workingdirectory",
    "ex": "executionpolicy", "ep": "executionpolicy", "executionpolicy": "executionpolicy",
    "in": "inputformat", "inputformat": "inputformat",
    "out": "outputformat", "outputformat": "outputformat",
    "win": "windowstyle", "windowstyle": "windowstyle",
    "version": "version",
    "psc": "psconsolefile", "psconsolefile": "psconsolefile",
}
_PS_VALUE_OPTIONS = frozenset({
    "command", "encodedcommand", "file", "inputformat", "outputformat",
    "workingdirectory", "executionpolicy", "windowstyle", "version",
    "psconsolefile",
})


def _ps_option_key(token: str) -> str | None:
    """Canonical PowerShell switch name, ``_`` for unknown flags, else ``None``."""
    if len(token) < 2 or token[0] not in "-/":
        return None
    raw = token[1:].lower()
    if raw.startswith("-"):
        raw = raw.lstrip("-")
    if not raw:
        return None
    if raw in _PS_OPTIONS:
        return _PS_OPTIONS[raw]
    # Unknown -Flag (e.g. -NoProfile): skip without consuming a value.
    if raw[0].isalpha():
        return "_"
    return None


def _ps_unwrap_scriptblock_text(script: str) -> str:
    """Drop a surrounding ``{ … }`` script-block wrapper when present."""
    text = script.strip()
    if text.startswith("{") and text.endswith("}"):
        return text[1:-1].strip()
    return text


def _ps_unwrap_scriptblock_tokens(tokens: list[str]) -> list[str]:
    """Drop a surrounding ``{ … }`` script-block wrapper when present."""
    if len(tokens) >= 2 and tokens[0] == "{" and tokens[-1] == "}":
        return tokens[1:-1]
    return tokens


def _powershell_payload(
    tokens: list[str], index: int, nesting: int,
) -> tuple[str, str | list[str]] | None:
    """Return ``('command', body)`` or ``('file', path)`` from powershell/pwsh argv."""
    i = index + 1
    while i < len(tokens):
        key = _ps_option_key(tokens[i])
        if key == "command":
            if nesting >= 3 or i + 1 >= len(tokens):
                return None
            rest = tokens[i + 1:]
            if len(rest) == 1 and rest[0] == "-":
                return None
            if len(rest) == 1:
                return ("command", _ps_unwrap_scriptblock_text(rest[0]))
            return ("command", _ps_unwrap_scriptblock_tokens(rest))
        if key == "file":
            if i + 1 >= len(tokens):
                return None
            return ("file", tokens[i + 1])
        if key == "encodedcommand":
            return None
        if key in _PS_VALUE_OPTIONS:
            i += 2
            continue
        if key is not None:
            i += 1
            continue
        return None
    return None


def _powershell_invocation_label(
    tokens: list[str], index: int, interpreter: str, nesting: int,
) -> str:
    """Label a powershell/pwsh invocation by ``-Command`` body or ``-File`` basename."""
    peeled = _powershell_payload(tokens, index, nesting)
    if peeled is None:
        return interpreter
    kind, body = peeled
    if kind == "file":
        return _path_basename(str(body)) or interpreter
    if isinstance(body, str):
        return primary_shell_command(body, _nesting=nesting + 1) or interpreter
    return _primary_from_segment(body, nesting + 1) or interpreter


def _powershell_runs_search(tokens: list[str], index: int, nesting: int) -> bool:
    """True when a powershell/pwsh wrapper's ``-Command`` body runs a search tool."""
    peeled = _powershell_payload(tokens, index, nesting)
    if peeled is None or peeled[0] != "command":
        return False
    body = peeled[1]
    if isinstance(body, str):
        return any(
            _segment_runs_search(segment, nesting + 1)
            for segment in _shell_segments(body)
        )
    return _segment_runs_search(body, nesting + 1)


def _segment_runs_search(tokens: list[str], nesting: int = 0) -> bool:
    """Recognize a search executable at the head of one shell segment."""
    index = 0
    # Bound wrapper traversal even for adversarially repetitive input.
    for _ in range(12):
        while index < len(tokens) and _is_shell_assignment(tokens[index]):
            index += 1
        if index >= len(tokens):
            return False

        command = _shell_name(tokens[index])
        if command in _SEARCH_BASH_PREFIXES:
            return True

        if command == "git":
            index = _skip_wrapper_options(tokens, index + 1, "git")
            return index < len(tokens) and _shell_name(tokens[index]) == "grep"

        if command == "command":
            index += 1
            if index < len(tokens) and tokens[index] in ("-v", "-V"):
                return False
            index = _skip_wrapper_options(tokens, index, "command")
            continue

        if command in ("env", "sudo", "time", "nice", "nohup", "xargs"):
            index = _skip_wrapper_options(tokens, index + 1, command)
            continue

        if command == "timeout":
            if _is_windows_timeout_invocation(tokens, index):
                return False
            index = _skip_wrapper_options(tokens, index + 1, command)
            # timeout's first positional argument is the duration.
            index += 1
            continue

        if command == "busybox":
            index += 1
            continue

        if command in ("cmd", "cmd.exe") and nesting < 3:
            return _wrapper_inner_runs_search(_cmd_script_tokens(tokens, index), nesting)

        if command in _PS_INTERPRETERS and nesting < 3:
            return _powershell_runs_search(tokens, index, nesting)

        if command in ("bash", "dash", "ksh", "sh", "zsh") and nesting < 3:
            # A quoted ``sh -c`` script is data to this process.  Lex it again
            # with the same non-executing tokenizer instead of invoking a shell.
            option_index = index + 1
            while option_index < len(tokens) and tokens[option_index].startswith("-"):
                flags = tokens[option_index].lstrip("-")
                if "c" in flags and option_index + 1 < len(tokens):
                    return any(
                        _segment_runs_search(segment, nesting + 1)
                        for segment in _shell_segments(tokens[option_index + 1])
                    )
                option_index += 1
            return False

        if command in ("if", "then", "elif", "while", "until", "do", "!"):
            index += 1
            continue

        return False
    return False


def primary_shell_command(command: str, *, _nesting: int = 0) -> str | None:
    """Return the primary executable basename for a shell command string.

    Peels common wrappers (``env``, ``sudo``, ``timeout``, ``sh -c``, ``cmd /c``,
    ``powershell -Command``, …) and
    skips leading directory-change segments (``cd`` / ``pushd`` / ``popd``)
    so ``cd src && git status`` reports ``git``. Returns ``None`` when nothing
    useful can be recovered.
    """
    if not isinstance(command, str) or not command.strip():
        return None

    segments = _shell_segments(command)
    if not segments:
        # Malformed quoting — fall back to the converge-style first token.
        for tok in command.strip().split():
            if _is_shell_assignment(tok):
                continue
            return _shell_name(tok) or None
        return None

    for segment in segments:
        name = _primary_from_segment(segment, _nesting)
        if name is None:
            continue
        # Agents often prefix real work with ``cd … && …``; attribute the call
        # to the following command when the lead segment is only navigation.
        if name in ("cd", "pushd", "popd") and len(segments) > 1:
            continue
        return name
    return None


def tool_call_hint(tc: dict) -> str:
    """Short per-call label (command/path/pattern) for chart hovers."""
    text = tc.get("title") if isinstance(tc.get("title"), str) else ""
    if not text.strip():
        inp = tc.get("input")
        if isinstance(inp, dict):
            for key in ("command", "file_path", "path", "pattern", "description", "prompt"):
                v = inp.get(key)
                if isinstance(v, str) and v.strip():
                    text = v
                    break
    text = " ".join(text.split())
    if len(text) > 64:
        text = text[:63] + "…"
    return text


def tool_chart_name(tc: dict) -> str:
    """Chart label for a tool call; expand Bash into the shell command/script."""
    name = tc.get("tool_name") or "(unnamed)"
    if name not in BASH_TOOL_NAMES:
        return name
    inp = tc.get("input")
    command = inp.get("command") if isinstance(inp, dict) else None
    if not (isinstance(command, str) and command.strip()):
        # Split Chrys calls keep the intended command on the stub's title.
        command = tc.get("title")
    if not (isinstance(command, str) and command.strip()):
        return name
    return primary_shell_command(command) or name


def _primary_from_segment(tokens: list[str], nesting: int = 0) -> str | None:
    """Resolve the executable at the head of one shell segment."""
    index = 0
    for _ in range(12):
        while index < len(tokens) and _is_shell_assignment(tokens[index]):
            index += 1
        if index >= len(tokens):
            return None

        command_name = _shell_name(tokens[index])
        if not command_name:
            return None

        if command_name == "command":
            index += 1
            if index < len(tokens) and tokens[index] in ("-v", "-V"):
                return "command"
            index = _skip_wrapper_options(tokens, index, "command")
            continue

        if command_name in ("env", "sudo", "time", "nice", "nohup", "xargs"):
            index = _skip_wrapper_options(tokens, index + 1, command_name)
            continue

        if command_name == "timeout":
            if _is_windows_timeout_invocation(tokens, index):
                return "timeout"
            index = _skip_wrapper_options(tokens, index + 1, command_name)
            index += 1
            continue

        if command_name == "busybox":
            index += 1
            continue

        if command_name in ("cmd", "cmd.exe"):
            if nesting >= 3:
                return command_name
            return _wrapper_inner_label(
                _cmd_script_tokens(tokens, index), nesting, command_name,
            )

        if command_name in _PS_INTERPRETERS:
            return _powershell_invocation_label(tokens, index, command_name, nesting)

        if command_name in ("bash", "dash", "ksh", "sh", "zsh"):
            option_index = index + 1
            while option_index < len(tokens) and tokens[option_index].startswith("-"):
                flags = tokens[option_index].lstrip("-")
                if "c" in flags and option_index + 1 < len(tokens):
                    if nesting >= 3:
                        return command_name
                    return primary_shell_command(
                        tokens[option_index + 1], _nesting=nesting + 1
                    )
                option_index += 1
            return command_name

        if command_name in ("if", "then", "elif", "while", "until", "do", "!"):
            index += 1
            continue

        if _PYTHON_INTERPRETER_RE.fullmatch(command_name):
            return _python_invocation_label(tokens, index, command_name)

        return command_name
    return None


def _python_invocation_label(tokens: list[str], index: int, interpreter: str) -> str:
    """Label a python/pypy invocation by script basename or ``-m`` module.

    ``python -c …`` and bare interpreters keep the interpreter name — there is
    no stable script identity to chart.
    """
    i = index + 1
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            i += 1
            break
        if token == "-m" and i + 1 < len(tokens):
            return tokens[i + 1]
        if token == "-c":
            return interpreter
        if token.startswith("-") and token != "-":
            if token in _PYTHON_VALUE_OPTIONS and i + 1 < len(tokens):
                i += 2
                continue
            if token.startswith("--") and "=" in token:
                i += 1
                continue
            i += 1
            continue
        break

    if i >= len(tokens):
        return interpreter
    script = tokens[i]
    base = _path_basename(script)
    return base or interpreter



def shell_runs_search(command: str) -> bool:
    """True when a shell command string runs a search-like executable."""
    if not isinstance(command, str) or not command.strip():
        return False
    return any(_segment_runs_search(segment) for segment in _shell_segments(command))
