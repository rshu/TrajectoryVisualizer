"""Launch the Converge Gradio app via `python -m trajviz.converge`.

The batch/report CLI is a separate entry point: the ``trajectory-converge``
console script, or ``python -m trajviz.converge.cli``. This module intentionally
launches the app (same precedent as ``trajviz.insight.__main__``), so do not
repoint it at the CLI — that would break every documented app invocation.
"""

from .app import LAUNCH_PRESENTATION, build_ui


def main():
    app = build_ui()
    app.launch(**LAUNCH_PRESENTATION)


if __name__ == "__main__":
    main()
