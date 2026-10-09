"""Entry point for `python -m trajviz.insight`."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Insight")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1). Use 0.0.0.0 to accept connections from any IP.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument(
        "--report",
        metavar="TRAJECTORY",
        help="Write a standalone HTML report for TRAJECTORY and exit (no dashboard)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="HTML report output file or directory (default: a temp file; prints the path)",
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Render the HTML report with dark charts and tokens (with --report)",
    )
    parser.add_argument(
        "--inline-plotly",
        action="store_true",
        help="Embed plotly.js in the report so it works offline (with --report; adds several MB)",
    )
    args = parser.parse_args()

    if args.report:
        from .report import ReportError, report_from_path

        try:
            path = report_from_path(args.report, args.output, dark=args.dark, inline_plotly=args.inline_plotly)
        except ReportError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        print(path)
        return

    try:
        from .insight import LAUNCH_PRESENTATION, build_ui
    except ImportError:
        print(
            "Error: Insight dependencies are not installed.\n"
            "Install them with:\n"
            "  pip install -e .            (package-managed)\n"
            "  pip install -r requirements.txt   (requirements file)",
            file=sys.stderr,
        )
        sys.exit(1)

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        # build_ui() returns a deliberately unstyled app; presentation is applied
        # at launch. Spread the shared dict so an embedder cannot drift from it.
        **LAUNCH_PRESENTATION,
    )


if __name__ == "__main__":
    main()
