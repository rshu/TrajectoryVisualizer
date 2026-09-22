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
    args = parser.parse_args()

    if args.report:
        from .report import ReportError, report_from_path

        try:
            path = report_from_path(args.report, args.output, dark=args.dark)
        except ReportError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        print(path)
        return

    try:
        from .insight import build_ui
        from .styles import APP_CSS
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
        css=APP_CSS,
        head='<meta name="color-scheme" content="light">',
    )


if __name__ == "__main__":
    main()
