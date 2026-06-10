"""Entry point for `python -m trajectory_visualizer.insight`."""

import argparse
import logging
import sys

from .._server import add_server_args, resolve_launch_security


def main():
    parser = argparse.ArgumentParser(description="Insight")
    add_server_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Enforce the auth/exposure policy before importing the (heavy) UI stack so
    # a misconfigured exposed launch fails fast.
    auth = resolve_launch_security(args, prog="insight")
    logging.getLogger(__name__).info(
        "Starting Insight on %s:%s (share=%s, auth=%s)",
        args.host, args.port, args.share, "yes" if auth else "no",
    )

    try:
        from .insight import build_ui, APP_CSS
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
    app.launch(server_name=args.host, server_port=args.port, share=args.share,
               auth=auth, css=APP_CSS)


if __name__ == "__main__":
    main()
