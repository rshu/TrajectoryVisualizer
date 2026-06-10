"""Launch the Converge Gradio app via `python -m trajectory_visualizer.converge`."""

import argparse

from .._server import add_server_args, resolve_launch_security


def main():
    parser = argparse.ArgumentParser(description="Converge — two-trajectory comparison")
    add_server_args(parser)
    args = parser.parse_args()

    # Same auth/exposure policy as Insight; Converge can read local files named
    # in an uploaded manifest, so an unauthenticated exposed launch is worse here.
    auth = resolve_launch_security(args, prog="converge")

    from .app import build_ui

    app = build_ui()
    app.launch(server_name=args.host, server_port=args.port, share=args.share, auth=auth)


if __name__ == "__main__":
    main()
