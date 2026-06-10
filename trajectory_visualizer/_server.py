"""Shared CLI launch helpers: auth resolution + exposure guard.

Kept dependency-light (no gradio import) so the launch policy is unit-testable
headless and shared by both the Insight and Converge entry points.

Policy: a trajectory can embed source code, shell I/O, file paths, and secrets,
and the dashboard has no per-resource access control. So whenever the app is
*exposed* (``--share`` or a non-loopback ``--host``) we require authentication,
unless the operator deliberately opts out with ``--allow-unauthenticated``.
"""

from __future__ import annotations

import argparse
import os
import sys

# Hosts that only accept local connections (no auth required by default).
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", ""})


def add_server_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared --host/--port/--share/--auth options on *parser*."""
    parser.add_argument("--port", type=int, default=7860,
                        help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Server host (default: 127.0.0.1). Use 0.0.0.0 to "
                             "accept connections from any IP.")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link (exposes the app to the internet)")
    parser.add_argument("--auth", type=str, default=None, metavar="USER:PASS",
                        help="Require basic auth (USER:PASS). May also be set via "
                             "$GRADIO_AUTH. Required whenever the app is exposed "
                             "(--share or a non-loopback --host).")
    parser.add_argument("--allow-unauthenticated", action="store_true",
                        help="Permit an exposed launch with NO auth (dangerous; off by default).")


def parse_auth(raw: str | None):
    """Parse ``'user:pass'`` into a ``(user, pass)`` tuple, or ``None`` when unset.

    Raises ``ValueError`` on a malformed value (no colon, or empty field).
    """
    if not raw:
        return None
    if ":" not in raw:
        raise ValueError("--auth/$GRADIO_AUTH must be in 'user:pass' form")
    user, _, pw = raw.partition(":")
    if not user or not pw:
        raise ValueError("--auth/$GRADIO_AUTH must have a non-empty user and password")
    return (user, pw)


def is_exposed(host: str, share: bool) -> bool:
    """True when the launch is reachable beyond the local loopback interface."""
    return bool(share) or (host not in _LOOPBACK_HOSTS)


def resolve_launch_security(args, *, prog: str):
    """Return the auth tuple (or ``None``) for ``app.launch(...)``.

    Enforces the exposure policy: if the launch is exposed without auth and the
    operator did not pass ``--allow-unauthenticated``, print guidance and exit(2).
    """
    raw = args.auth or os.environ.get("GRADIO_AUTH")
    try:
        auth = parse_auth(raw)
    except ValueError as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        sys.exit(2)

    if is_exposed(args.host, args.share) and auth is None:
        if getattr(args, "allow_unauthenticated", False):
            print(
                f"{prog}: WARNING — launching EXPOSED (host={args.host}, share={args.share}) "
                f"with NO authentication. Trajectories may contain source code, shell output, "
                f"paths, and secrets. Proceeding because --allow-unauthenticated was set.",
                file=sys.stderr,
            )
        else:
            print(
                f"{prog}: refusing to launch exposed (host={args.host}, share={args.share}) "
                f"without authentication.\n"
                f"  Trajectories can embed source, shell I/O, file paths, and secrets.\n"
                f"  Fix one of:\n"
                f"    - pass --auth USER:PASS  (or set $GRADIO_AUTH=USER:PASS)\n"
                f"    - bind to 127.0.0.1      (the default; local-only)\n"
                f"    - pass --allow-unauthenticated to override deliberately",
                file=sys.stderr,
            )
            sys.exit(2)
    return auth
