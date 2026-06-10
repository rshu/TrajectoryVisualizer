"""Tests for the shared launch auth/exposure policy (trajectory_visualizer._server)."""

import argparse
import unittest

from trajectory_visualizer._server import (
    add_server_args,
    is_exposed,
    parse_auth,
    resolve_launch_security,
)


def _args(**overrides):
    parser = argparse.ArgumentParser()
    add_server_args(parser)
    ns = parser.parse_args([])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class ParseAuthTests(unittest.TestCase):
    def test_none_when_unset(self):
        self.assertIsNone(parse_auth(None))
        self.assertIsNone(parse_auth(""))

    def test_splits_user_pass(self):
        self.assertEqual(parse_auth("alice:s3cret"), ("alice", "s3cret"))

    def test_password_may_contain_colons(self):
        self.assertEqual(parse_auth("alice:a:b:c"), ("alice", "a:b:c"))

    def test_rejects_missing_colon(self):
        with self.assertRaises(ValueError):
            parse_auth("nopassword")

    def test_rejects_empty_field(self):
        with self.assertRaises(ValueError):
            parse_auth(":secret")
        with self.assertRaises(ValueError):
            parse_auth("user:")


class IsExposedTests(unittest.TestCase):
    def test_loopback_not_exposed(self):
        for host in ("127.0.0.1", "localhost", "::1"):
            self.assertFalse(is_exposed(host, share=False))

    def test_share_is_exposed(self):
        self.assertTrue(is_exposed("127.0.0.1", share=True))

    def test_nonloopback_host_is_exposed(self):
        self.assertTrue(is_exposed("0.0.0.0", share=False))


class ResolveLaunchSecurityTests(unittest.TestCase):
    def test_loopback_no_auth_is_allowed(self):
        self.assertIsNone(resolve_launch_security(_args(), prog="test"))

    def test_exposed_without_auth_exits(self):
        with self.assertRaises(SystemExit) as cm:
            resolve_launch_security(_args(share=True), prog="test")
        self.assertEqual(cm.exception.code, 2)

    def test_nonloopback_without_auth_exits(self):
        with self.assertRaises(SystemExit):
            resolve_launch_security(_args(host="0.0.0.0"), prog="test")

    def test_exposed_with_auth_ok(self):
        auth = resolve_launch_security(_args(share=True, auth="u:p"), prog="test")
        self.assertEqual(auth, ("u", "p"))

    def test_exposed_with_explicit_override_ok(self):
        auth = resolve_launch_security(
            _args(share=True, allow_unauthenticated=True), prog="test"
        )
        self.assertIsNone(auth)

    def test_env_var_supplies_auth(self):
        import os
        os.environ["GRADIO_AUTH"] = "envuser:envpass"
        try:
            auth = resolve_launch_security(_args(host="0.0.0.0"), prog="test")
        finally:
            del os.environ["GRADIO_AUTH"]
        self.assertEqual(auth, ("envuser", "envpass"))


if __name__ == "__main__":
    unittest.main()
