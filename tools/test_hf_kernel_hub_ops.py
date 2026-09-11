"""Offline contract tests for Kernel Hub ops. No Hub writes, no weight loads."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import hf_kernel_hub_ops as ops


def _kernel(sha="a" * 40, branches=None):
    branches = branches or {"main": "b" * 40, "v1": "c" * 40}
    return (
        {"sha": sha, "repoType": "kernel", "lastModified": "2026-09-01T00:00:00.000Z"},
        {"branches": [{"name": n, "targetCommit": s} for n, s in branches.items()], "tags": []},
    )


class ClassifyTests(unittest.TestCase):
    def test_exact_revision_admitted(self):
        result = ops.classify_consumer_call({
            "repo_id": "SZLHOLDINGS/szl-kernels",
            "revision": "c" * 40,
            "repo_type": "kernel",
        })
        self.assertTrue(result["ok"])
        self.assertEqual(result["state"], "ADMITTED")
        self.assertFalse(result["executed"])

    def test_model_type_forbidden(self):
        result = ops.classify_consumer_call({
            "repo_id": "SZLHOLDINGS/szl-kernels",
            "revision": "v1",
            "repo_type": "model",
        })
        self.assertIn("LEGACY_MODEL_TYPE_FORBIDDEN", result["blockers"])

    def test_missing_revision_hold(self):
        result = ops.classify_consumer_call({
            "repo_id": "SZLHOLDINGS/szl-kernels",
        })
        self.assertIn("REVISION_REQUIRED", result["blockers"])

    def test_floating_main_hold(self):
        result = ops.classify_consumer_call({
            "repo_id": "SZLHOLDINGS/szl-kernels",
            "revision": "main",
        })
        self.assertIn("FLOATING_MAIN_NOT_ADMITTED_AT_RUNTIME", result["blockers"])

    def test_trust_remote_code_is_review_not_repair(self):
        result = ops.classify_consumer_call({
            "repo_id": "SZLHOLDINGS/szl-kernels",
            "revision": "v1",
            "trust_remote_code": True,
        })
        self.assertIn("TRUST_REMOTE_CODE_REQUIRES_REVIEW", result["blockers"])
