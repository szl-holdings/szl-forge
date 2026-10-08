"""Offline controls for the manual public-card release entry point."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from tools import check_receiptagent_v3_card_dispatch as dispatch


SOURCE = "a" * 40


def valid_environment() -> dict[str, str]:
    return {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": dispatch.writer.SOURCE_REPOSITORY,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": SOURCE,
        "CARD_SOURCE_REVISION": SOURCE,
        "CARD_EXPECTED_HUB_PARENT": dispatch.writer.card.HUB_PARENT,
        "CARD_CONFIRMATION": "README_ONLY_LOADER_WITHDRAWAL",
    }


class ManualDispatchTests(unittest.TestCase):
    def setUp(self):
        self.current = patch.object(dispatch.writer, "assert_current_main").start()
        self.committed = patch.object(dispatch.writer, "assert_writer_files_committed").start()
        self.addCleanup(patch.stopall)

    def test_valid_intent_binds_current_main_before_credentials(self):
        result = dispatch.validate_dispatch(valid_environment())
        self.assertEqual(result["source_revision"], SOURCE)
        self.assertEqual(result["expected_hub_parent"], dispatch.writer.card.HUB_PARENT)
        self.assertEqual(result["state"], "MANUAL_INTENT_AND_SOURCE_VERIFIED")
        self.assertEqual(result["candidate_kind"], "LOADER_EXAMPLE_WITHDRAWAL_ONLY")
        self.current.assert_called_once_with(SOURCE)
        self.committed.assert_called_once_with(SOURCE)

    def test_invalid_or_missing_intent_blocks_before_source_lookup(self):
        cases = {
            "GITHUB_EVENT_NAME": ["push", "pull_request", ""],
            "GITHUB_REPOSITORY": ["other/szl-forge", ""],
            "GITHUB_REF": ["refs/heads/szl/candidate", "refs/tags/v1", ""],
            "GITHUB_RUN_ATTEMPT": ["2", "01", ""],
            "GITHUB_SHA": ["short", "A" * 40, ""],
            "CARD_SOURCE_REVISION": ["b" * 40, SOURCE + " ", ""],
            "CARD_EXPECTED_HUB_PARENT": ["b" * 40, ""],
            "CARD_CONFIRMATION": ["publish", "README_ONLY_UNQUALIFIED", ""],
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    environment = valid_environment()
                    environment[name] = value
                    with self.assertRaises(dispatch.DispatchError):
                        dispatch.validate_dispatch(environment)
        self.current.assert_not_called()
        self.committed.assert_not_called()

    def test_superseded_main_blocks_before_writer_binding(self):
        self.current.side_effect = dispatch.writer.PublicationError("superseded main")
        with self.assertRaisesRegex(dispatch.writer.PublicationError, "superseded"):
            dispatch.validate_dispatch(valid_environment())
        self.committed.assert_not_called()

    def test_modified_release_source_is_rejected(self):
        self.committed.side_effect = dispatch.writer.PublicationError("modified writer")
        with self.assertRaisesRegex(dispatch.writer.PublicationError, "modified"):
            dispatch.validate_dispatch(valid_environment())

    def test_cli_failure_does_not_disclose_environment_or_credential(self):
        environment = valid_environment()
        environment["GITHUB_RUN_ATTEMPT"] = "2"
        environment["HF_TOKEN"] = "synthetic-secret-never-print"
        output = io.StringIO()
        with patch.dict(dispatch.os.environ, environment, clear=True), contextlib.redirect_stdout(output):
            result = dispatch.main()
        self.assertEqual(result, 1)
        self.assertIn("BLOCKED_NO_WRITE", output.getvalue())
        self.assertNotIn(environment["HF_TOKEN"], output.getvalue())
        self.current.assert_not_called()

    def test_workflow_is_manual_and_gates_credentials_on_source(self):
        text = (dispatch.writer.ROOT / dispatch.WORKFLOW_PATH).read_text(encoding="utf-8")
        self.assertIn("on:\n  workflow_dispatch:", text)
        self.assertNotIn("  push:", text)
        self.assertNotIn("  pull_request:", text)
        self.assertIn("github.ref == 'refs/heads/main'", text)
        self.assertIn("github.repository == 'szl-holdings/szl-forge'", text)
        self.assertIn("ref: ${{ github.sha }}", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertIn("huggingface-hub==1.23.0", text)
        self.assertIn("README_ONLY_LOADER_WITHDRAWAL", text)
        self.assertEqual(text.count("${{ secrets."), 1)
        self.assertIn("HF_ORG_TOKEN_CANDIDATE: ${{ secrets.HF_ORG_TOKEN }}", text)
        self.assertNotIn("--allow-create", text)
        self.assertNotIn("--oidc-resource", text)
        self.assertIn("mkdir -p reports", text)
        for name, input_name in (
            ("CARD_SOURCE_REVISION", "source_revision"),
            ("CARD_EXPECTED_HUB_PARENT", "expected_hub_parent"),
            ("CARD_CONFIRMATION", "confirmation"),
        ):
            self.assertIn(f"{name}: ${{{{ inputs.{input_name} }}}}", text)
        self.assertIn('reports/publication-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.json', text)
        self.assertIn("if-no-files-found: warn", text)
        self.assertLess(text.rindex("tools/check_receiptagent_v3_card_dispatch.py"),
                        text.index("HF_ORG_TOKEN_CANDIDATE:"))
        self.assertLess(text.index("tools/acquire_hf_publisher_token.py"),
                        text.index("--publish"))
        self.assertIn("if: always()", text)

    def test_dispatch_workflow_and_credential_selector_are_source_bound(self):
        for path in (
            "tools/check_receiptagent_v3_card_dispatch.py",
            "tools/acquire_hf_publisher_token.py",
            dispatch.WORKFLOW_PATH,
        ):
            self.assertIn(path, dispatch.writer.SOURCE_BOUND_WRITER_FILES)


if __name__ == "__main__":
    unittest.main()
