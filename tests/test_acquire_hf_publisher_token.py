from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import acquire_hf_publisher_token as auth


OIDC_TOKEN = "hf_jwt_oidc-valid"
ORG_TOKEN = "hf_org_expired"
WRITE_TOKEN = "hf_write_valid"


class PublisherCredentialSelectionTests(unittest.TestCase):
    @staticmethod
    def _result(source: str, access: str = "EXISTING_WRITE_CONFIRMED"):
        return auth.ValidationResult(
            source=source,
            identity_sha256="a" * 64,
            target_access=access,
        )

    def test_trusted_publisher_is_preferred(self) -> None:
        validated: list[str] = []

        def validator(token: str, **kwargs):
            validated.append(token)
            return self._result(kwargs["source"])

        token, selected, attempts = auth.select_credential(
            resource="spaces/SZLHOLDINGS/example",
            target_repo="SZLHOLDINGS/example",
            target_type="space",
            allow_create=True,
            environment={"HF_WRITE_TOKEN_CANDIDATE": WRITE_TOKEN},
            oidc_supplier=lambda _resource: OIDC_TOKEN,
            validator=validator,
        )

        self.assertEqual(OIDC_TOKEN, token)
        self.assertEqual("TRUSTED_PUBLISHER", selected.source)
        self.assertEqual([OIDC_TOKEN], validated)
        self.assertEqual(1, len(attempts))
        self.assertTrue(attempts[0].valid)

    def test_expired_earlier_secret_cannot_mask_valid_write_token(self) -> None:
        validated: list[tuple[str, str]] = []

        def validator(token: str, **kwargs):
            source = kwargs["source"]
            validated.append((source, token))
            if source == "HF_ORG_TOKEN":
                raise RuntimeError("expired credential")
            return self._result(source, "CREATE_OR_RECOVER_REQUIRED")

        token, selected, attempts = auth.select_credential(
            resource=None,
            target_repo="SZLHOLDINGS/example",
            target_type="space",
            allow_create=True,
            environment={
                "HF_ORG_TOKEN_CANDIDATE": ORG_TOKEN,
                "HF_WRITE_TOKEN_CANDIDATE": WRITE_TOKEN,
                "HF_TOKEN_CANDIDATE": "hf_stale_shadow",
            },
            validator=validator,
        )

        self.assertEqual(WRITE_TOKEN, token)
        self.assertEqual("HF_WRITE_TOKEN", selected.source)
        self.assertEqual(
            [("HF_ORG_TOKEN", ORG_TOKEN), ("HF_WRITE_TOKEN", WRITE_TOKEN)],
            validated,
        )
        self.assertFalse(attempts[0].valid)
        self.assertEqual("RuntimeError", attempts[0].failure_type)
        self.assertEqual("HF_ORG_TOKEN1", attempts[1].source)
        self.assertFalse(attempts[1].present)
        self.assertTrue(attempts[2].valid)

    def test_missing_candidates_are_explicit(self) -> None:
        def validator(token: str, **kwargs):
            return self._result(kwargs["source"])

        token, selected, attempts = auth.select_credential(
            resource=None,
            target_repo="SZLHOLDINGS/example",
            target_type="model",
            allow_create=False,
            environment={"HF_TOKEN_CANDIDATE": WRITE_TOKEN},
            validator=validator,
        )

        self.assertEqual(WRITE_TOKEN, token)
        self.assertEqual("HF_TOKEN", selected.source)
        by_source = {attempt.source: attempt for attempt in attempts}
        self.assertFalse(by_source["HF_ORG_TOKEN"].present)
        self.assertFalse(by_source["HF_WRITE_TOKEN"].present)
        self.assertTrue(by_source["HF_TOKEN"].valid)

    def test_report_contains_no_token_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "credential.json"
            selected = self._result("HF_WRITE_TOKEN")
            auth._write_report(
                report,
                target_repo="SZLHOLDINGS/example",
                target_type="space",
                resource="spaces/SZLHOLDINGS/example",
                selected=selected,
                attempts=[
                    auth.Attempt(
                        source="HF_WRITE_TOKEN",
                        present=True,
                        valid=True,
                        target_access=selected.target_access,
                    )
                ],
            )
            raw = report.read_text(encoding="utf-8")
            payload = json.loads(raw)

        self.assertNotIn(WRITE_TOKEN, raw)
        self.assertFalse(payload["token_persisted"])
        self.assertFalse(payload["token_logged"])
        self.assertEqual("HF_WRITE_TOKEN", payload["selected"]["source"])

    def test_github_environment_contains_only_selected_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            github_env = Path(directory) / "github.env"
            auth._append_github_environment(
                github_env,
                WRITE_TOKEN,
                "HF_WRITE_TOKEN",
            )
            text = github_env.read_text(encoding="utf-8")

        self.assertEqual(
            "HF_TOKEN=hf_write_valid\nHF_TOKEN_SOURCE=HF_WRITE_TOKEN\n",
            text,
        )
        self.assertNotIn(ORG_TOKEN, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
