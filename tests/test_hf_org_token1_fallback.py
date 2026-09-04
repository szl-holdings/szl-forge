from __future__ import annotations

import unittest

from tools import acquire_hf_publisher_token as auth

PRIMARY = "hf_primary_expired"
ALTERNATE = "hf_alternate_valid"
WRITE = "hf_write_should_not_be_reached"


class AlternateOrganizationCredentialTests(unittest.TestCase):
    @staticmethod
    def result(source: str) -> auth.ValidationResult:
        return auth.ValidationResult(
            source=source,
            identity_sha256="a" * 64,
            target_access="EXISTING_WRITE_CONFIRMED",
        )

    def test_alternate_org_credential_precedes_broader_fallbacks(self) -> None:
        order = [source for source, _variable in auth.TOKEN_ENV_ORDER]
        self.assertEqual(
            ["HF_ORG_TOKEN", "HF_ORG_TOKEN1", "HF_WRITE_TOKEN"],
            order[:3],
        )

    def test_expired_primary_falls_through_to_valid_alternate(self) -> None:
        validated: list[tuple[str, str]] = []

        def validator(token: str, **kwargs):
            source = kwargs["source"]
            validated.append((source, token))
            if source == "HF_ORG_TOKEN":
                raise RuntimeError("expired primary organization credential")
            return self.result(source)

        token, selected, attempts = auth.select_credential(
            resource=None,
            target_repo="SZLHOLDINGS/szl-model-inference-lab",
            target_type="space",
            allow_create=True,
            environment={
                "HF_ORG_TOKEN_CANDIDATE": PRIMARY,
                "HF_ORG_TOKEN1_CANDIDATE": ALTERNATE,
                "HF_WRITE_TOKEN_CANDIDATE": WRITE,
            },
            validator=validator,
        )

        self.assertEqual(ALTERNATE, token)
        self.assertEqual("HF_ORG_TOKEN1", selected.source)
        self.assertEqual(
            [
                ("HF_ORG_TOKEN", PRIMARY),
                ("HF_ORG_TOKEN1", ALTERNATE),
            ],
            validated,
        )
        self.assertEqual(2, len(attempts))
        self.assertFalse(attempts[0].valid)
        self.assertTrue(attempts[1].valid)
        self.assertNotIn(WRITE, [candidate for _source, candidate in validated])


if __name__ == "__main__":
    unittest.main(verbosity=2)
