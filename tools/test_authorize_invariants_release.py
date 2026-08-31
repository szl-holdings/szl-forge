from __future__ import annotations

import unittest

import authorize_invariants_release as authorization


class AuthorizeInvariantsReleaseTests(unittest.TestCase):
    source = "a" * 40
    publisher = "b" * 40

    def _get(self, path: str) -> dict[str, object]:
        if path.endswith("/szl-invariants/git/ref/heads/main"):
            return {"object": {"sha": self.source}}
        if path.endswith("/szl-forge/git/ref/heads/main"):
            return {"object": {"sha": self.publisher}}
        if path.endswith(f"/commits/{self.source}"):
            return {
                "commit": {
                    "verification": {"verified": True, "reason": "valid"}
                }
            }
        if path.endswith("/check-runs?per_page=100"):
            return {
                "check_runs": [
                    {
                        "name": "verify canonical kernel",
                        "status": "completed",
                        "conclusion": "success",
                        "details_url": "https://example.invalid/check",
                    }
                ]
            }
        raise AssertionError(path)

    def test_authorizes_exact_verified_mains_with_terminal_check(self) -> None:
        result = authorization.authorize_once(
            source_revision=self.source,
            publisher_revision=self.publisher,
            getter=self._get,
        )
        self.assertEqual(result["status"], "AUTHORIZED_PROTECTED_MAIN")
        self.assertEqual(result["source"]["revision"], self.source)
        self.assertEqual(len(result["source"]["checks"]), 1)

    def test_rejects_non_main_source(self) -> None:
        with self.assertRaisesRegex(authorization.AuthorizationError, "not current"):
            authorization.authorize_once(
                source_revision="c" * 40,
                publisher_revision=self.publisher,
                getter=self._get,
            )

    def test_rejects_unsigned_source(self) -> None:
        def unsigned(path: str) -> dict[str, object]:
            payload = self._get(path)
            if path.endswith(f"/commits/{self.source}"):
                payload["commit"]["verification"] = {
                    "verified": False,
                    "reason": "unsigned",
                }
            return payload

        with self.assertRaisesRegex(authorization.AuthorizationError, "not verified"):
            authorization.authorize_once(
                source_revision=self.source,
                publisher_revision=self.publisher,
                getter=unsigned,
            )

    def test_rejects_missing_required_check(self) -> None:
        def missing(path: str) -> dict[str, object]:
            payload = self._get(path)
            if path.endswith("/check-runs?per_page=100"):
                payload["check_runs"] = []
            return payload

        with self.assertRaisesRegex(authorization.AuthorizationError, "missing"):
            authorization.authorize_once(
                source_revision=self.source,
                publisher_revision=self.publisher,
                getter=missing,
            )

    def test_rejects_failed_required_check(self) -> None:
        def failed(path: str) -> dict[str, object]:
            payload = self._get(path)
            if path.endswith("/check-runs?per_page=100"):
                payload["check_runs"][0]["conclusion"] = "failure"
            return payload

        with self.assertRaisesRegex(authorization.AuthorizationError, "failed"):
            authorization.authorize_once(
                source_revision=self.source,
                publisher_revision=self.publisher,
                getter=failed,
            )


if __name__ == "__main__":
    unittest.main()
