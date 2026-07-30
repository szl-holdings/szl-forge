from __future__ import annotations

import unittest

import authorize_kernels_release as authorization


class AuthorizeKernelsReleaseTests(unittest.TestCase):
    source = "a" * 40
    publisher = "b" * 40

    def _get(self, path: str) -> dict[str, object]:
        if path.endswith("/szl-kernels/git/ref/heads/main"):
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
                        "name": name,
                        "status": "completed",
                        "conclusion": "success",
                        "details_url": f"https://example.invalid/{index}",
                    }
                    for index, name in enumerate(
                        sorted(authorization.REQUIRED_CHECKS)
                    )
                ]
            }
        raise AssertionError(path)

    def test_authorizes_exact_verified_mains_with_terminal_checks(self) -> None:
        result = authorization.authorize_once(
            source_revision=self.source,
            publisher_revision=self.publisher,
            getter=self._get,
        )
        self.assertEqual(result["status"], "AUTHORIZED_PROTECTED_MAIN")
        self.assertEqual(result["source"]["revision"], self.source)
        self.assertEqual(len(result["source"]["checks"]), 3)

    def test_rejects_non_main_source(self) -> None:
        with self.assertRaisesRegex(
            authorization.AuthorizationError,
            "not current protected main",
        ):
            authorization.authorize_once(
                source_revision="c" * 40,
                publisher_revision=self.publisher,
                getter=self._get,
            )

    def test_rejects_missing_required_check(self) -> None:
        def missing(path: str) -> dict[str, object]:
            payload = self._get(path)
            if path.endswith("/check-runs?per_page=100"):
                payload["check_runs"] = list(payload["check_runs"])[1:]
            return payload

        with self.assertRaisesRegex(
            authorization.AuthorizationError,
            "missing",
        ):
            authorization.authorize_once(
                source_revision=self.source,
                publisher_revision=self.publisher,
                getter=missing,
            )


if __name__ == "__main__":
    unittest.main()
