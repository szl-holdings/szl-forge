from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

import authorize_invariants_release as authorization


class AuthorizeInvariantsReleaseTests(unittest.TestCase):
    source = "a" * 40
    publisher = "b" * 40

    def _get(self, path: str) -> dict[str, object] | list[object]:
        if path.endswith("/szl-invariants/git/ref/heads/main"):
            return {"object": {"sha": self.source}}
        if path.endswith("/szl-forge/git/ref/heads/main"):
            return {"object": {"sha": self.publisher}}
        if path.endswith("/szl-invariants/branches/main"):
            return {"name": "main", "protected": True, "commit": {"sha": self.source}}
        if path.endswith("/szl-forge/branches/main"):
            return {"name": "main", "protected": True, "commit": {"sha": self.publisher}}
        if "/rules/branches/main?" in path:
            return [{"type": "required_status_checks", "parameters": {
                "required_status_checks": [{"context": "verify canonical kernel", "integration_id": None}]
            }}]
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
        self.assertIs(result["source"]["branch_protection_observed"], True)
        self.assertIs(result["publisher"]["branch_protection_observed"], True)
        self.assertEqual(result["source"]["required_check_enforcement"]["basis"], ["effective_branch_rules"])

    def test_unprotected_source_or_publisher_cannot_be_authorized(self) -> None:
        for repository in (authorization.SOURCE_REPOSITORY, authorization.PUBLISHER_REPOSITORY):
            with self.subTest(repository=repository):
                def unprotected(path: str):
                    payload = self._get(path)
                    if path == f"/repos/{repository}/branches/main":
                        payload["protected"] = False
                    return payload

                with self.assertRaisesRegex(authorization.AuthorizationError, "protection"):
                    authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=unprotected)

    def test_successful_check_without_enforced_check_policy_is_rejected(self) -> None:
        def unenforced(path: str):
            return [] if "/rules/branches/main?" in path else self._get(path)

        with self.assertRaisesRegex(authorization.AuthorizationError, "enforcement"):
            authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=unenforced)

    def test_malformed_or_non_boolean_protection_is_rejected(self) -> None:
        for bad in (None, [], {}, {"name": "main", "protected": "true"},
                    {"name": "main", "protected": 1}, {"name": "main", "protected": True, "commit": []}):
            with self.subTest(protection=bad):
                def malformed(path: str):
                    return bad if path.endswith("/szl-invariants/branches/main") else self._get(path)

                with self.assertRaises(authorization.AuthorizationError):
                    authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=malformed)

    def test_branch_sha_must_still_match_ref_when_protection_is_observed(self) -> None:
        def moved(path: str):
            payload = self._get(path)
            if path.endswith("/szl-invariants/branches/main"):
                payload["commit"]["sha"] = "c" * 40
            return payload

        with self.assertRaisesRegex(authorization.AuthorizationError, "changed"):
            authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=moved)

    def test_malformed_effective_rules_fail_closed(self) -> None:
        for bad in (None, {}, [None], [{}], [{"type": "required_status_checks"}],
                    [{"type": "required_status_checks", "parameters": {"required_status_checks": ["verify canonical kernel"]}}]):
            with self.subTest(rules=bad):
                def malformed(path: str):
                    return bad if "/rules/branches/main?" in path else self._get(path)

                with self.assertRaisesRegex(authorization.AuthorizationError, "malformed"):
                    authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=malformed)

    def test_actual_classic_required_check_protection_is_supported(self) -> None:
        def classic(path: str):
            if "/rules/branches/main?" in path:
                return []
            payload = self._get(path)
            if path.endswith("/szl-invariants/branches/main"):
                payload["protection"] = {"required_status_checks": {
                    "enforcement_level": "everyone", "contexts": ["verify canonical kernel"], "checks": []}}
            return payload

        result = authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=classic)
        self.assertEqual(result["source"]["required_check_enforcement"]["basis"], ["classic_branch_protection"])

    def test_disabled_or_malformed_classic_protection_does_not_authorize(self) -> None:
        for level in ("off", None, True, "unknown", []):
            with self.subTest(level=level):
                def classic(path: str):
                    if "/rules/branches/main?" in path:
                        return []
                    payload = self._get(path)
                    if path.endswith("/szl-invariants/branches/main"):
                        payload["protection"] = {"required_status_checks": {
                            "enforcement_level": level, "contexts": ["verify canonical kernel"], "checks": []}}
                    return payload

                with self.assertRaises(authorization.AuthorizationError):
                    authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=classic)

    def test_explicitly_disabled_classic_protection_is_not_enforcement(self) -> None:
        def disabled(path: str):
            if "/rules/branches/main?" in path:
                return []
            payload = self._get(path)
            if path.endswith("/szl-invariants/branches/main"):
                payload["protection"] = {"enabled": False, "required_status_checks": {
                    "enforcement_level": "everyone", "contexts": ["verify canonical kernel"], "checks": []}}
            return payload

        with self.assertRaisesRegex(authorization.AuthorizationError, "enforcement"):
            authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=disabled)

    def test_unavailable_protection_or_rules_fail_closed(self) -> None:
        for endpoint in ("/szl-invariants/branches/main", "/szl-forge/branches/main", "/rules/branches/main?"):
            with self.subTest(endpoint=endpoint):
                def unavailable(path: str):
                    if endpoint in path:
                        raise authorization.AuthorizationError("GitHub policy observation unavailable")
                    return self._get(path)

                with self.assertRaisesRegex(authorization.AuthorizationError, "unavailable"):
                    authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=unavailable)

    def test_rules_are_paginated_and_not_silently_truncated(self) -> None:
        observed = []

        def paginated(path: str):
            observed.append(path)
            if "/rules/branches/main?per_page=100&page=1" in path:
                return [{"type": "deletion"}] * 100
            return self._get(path)

        result = authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=paginated)
        self.assertEqual(result["status"], "AUTHORIZED_PROTECTED_MAIN")
        self.assertIn(f"/repos/{authorization.SOURCE_REPOSITORY}/rules/branches/main?per_page=100&page=2", observed)

    def test_unbounded_rules_fail_closed(self) -> None:
        def unbounded(path: str):
            return [{"type": "deletion"}] * 100 if "/rules/branches/main?" in path else self._get(path)

        with self.assertRaisesRegex(authorization.AuthorizationError, "bound"):
            authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=unbounded)

    def test_github_get_accepts_list_endpoint_but_rejects_scalar_or_invalid_json(self) -> None:
        for valid in ({"protected": True}, [{"type": "deletion"}]):
            with self.subTest(valid=valid), patch.object(authorization.urllib.request, "urlopen", return_value=io.StringIO(json.dumps(valid))):
                self.assertEqual(authorization.github_get("/fixture", None), valid)
        for invalid in ("null", "true", "1", '"text"', "{"):
            with self.subTest(invalid=invalid), patch.object(authorization.urllib.request, "urlopen", return_value=io.StringIO(invalid)):
                with self.assertRaises(authorization.AuthorizationError):
                    authorization.github_get("/fixture", None)

    def test_wait_does_not_retry_absent_branch_protection(self) -> None:
        def unprotected(path: str):
            payload = self._get(path)
            if path.endswith("/szl-invariants/branches/main"):
                payload["protected"] = False
            return payload

        with patch.object(authorization.time, "sleep") as sleeper:
            with self.assertRaisesRegex(authorization.AuthorizationError, "protection"):
                authorization.authorize_with_wait(source_revision=self.source, publisher_revision=self.publisher, wait_seconds=60, getter=unprotected)
            sleeper.assert_not_called()

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
