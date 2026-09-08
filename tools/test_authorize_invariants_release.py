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
        if "/check-runs?" in path:
            return {
                "check_runs": [
                    {
                        "name": "verify canonical kernel",
                        "id": 42,
                        "head_sha": self.source,
                        "app": {"id": 17},
                        "status": "completed",
                        "conclusion": "success",
                        "details_url": "https://example.invalid/check",
                    }
                ]
            }
        raise AssertionError(path)

    def _with_bindings(self, checks, runs, *, classic=False):
        def getter(path):
            if "/rules/branches/main?" in path:
                return [] if classic else [{"type": "required_status_checks", "parameters": {"required_status_checks": checks}}]
            payload = self._get(path)
            if "/check-runs?" in path:
                payload["check_runs"] = runs
            if classic and path.endswith("/szl-invariants/branches/main"):
                payload["protection"] = {"required_status_checks": {
                    "enforcement_level": "everyone", "contexts": [check["context"] for check in checks], "checks": checks}}
            return payload
        return authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=getter)

    def _run(self, *, run_id=42, app_id=17, conclusion="success", status="completed"):
        return {"id": run_id, "name": "verify canonical kernel", "app": {"id": app_id},
                "head_sha": self.source, "status": status, "conclusion": conclusion}

    def test_bound_check_cannot_be_satisfied_by_foreign_app(self) -> None:
        for classic, field in ((False, "integration_id"), (True, "app_id")):
            with self.subTest(classic=classic):
                with self.assertRaises(authorization.AuthorizationError):
                    self._with_bindings([{"context": "verify canonical kernel", field: 17}], [self._run(app_id=99)], classic=classic)

    def test_foreign_success_does_not_override_bound_failure_or_pending(self) -> None:
        for state in ("failure", "pending"):
            bound = self._run(run_id=50, conclusion="failure" if state == "failure" else None,
                              status="completed" if state == "failure" else "in_progress")
            for runs in ([bound, self._run(run_id=60, app_id=99)], [self._run(run_id=60, app_id=99), bound]):
                with self.subTest(state=state, order=[run["id"] for run in runs]):
                    with self.assertRaises(authorization.AuthorizationError):
                        self._with_bindings([{"context": "verify canonical kernel", "integration_id": 17}], runs)

    def test_newest_run_id_wins_independently_of_response_order(self) -> None:
        failed = self._run(run_id=50, conclusion="failure")
        old_success = self._run(run_id=40)
        for runs in ([failed, old_success], [old_success, failed]):
            with self.subTest(order=[run["id"] for run in runs]):
                with self.assertRaisesRegex(authorization.AuthorizationError, "failed"):
                    self._with_bindings([{"context": "verify canonical kernel", "integration_id": 17}], runs)

    def test_every_specific_binding_survives_duplicate_contexts_and_wildcard(self) -> None:
        bindings = [{"context": "verify canonical kernel", "integration_id": app} for app in (None, 17, 18)]
        with self.assertRaises(authorization.AuthorizationError):
            self._with_bindings(bindings, [self._run(app_id=18, run_id=50)])
        result = self._with_bindings(bindings, [self._run(app_id=17), self._run(app_id=18, run_id=50)])
        self.assertEqual(len(result["source"]["checks"]), 3)

    def test_specific_app_receipt_retains_bound_and_observed_identity(self) -> None:
        result = self._with_bindings([{"context": "verify canonical kernel", "integration_id": 17}], [self._run()])
        observed = result["source"]["checks"][0]
        self.assertEqual(observed["required_app_id"], 17)
        self.assertEqual(observed["app_id"], 17)
        self.assertEqual(observed["run_id"], 42)

    def test_documented_wildcard_bindings_accept_an_observed_app(self) -> None:
        for classic, check in ((False, {"context": "verify canonical kernel"}),
                               (False, {"context": "verify canonical kernel", "integration_id": None}),
                               (True, {"context": "verify canonical kernel", "app_id": -1}),
                               (True, {"context": "verify canonical kernel", "app_id": None})):
            with self.subTest(classic=classic, check=check):
                result = self._with_bindings([check], [self._run(app_id=99)], classic=classic)
                self.assertEqual(result["source"]["checks"][0]["required_app_id"], None)

    def test_malformed_policy_app_bindings_fail_closed(self) -> None:
        for classic, field, invalid in ((False, "integration_id", [True, "17", 0, -1, -2, [], {}]),
                                        (True, "app_id", [True, "17", 0, -2, [], {}])):
            for app_id in invalid:
                with self.subTest(classic=classic, app_id=app_id):
                    with self.assertRaises(authorization.AuthorizationError):
                        self._with_bindings([{"context": "verify canonical kernel", field: app_id}], [self._run()], classic=classic)

    def test_missing_or_malformed_observed_app_or_run_id_fails_closed(self) -> None:
        for field in ("app", "id"):
            for value in (None, True, "17", 0, -1, []):
                run = self._run()
                run[field] = {"id": value} if field == "app" else value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(authorization.AuthorizationError):
                        self._with_bindings([{"context": "verify canonical kernel", "integration_id": 17}], [run])

    def test_missing_classic_or_wrongly_named_policy_binding_is_rejected(self) -> None:
        for classic, check in ((True, {"context": "verify canonical kernel"}),
                               (False, {"context": "verify canonical kernel", "app_id": 17}),
                               (True, {"context": "verify canonical kernel", "integration_id": 17})):
            with self.subTest(classic=classic, check=check):
                with self.assertRaisesRegex(authorization.AuthorizationError, "binding"):
                    self._with_bindings([check], [self._run()], classic=classic)

    def test_latest_matching_success_can_replace_an_older_failure(self) -> None:
        result = self._with_bindings([{"context": "verify canonical kernel", "integration_id": 17}],
                                     [self._run(run_id=50), self._run(run_id=40, conclusion="failure")])
        self.assertEqual(result["source"]["checks"][0]["run_id"], 50)

    def test_all_enforced_contexts_are_verified_not_only_the_minimum_name(self) -> None:
        bindings = [{"context": name, "integration_id": 17} for name in ("verify canonical kernel", "secondary-check")]
        with self.assertRaisesRegex(authorization.AuthorizationError, "missing"):
            self._with_bindings(bindings, [self._run()])
        secondary = dict(self._run(run_id=50), name="secondary-check")
        result = self._with_bindings(bindings, [self._run(), secondary])
        self.assertEqual({check["name"] for check in result["source"]["checks"]}, {"verify canonical kernel", "secondary-check"})

    def test_later_check_page_can_invalidate_older_first_page_success(self) -> None:
        observed = []

        def paginated(path):
            observed.append(path)
            if "/check-runs?" in path:
                if path.endswith("page=1"):
                    return {"check_runs": [self._run(run_id=40)] + [{"name": "unrelated"}] * 99}
                return {"check_runs": [self._run(run_id=50, conclusion="failure")]}
            return self._get(path)

        with self.assertRaisesRegex(authorization.AuthorizationError, "failed"):
            authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=paginated)
        self.assertTrue(any("filter=all&page=2" in path for path in observed))

    def test_check_revision_and_duplicate_run_observations_fail_closed(self) -> None:
        for runs in ([dict(self._run(), head_sha="c" * 40)],
                     [self._run(), self._run(conclusion="failure")]):
            with self.subTest(runs=runs):
                with self.assertRaises(authorization.AuthorizationError):
                    self._with_bindings([{"context": "verify canonical kernel", "integration_id": 17}], runs)

    def test_legacy_context_only_summary_cannot_invent_any_app_permission(self) -> None:
        def ambiguous(path):
            if "/rules/branches/main?" in path:
                return []
            payload = self._get(path)
            if path.endswith("/szl-invariants/branches/main"):
                payload["protection"] = {"required_status_checks": {
                    "enforcement_level": "everyone", "contexts": ["verify canonical kernel"], "checks": []}}
            return payload

        with self.assertRaisesRegex(authorization.AuthorizationError, "binding"):
            authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=ambiguous)

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
                    "enforcement_level": "everyone", "contexts": ["verify canonical kernel"],
                    "checks": [{"context": "verify canonical kernel", "app_id": -1}]}}
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
            if "/check-runs?" in path:
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
            if "/check-runs?" in path:
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
