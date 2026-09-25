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
        if path.endswith("/szl-kernels/branches/main"):
            return {"protected": True, "commit": {"sha": self.source}}
        if path.endswith("/szl-forge/branches/main"):
            return {"protected": True, "commit": {"sha": self.publisher}}
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
                        "id": index + 1,
                        "head_sha": self.source,
                        "app": {"id": authorization.GITHUB_ACTIONS_APP_ID},
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
        self.assertTrue(result["source"]["branch_protection_observed"])
        self.assertTrue(result["publisher"]["branch_protection_observed"])

    def test_required_check_names_match_current_kernel_workflow(self) -> None:
        self.assertEqual(
            authorization.REQUIRED_CHECKS,
            {"Kernel contract", "MiniEmbed artifact replay", "Source binding dry run (no publication)"},
        )

    def test_rejects_unprotected_or_moved_main_on_either_repository(self) -> None:
        for repository in ("szl-kernels", "szl-forge"):
            for mutation, message in (({"protected": False}, "not protected"), ({"commit": {"sha": "c" * 40}}, "changed")):
                with self.subTest(repository=repository, mutation=mutation):
                    def changed(path: str) -> dict[str, object]:
                        payload = self._get(path)
                        if path.endswith(f"/{repository}/branches/main"):
                            payload.update(mutation)
                        return payload
                    with self.assertRaisesRegex(authorization.AuthorizationError, message):
                        authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=changed)

    def test_rejects_foreign_app_wrong_revision_and_missing_check_identity(self) -> None:
        for mutation, message in (
            ({"app": {"id": 999}}, "missing"),
            ({"head_sha": "c" * 40}, "exact source revision"),
            ({"id": None}, "invalid identity"),
        ):
            with self.subTest(mutation=mutation):
                def changed(path: str) -> dict[str, object]:
                    payload = self._get(path)
                    if path.endswith("/check-runs?per_page=100"):
                        payload["check_runs"][0].update(mutation)
                    return payload
                with self.assertRaisesRegex(authorization.AuthorizationError, message):
                    authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=changed)

    def test_newer_failed_or_pending_check_cannot_hide_behind_old_success(self) -> None:
        for status, conclusion, message in (("completed", "failure", "failed"), ("in_progress", None, "pending")):
            for reverse in (False, True):
                with self.subTest(status=status, reverse=reverse):
                    def changed(path: str) -> dict[str, object]:
                        payload = self._get(path)
                        if path.endswith("/check-runs?per_page=100"):
                            runs = payload["check_runs"]
                            newer = {**runs[0], "id": 100, "status": status, "conclusion": conclusion}
                            payload["check_runs"] = [newer, *runs] if reverse else [*runs, newer]
                        return payload
                    with self.assertRaisesRegex(authorization.AuthorizationError, message):
                        authorization.authorize_once(source_revision=self.source, publisher_revision=self.publisher, getter=changed)

    def test_rejects_either_main_advancing_during_check_query(self) -> None:
        for repository in ("szl-kernels", "szl-forge"):
            with self.subTest(repository=repository):
                checked = False

                def advancing(path: str) -> dict[str, object]:
                    nonlocal checked
                    payload = self._get(path)
                    if path.endswith("/check-runs?per_page=100"):
                        checked = True
                    if checked and path.endswith(f"/{repository}/git/ref/heads/main"):
                        payload["object"]["sha"] = "c" * 40
                    if checked and path.endswith(f"/{repository}/branches/main"):
                        payload["commit"]["sha"] = "c" * 40
                    return payload

                with self.assertRaisesRegex(authorization.AuthorizationError, "changed during authorization"):
                    authorization.authorize_once(
                        source_revision=self.source, publisher_revision=self.publisher, getter=advancing,
                    )

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
