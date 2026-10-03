import importlib.util
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oac_publication_contract", ROOT / "publish_space.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicationContracts(unittest.TestCase):
    def test_existing_team_entitlement(self):
        self.assertFalse(
            MODULE.verify_entitlement(
                {"orgs": [{"name": "SZLHOLDINGS", "plan": "team"}]}
            )["purchase_or_upgrade"]
        )

    def test_verified_git_blob_is_returned_as_immutable_bytes(self):
        revision = "a" * 40
        blob = b"reviewed bytes"
        plan = {
            "source_dir": "spaces/oac-system-health-lab",
            "files": {
                "app.py": {
                    "source_path": "spaces/oac-system-health-lab/app.py",
                    "sha256": hashlib.sha256(blob).hexdigest(),
                }
            },
        }
        with (
            mock.patch.object(
                MODULE.subprocess,
                "check_output",
                side_effect=[revision + "\n", str(len(blob)), blob],
            ) as read,
            mock.patch.object(MODULE.subprocess, "run"),
        ):
            self.assertEqual(
                MODULE.verify_head_and_bytes(revision, plan), {"app.py": blob}
            )
        self.assertEqual(
            read.call_args_list[-1].args[0],
            ["git", "show", revision + ":spaces/oac-system-health-lab/app.py"],
        )

    def test_oversize_blob_refused_before_git_show(self):
        revision = "a" * 40
        plan = {
            "source_dir": "spaces/oac-system-health-lab",
            "files": {
                "app.py": {
                    "source_path": "spaces/oac-system-health-lab/app.py",
                    "sha256": "b" * 64,
                }
            },
        }
        with (
            mock.patch.object(
                MODULE.subprocess,
                "check_output",
                side_effect=[revision, str(2 * 1024 * 1024 + 1)],
            ) as read,
            mock.patch.object(MODULE.subprocess, "run"),
            self.assertRaises(MODULE.PublicationRefused),
        ):
            MODULE.verify_head_and_bytes(revision, plan)
        self.assertEqual(read.call_count, 2)

    def test_wrong_blob_hash_refused(self):
        revision = "a" * 40
        plan = {
            "source_dir": "spaces/oac-system-health-lab",
            "files": {
                "app.py": {
                    "source_path": "spaces/oac-system-health-lab/app.py",
                    "sha256": "b" * 64,
                }
            },
        }
        with (
            mock.patch.object(
                MODULE.subprocess, "check_output", side_effect=[revision, "1", b"x"]
            ),
            mock.patch.object(MODULE.subprocess, "run"),
            self.assertRaises(MODULE.PublicationRefused),
        ):
            MODULE.verify_head_and_bytes(revision, plan)

    def test_existing_report_refuses_before_provider_write(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "existing.json"
            destination.write_bytes(b"original evidence")
            with mock.patch.object(MODULE, "publish") as publish:
                self.assertEqual(
                    MODULE.main(
                        ["--source-revision", "a" * 40, "--report", str(destination)]
                    ),
                    1,
                )
            publish.assert_not_called()
            self.assertEqual(destination.read_bytes(), b"original evidence")

    def test_unknown_or_free_entitlement_refused(self):
        for value in (
            {},
            {"orgs": [{"name": "SZLHOLDINGS", "plan": "free"}]},
            {"orgs": [{"name": "OTHER", "plan": "enterprise"}]},
        ):
            with (
                self.subTest(value=value),
                self.assertRaises(MODULE.PublicationRefused),
            ):
                MODULE.verify_entitlement(value)

    def test_free_runtime(self):
        value = MODULE.verify_hardware(
            SimpleNamespace(
                hardware="cpu-basic", requested_hardware="cpu-basic", storage=None
            ),
            running=True,
        )
        self.assertFalse(value["hardware_upgrade"])

    def test_paid_unknown_running_and_storage_refused(self):
        for current, requested, storage in (
            ("cpu-upgrade", "cpu-upgrade", None),
            ("cpu-basic", "t4-small", None),
            (None, "cpu-basic", None),
            ("cpu-basic", "cpu-basic", "small"),
        ):
            with (
                self.subTest(current=current, requested=requested, storage=storage),
                self.assertRaises(MODULE.PublicationRefused),
            ):
                MODULE.verify_hardware(
                    SimpleNamespace(
                        hardware=current, requested_hardware=requested, storage=storage
                    ),
                    running=True,
                )

    def test_attached_volumes_refused_without_changes(self):
        for attributes in (
            {"volumes": [{"bucket": "unrelated"}]},
            {"raw": {"volumes": [{"bucket": "unrelated"}]}},
        ):
            with (
                self.subTest(attributes=attributes),
                self.assertRaises(MODULE.PublicationRefused),
            ):
                MODULE.verify_hardware(
                    SimpleNamespace(
                        hardware="cpu-basic",
                        requested_hardware="cpu-basic",
                        storage=None,
                        **attributes,
                    ),
                    running=True,
                )

    def run_publication(
        self,
        *,
        remote_files=None,
        drifting=False,
        drift_after=False,
        drift_during_runtime=False,
        hardware="cpu-basic",
        final_hardware=None,
        readme_bytes=None,
        card_validation_error=None,
        runtime_wait_error=None,
        inspection=None,
    ):
        before, after = "1" * 40, "2" * 40
        api = mock.Mock()
        info = SimpleNamespace(id=MODULE.TARGET, sdk="docker", sha=before)
        api.whoami.return_value = {"orgs": [{"name": "SZLHOLDINGS", "plan": "team"}]}
        api.space_info.side_effect = [
            info,
            SimpleNamespace(sha=after if drifting else before),
            SimpleNamespace(sha=before if drift_after else after),
            SimpleNamespace(sha=before if drift_during_runtime else after),
        ]
        api.list_repo_files.return_value = (
            remote_files if remote_files is not None else [".gitattributes"]
        )
        api.get_space_runtime.side_effect = [
            SimpleNamespace(
                hardware=observed,
                requested_hardware=observed,
                storage=None,
            )
            for observed in (hardware, hardware, final_hardware or hardware)
        ]
        api.create_commit.return_value = SimpleNamespace(oid=after)
        api.get_space_variables.return_value = {
            MODULE.SOURCE_VARIABLE: SimpleNamespace(value="3" * 40)
        }
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "README.md"
            fixture.write_bytes(
                readme_bytes
                if readme_bytes is not None
                else (
                    b"---\nshort_description: "
                    + b"x" * 60
                    + b"\n---\n\n# Synthetic fixture\n"
                )
            )
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            plan = {
                "source_dir": "spaces/oac-system-health-lab",
                "files": {
                    "README.md": {
                        "sha256": digest,
                        "source_path": "spaces/oac-system-health-lab/README.md",
                    }
                },
            }
            helper = SimpleNamespace(
                build_plan=mock.Mock(return_value=plan),
                live_origin=mock.Mock(
                    return_value="https://szlholdings-oac-system-health-lab.hf.space"
                ),
                wait_for_exact_running_space=mock.Mock(
                    return_value=SimpleNamespace(sha=after)
                ),
                wait_for_exact_runtime_source=mock.Mock(
                    return_value=(
                        {
                            "build": {"state": "OBSERVED", "revision": "3" * 40},
                            "receipt_minted": False,
                        },
                        {
                            "state": "EXACT_SOURCE_OBSERVED",
                            "shared_publication_deadline": True,
                        },
                    ),
                    side_effect=runtime_wait_error,
                ),
            )
            session = mock.Mock()
            session.headers = {}
            card = mock.Mock()
            card.validate.side_effect = card_validation_error
            repo_card = mock.Mock(return_value=card)
            client = SimpleNamespace(
                CommitOperationAdd=lambda **kwargs: kwargs,
                RepoCard=repo_card,
                __version__="1.23.0",
                hf_hub_download=mock.Mock(return_value=str(fixture)),
            )
            utils = SimpleNamespace(
                RepositoryNotFoundError=type("AbsentRepository", (Exception,), {})
            )
            release = SimpleNamespace(
                verify_release=mock.Mock(return_value={"complete": True})
            )
            modules = {
                "publish_hf_space": helper,
                "huggingface_hub": client,
                "huggingface_hub.utils": utils,
                "requests": SimpleNamespace(Session=mock.Mock(return_value=session)),
                "verify_release": release,
            }
            result = {}
            if inspection is not None:
                inspection.update(
                    {"api": api, "helper": helper, "report": result, "session": session}
                )
            with (
                mock.patch.dict(MODULE.sys.modules, modules),
                mock.patch.object(
                    MODULE,
                    "verify_head_and_bytes",
                    return_value={"README.md": fixture.read_bytes()},
                ),
                mock.patch.object(MODULE.sys, "path", list(MODULE.sys.path)),
            ):
                try:
                    MODULE.publish(api, "3" * 40, result, wait_seconds=30)
                except MODULE.PublicationRefused:
                    return api, result, False, repo_card, card
            return api, result, True, repo_card, card

    def test_atomic_parent_and_verified_success(self):
        inspection = {}
        api, report, succeeded, _, _ = self.run_publication(
            inspection=inspection
        )
        self.assertTrue(succeeded)
        self.assertEqual(api.create_commit.call_args.kwargs["parent_commit"], "1" * 40)
        self.assertEqual(api.create_commit.call_args.kwargs["repo_id"], MODULE.TARGET)
        self.assertEqual(
            api.create_commit.call_args.kwargs["operations"][0]["path_or_fileobj"],
            (
                b"---\nshort_description: "
                + b"x" * 60
                + b"\n---\n\n# Synthetic fixture\n"
            ),
        )
        self.assertEqual(report["hub_commit"], "2" * 40)
        self.assertTrue(report["complete"])
        self.assertEqual(report["files"][0]["matches"], True)
        self.assertEqual(
            report["runtime_source_wait"]["state"], "EXACT_SOURCE_OBSERVED"
        )
        hub_wait = inspection["helper"].wait_for_exact_running_space.call_args
        runtime_wait = inspection["helper"].wait_for_exact_runtime_source.call_args
        self.assertEqual(hub_wait.kwargs["deadline"], runtime_wait.kwargs["deadline"])
        self.assertEqual(runtime_wait.args[2], "3" * 40)
        self.assertIs(runtime_wait.args[0], inspection["session"])
        self.assertEqual(
            runtime_wait.args[1],
            "https://szlholdings-oac-system-health-lab.hf.space",
        )
        self.assertEqual(inspection["session"].close.call_count, 1)
        self.assertEqual(report["hardware_final"]["current"], "cpu-basic")
        api.create_repo.assert_not_called()

    def test_runtime_source_timeout_never_marks_publication_complete(self):
        inspection = {}
        with self.assertRaisesRegex(RuntimeError, "runtime deadline exhausted"):
            self.run_publication(
                runtime_wait_error=RuntimeError("runtime deadline exhausted"),
                inspection=inspection,
            )
        self.assertNotIn("complete", inspection["report"])
        self.assertNotIn("status", inspection["report"])
        self.assertEqual(inspection["report"]["files"][0]["matches"], True)
        inspection["api"].create_commit.assert_called_once()
        inspection["session"].close.assert_called_once()

    def test_valid_provider_card_preflight_preserves_existing_adoption(self):
        readme = (
            b"---\nshort_description: "
            + b"x" * 60
            + b"\n---\n\n# Synthetic fixture\n"
        )

        api, report, succeeded, repo_card, card = self.run_publication(
            readme_bytes=readme
        )

        self.assertTrue(succeeded)
        repo_card.assert_called_once_with(readme.decode("utf-8"))
        card.validate.assert_called_once_with(repo_type="space")
        self.assertEqual(report["card_validation"]["state"], "VALIDATED")
        self.assertEqual(
            report["card_validation"]["huggingface_hub_version"], "1.23.0"
        )
        self.assertEqual(
            report["card_validation"]["sha256"], hashlib.sha256(readme).hexdigest()
        )
        api.create_repo.assert_not_called()

    def test_invalid_provider_card_refused_before_repository_creation(self):
        readme = (
            b"---\nshort_description: "
            + b"x" * 61
            + b"\n---\n\n# Synthetic fixture\n"
        )

        api, _, succeeded, repo_card, card = self.run_publication(
            readme_bytes=readme,
            card_validation_error=ValueError(
                "short_description must be at most 60 characters"
            ),
        )

        self.assertFalse(succeeded)
        repo_card.assert_called_once_with(readme.decode("utf-8"))
        card.validate.assert_called_once_with(repo_type="space")
        api.whoami.assert_not_called()
        api.space_info.assert_not_called()
        api.create_repo.assert_not_called()
        api.create_commit.assert_not_called()

    def test_concurrent_drift_refused_before_write(self):
        api, _, succeeded, _, _ = self.run_publication(drifting=True)
        self.assertFalse(succeeded)
        api.create_commit.assert_not_called()
        api.add_space_variable.assert_not_called()

    def test_unexpected_files_preserved_and_refused(self):
        api, _, succeeded, _, _ = self.run_publication(
            remote_files=["someone-elses-data.json"]
        )
        self.assertFalse(succeeded)
        api.create_commit.assert_not_called()

    def test_paid_hardware_refused_before_write(self):
        api, _, succeeded, _, _ = self.run_publication(hardware="cpu-upgrade")
        self.assertFalse(succeeded)
        api.create_commit.assert_not_called()

    def test_post_publication_drift_never_complete(self):
        _, report, succeeded, _, _ = self.run_publication(drift_after=True)
        self.assertFalse(succeeded)
        self.assertNotIn("complete", report)

    def test_hub_drift_during_runtime_rollout_never_complete(self):
        inspection = {}
        _, report, succeeded, _, _ = self.run_publication(
            drift_during_runtime=True, inspection=inspection
        )
        self.assertFalse(succeeded)
        self.assertNotIn("complete", report)
        self.assertEqual(report["runtime_source_wait"]["state"], "EXACT_SOURCE_OBSERVED")
        inspection["helper"].wait_for_exact_runtime_source.assert_called_once()

    def test_paid_hardware_during_runtime_rollout_never_complete(self):
        _, report, succeeded, _, _ = self.run_publication(
            final_hardware="cpu-upgrade"
        )
        self.assertFalse(succeeded)
        self.assertNotIn("complete", report)
        self.assertNotIn("hardware_final", report)


if __name__ == "__main__":
    unittest.main()
