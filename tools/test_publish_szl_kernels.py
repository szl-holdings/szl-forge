from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import publish_szl_kernels as publisher
import verify_szl_kernel_runtime as runtime_verifier


class FakeApi:
    model_revision = "d" * 40
    kernel_revision = "e" * 40

    def __init__(self, artifacts: dict[str, Path]) -> None:
        self.files = list(artifacts)
        self.commits: list[dict[str, object]] = []
        self.kernel_revisions = {
            branch: self.kernel_revision for branch in publisher.KERNEL_BRANCHES
        }
        self.kernel_branch_files = {
            branch: set(publisher.KERNEL_EXISTING_REQUIRED_FILES)
            for branch in publisher.KERNEL_BRANCHES
        }
        self.remote: dict[tuple[str, str], dict[str, bytes]] = {
            (publisher.LEGACY_REPO_TYPE, self.model_revision): {
                relative: path.read_bytes() for relative, path in artifacts.items()
            },
            (publisher.KERNEL_REPO_TYPE, self.kernel_revision): {
                kernel_path: b"previous"
                for kernel_path in publisher.FIRST_CLASS_KERNEL_FILES.values()
            },
        }

    def model_info(
        self,
        repo_id: str,
        files_metadata: bool = False,
        token: str | None = None,
    ) -> SimpleNamespace:
        del repo_id, files_metadata, token
        return SimpleNamespace(
            sha=self.model_revision,
            siblings=[SimpleNamespace(rfilename=path) for path in self.files],
        )

    def repo_info(
        self,
        repo_id: str,
        *,
        repo_type: str,
        revision: str | None = None,
        **_: object,
    ) -> SimpleNamespace:
        del repo_id
        self._assert_kernel(repo_type)
        return SimpleNamespace(
            sha=self.kernel_revisions[revision or publisher.KERNEL_BRANCHES[0]]
        )

    def list_repo_refs(
        self,
        repo_id: str,
        *,
        repo_type: str,
        **_: object,
    ) -> SimpleNamespace:
        del repo_id
        self._assert_kernel(repo_type)
        return SimpleNamespace(
            branches=[
                SimpleNamespace(
                    name=branch,
                    target_commit=self.kernel_revisions[branch],
                )
                for branch in publisher.KERNEL_BRANCHES
            ]
        )

    def list_repo_tree(
        self,
        repo_id: str,
        *,
        repo_type: str,
        revision: str,
        **_: object,
    ) -> list[SimpleNamespace]:
        del repo_id
        self._assert_kernel(repo_type)
        branch = next(
            branch
            for branch, target in self.kernel_revisions.items()
            if target == revision
        )
        return [
            SimpleNamespace(path=path)
            for path in self.kernel_branch_files[branch]
        ]

    def create_commit(
        self,
        repo_id: str,
        operations: list[object],
        *,
        repo_type: str,
        revision: str | None = None,
        parent_commit: str,
        **_: object,
    ) -> SimpleNamespace:
        del repo_id
        self.assert_parent(repo_type, parent_commit)
        self.commits.append({"repo_type": repo_type, "revision": revision})
        oid = f"{len(self.commits)}" * 40
        remote = dict(self.remote[(repo_type, parent_commit)])
        for operation in operations:
            source = operation.path_or_fileobj
            if isinstance(source, (str, Path)):
                payload = Path(source).read_bytes()
            elif isinstance(source, io.BytesIO):
                payload = source.getvalue()
            else:  # pragma: no cover - the publisher only uses paths and BytesIO
                raise AssertionError(type(source))
            remote[operation.path_in_repo] = payload
        self.remote[(repo_type, oid)] = remote
        return SimpleNamespace(oid=oid)

    def upload_kernel(self, staging_root: Path, token: str) -> None:
        if token != "test-token":
            raise AssertionError(token)
        main_revision = "1" * 40
        version_revision = "2" * 40
        self.remote[(publisher.KERNEL_REPO_TYPE, main_revision)] = {
            "README.md": (staging_root / "build/CARD.md").read_bytes(),
        }
        version_remote = {}
        for path in (staging_root / "build" / publisher.KERNEL_VARIANT).rglob("*"):
            if path.is_file():
                relative = path.relative_to(staging_root).as_posix()
                version_remote[relative] = path.read_bytes()
        self.remote[(publisher.KERNEL_REPO_TYPE, version_revision)] = version_remote
        self.kernel_revisions = {
            "main": main_revision,
            "v1": version_revision,
        }
        self.kernel_branch_files = {
            "main": set(self.remote[(publisher.KERNEL_REPO_TYPE, main_revision)]),
            "v1": set(self.remote[(publisher.KERNEL_REPO_TYPE, version_revision)]),
        }
        self.commits.extend(
            [
                {"repo_type": "kernel", "revision": "main"},
                {"repo_type": "kernel", "revision": "v1"},
            ]
        )

    def download(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str,
        revision: str,
        **_: object,
    ) -> str:
        if repo_id != publisher.EXPECTED_REPO_ID:
            raise AssertionError(repo_id)
        payload = self.remote[(repo_type, revision)][filename]
        destination = self.download_root / repo_type / revision / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return str(destination)

    def assert_parent(self, repo_type: str, parent_commit: str) -> None:
        expected = (
            self.kernel_revision
            if repo_type == publisher.KERNEL_REPO_TYPE
            else self.model_revision
        )
        if parent_commit != expected:
            raise AssertionError((repo_type, parent_commit, expected))

    @staticmethod
    def _assert_kernel(repo_type: str) -> None:
        if repo_type != publisher.KERNEL_REPO_TYPE:
            raise AssertionError(repo_type)


class PublishSzlKernelsTests(unittest.TestCase):
    def test_kernel_parent_revalidation_rejects_branch_drift(self) -> None:
        api = FakeApi({})
        observed = {
            branch: {"revision": api.kernel_revision}
            for branch in publisher.KERNEL_BRANCHES
        }
        api.kernel_revisions["v1"] = "f" * 40

        with self.assertRaisesRegex(
            publisher.PublicationError,
            "branch parents changed before upload",
        ):
            publisher.revalidate_kernel_branch_parents(
                api,
                observed,
                token="test-token",
            )

    def test_isolated_runtime_timeout_forces_container_cleanup(self) -> None:
        container_id = "d" * 64
        operations: list[str] = []

        def run_docker(command: list[str], **kwargs: object) -> SimpleNamespace:
            operation = command[1]
            operations.append(operation)
            if operation == "create":
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "start":
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "wait":
                self.assertEqual(
                    kwargs["timeout"],
                    publisher.KERNEL_RUNTIME_TIMEOUT_SECONDS,
                )
                raise publisher.subprocess.TimeoutExpired(
                    command,
                    publisher.KERNEL_RUNTIME_TIMEOUT_SECONDS,
                )
            if operation == "rm":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(command)

        with patch.object(
            publisher.subprocess,
            "run",
            side_effect=run_docker,
        ):
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "timed out after 300 seconds",
            ):
                publisher.verify_stable_kernel_runtime_isolated(
                    revision="2" * 40
                )

        self.assertEqual(operations, ["create", "start", "wait", "rm"])

    def test_isolated_runtime_cleanup_timeout_fails_after_valid_evidence(self) -> None:
        revision = "2" * 40
        container_id = "c" * 64
        evidence = {
            "status": "STABLE_GET_KERNEL_VERIFIED",
            "client_version": publisher.KERNEL_RUNTIME_CLIENT_VERSION,
            "revision": revision,
            "package_version": publisher.EXPECTED_KERNEL_PACKAGE_VERSION,
            "selfcheck_ok": True,
            "invalid_thresholds_rejected_before_receipt": 4,
            "inclusive_boundaries": {
                "0": {"passed": True, "receipt_depth": 1},
                "1": {"passed": False, "receipt_depth": 1},
            },
        }

        def run_docker(command: list[str], **kwargs: object) -> SimpleNamespace:
            operation = command[1]
            if operation in {"create", "start"}:
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "wait":
                return SimpleNamespace(returncode=0, stdout="0\n", stderr="")
            if operation == "inspect":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"ExitCode": 0, "OOMKilled": False}),
                    stderr="",
                )
            if operation == "cp":
                Path(command[-1]).write_text(json.dumps(evidence), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if operation == "rm":
                raise publisher.subprocess.TimeoutExpired(
                    command,
                    kwargs["timeout"],
                )
            raise AssertionError(command)

        with patch.object(publisher.subprocess, "run", side_effect=run_docker):
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "isolated stable Kernel runtime cleanup timed out",
            ):
                publisher.verify_stable_kernel_runtime_isolated(revision=revision)

    def test_isolated_runtime_scrubs_credentials_and_validates_evidence(self) -> None:
        revision = "2" * 40
        evidence = {
            "status": "STABLE_GET_KERNEL_VERIFIED",
            "client_version": "0.16.0",
            "revision": revision,
            "package_version": "0.1.1",
            "selfcheck_ok": True,
            "invalid_thresholds_rejected_before_receipt": 4,
            "inclusive_boundaries": {
                "0": {"passed": True, "receipt_depth": 1},
                "1": {"passed": False, "receipt_depth": 1},
            },
        }
        container_id = "c" * 64

        def run_docker(command: list[str], **_: object) -> SimpleNamespace:
            operation = command[1]
            if operation == "create":
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "start":
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "wait":
                return SimpleNamespace(returncode=0, stdout="0\n", stderr="")
            if operation == "inspect":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"ExitCode": 0, "OOMKilled": False}),
                    stderr="",
                )
            if operation == "cp":
                Path(command[-1]).write_text(json.dumps(evidence), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if operation == "rm":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(command)
        inherited = {
            "PATH": "trusted-path",
            "HF_TOKEN": "publisher-secret",
            "GITHUB_TOKEN": "github-secret",
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "oidc-secret",
            "SERVICE_API_KEY": "api-secret",
        }
        with patch.dict(publisher.os.environ, inherited, clear=True), patch.object(
            publisher.subprocess,
            "run",
            side_effect=run_docker,
        ) as run:
            observed = publisher.verify_stable_kernel_runtime_isolated(
                revision=revision
            )

        self.assertEqual(observed, evidence)
        create_call = next(
            call for call in run.call_args_list if call.args[0][1] == "create"
        )
        command = create_call.args[0]
        environment = create_call.kwargs["env"]
        self.assertEqual(command[0:2], ["docker", "create"])
        self.assertEqual(
            command[-5:],
            [
                publisher.KERNEL_RUNTIME_IMAGE,
                "--revision",
                revision,
                "--output",
                publisher.KERNEL_RUNTIME_EVIDENCE_PATH,
            ],
        )
        self.assertIn("--log-driver=json-file", command)
        self.assertIn("--log-opt=max-size=64k", command)
        self.assertIn("--log-opt=max-file=1", command)
        self.assertFalse(
            any(argument == "--pid" or argument.startswith("--pid=") for argument in command),
            "Docker's omitted PID option must preserve the default private namespace",
        )
        self.assertIn("--read-only", command)
        self.assertIn("--cap-drop=ALL", command)
        self.assertIn("--security-opt=no-new-privileges", command)
        self.assertFalse(
            any(
                argument == "--mount"
                or argument == "--volume"
                or argument.startswith("--mount=")
                or argument.startswith("--volume=")
                or argument.startswith("-v")
                for argument in command
            )
        )
        self.assertEqual(environment["PATH"], "trusted-path")
        self.assertIn("--env=HF_HUB_DISABLE_IMPLICIT_TOKEN=1", command)
        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("GITHUB_TOKEN", environment)
        self.assertNotIn("ACTIONS_ID_TOKEN_REQUEST_TOKEN", environment)
        self.assertNotIn("SERVICE_API_KEY", environment)
        operations = [call.args[0][1] for call in run.call_args_list]
        self.assertEqual(
            operations,
            ["create", "start", "wait", "inspect", "cp", "rm"],
        )
        for call in run.call_args_list:
            operation = call.args[0][1]
            expected_timeout = (
                publisher.KERNEL_RUNTIME_TIMEOUT_SECONDS
                if operation == "wait"
                else publisher.KERNEL_RUNTIME_CONTROL_TIMEOUT_SECONDS
            )
            self.assertEqual(call.kwargs.get("timeout"), expected_timeout)
        start_command = run.call_args_list[1].args[0]
        self.assertNotIn("--attach", start_command)

    def test_isolated_runtime_preserves_bounded_failure_evidence(self) -> None:
        container_id = "c" * 64
        failure = {
            "status": "FAILED",
            "error_type": "RuntimeError",
            "error": "stable load failed",
        }

        def run_docker(command: list[str], **_: object) -> SimpleNamespace:
            operation = command[1]
            if operation in {"create", "start"}:
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "wait":
                return SimpleNamespace(returncode=0, stdout="1\n", stderr="")
            if operation == "inspect":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"ExitCode": 1, "OOMKilled": False}),
                    stderr="",
                )
            if operation == "cp":
                Path(command[-1]).write_text(json.dumps(failure), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if operation == "rm":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(command)

        with patch.object(publisher.subprocess, "run", side_effect=run_docker) as run:
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "isolated stable Kernel runtime failed: RuntimeError: stable load failed",
            ):
                publisher.verify_stable_kernel_runtime_isolated(revision="2" * 40)

        operations = [call.args[0][1] for call in run.call_args_list]
        self.assertEqual(
            operations,
            ["create", "start", "wait", "inspect", "cp", "rm"],
        )
        create_command = run.call_args_list[0].args[0]
        self.assertNotIn("/output", create_command)
        self.assertIn("/tmp:rw,nosuid,nodev,noexec,size=64m", create_command)
        self.assertIn(publisher.KERNEL_RUNTIME_EVIDENCE_PATH, create_command)

    def test_isolated_runtime_recovers_success_evidence_from_bounded_logs(self) -> None:
        revision = "2" * 40
        container_id = "c" * 64
        evidence = {
            "status": "STABLE_GET_KERNEL_VERIFIED",
            "client_version": publisher.KERNEL_RUNTIME_CLIENT_VERSION,
            "revision": revision,
            "package_version": publisher.EXPECTED_KERNEL_PACKAGE_VERSION,
            "selfcheck_ok": True,
            "invalid_thresholds_rejected_before_receipt": 4,
            "inclusive_boundaries": {
                "0": {"passed": True, "receipt_depth": 1},
                "1": {"passed": False, "receipt_depth": 1},
            },
        }

        def run_docker(command: list[str], **_: object) -> SimpleNamespace:
            operation = command[1]
            if operation in {"create", "start"}:
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "wait":
                return SimpleNamespace(returncode=0, stdout="0\n", stderr="")
            if operation == "inspect":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"ExitCode": 0, "OOMKilled": False}),
                    stderr="",
                )
            if operation == "cp":
                return SimpleNamespace(returncode=1, stdout="", stderr="missing")
            if operation == "logs":
                payload = json.dumps(evidence, separators=(",", ":"), sort_keys=True)
                return SimpleNamespace(
                    returncode=0,
                    stdout=f"{publisher.KERNEL_RUNTIME_LOG_PREFIX}{payload}\n",
                    stderr="",
                )
            if operation == "rm":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(command)

        with patch.object(publisher.subprocess, "run", side_effect=run_docker) as run:
            observed = publisher.verify_stable_kernel_runtime_isolated(
                revision=revision
            )

        self.assertEqual(observed, evidence)
        for call in run.call_args_list:
            operation = call.args[0][1]
            expected_timeout = (
                publisher.KERNEL_RUNTIME_TIMEOUT_SECONDS
                if operation == "wait"
                else publisher.KERNEL_RUNTIME_CONTROL_TIMEOUT_SECONDS
            )
            self.assertEqual(call.kwargs.get("timeout"), expected_timeout)

    def test_isolated_runtime_reports_missing_evidence_with_exit_state(self) -> None:
        container_id = "c" * 64

        def run_docker(command: list[str], **_: object) -> SimpleNamespace:
            operation = command[1]
            if operation in {"create", "start"}:
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "wait":
                return SimpleNamespace(returncode=0, stdout="137\n", stderr="")
            if operation == "inspect":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"ExitCode": 137, "OOMKilled": False}),
                    stderr="",
                )
            if operation == "cp":
                return SimpleNamespace(returncode=1, stdout="", stderr="missing")
            if operation == "logs":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if operation == "rm":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(command)

        with patch.object(publisher.subprocess, "run", side_effect=run_docker):
            with self.assertRaisesRegex(
                publisher.PublicationError,
                r"exited without evidence \(exit_code=137, oom_killed=false\)",
            ):
                publisher.verify_stable_kernel_runtime_isolated(revision="2" * 40)

    def test_isolated_runtime_bounds_copy_and_log_timeouts(self) -> None:
        container_id = "c" * 64

        def run_docker(command: list[str], **kwargs: object) -> SimpleNamespace:
            operation = command[1]
            if operation in {"create", "start"}:
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "wait":
                return SimpleNamespace(returncode=0, stdout="137\n", stderr="")
            if operation == "inspect":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"ExitCode": 137, "OOMKilled": False}),
                    stderr="",
                )
            if operation in {"cp", "logs"}:
                raise publisher.subprocess.TimeoutExpired(
                    command,
                    kwargs["timeout"],
                )
            if operation == "rm":
                raise publisher.subprocess.TimeoutExpired(
                    command,
                    kwargs["timeout"],
                )
            raise AssertionError(command)

        with patch.object(publisher.subprocess, "run", side_effect=run_docker) as run:
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "evidence copy timed out; bounded logs: logs timed out",
            ):
                publisher.verify_stable_kernel_runtime_isolated(revision="2" * 40)

        operations = [call.args[0][1] for call in run.call_args_list]
        self.assertEqual(
            operations,
            ["create", "start", "wait", "inspect", "cp", "logs", "rm"],
        )

    def test_isolated_runtime_rejects_oom_even_with_zero_exit(self) -> None:
        container_id = "c" * 64

        def run_docker(command: list[str], **_: object) -> SimpleNamespace:
            operation = command[1]
            if operation in {"create", "start"}:
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "wait":
                return SimpleNamespace(returncode=0, stdout="0\n", stderr="")
            if operation == "inspect":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"ExitCode": 0, "OOMKilled": True}),
                    stderr="",
                )
            if operation == "rm":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(command)

        with patch.object(publisher.subprocess, "run", side_effect=run_docker) as run:
            with self.assertRaisesRegex(
                publisher.PublicationError,
                r"was OOM-killed \(exit_code=0, oom_killed=true\)",
            ):
                publisher.verify_stable_kernel_runtime_isolated(revision="2" * 40)

        operations = [call.args[0][1] for call in run.call_args_list]
        self.assertEqual(operations, ["create", "start", "wait", "inspect", "rm"])

    def test_isolated_runtime_rejects_malformed_evidence_with_exit_state(self) -> None:
        container_id = "c" * 64

        def run_docker(command: list[str], **_: object) -> SimpleNamespace:
            operation = command[1]
            if operation in {"create", "start"}:
                return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
            if operation == "wait":
                return SimpleNamespace(returncode=0, stdout="1\n", stderr="")
            if operation == "inspect":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"ExitCode": 1, "OOMKilled": False}),
                    stderr="",
                )
            if operation == "cp":
                Path(command[-1]).write_text("{", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if operation == "rm":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            raise AssertionError(command)

        with patch.object(publisher.subprocess, "run", side_effect=run_docker):
            with self.assertRaisesRegex(
                publisher.PublicationError,
                r"returned malformed evidence \(exit_code=1, oom_killed=false\)",
            ):
                publisher.verify_stable_kernel_runtime_isolated(revision="2" * 40)

    def test_runtime_verifier_writes_sanitized_bounded_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence.json"
            detail = "unsafe\n\x1b[31m" + ("x" * 2500)
            with patch.object(
                runtime_verifier,
                "verify_stable_kernel_runtime",
                side_effect=ValueError(detail),
            ):
                result = runtime_verifier.main(
                    ["--revision", "2" * 40, "--output", str(output)]
                )

            evidence = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, 1)
        self.assertEqual(evidence["status"], "FAILED")
        self.assertEqual(evidence["error_type"], "ValueError")
        self.assertEqual(len(evidence["error"]), 2000)
        self.assertTrue(all(32 <= ord(character) <= 126 for character in evidence["error"]))

    def test_runtime_verifier_records_publisher_bootstrap_base_exception(self) -> None:
        class BootstrapAbort(BaseException):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence.json"
            with patch.object(
                runtime_verifier,
                "verify_stable_kernel_runtime",
                side_effect=BootstrapAbort("publisher bootstrap aborted"),
            ):
                result = runtime_verifier.main(
                    ["--revision", "2" * 40, "--output", str(output)]
                )

            evidence = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, 1)
        self.assertEqual(evidence["status"], "FAILED")
        self.assertEqual(evidence["error_type"], "BootstrapAbort")
        self.assertEqual(evidence["error"], "publisher bootstrap aborted")

    def test_runtime_verifier_survives_unprintable_exception(self) -> None:
        class UnprintableError(Exception):
            def __str__(self) -> str:
                raise RuntimeError("formatter failed")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence.json"
            with patch.object(
                runtime_verifier,
                "verify_stable_kernel_runtime",
                side_effect=UnprintableError(),
            ):
                result = runtime_verifier.main(
                    ["--revision", "2" * 40, "--output", str(output)]
                )

            evidence = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, 1)
        self.assertEqual(evidence["error_type"], "UnprintableError")
        self.assertEqual(evidence["error"], "<unprintable>")

    def test_runtime_verifier_survives_hostile_string_iteration(self) -> None:
        class HostileString(str):
            def __iter__(self):
                raise SystemExit(9)

        class HostileError(Exception):
            def __str__(self) -> str:
                return HostileString("unsafe")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence.json"
            with patch.object(
                runtime_verifier,
                "verify_stable_kernel_runtime",
                side_effect=HostileError(),
            ):
                result = runtime_verifier.main(
                    ["--revision", "2" * 40, "--output", str(output)]
                )

            evidence = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, 1)
        self.assertEqual(evidence["error_type"], "HostileError")
        self.assertEqual(evidence["error"], "<unprintable>")

    def test_runtime_verifier_records_unprintable_system_exit(self) -> None:
        class UnprintableExitCode:
            def __str__(self) -> str:
                raise RuntimeError("formatter failed")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence.json"
            with patch.object(
                runtime_verifier,
                "verify_stable_kernel_runtime",
                side_effect=SystemExit(UnprintableExitCode()),
            ):
                result = runtime_verifier.main(
                    ["--revision", "2" * 40, "--output", str(output)]
                )

            evidence = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, 1)
        self.assertEqual(evidence["error_type"], "SystemExit")
        self.assertEqual(evidence["error"], "<unprintable>")

    def test_runtime_verifier_sanitizes_error_type_grammar(self) -> None:
        OddError = type("9Odd-Name", (Exception,), {})

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence.json"
            with patch.object(
                runtime_verifier,
                "verify_stable_kernel_runtime",
                side_effect=OddError("boom"),
            ):
                result = runtime_verifier.main(
                    ["--revision", "2" * 40, "--output", str(output)]
                )

            evidence = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, 1)
        self.assertEqual(evidence["error_type"], "_9Odd_Name")
        self.assertTrue(all(32 <= ord(character) <= 126 for character in evidence["error_type"]))

    def test_kernel_runtime_image_pins_canonical_numpy(self) -> None:
        dockerfile = Path(__file__).with_name("kernel-runtime.Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn('"numpy==2.3.5"', dockerfile)

    def test_first_class_before_accepts_split_builder_layout(self) -> None:
        api = FakeApi({})
        api.kernel_revisions = {
            "main": "1" * 40,
            "v1": "2" * 40,
        }
        api.kernel_branch_files = {
            branch: set(paths)
            for branch, paths in publisher.KERNEL_REQUIRED_FILES_BY_BRANCH.items()
        }

        evidence = publisher.first_class_kernel_before(
            api,
            {"artifact_files": sorted(publisher.FIRST_CLASS_KERNEL_FILES)},
            token=None,
        )

        self.assertEqual(evidence["branches"]["main"]["revision"], "1" * 40)
        self.assertEqual(evidence["branches"]["v1"]["revision"], "2" * 40)

    def test_first_class_before_rejects_incomplete_split_branch(self) -> None:
        api = FakeApi({})
        api.kernel_revisions = {
            "main": "1" * 40,
            "v1": "2" * 40,
        }
        api.kernel_branch_files = {
            branch: set(paths)
            for branch, paths in publisher.KERNEL_REQUIRED_FILES_BY_BRANCH.items()
        }
        api.kernel_branch_files["v1"].remove(
            f"build/{publisher.KERNEL_VARIANT}/metadata.json"
        )

        with self.assertRaisesRegex(
            publisher.PublicationError,
            "first-class Kernel v1 is missing package files",
        ):
            publisher.first_class_kernel_before(
                api,
                {"artifact_files": sorted(publisher.FIRST_CLASS_KERNEL_FILES)},
                token=None,
            )

    def test_stable_runtime_verifies_exact_load_and_threshold_contract(self) -> None:
        class Chain:
            def __init__(self) -> None:
                self.depth = 0

            def verify(self) -> tuple[bool, int, int]:
                return True, self.depth, -1

        class Module:
            UnifiedReceiptChain = Chain

            @staticmethod
            def selfcheck() -> dict[str, object]:
                return {"ok": True, "version": "0.1.1"}

            @staticmethod
            def governed_lambda_gate(
                chain: Chain,
                axes: list[float],
                *,
                threshold: float,
            ) -> dict[str, object]:
                del axes
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError("invalid threshold")
                chain.depth += 1
                return {
                    "threshold": threshold,
                    "passed": 0.5 >= threshold,
                }

        def get_kernel(repo_id: str, **kwargs: object) -> Module:
            self.assertEqual(repo_id, publisher.EXPECTED_REPO_ID)
            self.assertEqual(kwargs["revision"], "2" * 40)
            self.assertEqual(kwargs["backend"], "cpu")
            self.assertIs(kwargs["trust_remote_code"], True)
            return Module()

        evidence = publisher.verify_stable_kernel_runtime(
            revision="2" * 40,
            get_kernel_fn=get_kernel,
            tensor_fn=lambda values: values,
            client_version="0.16.0",
        )
        self.assertEqual(evidence["status"], "STABLE_GET_KERNEL_VERIFIED")
        self.assertEqual(
            evidence["invalid_thresholds_rejected_before_receipt"],
            4,
        )
        self.assertEqual(evidence["inclusive_boundaries"]["0"]["receipt_depth"], 1)
        self.assertEqual(evidence["inclusive_boundaries"]["1"]["receipt_depth"], 1)
        self.assertIs(evidence["inclusive_boundaries"]["0"]["passed"], True)
        self.assertIs(evidence["inclusive_boundaries"]["1"]["passed"], False)

    def test_stable_runtime_rejects_inverted_boundary_decision(self) -> None:
        class Chain:
            def __init__(self) -> None:
                self.depth = 0

            def verify(self) -> tuple[bool, int, int]:
                return True, self.depth, -1

        class Module:
            UnifiedReceiptChain = Chain

            @staticmethod
            def selfcheck() -> dict[str, object]:
                return {"ok": True, "version": "0.1.1"}

            @staticmethod
            def governed_lambda_gate(
                chain: Chain,
                axes: list[float],
                *,
                threshold: float,
            ) -> dict[str, object]:
                del axes
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError("invalid threshold")
                chain.depth += 1
                return {
                    "threshold": threshold,
                    "passed": threshold == 1.0,
                }

        with self.assertRaisesRegex(
            publisher.PublicationError,
            "inclusive threshold boundary contract failed",
        ):
            publisher.verify_stable_kernel_runtime(
                revision="2" * 40,
                get_kernel_fn=lambda *_args, **_kwargs: Module(),
                tensor_fn=lambda values: values,
                client_version="0.16.0",
            )

    def test_supported_builder_uses_official_package_version_identity(self) -> None:
        version = SimpleNamespace(
            returncode=0,
            stdout="hf-kernel-builder 0.17.0-dev0\n",
            stderr="",
        )
        with patch.object(
            publisher.shutil,
            "which",
            return_value="/trusted/kernel-builder",
        ), patch.object(publisher.subprocess, "run", return_value=version) as run:
            executable = publisher.require_kernel_builder_executable()

        self.assertEqual(executable, "/trusted/kernel-builder")
        run.assert_called_once_with(
            ["/trusted/kernel-builder", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_supported_builder_rejects_executable_name_as_identity(self) -> None:
        version = SimpleNamespace(
            returncode=0,
            stdout="kernel-builder 0.17.0-dev0\n",
            stderr="",
        )
        with patch.object(
            publisher.shutil,
            "which",
            return_value="/trusted/kernel-builder",
        ), patch.object(publisher.subprocess, "run", return_value=version):
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "hf-kernel-builder 0.17.0-dev0",
            ):
                publisher.require_kernel_builder_executable()

    def test_gateway_installs_hub_client_before_authorization_tests(self) -> None:
        workflow = (
            Path(__file__).parents[1]
            / ".github"
            / "workflows"
            / "publish-szl-kernels.yml"
        ).read_text(encoding="utf-8")
        dockerfile = (
            Path(__file__).parent / "kernel-runtime.Dockerfile"
        ).read_text(encoding="utf-8")
        install = workflow.index("Install trusted gateway test dependency")
        tests = workflow.index("Test trusted gateway contracts")
        dependency = workflow.index('"huggingface-hub==1.26.0"', install)
        uploader = workflow.index(
            "Install exact publication client without publisher secret"
        )
        upstream_pin = workflow.index(
            "633246310320d85def0c67d62c7912fd444a842f",
            uploader,
        )
        publish = workflow.index(
            "Publish declared data with trusted code and verify exact readback"
        )
        sandbox = workflow.index(
            "Build credentialless stable runtime sandbox",
            uploader,
        )
        self.assertLess(install, dependency)
        self.assertLess(dependency, tests)
        self.assertLess(uploader, upstream_pin)
        self.assertLess(upstream_pin, publish)
        self.assertLess(sandbox, publish)
        self.assertIn('"torch==2.9.1"', dockerfile)
        self.assertIn('"kernels==0.16.0"', dockerfile)
        self.assertIn(
            "sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7",
            dockerfile,
        )
        self.assertIn(
            "--file tools/kernel-runtime.Dockerfile",
            workflow[sandbox:publish],
        )
        self.assertIn(
            'test "$(kernel-builder --version)" = '
            f'"{publisher.KERNEL_BUILDER_VERSION_OUTPUT}"',
            workflow[uploader:publish],
        )

    source_revision = "a" * 40
    publisher_revision = "b" * 40

    def _fixture(self, root: Path) -> tuple[Path, dict[str, Path]]:
        (root / "publishing").mkdir()
        artifacts = {
            ".gitattributes": root / ".gitattributes",
            "LICENSE": root / "LICENSE",
            "README.md": root / "README.md",
            "build/torch-universal/szl_kernels/__init__.py": (
                root / "build/torch-universal/szl_kernels/__init__.py"
            ),
            "build/torch-universal/szl_kernels/_chain.py": (
                root / "build/torch-universal/szl_kernels/_chain.py"
            ),
            "build/torch-universal/szl_kernels/_ops.py": (
                root / "build/torch-universal/szl_kernels/_ops.py"
            ),
            "build/torch-universal/szl_kernels/metadata.json": (
                root / "build/torch-universal/szl_kernels/metadata.json"
            ),
            "vectors.npz": root / "vectors.npz",
        }
        for path in artifacts.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        artifacts[".gitattributes"].write_text("*.bin lfs\n", encoding="utf-8")
        artifacts["LICENSE"].write_text("Apache-2.0\n", encoding="utf-8")
        artifacts["README.md"].write_text("kernel\n", encoding="utf-8")
        artifacts["build/torch-universal/szl_kernels/__init__.py"].write_text(
            '__version__ = "0.1.1"\n', encoding="utf-8"
        )
        artifacts["build/torch-universal/szl_kernels/_chain.py"].write_text(
            "GENESIS = '0' * 64\n", encoding="utf-8"
        )
        artifacts["build/torch-universal/szl_kernels/_ops.py"].write_text(
            "def op(): return True\n", encoding="utf-8"
        )
        artifacts["build/torch-universal/szl_kernels/metadata.json"].write_text(
            json.dumps({"name": "szl_kernels", "version": "0.1.1"}),
            encoding="utf-8",
        )
        artifacts["vectors.npz"].write_bytes(b"weights")
        contract = {
            "schema": "szl.kernel-source-binding/v1",
            "repo_id": publisher.EXPECTED_REPO_ID,
            "source_repository": publisher.EXPECTED_SOURCE_REPOSITORY,
            "artifact_files": list(artifacts),
            "expected_artifact_sha256": {
                "vectors.npz": hashlib.sha256(b"weights").hexdigest()
            },
            "claims": {"scope": "test"},
            "limitations": ["test fixture"],
        }
        (root / publisher.CONTRACT_RELATIVE).write_text(
            json.dumps(contract),
            encoding="utf-8",
        )
        authorization = root / "authorization.json"
        authorization.write_text(
            json.dumps(
                {
                    "schema": "szl.kernels-release-authorization/v1",
                    "status": "AUTHORIZED_PROTECTED_MAIN",
                    "source": {
                        "repository": publisher.EXPECTED_SOURCE_REPOSITORY,
                        "revision": self.source_revision,
                        "protected_main": self.source_revision,
                        "signature_verified": True,
                        "checks": [],
                    },
                    "publisher": {
                        "repository": publisher.EXPECTED_PUBLISHER_REPOSITORY,
                        "revision": self.publisher_revision,
                        "protected_main": self.publisher_revision,
                    },
                }
            ),
            encoding="utf-8",
        )
        return authorization, artifacts

    def test_dry_run_uses_authorized_data_and_immutable_publisher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, artifacts = self._fixture(root)

            api = FakeApi(artifacts)
            api.download_root = root / "downloads"

            identity = publisher.publisher_identity(
                repository=publisher.EXPECTED_PUBLISHER_REPOSITORY,
                revision=self.publisher_revision,
                workflow_ref=(
                    f"{publisher.EXPECTED_PUBLISHER_REPOSITORY}/"
                    ".github/workflows/publish-szl-kernels.yml@refs/heads/main"
                ),
                run_id="123",
                run_attempt="1",
            )
            result = publisher.run(
                source_root=root,
                report_path=root / "report.json",
                authorization_path=authorization,
                source_revision=self.source_revision,
                publisher=identity,
                publish=False,
                token=None,
                api=api,
                download_fn=api.download,
            )
            self.assertEqual(result["status"], "VERIFIED_DRY_RUN")
            self.assertEqual(
                result["targets"]["first_class_kernel"]["mapped_file_count"],
                1 + 2 * len(publisher.FIRST_CLASS_KERNEL_FILES),
            )
            binding = result["targets"]["first_class_kernel"]["binding"]
            self.assertEqual(
                len(binding["source"]["kernel_files"]),
                1 + 2 * len(publisher.FIRST_CLASS_KERNEL_FILES),
            )
            self.assertIn(
                {
                    "source_path": "README.md",
                    "kernel_path": "README.md",
                    "bytes": artifacts["README.md"].stat().st_size,
                    "sha256": publisher.file_sha256(artifacts["README.md"]),
                },
                binding["source"]["kernel_files"],
            )
            self.assertEqual(
                binding["schema"],
                "szl.hf-first-class-kernel-binding/v1",
            )
            self.assertEqual(binding["source_revision"], self.source_revision)
            self.assertIn(self.publisher_revision, identity["workflow_url"])

    def test_publish_updates_kernel_main_and_v1_then_legacy_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, artifacts = self._fixture(root)
            api = FakeApi(artifacts)
            api.download_root = root / "downloads"
            identity = publisher.publisher_identity(
                repository=publisher.EXPECTED_PUBLISHER_REPOSITORY,
                revision=self.publisher_revision,
                workflow_ref=(
                    f"{publisher.EXPECTED_PUBLISHER_REPOSITORY}/"
                    ".github/workflows/publish-szl-kernels.yml@refs/heads/main"
                ),
                run_id="123",
                run_attempt="1",
            )

            def verify_runtime(*, revision: str) -> dict[str, object]:
                self.assertEqual(revision, "2" * 40)
                return {
                    "status": "STABLE_GET_KERNEL_VERIFIED",
                    "client_version": "0.16.0",
                    "revision": revision,
                    "package_version": "0.1.1",
                }

            result = publisher.run(
                source_root=root,
                report_path=root / "report.json",
                authorization_path=authorization,
                source_revision=self.source_revision,
                publisher=identity,
                publish=True,
                token="test-token",
                api=api,
                download_fn=api.download,
                kernel_upload_fn=api.upload_kernel,
                kernel_runtime_fn=verify_runtime,
            )
            self.assertEqual(
                result["status"], "PUBLISHED_AND_EXACT_READBACK_VERIFIED"
            )
            self.assertEqual(
                api.commits,
                [
                    {"repo_type": "kernel", "revision": "main"},
                    {"repo_type": "kernel", "revision": "v1"},
                    {"repo_type": "model", "revision": None},
                ],
            )
            branches_after = result["targets"]["first_class_kernel"][
                "branches_after"
            ]
            self.assertEqual(set(branches_after), set(publisher.KERNEL_BRANCHES))
            self.assertEqual(
                set(
                    result["targets"]["first_class_kernel"]["readback"].values()
                ),
                {"EXACT_BYTES_VERIFIED"},
            )
            self.assertEqual(
                result["targets"]["legacy_model"]["readback"],
                "EXACT_BYTES_VERIFIED",
            )
            self.assertEqual(
                result["targets"]["first_class_kernel"]["runtime"]["status"],
                "STABLE_GET_KERNEL_VERIFIED",
            )
            metadata = json.loads(
                api.remote[
                    (publisher.KERNEL_REPO_TYPE, branches_after["v1"])
                ][f"build/{publisher.KERNEL_VARIANT}/metadata.json"]
            )
            self.assertEqual(metadata["version"], 1)
            self.assertEqual(metadata["digest"]["algorithm"], "sha256")
            self.assertIn("__init__.py", metadata["digest"]["files"])
            self.assertIn(
                "szl_kernels/__init__.py",
                metadata["digest"]["files"],
            )

    def test_failed_readback_preserves_the_created_kernel_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization, artifacts = self._fixture(root)
            api = FakeApi(artifacts)
            api.download_root = root / "downloads"
            corrupt = root / "corrupt"
            corrupt.write_bytes(b"corrupt")

            def fail_main_readback(*args: object, **kwargs: object) -> str:
                if (
                    kwargs.get("repo_type") == publisher.KERNEL_REPO_TYPE
                    and kwargs.get("revision") == "1" * 40
                ):
                    return str(corrupt)
                return api.download(*args, **kwargs)

            identity = publisher.publisher_identity(
                repository=publisher.EXPECTED_PUBLISHER_REPOSITORY,
                revision=self.publisher_revision,
                workflow_ref=(
                    f"{publisher.EXPECTED_PUBLISHER_REPOSITORY}/"
                    ".github/workflows/publish-szl-kernels.yml@refs/heads/main"
                ),
                run_id="123",
                run_attempt="1",
            )
            report = root / "report.json"
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "main readback mismatch",
            ):
                publisher.run(
                    source_root=root,
                    report_path=report,
                    authorization_path=authorization,
                    source_revision=self.source_revision,
                    publisher=identity,
                    publish=True,
                    token="test-token",
                    api=api,
                    download_fn=fail_main_readback,
                    kernel_upload_fn=api.upload_kernel,
                )
            partial = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(partial["status"], "PUBLICATION_IN_PROGRESS")
            self.assertEqual(
                partial["targets"]["first_class_kernel"]["branches_after"],
                {"main": "1" * 40, "v1": "2" * 40},
            )
            self.assertEqual(
                partial["targets"]["first_class_kernel"]["readback"],
                {"main": "PENDING", "v1": "PENDING"},
            )

    def test_contract_cannot_redirect_publisher_to_another_repo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            contract_path = root / publisher.CONTRACT_RELATIVE
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["repo_id"] = "attacker/target"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "another Hub repository",
            ):
                publisher.load_contract(root)

    def test_contract_must_declare_kernel_readme_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            contract_path = root / publisher.CONTRACT_RELATIVE
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["artifact_files"].remove("README.md")
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "first-class Kernel source inputs",
            ):
                publisher.load_contract(root)


if __name__ == "__main__":
    unittest.main()
