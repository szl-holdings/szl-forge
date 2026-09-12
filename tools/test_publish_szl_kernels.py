from __future__ import annotations

import copy
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import publish_szl_kernels as publisher
import verify_szl_kernel_runtime as runtime_verifier


def retrieval_evidence() -> dict[str, object]:
    byte_prefix = "<" if sys.byteorder == "little" else ">"
    attrs = {
        "schema": "szl.governed-cosine-topk/v1",
        "query_shape": [2], "documents_shape": [4, 2], "output_shape": [1, 4],
        "dtype": "float32", "index_dtype": "int64", "device": "cpu",
        "byte_order": sys.byteorder, "input_hash_format": "logical_c_order_raw_bytes",
        "hash_algorithm": "sha256", "k": 4, "block_rows": 1,
        "actual_block_rows": 1, "max_similarity_elements": 1,
        "zero_document_count": 0, "zero_document_policy": "score_zero",
        "tie_break": "ascending_document_index",
        "implementation": "pytorch_float32_blocked_reference",
        "torch_version": "2.9.1+cpu", "matmul_precision": "highest",
        "cuda_tf32_allowed": None, "receipt_authenticity": "UNSIGNED",
        "retrieval_quality": "NOT_MEASURED", "acceleration_claim": False,
    }
    for name, kind, values in (
        ("query", "f", [1.0, 0.0]),
        ("documents", "f", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0]),
        ("scores", "f", [1.0, 1.0, 0.0, -1.0]),
        ("indices", "q", [1, 2, 0, 3]),
    ):
        attrs[f"{name}_sha256"] = hashlib.sha256(
            struct.pack(f"{byte_prefix}{len(values)}{kind}", *values)
        ).hexdigest()
    body = {
        "seq": 0, "kernel": "governed_retrieval", "op": "cosine_topk",
        "attrs": attrs, "prev": "0" * 64,
    }
    receipt = dict(body, ts=1.0, digest=hashlib.sha3_256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest())
    return {
        "indices": [[1, 2, 0, 3]], "scores": [[1.0, 1.0, 0.0, -1.0]],
        "receipt": receipt, "receipt_depth": 1, "chain_verified": True,
    }


class FakeRuntimeChain:
    def __init__(self) -> None:
        self.depth = 0
        self.records = []

    def verify(self) -> tuple[bool, int, int]:
        return True, self.depth, -1

    def head(self) -> str:
        return self.records[-1]["digest"] if self.records else "0" * 64

    def to_json(self) -> str:
        return json.dumps(self.records)


def runtime_module() -> SimpleNamespace:
    def gate(chain, _axes, *, threshold):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("invalid threshold")
        chain.depth += 1
        return {"threshold": threshold, "passed": 0.5 >= threshold}

    def retrieval(chain, query, documents, *, k, block_rows):
        if query != [1.0, 0.0] or documents != [
            [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]
        ] or k != 4 or block_rows != 1:
            raise AssertionError("retrieval smoke inputs changed")
        evidence = retrieval_evidence()
        chain.depth += 1
        chain.records.append(evidence["receipt"])
        return {
            "indices": SimpleNamespace(
                dtype="torch.int64", device="cpu", tolist=lambda: evidence["indices"]
            ),
            "scores": SimpleNamespace(
                dtype="torch.float32", device="cpu", tolist=lambda: evidence["scores"]
            ),
            "receipt": evidence["receipt"],
        }

    module = SimpleNamespace(**{
        name: (lambda *_args, **_kwargs: None)
        for name in publisher.KERNEL_REQUIRED_EXPORTS
    })
    module.__version__ = "0.2.0"
    module.GENESIS = "0" * 64
    module.UnifiedReceiptChain = FakeRuntimeChain
    module.selfcheck = lambda: {"ok": True, "version": "0.2.0"}
    module.governed_lambda_gate = gate
    module.governed_cosine_topk = retrieval
    return module


def runtime_evidence(revision: str = "2" * 40) -> dict[str, object]:
    return {
        "status": "STABLE_GET_KERNEL_VERIFIED", "client_version": "0.16.0",
        "revision": revision, "package_version": "0.2.0", "selfcheck_ok": True,
        "verified_exports": list(publisher.KERNEL_REQUIRED_EXPORTS),
        "invalid_thresholds_rejected_before_receipt": 4,
        "inclusive_boundaries": {
            "0": {"passed": True, "receipt_depth": 1},
            "1": {"passed": False, "receipt_depth": 1},
        },
        "retrieval": retrieval_evidence(),
    }


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
        evidence = runtime_evidence(revision)

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
        evidence = runtime_evidence(revision)
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
        evidence = runtime_evidence(revision)

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
        def get_kernel(repo_id: str, **kwargs: object) -> SimpleNamespace:
            self.assertEqual(repo_id, publisher.EXPECTED_REPO_ID)
            self.assertEqual(kwargs["revision"], "2" * 40)
            self.assertEqual(kwargs["backend"], "cpu")
            self.assertIs(kwargs["trust_remote_code"], True)
            return runtime_module()

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
        self.assertEqual(evidence, runtime_evidence())

    def test_stable_runtime_rejects_inverted_boundary_decision(self) -> None:
        module = runtime_module()
        valid_gate = module.governed_lambda_gate

        def inverted_gate(*args, **kwargs):
            result = valid_gate(*args, **kwargs)
            result["passed"] = not result["passed"]
            return result

        module.governed_lambda_gate = inverted_gate

        with self.assertRaisesRegex(
            publisher.PublicationError,
            "inclusive threshold boundary contract failed",
        ):
            publisher.verify_stable_kernel_runtime(
                revision="2" * 40,
                get_kernel_fn=lambda *_args, **_kwargs: module,
                tensor_fn=lambda values: values,
                client_version="0.16.0",
            )

    def test_stable_runtime_requires_every_legacy_and_retrieval_export(self) -> None:
        self.assertEqual(len(publisher.KERNEL_REQUIRED_EXPORTS), 16)
        for missing in publisher.KERNEL_REQUIRED_EXPORTS:
            with self.subTest(missing=missing):
                module = runtime_module()
                delattr(module, missing)
                with self.assertRaisesRegex(publisher.PublicationError, "missing public exports"):
                    publisher.verify_stable_kernel_runtime(
                        revision="2" * 40,
                        get_kernel_fn=lambda *_args, **_kwargs: module,
                        tensor_fn=lambda values: values,
                        client_version="0.16.0",
                    )

    def test_stable_runtime_rejects_package_or_selfcheck_version_drift(self) -> None:
        for field in ("__version__", "selfcheck"):
            with self.subTest(field=field):
                module = runtime_module()
                setattr(module, field, "0.1.1" if field == "__version__" else (
                    lambda: {"ok": True, "version": "0.1.1"}
                ))
                with self.assertRaisesRegex(publisher.PublicationError, "unexpected package version"):
                    publisher.verify_stable_kernel_runtime(
                        revision="2" * 40,
                        get_kernel_fn=lambda *_args, **_kwargs: module,
                        tensor_fn=lambda values: values,
                        client_version="0.16.0",
                    )

    def test_retrieval_rejects_wrong_values_raw_hashes_or_unsigned_claims(self) -> None:
        mutations = (
            ("indices", [[2, 1, 0, 3]]), ("scores", [[1.0, 1.0, 0.1, -1.0]]),
            ("indices", [[True, 2, 0, 3]]), ("indices", [[1.0, 2, 0, 3]]),
            ("indices", [(1, 2, 0, 3)]),
            ("scores", [[True, 1.0, 0.0, -1.0]]),
            ("scores", [[1, 1, 0, -1]]),
            ("scores", [[1.0, 1.0, -0.0, -1.0]]),
            ("scores", [(1.0, 1.0, 0.0, -1.0)]),
            ("receipt_depth", 2), ("chain_verified", False),
            ("receipt_depth", True), ("receipt_depth", 1.0),
            ("chain_verified", 1),
            ("receipt.attrs.query_sha256", "0" * 64),
            ("receipt.attrs.documents_sha256", "0" * 64),
            ("receipt.attrs.scores_sha256", "0" * 64),
            ("receipt.attrs.indices_sha256", "0" * 64),
            ("receipt.attrs.receipt_authenticity", "SIGNED"),
            ("receipt.attrs.retrieval_quality", "MEASURED"),
            ("receipt.attrs.acceleration_claim", True),
            ("receipt.attrs.acceleration_claim", 0),
            ("receipt.attrs.query_shape", [2.0]),
            ("receipt.attrs.documents_shape", [4.0, 2]),
            ("receipt.attrs.output_shape", [True, 4]),
            ("receipt.attrs.output_shape", [1.0, 4]),
            ("receipt.attrs.output_shape", (1, 4)),
            ("receipt.attrs.k", 4.0),
            ("receipt.attrs.block_rows", True),
            ("receipt.attrs.actual_block_rows", 1.0),
            ("receipt.attrs.max_similarity_elements", True),
            ("receipt.attrs.zero_document_count", False),
            ("receipt.attrs.zero_document_count", 0.0),
            ("receipt.attrs.byte_order", "unknown"),
            ("receipt.attrs.input_hash_format", "rounded_decimal"),
            ("receipt.attrs.device", "cuda:0"),
            ("receipt.attrs.tie_break", "unspecified"),
            ("receipt.attrs.torch_version", None),
            ("receipt.attrs.torch_version", 2.9),
            ("receipt.attrs.torch_version", ""),
            ("receipt.attrs.matmul_precision", True),
            ("receipt.attrs.matmul_precision", "unknown"),
            ("receipt.ts", None), ("receipt.ts", True),
            ("receipt.ts", 1), ("receipt.ts", "1.0"),
            ("receipt.ts", float("nan")),
            ("receipt.ts", float("inf")),
            ("receipt.ts", float("-inf")),
            ("receipt.seq", 1), ("receipt.prev", "f" * 64),
            ("receipt.seq", False), ("receipt.seq", 0.0),
            ("receipt.digest", "0" * 64), ("receipt.kernel", "other"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                evidence = retrieval_evidence()
                target = evidence
                parts = field.split(".")
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                # A self-consistent modified receipt must still fail the known
                # fixture's data/meaning contract, independently of its digest.
                if field != "receipt.digest":
                    receipt = evidence["receipt"]
                    body = {key: receipt[key] for key in ("seq", "kernel", "op", "attrs", "prev")}
                    receipt["digest"] = hashlib.sha3_256(json.dumps(
                        body, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")).hexdigest()
                with self.assertRaisesRegex(publisher.PublicationError, "retrieval runtime evidence"):
                    publisher.validate_retrieval_runtime_evidence(evidence)

    def test_retrieval_rejects_self_consistent_opposite_byte_order(self) -> None:
        evidence = retrieval_evidence()
        receipt = evidence["receipt"]
        attrs = receipt["attrs"]
        opposite = "big" if sys.byteorder == "little" else "little"
        attrs["byte_order"] = opposite
        prefix = ">" if opposite == "big" else "<"
        for name, kind, values in (
            ("query", "f", [1.0, 0.0]),
            ("documents", "f", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0]),
            ("scores", "f", [1.0, 1.0, 0.0, -1.0]),
            ("indices", "q", [1, 2, 0, 3]),
        ):
            attrs[f"{name}_sha256"] = hashlib.sha256(
                struct.pack(f"{prefix}{len(values)}{kind}", *values)
            ).hexdigest()
        body = {key: receipt[key] for key in ("seq", "kernel", "op", "attrs", "prev")}
        receipt["digest"] = hashlib.sha3_256(json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(publisher.PublicationError, "retrieval runtime evidence"):
            publisher.validate_retrieval_runtime_evidence(evidence)

    def test_retrieval_rejects_extra_or_missing_schema_fields(self) -> None:
        valid = retrieval_evidence()
        for level in ((), ("receipt",), ("receipt", "attrs")):
            valid_target = valid
            for key in level:
                valid_target = valid_target[key]
            for missing in (None, *valid_target):
                with self.subTest(level=level, missing=missing):
                    evidence = copy.deepcopy(valid)
                    target = evidence
                    for key in level:
                        target = target[key]
                    if missing is None:
                        target["authenticated"] = True
                    else:
                        del target[missing]
                    receipt = evidence.get("receipt", {})
                    body_keys = ("seq", "kernel", "op", "attrs", "prev")
                    if all(key in receipt for key in body_keys) and "digest" in receipt:
                        body = {key: receipt[key] for key in body_keys}
                        receipt["digest"] = hashlib.sha3_256(json.dumps(
                            body, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")).hexdigest()
                    with self.assertRaisesRegex(publisher.PublicationError, "retrieval runtime evidence"):
                        publisher.validate_retrieval_runtime_evidence(evidence)

    def test_retrieval_accepts_legitimate_runtime_metadata(self) -> None:
        for precision in ("highest", "high", "medium"):
            with self.subTest(precision=precision):
                evidence = retrieval_evidence()
                receipt = evidence["receipt"]
                receipt["attrs"]["torch_version"] = "2.11.0+cu128"
                receipt["attrs"]["matmul_precision"] = precision
                # Timestamp is deliberately not digest-bound or a freshness gate.
                receipt["ts"] = 1_800_000_000.5
                body = {key: receipt[key] for key in ("seq", "kernel", "op", "attrs", "prev")}
                receipt["digest"] = hashlib.sha3_256(json.dumps(
                    body, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")).hexdigest()
                publisher.validate_retrieval_runtime_evidence(evidence)

    def test_runtime_readback_rejects_missing_exports_or_retrieval_evidence(self) -> None:
        valid = runtime_evidence()
        for field, value in (
            ("verified_exports", valid["verified_exports"][:-1]),
            ("retrieval", None), ("retrieval", []),
            ("retrieval", {"receipt": None}),
            ("package_version", "0.1.1"),
            ("inclusive_boundaries", []),
            ("inclusive_boundaries", {"0": None, "1": {}}),
            ("invalid_thresholds_rejected_before_receipt", 4.0),
            ("selfcheck_ok", 1),
            ("inclusive_boundaries", {
                "0": {"passed": True, "receipt_depth": True},
                "1": {"passed": False, "receipt_depth": 1},
            }),
            ("inclusive_boundaries", {
                "0": {"passed": True, "receipt_depth": 1},
                "1": {"passed": False, "receipt_depth": 1.0},
            }),
            ("inclusive_boundaries", {
                "0": {"passed": 1, "receipt_depth": 1},
                "1": {"passed": 0, "receipt_depth": 1},
            }),
        ):
            with self.subTest(field=field, value=value):
                evidence = copy.deepcopy(valid)
                evidence[field] = value
                with self.assertRaises(publisher.PublicationError):
                    publisher.validate_stable_kernel_runtime_evidence(evidence, revision="2" * 40)

    def test_runtime_readback_requires_complete_outer_schema(self) -> None:
        valid = runtime_evidence()
        publisher.validate_stable_kernel_runtime_evidence(valid, revision="2" * 40)
        for missing in valid:
            with self.subTest(missing=missing):
                evidence = copy.deepcopy(valid)
                del evidence[missing]
                with self.assertRaises(publisher.PublicationError):
                    publisher.validate_stable_kernel_runtime_evidence(evidence, revision="2" * 40)
        for field, value in (("authenticated", True), ("error", "contradictory failure")):
            with self.subTest(extra=field):
                evidence = copy.deepcopy(valid)
                evidence[field] = value
                with self.assertRaises(publisher.PublicationError):
                    publisher.validate_stable_kernel_runtime_evidence(evidence, revision="2" * 40)

    def test_stable_runtime_rejects_unrecorded_retrieval_receipt(self) -> None:
        module = runtime_module()
        valid_retrieval = module.governed_cosine_topk

        def unrecorded_retrieval(*args, **kwargs):
            result = valid_retrieval(*args, **kwargs)
            args[0].records.clear()
            return result

        module.governed_cosine_topk = unrecorded_retrieval
        with self.assertRaisesRegex(publisher.PublicationError, "receipt chain contract"):
            publisher.verify_stable_kernel_runtime(
                revision="2" * 40,
                get_kernel_fn=lambda *_args, **_kwargs: module,
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
            "KERNEL_HUB.md": root / "KERNEL_HUB.md",
            "build/torch-universal/szl_kernels/__init__.py": (
                root / "build/torch-universal/szl_kernels/__init__.py"
            ),
            "build/torch-universal/szl_kernels/_kernel_api.py": (
                root / "build/torch-universal/szl_kernels/_kernel_api.py"
            ),
            "build/torch-universal/szl_kernels/_chain.py": (
                root / "build/torch-universal/szl_kernels/_chain.py"
            ),
            "build/torch-universal/szl_kernels/_ops.py": (
                root / "build/torch-universal/szl_kernels/_ops.py"
            ),
            "build/torch-universal/szl_kernels/retrieval.py": (
                root / "build/torch-universal/szl_kernels/retrieval.py"
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
        artifacts["README.md"].write_text("legacy distribution\n", encoding="utf-8")
        artifacts["KERNEL_HUB.md"].write_text("CPU Kernel Hub card\n", encoding="utf-8")
        artifacts["build/torch-universal/szl_kernels/__init__.py"].write_text(
            'import numpy\n__version__ = "0.2.0"\n', encoding="utf-8"
        )
        artifacts["build/torch-universal/szl_kernels/_kernel_api.py"].write_text(
            'from .retrieval import governed_cosine_topk\n__version__ = "0.2.0"\n',
            encoding="utf-8",
        )
        artifacts["build/torch-universal/szl_kernels/_chain.py"].write_text(
            "GENESIS = '0' * 64\n", encoding="utf-8"
        )
        artifacts["build/torch-universal/szl_kernels/_ops.py"].write_text(
            "def op(): return True\n", encoding="utf-8"
        )
        artifacts["build/torch-universal/szl_kernels/retrieval.py"].write_text(
            "def governed_cosine_topk(): return {}\n", encoding="utf-8"
        )
        artifacts["build/torch-universal/szl_kernels/metadata.json"].write_text(
            json.dumps({"name": "szl_kernels", "version": "0.2.0"}),
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

    def test_staging_uses_cpu_facade_and_all_four_modules_in_both_layouts(self) -> None:
        source_prefix = "build/torch-universal/szl_kernels/"
        mapping = {
            "_kernel_api.py": "__init__.py", "_chain.py": "_chain.py",
            "_ops.py": "_ops.py", "retrieval.py": "retrieval.py",
        }
        self.assertEqual(publisher.FIRST_CLASS_KERNEL_FILES, {
            source_prefix + source: "build/torch-cpu/" + target
            for source, target in mapping.items()
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, artifacts = self._fixture(root)
            staged = root / "staged"
            expected = publisher.stage_first_class_kernel(root, b"binding", staged)
            self.assertEqual(expected["main"]["README.md"], artifacts["KERNEL_HUB.md"].read_bytes())
            self.assertNotEqual(expected["main"]["README.md"], artifacts["README.md"].read_bytes())
            metadata = json.loads(expected["v1"]["build/torch-cpu/metadata.json"])
            self.assertEqual(metadata["python-depends"], [])
            self.assertEqual(metadata["backend"], {"type": "cpu"})
            self.assertEqual(len(expected["v1"]), 10)
            for source, target in mapping.items():
                for layout in ("", "szl_kernels/"):
                    relative = f"build/torch-cpu/{layout}{target}"
                    self.assertEqual(expected["v1"][relative], artifacts[source_prefix + source].read_bytes())
                    self.assertEqual((staged / relative).read_bytes(), expected["v1"][relative])
                    self.assertEqual(metadata["digest"]["files"][layout + target],
                                     publisher.digest_base64(expected["v1"][relative]))
                    if target == "__init__.py":
                        self.assertNotIn(b"import numpy", expected["v1"][relative])

    def test_readback_rejects_drift_in_card_or_either_four_module_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            expected = publisher.stage_first_class_kernel(root, b"binding", root / "staged")
            download_path = root / "downloaded"
            for branch, files in expected.items():
                for corrupt_path in files:
                    with self.subTest(branch=branch, corrupt_path=corrupt_path):
                        def download(_repo, relative, **kwargs):
                            self.assertEqual(kwargs["revision"], "2" * 40)
                            download_path.write_bytes(
                                b"drift" if relative == corrupt_path else files[relative]
                            )
                            return str(download_path)

                        with self.assertRaisesRegex(publisher.PublicationError, "readback mismatch"):
                            publisher.verify_kernel_readback(
                                files, branch=branch, revision="2" * 40,
                                token="test-token", download_fn=download,
                            )

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
                    "source_path": "KERNEL_HUB.md",
                    "kernel_path": "README.md",
                    "bytes": artifacts["KERNEL_HUB.md"].stat().st_size,
                    "sha256": publisher.file_sha256(artifacts["KERNEL_HUB.md"]),
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
                return runtime_evidence(revision)

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

    def test_extra_runtime_claim_is_not_persisted_or_promoted_to_legacy(self) -> None:
        for field, value in (("authenticated", True), ("error", "UNTRUSTED_ERROR_MARKER")):
            with self.subTest(extra=field), tempfile.TemporaryDirectory() as temporary:
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
                    run_id="123", run_attempt="1",
                )

                def invalid_runtime(*, revision: str) -> dict[str, object]:
                    evidence = runtime_evidence(revision)
                    evidence[field] = value
                    return evidence

                report = root / "report.json"
                with self.assertRaisesRegex(publisher.PublicationError, "runtime evidence failed validation"):
                    publisher.run(
                        source_root=root, report_path=report,
                        authorization_path=authorization, source_revision=self.source_revision,
                        publisher=identity, publish=True, token="test-token", api=api,
                        download_fn=api.download, kernel_upload_fn=api.upload_kernel,
                        kernel_runtime_fn=invalid_runtime,
                    )
                partial = json.loads(report.read_text(encoding="utf-8"))
                self.assertEqual(partial["status"], "PUBLICATION_IN_PROGRESS")
                observed = partial["targets"]["first_class_kernel"]["runtime"]
                self.assertEqual(observed["status"], "FAILED")
                self.assertEqual(set(observed), {"status", "client_version", "error"})
                self.assertNotIn("UNTRUSTED_ERROR_MARKER", report.read_text(encoding="utf-8"))
                self.assertEqual(api.commits, [
                    {"repo_type": "kernel", "revision": "main"},
                    {"repo_type": "kernel", "revision": "v1"},
                ])

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
            contract["artifact_files"].remove("KERNEL_HUB.md")
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(
                publisher.PublicationError,
                "first-class Kernel source inputs",
            ):
                publisher.load_contract(root)


if __name__ == "__main__":
    unittest.main()
