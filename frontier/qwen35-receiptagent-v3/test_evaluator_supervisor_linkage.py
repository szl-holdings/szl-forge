from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = "a" * 40
RUN_ID = "b" * 32
GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"
ADAPTER_SHA = "c" * 64
ADAPTER_FILES = [
    {
        "path": "adapter_config.json",
        "bytes": 2,
        "sha256": "d" * 64,
        "jsonKeys": 0,
    },
    {
        "path": "adapter_model.safetensors",
        "bytes": 128,
        "sha256": "e" * 64,
        "tensorCount": 1,
    },
]
COMPONENT_BYTES = {
    "launch_supervised_training.py": b"exact-launcher-source",
    "supervisor_bootstrap.py": b"exact-bootstrap-source",
    "supervise_training.py": b"exact-supervisor-source",
    "containment_probe.py": b"exact-containment-source",
    "train_candidate.py": b"exact-worker-source",
    "supervisor_validation.py": b"exact-validator-source",
    "candidate.json": b"exact-candidate-source",
}
TRAINING_RECIPE = {
    "smoke_optimizer_steps": 1,
    "full_optimizer_steps": 135,
    "full_scheduled_examples": 540,
    "full_epochs_over_unique_rows": 3,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_length": 2048,
    "learning_rate": 0.0001,
    "warmup_steps": 10,
    "optimizer": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler": "constant_with_warmup",
    "seed": 11,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "response_only_loss": True,
    "finetune_vision_layers": False,
    "finetune_language_layers": True,
    "finetune_attention_modules": True,
    "finetune_mlp_modules": True,
    "enable_thinking": False,
    "minimum_free_gpu_gib": 4.0,
    "maximum_gpu_temperature_c": 80,
}
RUNTIME_LOCK = {
    "unsloth": "2026.7.4",
    "torch": "2.10.0",
    "transformers": "5.5.0",
    "trl": "0.24.0",
    "datasets": "4.3.0",
    "peft": "0.19.1",
    "bitsandbytes": "0.49.2",
    "safetensors": "0.8.0",
    "accelerate": "1.14.0",
    "huggingface-hub": "1.24.0",
    "unsloth-zoo": "2026.7.4",
}


def load_evaluator():
    here_text = str(HERE)
    if here_text not in sys.path:
        sys.path.insert(0, here_text)
    spec = importlib.util.spec_from_file_location(
        "v3_evaluator_supervisor_linkage_under_test", HERE / "evaluate_candidate.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluator = load_evaluator()


def candidate() -> dict:
    return {
        "candidate_id": "SZL-ReceiptAgent-Qwen3.5-0.8B-v3",
        "training_recipe": copy.deepcopy(TRAINING_RECIPE),
        "runtime_lock": copy.deepcopy(RUNTIME_LOCK),
        "supervision_policy": {
            "python_executable": "/home/rosie/.venvs/szl-unsloth/bin/python",
            "schema": "test-policy",
            "required_containment": "SYSTEMD_USER_SERVICE_CGROUP_V2",
            "security_boundary": "COOPERATIVE_SAME_ACCOUNT",
            "filesystem_isolation": "ROOT_DIRECTORY_EXPLICIT_BIND_ALLOWLIST",
            "worker_mount_root": "/opt/szl-ra3",
            "thermal_sample_interval_seconds": 2.0,
            "maximum_telemetry_gap_seconds": 8.0,
            "full_wall_timeout_seconds": 10800.0,
        },
    }


def source_identity() -> dict:
    return {
        "repository": "szl-holdings/szl-forge",
        "revision": SOURCE,
        "branch": "main",
        "originIdentityVerified": True,
        "freshRemoteMainObserved": True,
        "cachedRemoteTrackingMatches": True,
        "workingTreeClean": True,
        "commitSignatureVerifiedByThisTool": False,
    }


def supervisor_source_identity() -> dict:
    return {
        **source_identity(),
        "components": {
            filename: {
                "bytes": len(COMPONENT_BYTES[filename]),
                "sha256": evaluator.sha256_bytes(COMPONENT_BYTES[filename]),
            }
            for filename in evaluator.SUPERVISOR_SOURCE_COMPONENTS
        },
    }


def child_report() -> dict:
    report = {
        "schema": "szl.frontier-training-run/v3",
        "candidateId": candidate()["candidate_id"],
        "supervisorRunId": RUN_ID,
        "runKind": "FULL",
        "state": "MEASURED_FULL_TRAINING_COMPLETED_UNATTESTED",
        "adapter": {"aggregateSha256": ADAPTER_SHA},
        "integrityDigestIsAuthentication": False,
        "authenticatedTrainingEnvelopePresent": False,
        "qualificationEligible": True,
        "receiptEligible": False,
        "publicationEligible": False,
        "autonomyEligible": False,
    }
    report["reportSha256"] = evaluator.sha256_json(report)
    return report


def supervisor_report(child: dict, child_bytes: bytes) -> dict:
    expected_identities = {
        "supervisionPolicySha256": evaluator.sha256_json(
            candidate()["supervision_policy"]
        ),
        "supervisorSourceSha256": evaluator.sha256_bytes(
            COMPONENT_BYTES["supervise_training.py"]
        ),
        "workerSourceSha256": evaluator.sha256_bytes(
            COMPONENT_BYTES["train_candidate.py"]
        ),
        "validatorSourceSha256": evaluator.sha256_bytes(
            COMPONENT_BYTES["supervisor_validation.py"]
        ),
        "candidateSourceSha256": evaluator.sha256_bytes(
            COMPONENT_BYTES["candidate.json"]
        ),
        "pythonExecutable": {
            "path": candidate()["supervision_policy"]["python_executable"],
            "resolvedPath": "/usr/bin/python3.12",
            "bytes": 8_000_000,
            "sha256": "f" * 64,
        },
        "workerEnvironmentSha256": evaluator.sha256_json(
            evaluator.expected_worker_environment(candidate())
        ),
        "admissionRecordSha256": "2" * 64,
    }
    report = {
        "schema": "szl.frontier-training-supervisor/v1",
        "candidateId": candidate()["candidate_id"],
        "runId": RUN_ID,
        "runKind": "FULL",
        "observedAt": "2026-08-13T12:00:00+00:00",
        "source": supervisor_source_identity(),
        "identities": expected_identities,
        "containment": {
            "unit": f"szl-ra3-supervisor-{RUN_ID}.service",
            "controlGroup": (
                "/user.slice/user-1000.slice/user@1000.service/app.slice/"
                f"szl-ra3-supervisor-{RUN_ID}.service"
            ),
            "MainPID": "4321",
            "KillMode": "control-group",
            "SendSIGKILL": "yes",
            "NoNewPrivileges": "yes",
            "ProtectControlGroups": "yes",
            "PrivateTmp": "yes",
            "RestrictSUIDSGID": "yes",
            "workerNamespaceProbe": {
                "relativePath": "runtime-cache/containment-probe.json",
                "state": "PASS",
                "fileSha256": "3" * 64,
                "bytes": 512,
                "canonicalReportSha256": evaluator.sha256_json(
                    evaluator.CONTAINMENT_PROBE_SEMANTICS
                ),
                "sameUnitPreExecGate": True,
            },
            "credentialCanarySha256": "4" * 64,
        },
        "provenance": {
            "trainingBundleSha256": "6" * 64,
            "credentialCanarySha256": "4" * 64,
        },
        "securityBoundary": "COOPERATIVE_SAME_ACCOUNT",
        "integrityDigestIsAuthentication": False,
        "authenticatedSupervisorEnvelopePresent": False,
        "qualificationEligible": False,
        "receiptEligible": False,
        "publicationEligible": False,
        "runtimeWitnessPresent": False,
        "autonomyEligible": False,
        "evaluationPerformed": False,
        "comparisonCriteriaSatisfied": False,
        "launch": {
            "workerUnit": f"szl-ra3-worker-{RUN_ID}.service",
            "workerControlGroup": (
                "/user.slice/user-1000.slice/user@1000.service/app.slice/"
                f"szl-ra3-worker-{RUN_ID}.service"
            ),
            "workerArgvSha256": "5" * 64,
            "startedAt": "2026-08-13T12:00:00+00:00",
            "endedAt": "2026-08-13T12:02:00+00:00",
            "durationSeconds": 120.0,
            "wallTimeoutSeconds": 10800.0,
            "workerExitStatus": 0,
            "workerResult": "success",
            "triggerError": None,
            "termination": None,
            "cgroupEmptyConfirmed": True,
        },
        "telemetry": {
            "source": "INDEPENDENT_SUPERVISOR_FIXED_NVIDIA_SMI",
            "gpuUuid": GPU_UUID,
            "maximumTemperaturePolicyC": 80,
            "sampleIntervalSeconds": 2.0,
            "maximumTelemetryGapSeconds": 8.0,
            "maximumObservedSampleGapSeconds": 2.1,
            "samples": [
                {
                    "offsetSeconds": -0.1,
                    "observedAt": "2026-08-13T12:00:00+00:00",
                    "gpuUuid": GPU_UUID,
                    "temperatureC": 55,
                    "freeMiB": 7000,
                    "totalMiB": 8192,
                },
                {
                    "offsetSeconds": 2.0,
                    "observedAt": "2026-08-13T12:00:02+00:00",
                    "gpuUuid": GPU_UUID,
                    "temperatureC": 60,
                    "freeMiB": 5000,
                    "totalMiB": 8192,
                },
            ],
            "maximumObservedTemperatureC": 60,
        },
        "logs": {},
        "trainingReport": {
            "relativePath": "payload/training-report.json",
            "fileSha256": evaluator.sha256_bytes(child_bytes),
            "bytes": len(child_bytes),
            "canonicalReportSha256": child["reportSha256"],
            "state": "MEASURED_FULL_TRAINING_COMPLETED_UNATTESTED",
            "provenance": "CHILD_REPORTED_UNATTESTED",
        },
        "bindings": copy.deepcopy(evaluator.SUPERVISOR_BINDINGS),
        "adapter": {
            "aggregateSha256": ADAPTER_SHA,
            "matchesTrainingReport": True,
            "safeTensorsParsed": True,
            "allowlistedFilesOnly": True,
            "symlinksAbsent": True,
            "files": copy.deepcopy(ADAPTER_FILES),
        },
        "state": evaluator.SUPERVISOR_FULL_STATE,
        "localEvaluationInputBindingSatisfied": True,
        "primaryCause": "SUCCESS",
        "workerPayloadDisposition": "BOUND_UNATTESTED",
        "claimBoundary": "bounded supervisor evidence",
    }
    report["reportSha256"] = evaluator.sha256_json(report)
    return report


class SupervisorLinkageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = pathlib.Path(self.temporary.name)
        self.adapter_dir = root / "adapter"
        self.adapter_dir.mkdir()
        self.child_path = root / "training-report.json"
        self.supervisor_path = root / "supervisor-report.json"
        self.child = child_report()
        self.child_bytes = (
            json.dumps(self.child, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        self.child_path.write_bytes(self.child_bytes)
        self.supervisor = supervisor_report(self.child, self.child_bytes)
        self.write_supervisor(self.supervisor)

    def write_supervisor(self, report: dict) -> None:
        unsigned = copy.deepcopy(report)
        unsigned.pop("reportSha256", None)
        report["reportSha256"] = evaluator.sha256_json(unsigned)
        self.supervisor_path.write_bytes(
            (json.dumps(report, sort_keys=True) + "\n").encode("utf-8")
        )

    def committed_bytes(self, _source: str, path: str) -> bytes:
        return COMPONENT_BYTES[path.rsplit("/", 1)[-1]]

    @contextlib.contextmanager
    def linkage_mocks(self):
        validated = types.SimpleNamespace(report_sha256=self.child["reportSha256"])
        with (
            mock.patch.object(
                evaluator, "committed_bytes", side_effect=self.committed_bytes
            ),
            mock.patch.object(
                evaluator,
                "hash_adapter",
                return_value=(ADAPTER_SHA, copy.deepcopy(ADAPTER_FILES)),
            ),
            mock.patch.object(evaluator, "curriculum", return_value=({"bound": True}, [])),
            mock.patch.object(
                evaluator, "validate_successful_report", return_value=validated
            ) as validate,
        ):
            yield validate

    def verify(self) -> dict:
        return evaluator.verify_supervisor_linkage(
            self.supervisor_path,
            training_report_path=self.child_path,
            training_report=self.child,
            adapter_dir=self.adapter_dir,
            source_commit=SOURCE,
            candidate=candidate(),
            source=source_identity(),
        )

    def test_valid_full_supervisor_report_binds_child_and_fresh_adapter(self):
        with self.linkage_mocks() as validate:
            linkage = self.verify()
        self.assertEqual(RUN_ID, linkage["runId"])
        self.assertEqual(ADAPTER_SHA, linkage["adapterAggregateSha256"])
        self.assertEqual(self.child["reportSha256"], linkage["trainingReportCanonicalSha256"])
        self.assertFalse(linkage["integrityDigestIsAuthentication"])
        self.assertFalse(linkage["authenticatedSupervisorEnvelopePresent"])
        self.assertFalse(linkage["runtimeWitnessPresent"])
        self.assertFalse(linkage["receiptEligible"])
        self.assertEqual(GPU_UUID, linkage["gpuUuid"])
        self.assertEqual(60, linkage["maximumObservedTemperatureC"])
        self.assertEqual(5000, linkage["minimumObservedFreeMiB"])
        self.assertEqual(2.1, linkage["maximumObservedSampleGapSeconds"])
        self.assertEqual("6" * 64, linkage["trainingBundleSha256"])
        self.assertEqual("4" * 64, linkage["credentialCanarySha256"])
        self.assertEqual(
            set(COMPONENT_BYTES) - {"candidate.json"},
            set(evaluator.SUPERVISOR_SOURCE_COMPONENTS),
        )
        self.assertEqual(
            evaluator.sha256_json(evaluator.CONTAINMENT_PROBE_SEMANTICS),
            linkage["containmentProbeCanonicalSha256"],
        )
        validate.assert_called_once()
        call = validate.call_args
        self.assertEqual("FULL", call.kwargs["expected_run_kind"])
        self.assertEqual(RUN_ID, call.kwargs["expected_supervisor_run_id"])
        self.assertEqual(GPU_UUID, call.kwargs["expected_gpu_uuid"])
        self.assertEqual(ADAPTER_FILES, call.kwargs["expected_adapter_files"])

    def test_tampered_self_digest_is_rejected(self):
        self.supervisor["primaryCause"] = "TAMPERED"
        self.supervisor_path.write_bytes(
            json.dumps(self.supervisor).encode("utf-8")
        )
        with self.linkage_mocks():
            with self.assertRaisesRegex(
                evaluator.QualificationError, "integrity digest is invalid"
            ):
                self.verify()

    def test_recomputed_tampering_of_every_required_link_is_rejected(self):
        mutations = {
            "state": lambda report: report.__setitem__("state", "WRONG"),
            "primary cause": lambda report: report.__setitem__(
                "primaryCause", "WORKER_EXIT_FAILURE"
            ),
            "runtime witness": lambda report: report.__setitem__(
                "runtimeWitnessPresent", True
            ),
            "source": lambda report: report["source"].__setitem__("revision", "9" * 40),
            "source component": lambda report: report["source"]["components"][
                "supervisor_bootstrap.py"
            ].__setitem__("sha256", "8" * 64),
            "launcher source component": lambda report: report["source"]["components"][
                "launch_supervised_training.py"
            ].__setitem__("sha256", "8" * 64),
            "containment source component": lambda report: report["source"]["components"][
                "containment_probe.py"
            ].__setitem__("sha256", "8" * 64),
            "training bundle provenance": lambda report: report["provenance"].__setitem__(
                "trainingBundleSha256", "not-a-digest"
            ),
            "credential provenance": lambda report: report["provenance"].__setitem__(
                "credentialCanarySha256", "6" * 64
            ),
            "run": lambda report: report.__setitem__("runId", "8" * 32),
            "worker": lambda report: report["identities"].__setitem__(
                "workerSourceSha256", "7" * 64
            ),
            "validator": lambda report: report["identities"].__setitem__(
                "validatorSourceSha256", "6" * 64
            ),
            "policy": lambda report: report["identities"].__setitem__(
                "supervisionPolicySha256", "5" * 64
            ),
            "worker environment": lambda report: report["identities"].__setitem__(
                "workerEnvironmentSha256", "6" * 64
            ),
            "systemd property": lambda report: report["containment"].__setitem__(
                "NoNewPrivileges", "no"
            ),
            "containment probe": lambda report: report["containment"][
                "workerNamespaceProbe"
            ].__setitem__("state", "FAIL"),
            "containment probe digest": lambda report: report["containment"][
                "workerNamespaceProbe"
            ].__setitem__("canonicalReportSha256", "7" * 64),
            "worker cgroup": lambda report: report["launch"].__setitem__(
                "workerControlGroup", "/wrong.service"
            ),
            "telemetry GPU": lambda report: report["telemetry"]["samples"][1].__setitem__(
                "gpuUuid", "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
            ),
            "telemetry temperature": lambda report: report["telemetry"].__setitem__(
                "maximumObservedTemperatureC", 59
            ),
            "telemetry memory": lambda report: report["telemetry"]["samples"][0].__setitem__(
                "freeMiB", 4095
            ),
            "telemetry gap": lambda report: report["telemetry"].__setitem__(
                "maximumObservedSampleGapSeconds", 7.9
            ),
            "child file": lambda report: report["trainingReport"].__setitem__(
                "fileSha256", "4" * 64
            ),
            "child canonical": lambda report: report["trainingReport"].__setitem__(
                "canonicalReportSha256", "3" * 64
            ),
            "adapter aggregate": lambda report: report["adapter"].__setitem__(
                "aggregateSha256", "2" * 64
            ),
            "adapter files": lambda report: report["adapter"].__setitem__("files", []),
            "binding": lambda report: report["bindings"].__setitem__(
                "adapterMatchesTrainingReport", False
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(self.supervisor)
                mutate(changed)
                self.write_supervisor(changed)
                with self.linkage_mocks():
                    with self.assertRaises(evaluator.QualificationError):
                        self.verify()

    def test_duplicate_supervisor_keys_are_rejected_before_semantics(self):
        raw = self.supervisor_path.read_text(encoding="utf-8").rstrip()
        self.supervisor_path.write_bytes(
            ('{"schema":"first",' + raw[1:]).encode("utf-8")
        )
        with self.linkage_mocks():
            with self.assertRaisesRegex(evaluator.QualificationError, "duplicate JSON key"):
                self.verify()

    def test_raw_child_report_without_supervisor_report_fails_before_source_work(self):
        args = argparse.Namespace(
            model_kind="v3",
            training_report=self.child_path,
            supervisor_report=None,
            adapter_dir=self.adapter_dir,
            source_commit=SOURCE,
            split="test",
        )
        with mock.patch.object(evaluator, "fresh_exact_source") as fresh:
            with self.assertRaisesRegex(
                evaluator.QualificationError, "--supervisor-report"
            ):
                evaluator.evaluate(args)
        fresh.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
