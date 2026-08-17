from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "v3_supervisor_validation", HERE / "supervisor_validation.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validation = load_validator()


class SuccessfulReportValidationTests(unittest.TestCase):
    SOURCE = "a" * 40
    RUN_ID = "b" * 32
    GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"
    WORKER_SHA = "c" * 64
    ADAPTER_SHA = "d" * 64

    def setUp(self):
        self.candidate = {
            "candidate_id": "SZL-ReceiptAgent-Qwen3.5-0.8B-v3",
            "state": "SOURCE_READY_NOT_TRAINED",
            "actual_training_base": {
                "repo_id": "unsloth/Qwen3.5-0.8B",
                "revision": "1" * 40,
                "license": "apache-2.0",
                "runtime": "Unsloth FastVisionModel",
                "load_in_4bit": True,
            },
            "training_data": {"train_rows": 180, "dev_rows": 36, "test_rows": 72},
            "training_recipe": {
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
            },
            "supervision_policy": {
                "schema": "szl.receiptagent-v3-supervision-policy/v1",
                "required_containment": "SYSTEMD_USER_SERVICE_CGROUP_V2",
            },
            "runtime_lock": {"torch": "2.10.0", "unsloth": "2026.7.4"},
            "publication_eligible": False,
            "autonomy_eligible": False,
        }
        source_unsigned = {
            "manifestSha256": "2" * 64,
            "trainSha256": "3" * 64,
            "trainBytes": 1000,
            "uniqueTrainingRows": 180,
            "kindCounts": {"DRAFT": 60, "RECOVERY": 60, "REFUSAL": 60},
            "heldOutCommitments": {
                "dev.jsonl": {"rows": 36, "sha256": "4" * 64},
                "test.jsonl": {"rows": 72, "sha256": "5" * 64},
            },
            "trainerOpenedSplitContent": ["TRAIN"],
        }
        self.source_bundle = {
            **source_unsigned,
            "bundleSha256": validation.sha256_json(source_unsigned),
        }
        self.adapter_files = [
            {
                "path": "adapter_config.json",
                "bytes": 10,
                "sha256": "6" * 64,
                "jsonKeys": 2,
            },
            {
                "path": "adapter_model.safetensors",
                "bytes": 20,
                "sha256": "7" * 64,
                "tensorCount": 4,
            },
        ]

    def make_report(self, run_kind: str = "FULL"):
        recipe = self.candidate["training_recipe"]
        full = run_kind == "FULL"
        steps = (
            recipe["full_optimizer_steps"] if full else recipe["smoke_optimizer_steps"]
        )
        warmup = recipe["warmup_steps"] if full else 0
        report = {
            "schema": validation.REPORT_SCHEMA,
            "candidateId": self.candidate["candidate_id"],
            "supervisorRunId": self.RUN_ID,
            "supervisionPolicySha256": validation.sha256_json(
                self.candidate["supervision_policy"]
            ),
            "workerSourceSha256": self.WORKER_SHA,
            "trainingRecipeSha256": validation.sha256_json(recipe),
            "state": validation.SUCCESS_STATE_BY_KIND[run_kind],
            "runKind": run_kind,
            "measuredAt": "2026-08-13T12:00:00+00:00",
            "hostClass": "LOCAL_GPU_RUNNER_REDACTED",
            "source": {
                "repository": "szl-holdings/szl-forge",
                "revision": self.SOURCE,
                "branch": "main",
                "originIdentityVerified": True,
                "freshRemoteMainObserved": False,
                "freshRemoteMainObservationDelegatedToSupervisor": True,
                "cachedRemoteTrackingMatches": True,
                "workingTreeClean": True,
                "commitSignatureVerifiedByThisTool": False,
            },
            "sourceBundle": copy.deepcopy(self.source_bundle),
            "implementation": copy.deepcopy(self.candidate["actual_training_base"]),
            "runtimePackages": copy.deepcopy(self.candidate["runtime_lock"]),
            "uniqueTrainingRows": 180,
            "scheduledExamples": steps * recipe["gradient_accumulation_steps"],
            "optimizerSteps": steps,
            "configuration": {
                "per_device_train_batch_size": recipe["per_device_batch_size"],
                "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
                "warmup_steps": warmup,
                "max_steps": steps,
                "learning_rate": recipe["learning_rate"],
                "logging_steps": 1,
                "optim": recipe["optimizer"],
                "weight_decay": recipe["weight_decay"],
                "lr_scheduler_type": recipe["lr_scheduler"],
                "seed": recipe["seed"],
                "output_dir": "<OUTSIDE_REPOSITORY>/checkpoints",
                "report_to": "none",
                "remove_unused_columns": False,
                "dataset_text_field": "",
                "dataset_kwargs": {"skip_prepare_dataset": True},
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "max_length": recipe["max_length"],
                "save_strategy": "no",
                "finetuneVisionLayers": recipe["finetune_vision_layers"],
                "loraR": recipe["lora_r"],
                "loraAlpha": recipe["lora_alpha"],
                "responseOnlyLoss": recipe["response_only_loss"],
                "enableThinking": recipe["enable_thinking"],
            },
            "gpu": {
                "name": "NVIDIA Test GPU",
                "computeCapability": "12.0",
                "totalBytes": 8 * 1024**3,
                "freeBytesBeforeLoad": 6 * 1024**3,
                "temperatureCBeforeLoad": 60,
                "maximumTemperaturePolicyC": 80,
                "minimumFreeMemoryPolicyGiB": 4.0,
                "torchVersion": "2.10.0",
                "cudaRuntime": "13.0",
                "preRuntimeImport": {
                    "uuid": self.GPU_UUID,
                    "name": "NVIDIA Test GPU",
                    "temperatureCBeforeRuntimeImport": 59,
                    "freeMiBBeforeRuntimeImport": 6144,
                    "totalMiB": 8192,
                },
                "temperatureSamplesC": [60, 65, 63],
                "maximumObservedTemperatureC": 65,
                "temperatureCAfterRun": 63,
                "peakReservedBytesTraining": 1024,
            },
            "training": {"durationSeconds": 12.5, "metrics": {"train_loss": 0.5}},
            "adapter": {
                "relativePath": "adapter",
                "formatPolicy": "PARSED_SAFETENSORS_AND_ALLOWLISTED_METADATA",
                "aggregateSha256": self.ADAPTER_SHA,
                "files": copy.deepcopy(self.adapter_files),
            },
            "integrityDigestIsAuthentication": False,
            "authenticatedTrainingEnvelopePresent": False,
            "qualificationEligible": full,
            "receiptEligible": False,
            "publicationEligible": False,
            "autonomyEligible": False,
            "claimBoundary": validation.CLAIM_BOUNDARY,
        }
        report["reportSha256"] = validation.sha256_json(report)
        return report

    def validate(self, report, run_kind="FULL", **overrides):
        expected = {
            "candidate": self.candidate,
            "expected_source_revision": self.SOURCE,
            "expected_source_bundle": self.source_bundle,
            "expected_supervisor_run_id": self.RUN_ID,
            "expected_gpu_uuid": self.GPU_UUID,
            "expected_worker_source_sha256": self.WORKER_SHA,
            "expected_adapter_aggregate_sha256": self.ADAPTER_SHA,
            "expected_adapter_files": self.adapter_files,
            "expected_run_kind": run_kind,
        }
        expected.update(overrides)
        return validation.validate_successful_report(report, **expected)

    def test_full_report_is_bound_but_never_promotion_eligible(self):
        result = self.validate(self.make_report())
        self.assertEqual(
            "SUPERVISOR_OBSERVED_FULL_OUTPUT_BOUND_UNATTESTED",
            result.observation_state,
        )
        self.assertTrue(result.local_evaluation_input_binding_satisfied)
        self.assertEqual(
            self.source_bundle["bundleSha256"], result.source_bundle_sha256
        )
        self.assertEqual(
            validation.sha256_json(self.candidate["training_recipe"]),
            result.training_recipe_sha256,
        )
        self.assertFalse(result.receipt_eligible)
        self.assertFalse(result.publication_eligible)
        self.assertFalse(result.autonomy_eligible)

    def test_smoke_report_is_never_qualified_or_evaluation_input(self):
        result = self.validate(self.make_report("SMOKE"), "SMOKE")
        self.assertEqual(
            "SUPERVISOR_OBSERVED_SMOKE_OUTPUT_BOUND_NOT_QUALIFIED",
            result.observation_state,
        )
        self.assertFalse(result.local_evaluation_input_binding_satisfied)

    def test_run_source_gpu_recipe_and_adapter_tampering_fail(self):
        mutations = {
            "run ID": lambda report: report.__setitem__("supervisorRunId", "0" * 32),
            "source": lambda report: report["source"].__setitem__("revision", "0" * 40),
            "GPU UUID": lambda report: report["gpu"]["preRuntimeImport"].__setitem__(
                "uuid", "GPU-00000000-0000-0000-0000-000000000000"
            ),
            "recipe": lambda report: report.__setitem__(
                "trainingRecipeSha256", "0" * 64
            ),
            "configuration": lambda report: report["configuration"].__setitem__(
                "learning_rate", 0.9
            ),
            "adapter": lambda report: report["adapter"].__setitem__(
                "aggregateSha256", "0" * 64
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                report = self.make_report()
                mutate(report)
                report["reportSha256"] = validation.sha256_json(
                    {
                        key: value
                        for key, value in report.items()
                        if key != "reportSha256"
                    }
                )
                with self.assertRaises(validation.SupervisorValidationError):
                    self.validate(report)

    def test_self_digest_unknown_field_and_promotion_boundary_fail(self):
        report = self.make_report()
        report["reportSha256"] = "0" * 64
        with self.assertRaisesRegex(
            validation.SupervisorValidationError, "self-digest"
        ):
            self.validate(report)

        report = self.make_report()
        report["inventedEvidence"] = True
        report["reportSha256"] = validation.sha256_json(
            {key: value for key, value in report.items() if key != "reportSha256"}
        )
        with self.assertRaisesRegex(validation.SupervisorValidationError, "extra"):
            self.validate(report)

        report = self.make_report()
        report["publicationEligible"] = True
        report["reportSha256"] = validation.sha256_json(
            {key: value for key, value in report.items() if key != "reportSha256"}
        )
        with self.assertRaisesRegex(
            validation.SupervisorValidationError, "publication"
        ):
            self.validate(report)

    def test_smoke_cannot_claim_full_state_or_qualification(self):
        report = self.make_report("SMOKE")
        report["state"] = validation.SUCCESS_STATE_BY_KIND["FULL"]
        report["qualificationEligible"] = True
        report["reportSha256"] = validation.sha256_json(
            {key: value for key, value in report.items() if key != "reportSha256"}
        )
        with self.assertRaises(validation.SupervisorValidationError):
            self.validate(report, "SMOKE")

    def test_expected_source_bundle_must_itself_match_candidate(self):
        bundle = copy.deepcopy(self.source_bundle)
        bundle["heldOutCommitments"]["dev.jsonl"]["rows"] = 35
        unsigned = {
            key: value for key, value in bundle.items() if key != "bundleSha256"
        }
        bundle["bundleSha256"] = validation.sha256_json(unsigned)
        report = self.make_report()
        report["sourceBundle"] = copy.deepcopy(bundle)
        report["reportSha256"] = validation.sha256_json(
            {key: value for key, value in report.items() if key != "reportSha256"}
        )
        with self.assertRaisesRegex(
            validation.SupervisorValidationError, "held-out rows"
        ):
            self.validate(report, expected_source_bundle=bundle)

    def test_expected_adapter_evidence_must_itself_obey_allowlist(self):
        files = copy.deepcopy(self.adapter_files)
        files.append({"path": "unreviewed.bin", "bytes": 1, "sha256": "8" * 64})
        report = self.make_report()
        report["adapter"]["files"] = copy.deepcopy(files)
        report["reportSha256"] = validation.sha256_json(
            {key: value for key, value in report.items() if key != "reportSha256"}
        )
        with self.assertRaisesRegex(
            validation.SupervisorValidationError, "allowlisted"
        ):
            self.validate(report, expected_adapter_files=files)


if __name__ == "__main__":
    unittest.main()
