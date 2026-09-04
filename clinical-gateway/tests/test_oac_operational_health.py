from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest

import oac_operational_health as model


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "operational-model" / "artifacts" / "model.json"
MODEL_RECEIPT_PATH = ROOT / "operational-model" / "artifacts" / "model-receipt.json"
DATASET_RECEIPT_PATH = ROOT / "operational-model" / "artifacts" / "dataset-receipt.json"


def healthy_features() -> dict[str, int | float]:
    return {
        "listener_running": 1,
        "tls_enabled": 1,
        "peer_allowlist_configured": 1,
        "queue_utilization": 0.05,
        "consecutive_failures": 0,
        "seconds_since_last_success": 10.0,
        "ledger_integrity_ok": 1,
        "configuration_valid": 1,
    }


def attention_features() -> dict[str, int | float]:
    return {
        "listener_running": 0,
        "tls_enabled": 0,
        "peer_allowlist_configured": 0,
        "queue_utilization": 1.0,
        "consecutive_failures": 20,
        "seconds_since_last_success": 86400.0,
        "ledger_integrity_ok": 0,
        "configuration_valid": 0,
    }


class OperationalHealthKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kernel = model.OperationalHealthKernel(MODEL_PATH, MODEL_RECEIPT_PATH)

    def test_healthy_and_attention_examples_are_ordered(self) -> None:
        healthy = self.kernel.score(healthy_features())
        attention = self.kernel.score(attention_features())
        self.assertLess(healthy["operator_attention_score"], attention["operator_attention_score"])
        self.assertFalse(healthy["operator_attention_required"])
        self.assertTrue(attention["operator_attention_required"])

    def test_output_has_explicitly_zero_operational_authority(self) -> None:
        output = self.kernel.score(healthy_features())
        self.assertEqual(model.ADVISORY_SCHEMA, output["schema"])
        self.assertTrue(output["authority"])
        self.assertTrue(all(value is False for value in output["authority"].values()))
        self.assertEqual(
            "synthetic_attention_score_not_production_calibrated",
            output["score_semantics"],
        )
        self.assertFalse(hasattr(self.kernel, "acknowledge"))
        self.assertFalse(hasattr(self.kernel, "release"))

    def test_unknown_missing_sensitive_and_hl7_fields_fail_closed(self) -> None:
        cases = []
        missing = healthy_features()
        missing.pop("tls_enabled")
        cases.append(missing)
        unknown = healthy_features()
        unknown["temperature"] = 37
        cases.append(unknown)
        sensitive = healthy_features()
        sensitive["patient_id"] = "DEID-1"
        cases.append(sensitive)
        raw = healthy_features()
        raw["queue_utilization"] = "MSH|^~\\&|DEVICE"
        cases.append(raw)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(model.ModelInputError):
                self.kernel.score(case)

    def test_non_finite_and_out_of_range_values_fail_closed(self) -> None:
        for invalid in (-0.01, 1.01, float("nan"), float("inf")):
            features = healthy_features()
            features["queue_utilization"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(model.ModelInputError):
                self.kernel.score(features)

    def test_receipt_detects_model_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tampered = Path(temporary) / "model.json"
            tampered.write_bytes(MODEL_PATH.read_bytes() + b" ")
            with self.assertRaisesRegex(model.ModelArtifactError, "hash"):
                model.OperationalHealthKernel(tampered, MODEL_RECEIPT_PATH)

    def test_training_provenance_is_closed_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
            artifact["training"] = {}
            artifact_path = root / "model.json"
            artifact_path.write_text(
                json.dumps(artifact, sort_keys=True, separators=(",", ":")), encoding="utf-8"
            )
            receipt = json.loads(MODEL_RECEIPT_PATH.read_text(encoding="utf-8"))
            receipt["model_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(model.ModelArtifactError, "provenance"):
                model.OperationalHealthKernel(artifact_path, receipt_path)

    def test_model_outputs_are_finite_and_deterministic(self) -> None:
        first = self.kernel.score(healthy_features())
        second = self.kernel.score(dict(reversed(list(healthy_features().items()))))
        self.assertEqual(first, second)
        self.assertTrue(math.isfinite(first["operator_attention_score"]))

    def test_cli_uses_required_receipt(self) -> None:
        command = [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "src" / "oac_operational_health.py"),
            "--model",
            str(MODEL_PATH),
            "--receipt",
            str(MODEL_RECEIPT_PATH),
            "--input",
            str(ROOT / "operational-model" / "example-input.json"),
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(0, proc.returncode, proc.stderr)
        output = json.loads(proc.stdout)
        self.assertEqual(model.ADVISORY_SCHEMA, output["schema"])

    def test_cli_normalizes_missing_and_invalid_utf8_files(self) -> None:
        base = [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "src" / "oac_operational_health.py"),
        ]
        missing = subprocess.run(
            base
            + [
                "--model",
                str(ROOT / "does-not-exist.json"),
                "--receipt",
                str(MODEL_RECEIPT_PATH),
                "--input",
                str(ROOT / "operational-model" / "example-input.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(2, missing.returncode)
        self.assertNotIn("Traceback", missing.stderr)
        self.assertFalse(json.loads(missing.stderr)["ok"])

        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid.json"
            invalid.write_bytes(b"\xff\xfe")
            bad_input = subprocess.run(
                base
                + [
                    "--model",
                    str(MODEL_PATH),
                    "--receipt",
                    str(MODEL_RECEIPT_PATH),
                    "--input",
                    str(invalid),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(2, bad_input.returncode)
        self.assertNotIn("Traceback", bad_input.stderr)
        self.assertFalse(json.loads(bad_input.stderr)["ok"])


class SyntheticDatasetTests(unittest.TestCase):
    def test_dataset_has_exact_splits_schema_and_no_sensitive_fields(self) -> None:
        forbidden = {
            "address",
            "dob",
            "email",
            "fhir",
            "hl7",
            "mrn",
            "name",
            "obx",
            "order",
            "patient",
            "phone",
            "pid",
            "result",
            "specimen",
            "ssn",
        }
        expected_counts = {"train": 768, "validation": 192, "test": 240}
        sample_ids: set[str] = set()
        for split, expected_count in expected_counts.items():
            path = ROOT / "operational-model" / "data" / f"{split}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(expected_count, len(rows))
            for row in rows:
                self.assertEqual(
                    {"schema", "synthetic", "sample_id", "features", "label"}, set(row)
                )
                self.assertEqual("szl-oac/transport-health-observation/v1", row["schema"])
                self.assertIs(row["synthetic"], True)
                self.assertEqual(set(model.FEATURE_NAMES), set(row["features"]))
                self.assertNotIn(row["sample_id"], sample_ids)
                sample_ids.add(row["sample_id"])
                serialized_keys = {
                    key.lower()
                    for mapping in (row, row["features"], row["label"])
                    for key in mapping
                }
                self.assertFalse(serialized_keys & forbidden)
                model.normalize_features(row["features"])

    def test_dataset_receipt_hashes_every_split(self) -> None:
        receipt = json.loads(DATASET_RECEIPT_PATH.read_text(encoding="utf-8"))
        self.assertFalse(receipt["contains_phi"])
        self.assertFalse(receipt["contains_clinical_results"])
        self.assertEqual("synthetic_only", receipt["generated_data"])
        self.assertEqual({"train": 768, "validation": 192, "test": 240}, receipt["split_rows"])
        for relative, expected_hash in receipt["files"].items():
            path = ROOT / "operational-model" / relative
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(expected_hash, observed)

    def test_model_receipt_labels_metrics_as_synthetic_only(self) -> None:
        receipt = json.loads(MODEL_RECEIPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(model.RECEIPT_SCHEMA, receipt["schema"])
        self.assertEqual(
            "fixed_seed_synthetic_splits_only_not_production_validation",
            receipt["metrics_scope"],
        )
        self.assertEqual({"train", "validation", "test"}, set(receipt["metrics"]))
        for split in receipt["metrics"].values():
            self.assertGreaterEqual(split["roc_auc"], 0.5)
            self.assertLessEqual(split["roc_auc"], 1.0)

    def test_all_generated_artifacts_are_byte_reproducible(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(ROOT / "tools" / "train_operational_health_model.py"),
                "--verify",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)
        self.assertTrue(json.loads(proc.stdout)["ok"])

    def test_hugging_face_staging_manifests_are_closed(self) -> None:
        model_stage = ROOT / "huggingface" / "model" / "oac-clinical-transport-health-v1"
        dataset_stage = (
            ROOT
            / "huggingface"
            / "dataset"
            / "oac-clinical-transport-observability-synthetic"
        )
        observed_model = {
            path.relative_to(model_stage).as_posix()
            for path in model_stage.rglob("*")
            if path.is_file()
        }
        observed_dataset = {
            path.relative_to(dataset_stage).as_posix()
            for path in dataset_stage.rglob("*")
            if path.is_file()
        }
        self.assertEqual(
            {
                "LICENSE",
                "README.md",
                "artifact_receipt.json",
                "example_input.json",
                "model.json",
                "oac_operational_health.py",
            },
            observed_model,
        )
        self.assertEqual(
            {
                "LICENSE",
                "README.md",
                "data/test.jsonl",
                "data/train.jsonl",
                "data/validation.jsonl",
                "dataset_receipt.json",
                "schema.json",
                "training_source_snapshot.py",
            },
            observed_dataset,
        )


if __name__ == "__main__":
    unittest.main()
