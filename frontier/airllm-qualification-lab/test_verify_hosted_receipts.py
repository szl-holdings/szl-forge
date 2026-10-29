"""SYNTHETIC receipt fixtures; no model execution or qualification evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import hosted_cpu_smoke as smoke
import verify_hosted_receipts as verifier


SOURCE_SHA = "a" * 40
HARNESS_SHA256 = hashlib.sha256(Path(smoke.__file__).read_bytes()).hexdigest()
RUN_ID = "123456789"
RUN_ATTEMPT = "1"


def encode(document):
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")


def fixture(profile=smoke.DEFAULT_PROFILE):
    """Make declarations shaped like hosted output, without reading any weights."""
    model_id, revision, license_card, files = smoke._model_profile(profile)
    generation = {"model_profile": profile, "do_sample": False,
                  "max_new_tokens": 1, "seed": 0, "seed_applied": False}
    common = {
        "schema": smoke.SCHEMA,
        "captured_at_utc": "2026-10-29T00:00:00+00:00",
        "evidence_class": "MEASURED", "disposition": "MEASURED",
        "model_profile": profile, "model_id": model_id,
        "model_revision": revision, "base_revision": revision,
        "license_card_metadata": license_card,
        "dataset_sha256": smoke.INPUT_DATASET_SHA256,
        "dataset_evidence_class": smoke.DATASET_EVIDENCE_CLASS,
        "dataset_scope": smoke.DATASET_SCOPE,
        "input_sha256": hashlib.sha256(smoke.PROMPT.encode("utf-8")).hexdigest(),
        "harness_sha256": HARNESS_SHA256,
        "seed": 0, "seed_scope": "PYTHON_AND_TORCH_GREEDY_WORKER_ONLY",
        "generation": generation,
        "loss": {"evidence_class": "UNAVAILABLE", "status": "NOT_RUN", "reason": "INFERENCE_ONLY"},
        "airllm_source_commit": smoke.AIRLLM_SOURCE_COMMIT,
        "delete_original": False, "training_eligible": False,
        "publication_eligible": False, "execution_authority": False,
        "held_out_gate": "NOT_RUN", "numerical_parity": "NOT_RUN",
        "signing_status": "UNSIGNED", "model_quality": "UNKNOWN",
        "source": {"repository": "szl-holdings/szl-forge", "source_sha": SOURCE_SHA,
                   "run_id": RUN_ID, "run_attempt": RUN_ATTEMPT},
        "hardware": {"runner_os": "Linux", "runner_arch": "X64",
                     "cpu_model": "SYNTHETIC_TEST_CPU", "logical_cpu_count": 4,
                     "accelerator": "CPU_ONLY_NO_GPU_REQUESTED"},
        "snapshot": {"model_profile": profile, "model_type": "llama",
                     "architecture": "LlamaForCausalLM", "tokenizer_class": "GPT2Tokenizer",
                     "file_bytes": {name: size for name, (size, _) in files.items()},
                     "file_sha256": {name: digest for name, (_, digest) in files.items()}},
        "capacity_before": {"observed_free_bytes": {"disk": 20 * smoke.GIB, "ram": 8 * smoke.GIB},
                            "required_free_bytes": {"disk": 8 * smoke.GIB, "ram": 4 * smoke.GIB},
                            "bound_class": "MODELED",
                            "bound_scope": "CONSERVATIVE_ADMISSION_NOT_MEASURED_CONSUMPTION"},
    }
    prepare = deepcopy(common)
    prepare.update({"stage": "PREPARE", "measurement_scope": "PINNED_SOURCE_BYTES_PREPARED_ONLY",
                    "inference_run": "UNAVAILABLE", "inference_attempted": False,
                    "prepare_receipt_sha256": "UNAVAILABLE", "previous_receipt_sha256": "UNAVAILABLE",
                    "wheel_sha256": smoke.WHEEL_SHA256, "task_cache_root": "/synthetic/task/cache",
                    "do_sample": False, "max_new_tokens": 1, "seed_applied": False,
                    "metrics": {"evidence_class": "UNAVAILABLE", "generated_token_count": None,
                                "elapsed_seconds": None, "peak_rss_kib": None, "token_ids_sha256": None}})
    run = deepcopy(common)
    run["generation"]["seed_applied"] = True
    run.update({"stage": "RUN", "measurement_scope": "HOSTED_CPU_ONE_TOKEN_SMOKE_ONLY",
                "inference_run": "MEASURED", "inference_attempted": True, "worker_exit_code": 0,
                "airllm": {"version": smoke.AIRLLM_VERSION, "wheel_sha256": smoke.WHEEL_SHA256,
                           "source_commit": smoke.AIRLLM_SOURCE_COMMIT, "installed_package_files_matched": 12},
                "dependency_versions": dict(smoke.DEPENDENCIES),
                "dependency_integrity": "VERSION_OBSERVED_HASH_CLOSURE_NOT_VERIFIED",
                "capacity_after": {"disk_free_bytes": 19 * smoke.GIB, "ram_available_bytes": 7 * smoke.GIB}})
    run["capacity_before"]["required_free_bytes"]["disk"] = 6 * smoke.GIB
    metrics = {"generated_token_count": 1, "elapsed_seconds": 1.25,
               "peak_rss_kib": 123456, "token_ids_sha256": hashlib.sha256(b"[7,9]").hexdigest()}
    run["metrics"] = {"evidence_class": "MEASURED", **metrics}
    run["worker_result"] = {
        "status": "MEASURED", "scope": "HOSTED_CPU_ONE_TOKEN_SMOKE_ONLY", **metrics,
        **run["generation"], "input_sha256": run["input_sha256"],
        "dataset_sha256": run["dataset_sha256"], "harness_sha256": HARNESS_SHA256,
        "model_quality": "UNKNOWN", "numerical_parity": "NOT_RUN", "execution_authority": False,
    }
    return prepare, run


def chained(prepare, run):
    raw = encode(prepare)
    result = deepcopy(run)
    result["prepare_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    result["previous_receipt_sha256"] = result["prepare_receipt_sha256"]
    return raw, encode(result)


def expectations(profile=smoke.DEFAULT_PROFILE):
    return {"expected_source_sha": SOURCE_SHA, "expected_profile": profile,
            "expected_run_id": RUN_ID, "expected_run_attempt": RUN_ATTEMPT,
            "expected_harness_sha256": HARNESS_SHA256}


def change(document, keys, value):
    target = document
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


class HostedReceiptReadbackTests(unittest.TestCase):
    def verify(self, prepare, run, **overrides):
        expected = expectations(prepare.get("model_profile", smoke.DEFAULT_PROFILE))
        expected.update(overrides)
        return verifier.verify_receipts(*chained(prepare, run), **expected)

    def test_both_profile_pairs_accept_only_as_unsigned_authority_free_bindings(self):
        for profile in smoke.PROFILE_NAMES:
            with self.subTest(profile=profile):
                prepare, run = fixture(profile)
                result = self.verify(prepare, run)
                for key in ("training_eligible", "publication_eligible", "execution_authority",
                            "signature_verified", "inference_replayed"):
                    self.assertIs(result[key], False)
                self.assertEqual(result["model_qualification"], "UNKNOWN")

    def test_both_chain_fields_bind_exact_prepare_bytes(self):
        prepare, run = fixture()
        raw, run_raw = chained(prepare, run)
        for field in ("prepare_receipt_sha256", "previous_receipt_sha256"):
            changed = json.loads(run_raw)
            changed[field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(verifier.ReceiptError):
                verifier.verify_receipts(raw, encode(changed), **expectations())
        # Semantically identical JSON is still different raw evidence.
        with self.assertRaises(verifier.ReceiptError):
            verifier.verify_receipts(raw + b"\n", run_raw, **expectations())

    def test_external_expected_identity_cannot_be_self_asserted_by_receipts(self):
        prepare, run = fixture()
        wrong = {"expected_source_sha": "c" * 40, "expected_profile": smoke.PROFILE_360M,
                 "expected_run_id": "987654321", "expected_run_attempt": "2",
                 "expected_harness_sha256": "d" * 64}
        for field, value in wrong.items():
            with self.subTest(field=field), self.assertRaises(verifier.ReceiptError):
                self.verify(prepare, run, **{field: value})

    def test_identity_and_pinned_model_fields_reject_drift_in_either_receipt(self):
        cases = [(('source', 'repository'), "fork/forge"), (('source', 'source_sha'), "c" * 40),
                 (('source', 'run_id'), "2"), (('source', 'run_attempt'), "2"),
                 (('model_profile',), smoke.PROFILE_360M), (('model_id',), smoke.MODEL_360M_ID),
                 (('model_revision',), "c" * 40), (('base_revision',), "c" * 40),
                 (('harness_sha256',), "c" * 64), (('input_sha256',), "c" * 64),
                 (('dataset_sha256',), "c" * 64), (('dataset_evidence_class',), "MEASURED"),
                 (('dataset_scope',), "TRAINING"), (('airllm_source_commit',), "c" * 40)]
        for stage in (0, 1):
            for keys, value in cases:
                pair = fixture()
                change(pair[stage], keys, value)
                with self.subTest(stage=stage, keys=keys), self.assertRaises(verifier.ReceiptError):
                    verifier.verify_receipts(*chained(*pair), **expectations())

    def test_snapshot_exact_file_set_sizes_and_hashes_are_required(self):
        for profile in smoke.PROFILE_NAMES:
            for stage in (0, 1):
                for map_name in ("file_bytes", "file_sha256"):
                    for mutation in ("missing", "extra", "wrong"):
                        pair = fixture(profile)
                        mapping = pair[stage]["snapshot"][map_name]
                        if mutation == "missing":
                            del mapping["model.safetensors"]
                        elif mutation == "extra":
                            mapping["modeling_custom.py"] = 1 if map_name == "file_bytes" else "0" * 64
                        else:
                            mapping["model.safetensors"] = 1 if map_name == "file_bytes" else "0" * 64
                        with self.subTest(profile=profile, stage=stage, map_name=map_name, mutation=mutation):
                            with self.assertRaises(verifier.ReceiptError):
                                self.verify(*pair)

    def test_wheel_and_dependency_bindings_are_required(self):
        cases = [(0, ("wheel_sha256",), "0" * 64),
                 (1, ("airllm", "wheel_sha256"), "0" * 64),
                 (1, ("airllm", "version"), "4.0.1"),
                 (1, ("airllm", "source_commit"), "0" * 40),
                 (1, ("dependency_versions", "torch"), "2.7.0")]
        for stage, keys, value in cases:
            pair = fixture()
            change(pair[stage], keys, value)
            with self.subTest(keys=keys), self.assertRaises(verifier.ReceiptError):
                self.verify(*pair)

    def test_seed_stage_and_worker_bindings_must_agree(self):
        cases = [(0, ("generation", "seed_applied"), True),
                 (0, ("seed_applied",), True), (1, ("generation", "seed_applied"), False),
                 (1, ("generation", "seed_applied"), "UNKNOWN"),
                 (1, ("worker_result", "seed_applied"), False),
                 (1, ("worker_result", "seed"), 1), (1, ("worker_result", "do_sample"), True),
                 (1, ("worker_result", "max_new_tokens"), 2),
                 (1, ("worker_result", "model_profile"), smoke.PROFILE_360M),
                 (1, ("worker_result", "harness_sha256"), "c" * 64),
                 (1, ("worker_result", "input_sha256"), "c" * 64),
                 (1, ("worker_result", "dataset_sha256"), "c" * 64)]
        for stage, keys, value in cases:
            pair = fixture()
            change(pair[stage], keys, value)
            with self.subTest(keys=keys, value=value), self.assertRaises(verifier.ReceiptError):
                self.verify(*pair)

    def test_authority_flags_require_actual_false_and_unsigned_status(self):
        for stage in (0, 1):
            for key in ("training_eligible", "publication_eligible", "execution_authority", "delete_original"):
                for value in (True, 0, "false", None):
                    pair = fixture()
                    pair[stage][key] = value
                    with self.subTest(stage=stage, key=key, value=value), self.assertRaises(verifier.ReceiptError):
                        self.verify(*pair)
            pair = fixture()
            pair[stage]["signing_status"] = "SIGNED"
            with self.subTest(stage=stage), self.assertRaises(verifier.ReceiptError):
                self.verify(*pair)
        for value in (True, 0, "false", None):
            pair = fixture()
            pair[1]["worker_result"]["execution_authority"] = value
            with self.subTest(worker_authority=value), self.assertRaises(verifier.ReceiptError):
                self.verify(*pair)

    def test_missing_required_fields_fail_closed(self):
        for stage in (0, 1):
            for key in ("source", "model_profile", "base_revision", "snapshot", "hardware",
                        "harness_sha256", "dataset_sha256", "generation", "loss", "metrics",
                        "training_eligible", "publication_eligible", "execution_authority", "signing_status"):
                pair = fixture()
                del pair[stage][key]
                with self.subTest(stage=stage, key=key), self.assertRaises(verifier.ReceiptError):
                    self.verify(*pair)

    def test_hardware_identity_is_complete_and_bound_across_stages(self):
        for stage in (0, 1):
            for key in ("runner_arch", "cpu_model", "logical_cpu_count"):
                pair = fixture()
                del pair[stage]["hardware"][key]
                with self.subTest(stage=stage, missing=key), self.assertRaises(verifier.ReceiptError):
                    self.verify(*pair)
            for key, value in (("runner_arch", "ARM64"), ("cpu_model", "OTHER_SYNTHETIC_CPU"),
                               ("logical_cpu_count", 8)):
                pair = fixture()
                pair[stage]["hardware"][key] = value
                with self.subTest(stage=stage, mismatch=key), self.assertRaises(verifier.ReceiptError):
                    self.verify(*pair)
        for value in (-1, 0, True, False):
            pair = fixture()
            for receipt in pair:
                receipt["hardware"]["logical_cpu_count"] = value
            with self.subTest(cpu_count=value), self.assertRaises(verifier.ReceiptError):
                self.verify(*pair)
        pair = fixture()
        for receipt in pair:
            receipt["hardware"]["cpu_model"] = "UNAVAILABLE"
        self.assertIs(self.verify(*pair)["execution_authority"], False)

    def test_optional_signature_and_replay_claims_cannot_assert_verification(self):
        for stage, worker in ((0, False), (1, False), (1, True)):
            for key in ("signature_verified", "inference_replayed"):
                for value in (False, True):
                    pair = fixture()
                    target = pair[stage]["worker_result"] if worker else pair[stage]
                    target[key] = value
                    with self.subTest(stage=stage, worker=worker, key=key, value=value):
                        if value:
                            with self.assertRaises(verifier.ReceiptError):
                                self.verify(*pair)
                        else:
                            self.assertIs(self.verify(*pair)[key], False)

    def test_non_success_or_inconsistent_status_cannot_pass(self):
        cases = [(0, ("stage",), "RUN"), (1, ("stage",), "PREPARE"),
                 (0, ("evidence_class",), "DECLARED"), (1, ("disposition",), "BLOCKED"),
                 (1, ("inference_run",), "UNKNOWN"), (1, ("inference_attempted",), False),
                 (1, ("worker_exit_code",), 1), (1, ("worker_result", "status"), "BLOCKED"),
                 (1, ("worker_result", "scope"), "MODEL_QUALIFICATION"),
                 (1, ("metrics", "evidence_class"), "UNAVAILABLE"),
                 (0, ("loss", "status"), "MEASURED"),
                 (1, ("held_out_gate",), "PASSED"), (1, ("numerical_parity",), "PASSED")]
        for stage, keys, value in cases:
            pair = fixture()
            change(pair[stage], keys, value)
            with self.subTest(keys=keys, value=value), self.assertRaises(verifier.ReceiptError):
                self.verify(*pair)

    def test_boolean_integer_confusion_is_rejected(self):
        cases = [(0, ("seed",), False), (0, ("generation", "seed"), False),
                 (0, ("max_new_tokens",), True), (1, ("generation", "max_new_tokens"), True),
                 (1, ("worker_exit_code",), False), (1, ("worker_result", "generated_token_count"), True),
                 (1, ("metrics", "generated_token_count"), True),
                 (1, ("metrics", "peak_rss_kib"), True), (1, ("metrics", "elapsed_seconds"), True)]
        for stage, keys, value in cases:
            pair = fixture()
            change(pair[stage], keys, value)
            with self.subTest(keys=keys), self.assertRaises(verifier.ReceiptError):
                self.verify(*pair)

    def test_invalid_and_disagreeing_metrics_fail_closed(self):
        cases = [("generated_token_count", 0), ("generated_token_count", 2),
                 ("elapsed_seconds", -1), ("elapsed_seconds", float("nan")),
                 ("elapsed_seconds", float("inf")), ("elapsed_seconds", 10 ** 400),
                 ("elapsed_seconds", 10 ** 1000), ("peak_rss_kib", -1),
                 ("peak_rss_kib", None), ("peak_rss_kib", 1.5), ("token_ids_sha256", "not-a-hash")]
        for key, value in cases:
            pair = fixture()
            pair[1]["metrics"][key] = value
            pair[1]["worker_result"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(verifier.ReceiptError):
                self.verify(*pair)
        for key, value in (("elapsed_seconds", 2.5), ("peak_rss_kib", 999), ("token_ids_sha256", "d" * 64)):
            pair = fixture()
            pair[1]["metrics"][key] = value
            with self.subTest(disagreement=key), self.assertRaises(verifier.ReceiptError):
                self.verify(*pair)

    def test_strict_json_rejects_duplicate_keys_nonobjects_nonfinite_and_oversize(self):
        raw, run_raw = chained(*fixture())
        bad = [b"[]", b"null", b"false", b"7", b'"receipt"', b"{", b"\xff",
               b'{"value":NaN}', b'{"value":Infinity}']
        for invalid in bad:
            for stage in (0, 1):
                pair = [raw, run_raw]
                pair[stage] = invalid
                with self.subTest(stage=stage, prefix=invalid[:40]), self.assertRaises(verifier.ReceiptError):
                    verifier.verify_receipts(*pair, **expectations())
        # Each of these parses to the valid fixture under permissive json.loads;
        # update the chain after tampering so a hash mismatch cannot mask the parser gap.
        for stage in (0, 1):
            original = (raw, run_raw)[stage]
            duplicates = [
                b'{"schema":' + json.dumps(smoke.SCHEMA).encode() + b"," + original.lstrip()[1:],
                original.replace(b'"run_id": "123456789"',
                                 b'"run_id": "123456789", "run_id": "123456789"', 1),
                b'{"extra":NaN,' + original.lstrip()[1:],
                b'{"extra":Infinity,' + original.lstrip()[1:],
                b'{"extra":1e9999,' + original.lstrip()[1:],
                original + b" " * (1024 * 1024 + 1),
            ]
            for invalid in duplicates:
                pair = [raw, run_raw]
                pair[stage] = invalid
                if stage == 0:
                    adjusted_run = json.loads(run_raw)
                    digest = hashlib.sha256(invalid).hexdigest()
                    adjusted_run["prepare_receipt_sha256"] = digest
                    adjusted_run["previous_receipt_sha256"] = digest
                    pair[1] = encode(adjusted_run)
                with self.subTest(stage=stage, size=len(invalid)), self.assertRaises(verifier.ReceiptError):
                    verifier.verify_receipts(*pair, **expectations())

    def test_cli_readback_pass_and_fail_are_read_only_and_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare = root / "PREPARE.json"
            run = root / "SMOKE.json"
            raw, run_raw = chained(*fixture())
            prepare.write_bytes(raw)
            run.write_bytes(run_raw)
            argv = [str(Path(verifier.__file__)), "--prepare", str(prepare), "--smoke", str(run)]
            for key, value in expectations().items():
                argv.extend(["--" + key.replace("_", "-"), value])
            for failure in (None, "source", "elapsed_overflow"):
                selected = list(argv)
                run.write_bytes(run_raw)
                if failure == "source":
                    selected[selected.index("--expected-source-sha") + 1] = "c" * 40
                elif failure == "elapsed_overflow":
                    overflowing = json.loads(run_raw)
                    overflowing["worker_result"]["elapsed_seconds"] = 10 ** 400
                    overflowing["metrics"]["elapsed_seconds"] = 10 ** 400
                    run.write_bytes(encode(overflowing))
                before = {path.name: path.read_bytes() for path in root.iterdir()}
                output = io.StringIO()
                with patch.object(sys, "argv", selected), redirect_stdout(output), \
                        patch("socket.create_connection", side_effect=AssertionError("network forbidden")), \
                        patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), \
                        patch("subprocess.run", side_effect=AssertionError("subprocess forbidden")), \
                        patch("os.system", side_effect=AssertionError("shell forbidden")), \
                        patch.object(Path, "write_bytes", side_effect=AssertionError("write forbidden")), \
                        patch.object(Path, "write_text", side_effect=AssertionError("write forbidden")), \
                        patch.object(Path, "mkdir", side_effect=AssertionError("mkdir forbidden")):
                    code = verifier.main()
                self.assertEqual(code, 2 if failure else 0)
                self.assertLess(len(output.getvalue()), 8192)
                result = json.loads(output.getvalue())
                self.assertIs(result["execution_authority"], False)
                self.assertIs(result["signature_verified"], False)
                self.assertEqual({path.name: path.read_bytes() for path in root.iterdir()}, before)

    def test_cli_missing_arguments_returns_bounded_blocked_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = verifier.main([])
        self.assertEqual(code, 2)
        self.assertLess(len(output.getvalue()), 8192)
        result = json.loads(output.getvalue())
        self.assertEqual(result["disposition"], "BLOCKED")
        self.assertIs(result["execution_authority"], False)
        self.assertIs(result["signature_verified"], False)


if __name__ == "__main__":
    unittest.main()
