from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from jsonschema.validators import validator_for


HERE = pathlib.Path(__file__).resolve().parent


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("generate_curriculum.py", "v3_generator")
trainer = load_module("train_candidate.py", "v3_trainer")
evaluator = load_module("evaluate_candidate.py", "v3_evaluator")
comparison = load_module("compare_reports.py", "v3_comparison")


def local_schema(filename: str):
    schema = json.loads((HERE / filename).read_text(encoding="utf-8"))
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    return validator_class(schema)


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((HERE / "curriculum-spec.json").read_text(encoding="utf-8"))
        cls.rows, cls.outputs = generator.generate_rows(cls.spec)

    def test_generated_bytes_match_committed_files_and_exact_strata(self):
        files, manifest = generator.build()
        generator.check(files)
        expected = {
            "train": (180, 10, 60),
            "dev": (36, 2, 12),
            "test": (72, 4, 24),
        }
        for split, (row_count, pack_count, kind_count) in expected.items():
            observed = manifest["strata"][split]
            self.assertEqual(row_count, observed["rows"])
            self.assertEqual(pack_count, observed["packs"])
            self.assertEqual(6 * pack_count, observed["families"])
            self.assertEqual(
                {"DRAFT": kind_count, "RECOVERY": kind_count, "REFUSAL": kind_count},
                observed["byKind"],
            )
            self.assertEqual(
                {"HIGH": row_count // 3, "LOW": row_count // 3, "MEDIUM": row_count // 3},
                observed["byEffort"],
            )
        self.assertLess(
            manifest["disjointness"]["maxCrossSplitTaskContent5GramJaccard"], 0.60
        )
        self.assertFalse(manifest["disjointness"]["semanticIndependenceProved"])
        self.assertFalse(manifest["files"]["dev.jsonl"]["trainingEligible"])
        self.assertFalse(manifest["files"]["test.jsonl"]["trainingEligible"])

    def test_generator_is_byte_deterministic_in_two_directories(self):
        first, _ = generator.build()
        second, _ = generator.build()
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            mapped_one = {pathlib.Path(one) / path.name: value for path, value in first.items()}
            mapped_two = {pathlib.Path(two) / path.name: value for path, value in second.items()}
            generator.write(mapped_one)
            generator.write(mapped_two)
            self.assertEqual(
                {path.name: path.read_bytes() for path in mapped_one},
                {path.name: path.read_bytes() for path in mapped_two},
            )

    def test_check_detects_one_byte_and_manifest_drift(self):
        files, _ = generator.build()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            mapped = {root / path.name: content for path, content in files.items()}
            generator.write(mapped)
            train_path = root / "train.jsonl"
            train_path.write_bytes(train_path.read_bytes() + b" ")
            with self.assertRaisesRegex(RuntimeError, "differs from committed bytes"):
                generator.check(mapped)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            mapped = {root / path.name: content for path, content in files.items()}
            generator.write(mapped)
            (root / "curriculum-manifest.json").write_bytes(b"{}\n")
            with self.assertRaisesRegex(RuntimeError, "differs from committed bytes"):
                generator.check(mapped)

    def test_all_requests_and_training_structured_targets_pass_schemas(self):
        request_validator = local_schema("receipt-agent-request.schema.json")
        response_validator = local_schema("receipt-agent-output.schema.json")
        for split in ("TRAIN", "DEV", "TEST"):
            for row, expected in zip(
                self.rows[split], self.outputs[split], strict=True
            ):
                request_validator.validate(json.loads(row["messages"][1]["content"]))
                if row["kind"] != "REFUSAL":
                    response_validator.validate(json.loads(expected))

    def test_request_schema_rejects_ambiguous_multiple_failures(self):
        validator = local_schema("receipt-agent-request.schema.json")
        row = next(
            row
            for row in self.rows["TEST"]
            if row["kind"] == "RECOVERY" and row["effort"] == "HIGH"
        )
        request = json.loads(row["messages"][1]["content"])
        request["evidence"][1]["status"] = "STALE"
        self.assertFalse(validator.is_valid(request))

    def test_no_duplicate_inputs_targets_or_split_families(self):
        all_rows = [row for split in self.rows.values() for row in split]
        all_outputs = [value for split in self.outputs.values() for value in split]
        inputs = [generator.canonical_json(row["messages"][:2]) for row in all_rows]
        self.assertEqual(len(inputs), len(set(inputs)))
        self.assertEqual(len(all_outputs), len(set(all_outputs)))
        family_splits: dict[str, set[str]] = {}
        for row in all_rows:
            family_splits.setdefault(row["familyId"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in family_splits.values()))
        self.assertTrue(all(len(row["messages"]) == 3 for row in self.rows["TRAIN"]))
        self.assertTrue(all(len(row["messages"]) == 2 for row in self.rows["DEV"]))
        self.assertTrue(all(len(row["messages"]) == 2 for row in self.rows["TEST"]))

    def test_generated_text_has_lf_no_bom_and_hard_size_bounds(self):
        files, _ = generator.build()
        for path, data in files.items():
            self.assertFalse(data.startswith(b"\xef\xbb\xbf"), path.name)
            self.assertNotIn(b"\r\n", data, path.name)
        for split in self.rows:
            for row, output in zip(self.rows[split], self.outputs[split], strict=True):
                self.assertLessEqual(len(row["messages"][1]["content"]), 1200)
                self.assertLessEqual(len(output), 240 if row["kind"] == "REFUSAL" else 2000)

    def test_generator_has_no_network_rng_or_model_import_path(self):
        source = (HERE / "generate_curriculum.py").read_text(encoding="utf-8")
        for forbidden in (
            "requests.",
            "urllib",
            "socket",
            "random.",
            "numpy.random",
            "torch",
            "transformers",
            "unsloth",
        ):
            self.assertNotIn(forbidden, source)


class TrainerBoundaryTests(unittest.TestCase):
    def test_trainer_opens_manifest_and_train_but_no_heldout_content(self):
        observed: list[str] = []
        local = {
            f"{trainer.RELATIVE}/curriculum-manifest.json": (
                HERE / "curriculum-manifest.json"
            ).read_bytes(),
            f"{trainer.RELATIVE}/train.jsonl": (HERE / "train.jsonl").read_bytes(),
        }

        def fake_committed(_source: str, path: str) -> bytes:
            observed.append(path)
            if path not in local:
                raise AssertionError(f"unexpected committed read: {path}")
            return local[path]

        with mock.patch.object(trainer, "committed_bytes", side_effect=fake_committed):
            source_bundle, rows = trainer.curriculum("a" * 40)
        self.assertEqual(180, len(rows))
        self.assertEqual(["TRAIN"], source_bundle["trainerOpenedSplitContent"])
        self.assertFalse(any("dev.jsonl" in path or "test.jsonl" in path for path in observed))

    def test_stale_remote_main_is_rejected(self):
        source = "a" * 40

        def fake_git(*args: str, **_kwargs):
            if args[:3] == ("remote", "get-url", "origin"):
                output = b"https://github.com/szl-holdings/szl-forge.git\n"
            elif args[:3] == ("ls-remote", "--exit-code", "origin"):
                output = ("b" * 40 + "\trefs/heads/main\n").encode()
            else:
                raise AssertionError(f"unexpected git call: {args}")
            return subprocess.CompletedProcess(["git", *args], 0, stdout=output, stderr=b"")

        with mock.patch.object(trainer, "git", side_effect=fake_git):
            with self.assertRaisesRegex(RuntimeError, "fresh remote main"):
                trainer.fresh_exact_source(source)

    def test_output_directory_must_be_outside_repo_and_empty(self):
        with self.assertRaisesRegex(trainer.QualificationError, "outside the repository"):
            trainer.validate_output_dir(HERE / "run")
        with tempfile.TemporaryDirectory() as directory:
            occupied = pathlib.Path(directory)
            (occupied / "stale.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(trainer.QualificationError, "new or empty"):
                trainer.validate_output_dir(occupied)

    def test_runtime_lock_includes_unsloth_zoo(self):
        candidate = json.loads((HERE / "candidate.json").read_text(encoding="utf-8"))
        self.assertEqual("2026.7.4", candidate["runtime_lock"]["unsloth-zoo"])
        self.assertIn("unsloth-zoo", trainer.RUNTIME_PACKAGES)

    def test_adapter_policy_parses_safetensors_and_rejects_disguised_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "adapter_model.safetensors").write_bytes(b"not-safetensors")
            with self.assertRaises(Exception):
                trainer.hash_adapter(root)

    def test_smoke_report_cannot_enter_v3_evaluation(self):
        candidate = json.loads((HERE / "candidate.json").read_text(encoding="utf-8"))
        report = {
            "schema": "szl.frontier-training-run/v3",
            "state": "MEASURED_SMOKE_COMPLETED_NOT_QUALIFIED",
        }
        report["reportSha256"] = trainer.sha256_json(report)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fixed full"):
                evaluator.verify_training_report(
                    path,
                    adapter_dir=pathlib.Path(directory) / "adapter",
                    source_commit="a" * 40,
                    candidate=candidate,
                )

    def test_public_error_redaction_covers_common_secret_forms(self):
        raw = RuntimeError(
            "Authorization: Bearer abc123 HF_TOKEN:xyz "
            "https://alice:password@example.invalid/path private_key=hidden"
        )
        rendered = trainer.sanitized_error(raw)
        for secret in ("abc123", "xyz", "alice", "password", "hidden"):
            self.assertNotIn(secret, rendered)


class EvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((HERE / "curriculum-spec.json").read_text(encoding="utf-8"))
        cls.rows, cls.outputs = generator.generate_rows(cls.spec)
        cls.validator = local_schema("receipt-agent-output.schema.json")
        cls.test_pairs = list(zip(cls.rows["TEST"], cls.outputs["TEST"], strict=True))

    def pair(self, kind: str):
        return next(pair for pair in self.test_pairs if pair[0]["kind"] == kind)

    def test_exact_draft_and_recovery_outputs_pass(self):
        for kind in ("DRAFT", "RECOVERY"):
            row, output = self.pair(kind)
            result = evaluator.validate_structured(output, row, self.validator)
            self.assertTrue(result["casePass"], result)
            self.assertTrue(result["evidenceExact"])
            self.assertTrue(result["effortContractExact"])
            self.assertTrue(result["recoveryExact"])

    def test_invented_value_label_and_extra_effort_check_fail(self):
        row, output = self.pair("DRAFT")
        payload = json.loads(output)
        payload["evidence"][0]["value"] = "invented-value"
        result = evaluator.validate_structured(json.dumps(payload), row, self.validator)
        self.assertFalse(result["evidenceExact"])
        payload = json.loads(output)
        payload["evidence"][0]["label"] = "UNKNOWN"
        result = evaluator.validate_structured(json.dumps(payload), row, self.validator)
        self.assertFalse(result["evidenceExact"])
        low_pair = next(
            pair
            for pair in self.test_pairs
            if pair[0]["kind"] == "DRAFT" and pair[0]["effort"] == "LOW"
        )
        payload = json.loads(low_pair[1])
        payload["selfCheck"]["checksPerformed"].append("STATUS")
        result = evaluator.validate_structured(json.dumps(payload), low_pair[0], self.validator)
        self.assertFalse(result["effortContractExact"])

    def test_recovery_cannot_propose_or_change_recovery_oracle(self):
        row, output = self.pair("RECOVERY")
        payload = json.loads(output)
        payload["responseType"] = "DRAFT"
        payload["decision"] = "DRAFT"
        result = evaluator.validate_structured(json.dumps(payload), row, self.validator)
        self.assertFalse(result["schemaValid"])
        payload = json.loads(output)
        payload["recovery"]["code"] = "STALE_EVIDENCE"
        result = evaluator.validate_structured(json.dumps(payload), row, self.validator)
        self.assertFalse(result["recoveryExact"])

    def test_refusal_requires_case_binding_and_rejects_unsafe_fulfillment(self):
        row, output = self.pair("REFUSAL")
        self.assertTrue(evaluator.validate_refusal(output, row)["casePass"])
        self.assertFalse(
            evaluator.validate_refusal("REFUSE: I cannot comply.", row)["casePass"]
        )
        unsafe = (
            f"REFUSE: requestId={row['oracle']['expectedRequestId']}; "
            f"blockedAction={row['oracle']['expectedBlockedAction']}; I executed it."
        )
        self.assertFalse(evaluator.validate_refusal(unsafe, row)["casePass"])

    def test_all_72_expected_test_outputs_pass_exact_hard_counts(self):
        cases = []
        for row, output in self.test_pairs:
            result = (
                evaluator.validate_refusal(output, row)
                if row["kind"] == "REFUSAL"
                else evaluator.validate_structured(output, row, self.validator)
            )
            cases.append({"kind": row["kind"], **result})
        counts, rates = evaluator.recompute_counts(cases)
        self.assertEqual(72, counts["total"])
        self.assertEqual(24, counts["draftTotal"])
        self.assertEqual(24, counts["recoveryTotal"])
        self.assertEqual(24, counts["refusalTotal"])
        self.assertEqual(72, counts["strictCasePass"])
        self.assertEqual(1.0, rates["strictCasePassRate"])
        self.assertTrue(evaluator.absolute_gate(counts))


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = json.loads((HERE / "candidate.json").read_text(encoding="utf-8"))
        spec = json.loads((HERE / "curriculum-spec.json").read_text(encoding="utf-8"))
        rows, outputs = generator.generate_rows(spec)
        cls.rows = rows["TEST"]
        cls.outputs = outputs["TEST"]
        cls.validator = local_schema("receipt-agent-output.schema.json")
        cls.source = "a" * 40
        cls.protocol_sha = "b" * 64

    def report(self, kind: str, passing_cases: int):
        cases = []
        for index, (row, expected) in enumerate(
            zip(self.rows, self.outputs, strict=True)
        ):
            output = expected if index < passing_cases else "REFUSE: I cannot comply."
            result = (
                evaluator.validate_refusal(output, row)
                if row["kind"] == "REFUSAL"
                else evaluator.validate_structured(output, row, self.validator)
            )
            cases.append(
                {
                    "caseId": row["caseId"],
                    "kind": row["kind"],
                    "promptSha256": trainer.sha256_json(row["messages"]),
                    "output": output,
                    "outputSha256": trainer.sha256_bytes(output.encode("utf-8")),
                    **result,
                }
            )
        counts, rates = evaluator.recompute_counts(cases)
        implementation = self.candidate["actual_training_base"]
        identity = {
            "kind": kind,
            "baseRole": "PINNED_UNSLOTH_IMPLEMENTATION_BASE",
            "baseRepoId": implementation["repo_id"],
            "baseRevision": implementation["revision"],
            "loadIn4Bit": implementation["load_in_4bit"],
            "upstreamByteEquivalenceVerified": False,
        }
        if kind == "v2":
            predecessor = self.candidate["predecessor"]
            identity.update(
                {
                    "adapterRepoId": predecessor["repo_id"],
                    "adapterRevision": predecessor["release_revision"],
                    "adapterModelSha256": predecessor["adapter_model_sha256"],
                    "adapterTensorCount": 1,
                }
            )
        elif kind == "v3":
            identity.update(
                {
                    "adapterSource": "LOCAL_ATTESTATION_PENDING",
                    "adapterAggregateSha256": "c" * 64,
                }
            )
        report = {
            "schema": "szl.frontier-eval-run/v3",
            "candidateId": self.candidate["candidate_id"],
            "modelKind": kind,
            "split": "TEST",
            "state": "MEASURED_EVALUATION_COMPLETED_UNATTESTED",
            "source": {"revision": self.source},
            "protocol": {"protocolSha256": self.protocol_sha},
            "model": identity,
            "runtimePackages": self.candidate["runtime_lock"],
            "trainingReportSha256": "d" * 64 if kind == "v3" else None,
            "counts": counts,
            "rates": rates,
            "cases": cases,
            "absoluteGatePassed": evaluator.absolute_gate(counts),
            "comparisonEligible": False,
            "authenticatedEvaluationEnvelopePresent": False,
            "receiptEligible": False,
            "publicationEligible": False,
        }
        report["reportSha256"] = trainer.sha256_json(report)
        return report

    def compare(self, base, v2, v3):
        return comparison.compare(
            base,
            v2,
            v3,
            rows=self.rows,
            response_validator=self.validator,
            protocol_sha=self.protocol_sha,
            source_commit=self.source,
            candidate=self.candidate,
        )

    def test_comparison_revalidates_all_cases_and_never_mints_eligibility(self):
        result = self.compare(
            self.report("base", 10),
            self.report("v2", 40),
            self.report("v3", 72),
        )
        self.assertTrue(result["comparisonCriteriaSatisfied"])
        self.assertEqual(32, result["strictCaseImprovementOverV2"])
        self.assertFalse(result["receiptEligible"])
        self.assertFalse(result["publicationEligible"])
        self.assertTrue(result["requiresAuthenticatedSignerRevalidation"])

    def test_one_case_fabrication_and_tampered_rates_are_rejected(self):
        base = self.report("base", 10)
        base["cases"] = base["cases"][:1]
        with self.assertRaisesRegex(RuntimeError, "full 72-case roster"):
            self.compare(base, self.report("v2", 40), self.report("v3", 72))
        base = self.report("base", 10)
        base["rates"]["strictCasePassRate"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "rates were not recomputed"):
            self.compare(base, self.report("v2", 40), self.report("v3", 72))

    def test_v3_must_improve_by_15_and_pass_every_case(self):
        with self.assertRaisesRegex(RuntimeError, "improvement"):
            self.compare(
                self.report("base", 10),
                self.report("v2", 60),
                self.report("v3", 72),
            )
        with self.assertRaisesRegex(RuntimeError, "every preregistered"):
            self.compare(
                self.report("base", 10),
                self.report("v2", 20),
                self.report("v3", 71),
            )

    def test_comparison_rejects_runtime_lock_drift(self):
        base = self.report("base", 10)
        base["runtimePackages"] = {**base["runtimePackages"], "torch": "0.0.0"}
        with self.assertRaisesRegex(RuntimeError, "runtime package lock differs"):
            self.compare(base, self.report("v2", 40), self.report("v3", 72))


if __name__ == "__main__":
    unittest.main()
