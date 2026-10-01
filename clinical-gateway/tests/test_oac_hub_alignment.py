"""Adversarial, network-free tests for the immutable OAC Hub alignment gate."""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from urllib.parse import unquote, urlsplit


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "verify_hub_alignment.py"
SPEC = importlib.util.spec_from_file_location("oac_verify_hub_alignment", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
alignment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alignment)

REVISION = "1234567890abcdef1234567890abcdef12345678"
OTHER_REVISION = "abcdef1234567890abcdef1234567890abcdef12"
MODEL_ID = "SZLHOLDINGS/oac-system-health-v1"
DATASET_ID = "SZLHOLDINGS/oac-clinical-transport-observability-synthetic"


class FakeHub:
    """Serve explicit in-memory metadata/files; an unexpected URL is an error."""

    def __init__(self, *, kind: str = "model", metadata: dict | None = None) -> None:
        self.kind = kind
        self.repo_id = MODEL_ID if kind == "model" else DATASET_ID
        paths = alignment.MODEL_FILES if kind == "model" else alignment.DATASET_FILES
        self.files = {path: ("fixture:" + path).encode("utf-8") for path in paths}
        self.metadata = metadata if metadata is not None else {
            "id": self.repo_id,
            "sha": REVISION,
            "siblings": [{"rfilename": path} for path in sorted(self.files)],
        }
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
            raise AssertionError(f"unexpected origin: {url}")
        api_kind = "models" if self.kind == "model" else "datasets"
        if parsed.path == f"/api/{api_kind}/{self.repo_id}/revision/{REVISION}":
            return json.dumps(self.metadata).encode("utf-8")
        prefix = "" if self.kind == "model" else "datasets/"
        resolve_prefix = f"/{prefix}{self.repo_id}/resolve/{REVISION}/"
        if parsed.path.startswith(resolve_prefix):
            path = unquote(parsed.path[len(resolve_prefix):])
            if path in self.files:
                return self.files[path]
        raise AssertionError(f"unexpected or unpinned URL: {url}")


class RevisionTests(unittest.TestCase):
    def test_full_lowercase_revision_is_preserved(self) -> None:
        self.assertEqual(REVISION, alignment.require_revision(REVISION))

    def test_refs_partial_uppercase_and_nonstring_values_are_rejected(self) -> None:
        invalid = [
            "main", "refs/heads/main", "HEAD", "", REVISION[:7], REVISION[:-1],
            REVISION + "0", REVISION.upper(), "g" * 40, " " + REVISION,
            REVISION + "\n", "../" + REVISION, None, True, 40, b"a" * 40,
        ]
        for revision in invalid:
            with self.subTest(revision=revision), self.assertRaises(alignment.AlignmentError):
                alignment.require_revision(revision)


class StrictJsonTests(unittest.TestCase):
    def test_valid_nested_json_is_decoded(self) -> None:
        self.assertEqual(
            {"items": [1, 0.5, None, True, {"name": "OAC"}]},
            alignment.strict_json(b'{"items":[1,0.5,null,true,{"name":"OAC"}]}'),
        )

    def test_duplicate_keys_are_rejected_at_every_nesting_level(self) -> None:
        for raw in (
            b'{"sha":"first","sha":"second"}',
            b'{"nested":{"sha":"first","sha":"second"}}',
            b'[{"rfilename":"model.json","rfilename":"README.md"}]',
        ):
            with self.subTest(raw=raw), self.assertRaises(alignment.AlignmentError):
                alignment.strict_json(raw)

    def test_all_nonfinite_numbers_are_rejected(self) -> None:
        for token in (b"NaN", b"Infinity", b"-Infinity", b"1e999", b"-1e999"):
            for raw in (token, b'{"value":[' + token + b"]}"):
                with self.subTest(raw=raw), self.assertRaises(alignment.AlignmentError):
                    alignment.strict_json(raw)

    def test_malformed_json_and_invalid_utf8_fail_with_alignment_error(self) -> None:
        for raw in (b"", b"{", b'{}{}', b'{"x":}', b'"\xff"'):
            with self.subTest(raw=raw), self.assertRaises(alignment.AlignmentError):
                alignment.strict_json(raw)

    def test_deeply_nested_json_is_a_controlled_parse_failure(self) -> None:
        raw = b"[" * 100_000 + b"0" + b"]" * 100_000
        with self.assertRaises(alignment.AlignmentError):
            alignment.strict_json(raw)


class ClosedFileSetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = {"README.md", "model.json"}

    def test_exact_set_is_accepted_independent_of_order(self) -> None:
        alignment.validate_paths(["model.json", "README.md"], self.expected)

    def test_missing_and_unexpected_files_are_rejected(self) -> None:
        for paths in (["README.md"], ["README.md", "model.json", "extra.py"], []):
            with self.subTest(paths=paths), self.assertRaises(alignment.AlignmentError):
                alignment.validate_paths(paths, self.expected)

    def test_duplicate_paths_are_rejected_before_set_comparison(self) -> None:
        with self.assertRaises(alignment.AlignmentError):
            alignment.validate_paths(["README.md", "model.json", "model.json"], self.expected)

    def test_hub_metadata_requires_explicit_allowance(self) -> None:
        paths = ["README.md", "model.json", ".gitattributes"]
        with self.assertRaises(alignment.AlignmentError):
            alignment.validate_paths(paths, self.expected)
        alignment.validate_paths(paths, self.expected, allow_hub_metadata=True)

    def test_hub_metadata_allowance_does_not_allow_duplicates_or_other_paths(self) -> None:
        for extras in (
            [".gitattributes", ".gitattributes"], ["data/.gitattributes"],
            [".gitignore"], [".gitattributes", "unexpected.py"],
        ):
            with self.subTest(extras=extras), self.assertRaises(alignment.AlignmentError):
                alignment.validate_paths(
                    ["README.md", "model.json", *extras], self.expected,
                    allow_hub_metadata=True,
                )

    def test_unsafe_paths_are_rejected_even_if_claimed_as_expected(self) -> None:
        for path in (
            "../model.json", "data/../model.json", "/model.json", "./model.json",
            "data//model.json", "data\\model.json", "C:\\model.json", "model.json\x00",
        ):
            with self.subTest(path=path), self.assertRaises(alignment.AlignmentError):
                alignment.validate_paths([path], {path})

    def test_nonstring_paths_are_rejected(self) -> None:
        for path in (None, 123, True):
            with self.subTest(path=path), self.assertRaises(alignment.AlignmentError):
                alignment.validate_paths(["README.md", path], self.expected)

    def test_actual_packages_have_closed_six_and_eight_file_contracts(self) -> None:
        self.assertEqual(6, len(alignment.MODEL_FILES))
        self.assertEqual(8, len(alignment.DATASET_FILES))
        self.assertIn("model.json", alignment.MODEL_FILES)
        self.assertIn("artifact_receipt.json", alignment.MODEL_FILES)
        self.assertIn("data/test.jsonl", alignment.DATASET_FILES)
        self.assertIn("training_source_snapshot.py", alignment.DATASET_FILES)


class ComparePackageTests(unittest.TestCase):
    def test_identical_packages_report_hashes_for_every_file(self) -> None:
        package = {"README.md": b"card", "model.json": b"{}"}
        rows = alignment.compare_package(package, dict(package))
        self.assertEqual(set(package), {row["path"] for row in rows})
        self.assertEqual(len(package), len(rows))
        for row in rows:
            digest = hashlib.sha256(package[row["path"]]).hexdigest()
            self.assertEqual(digest, row["expected_sha256"])
            self.assertEqual(digest, row["observed_sha256"])
            self.assertIs(row["matches"], True)

    def test_changed_bytes_report_failure_not_merely_file_presence(self) -> None:
        expected = {"README.md": b"card", "model.json": b'{"weight":1}'}
        observed = {"README.md": b"card", "model.json": b'{"weight":2}'}
        rows = {row["path"]: row for row in alignment.compare_package(expected, observed)}
        self.assertIs(rows["README.md"]["matches"], True)
        self.assertIs(rows["model.json"]["matches"], False)
        self.assertNotEqual(rows["model.json"]["expected_sha256"], rows["model.json"]["observed_sha256"])

    def test_missing_or_extra_files_cannot_be_silently_ignored(self) -> None:
        expected = {"README.md": b"card", "model.json": b"{}"}
        for observed in (
            {"README.md": b"card"},
            {**expected, "extra.py": b"unexpected"},
        ):
            with self.subTest(paths=list(observed)), self.assertRaises(alignment.AlignmentError):
                alignment.compare_package(expected, observed)


class HubPackageTests(unittest.TestCase):
    def test_each_package_is_downloaded_only_from_its_immutable_revision(self) -> None:
        for kind, repo_id, paths in (
            ("model", MODEL_ID, alignment.MODEL_FILES),
            ("dataset", DATASET_ID, alignment.DATASET_FILES),
        ):
            with self.subTest(kind=kind):
                hub = FakeHub(kind=kind)
                observed = alignment.hub_package(kind, repo_id, REVISION, set(paths), hub)
                self.assertEqual(hub.files, observed)
                self.assertEqual(len(paths) + 1, len(hub.urls))
                self.assertTrue(all(REVISION in url for url in hub.urls))
                self.assertTrue(all("/resolve/main/" not in url for url in hub.urls))

    def test_revision_mismatch_missing_revision_and_nonstring_revision_fail(self) -> None:
        for value in (OTHER_REVISION, REVISION.upper(), None, 123):
            hub = FakeHub()
            hub.metadata["sha"] = value
            with self.subTest(value=value), self.assertRaises(alignment.AlignmentError):
                alignment.hub_package("model", MODEL_ID, REVISION, set(alignment.MODEL_FILES), hub)
            self.assertEqual(1, len(hub.urls), "file downloads must not precede metadata validation")
        hub = FakeHub()
        del hub.metadata["sha"]
        with self.assertRaises(alignment.AlignmentError):
            alignment.hub_package("model", MODEL_ID, REVISION, set(alignment.MODEL_FILES), hub)

    def test_extra_sibling_missing_sibling_and_duplicate_sibling_fail(self) -> None:
        for mutation in ("extra", "missing", "duplicate"):
            hub = FakeHub()
            if mutation == "extra":
                hub.metadata["siblings"].append({"rfilename": "surprise.py"})
            elif mutation == "missing":
                hub.metadata["siblings"].pop()
            else:
                hub.metadata["siblings"].append(dict(hub.metadata["siblings"][0]))
            with self.subTest(mutation=mutation), self.assertRaises(alignment.AlignmentError):
                alignment.hub_package("model", MODEL_ID, REVISION, set(alignment.MODEL_FILES), hub)
            self.assertEqual(1, len(hub.urls))

    def test_optional_gitattributes_is_not_part_of_the_model_payload(self) -> None:
        hub = FakeHub()
        hub.metadata["siblings"].append({"rfilename": ".gitattributes"})
        observed = alignment.hub_package("model", MODEL_ID, REVISION, set(alignment.MODEL_FILES), hub)
        self.assertEqual(hub.files, observed)
        self.assertFalse(any(url.endswith("/.gitattributes") for url in hub.urls))

    def test_malformed_sibling_metadata_fails_closed(self) -> None:
        for siblings in (None, {}, "README.md", [None], [{}], [{"rfilename": 1}]):
            hub = FakeHub()
            hub.metadata["siblings"] = siblings
            with self.subTest(siblings=siblings), self.assertRaises(alignment.AlignmentError):
                alignment.hub_package("model", MODEL_ID, REVISION, set(alignment.MODEL_FILES), hub)
            self.assertEqual(1, len(hub.urls))

    def test_other_repository_ids_are_rejected_without_fetching(self) -> None:
        for kind, repo_id in (
            ("model", "another-org/oac-system-health-v1"),
            ("model", "SZLHOLDINGS/oac-clinical-transport-health-v1"),
            ("dataset", MODEL_ID), ("space", MODEL_ID),
        ):
            hub = FakeHub()
            with self.subTest(kind=kind, repo_id=repo_id), self.assertRaises(alignment.AlignmentError):
                alignment.hub_package(kind, repo_id, REVISION, set(alignment.MODEL_FILES), hub)
            self.assertEqual([], hub.urls)

    def test_mutable_requested_revision_is_rejected_without_fetching(self) -> None:
        hub = FakeHub()
        with self.assertRaises(alignment.AlignmentError):
            alignment.hub_package("model", MODEL_ID, "main", set(alignment.MODEL_FILES), hub)
        self.assertEqual([], hub.urls)


class AlignmentReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeHub()
        self.dataset = FakeHub(kind="dataset")
        self.expected = {"model": dict(self.model.files), "dataset": dict(self.dataset.files)}

    def fetch(self, url: str) -> bytes:
        return self.dataset(url) if DATASET_ID in url else self.model(url)

    def verify(self) -> dict:
        with mock.patch.object(alignment, "manifest_from_git", return_value=self.expected):
            return alignment.verify_alignment(
                Path("unused-fixture-repository"), REVISION, REVISION, REVISION, self.fetch,
            )

    def test_exact_parity_is_complete_but_grants_no_extra_trust_or_authority(self) -> None:
        receipt = self.verify()
        self.assertIs(receipt["complete"], True)
        self.assertEqual(REVISION, receipt["source_revision"])
        self.assertEqual("https://github.com/szl-holdings/szl-forge", receipt["source_repository"])
        self.assertEqual("exact_pinned_source_and_hub_byte_parity_only", receipt["scope"])
        self.assertEqual([".gitattributes"], receipt["excluded_hub_metadata"])
        for field in (
            "source_signature_verified", "training_rerun",
            "production_promotion_allowed", "clinical_use_authorized",
        ):
            self.assertIs(receipt[field], False, field)
        self.assertEqual({"model", "dataset"}, {item["kind"] for item in receipt["packages"]})
        self.assertEqual(14, sum(len(item["files"]) for item in receipt["packages"]))

    def test_changed_model_bytes_make_receipt_incomplete(self) -> None:
        self.model.files["model.json"] += b"tampered"
        receipt = self.verify()
        self.assertIs(receipt["complete"], False)
        packages = {item["kind"]: item for item in receipt["packages"]}
        self.assertIs(packages["model"]["matches"], False)
        self.assertIs(packages["dataset"]["matches"], True)
        self.assertIs(receipt["production_promotion_allowed"], False)

    def test_changed_dataset_bytes_make_receipt_incomplete(self) -> None:
        self.dataset.files["data/test.jsonl"] += b"tampered"
        receipt = self.verify()
        self.assertIs(receipt["complete"], False)
        packages = {item["kind"]: item for item in receipt["packages"]}
        self.assertIs(packages["model"]["matches"], True)
        self.assertIs(packages["dataset"]["matches"], False)

    def test_any_mutable_revision_fails_before_source_or_network_access(self) -> None:
        for revisions in (
            ("main", REVISION, REVISION), (REVISION, "main", REVISION),
            (REVISION, REVISION, "main"),
        ):
            with self.subTest(revisions=revisions):
                with mock.patch.object(alignment, "manifest_from_git") as manifest:
                    with self.assertRaises(alignment.AlignmentError):
                        alignment.verify_alignment(Path("unused"), *revisions, self.fetch)
                    manifest.assert_not_called()
                self.assertEqual([], self.model.urls)
                self.assertEqual([], self.dataset.urls)

    def test_invalid_source_binding_fails_before_any_network_access(self) -> None:
        with mock.patch.object(
            alignment, "manifest_from_git", side_effect=alignment.AlignmentError("fixture mismatch"),
        ):
            with self.assertRaises(alignment.AlignmentError):
                alignment.verify_alignment(Path("unused"), REVISION, REVISION, REVISION, self.fetch)
        self.assertEqual([], self.model.urls)
        self.assertEqual([], self.dataset.urls)


class SourceBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = b"reviewed synthetic kernel fixture"
        self.trainer = b"reviewed synthetic trainer fixture"
        self.model = dict(FakeHub().files)
        self.dataset = dict(FakeHub(kind="dataset").files)
        self.model["oac_operational_health.py"] = self.source
        self.dataset["training_source_snapshot.py"] = self.trainer
        self.dataset_receipt = {
            "schema": "szl-oac/transport-health-dataset-receipt/v1",
            "purpose": "synthetic_operational_transport_observability_only",
            "authority": alignment.AUTHORITY,
            "contains_phi": False,
            "contains_clinical_results": False,
            "generated_data": "synthetic_only",
            "generator_sha256": hashlib.sha256(self.trainer).hexdigest(),
            "schema_sha256": hashlib.sha256(self.dataset["schema.json"]).hexdigest(),
            "files": {
                path: hashlib.sha256(raw).hexdigest()
                for path, raw in self.dataset.items() if path.startswith("data/")
            },
        }
        self.model_receipt = {
            "schema": "szl-oac/transport-health-artifact-receipt/v1",
            "purpose": "synthetic_operational_transport_attention_only",
            "authority": alignment.AUTHORITY,
            "metrics_scope": "fixed_seed_synthetic_splits_only_not_production_validation",
            "model_sha256": hashlib.sha256(self.model["model.json"]).hexdigest(),
            "kernel_sha256": hashlib.sha256(self.source).hexdigest(),
            "generator_sha256": hashlib.sha256(self.trainer).hexdigest(),
        }
        self.write_receipts()

    def write_receipts(self) -> None:
        self.dataset["dataset_receipt.json"] = json.dumps(self.dataset_receipt).encode("utf-8")
        self.model_receipt["dataset_receipt_sha256"] = hashlib.sha256(
            self.dataset["dataset_receipt.json"],
        ).hexdigest()
        self.model["artifact_receipt.json"] = json.dumps(self.model_receipt).encode("utf-8")

    def manifest(self) -> dict:
        def package(repo_root, revision, directory, files):
            return self.model if "/model/" in directory else self.dataset

        def blob(repo_root, revision, path):
            if path.endswith("src/oac_operational_health.py"):
                return self.source
            if path.endswith("tools/train_operational_health_model.py"):
                return self.trainer
            raise AssertionError(f"unexpected source read: {path}")

        with mock.patch.object(alignment, "_git", return_value=b"commit\n"):
            with mock.patch.object(alignment, "_package", side_effect=package):
                with mock.patch.object(alignment, "_blob", side_effect=blob):
                    return alignment.manifest_from_git(Path("fixture"), REVISION)

    def test_consistently_bound_source_and_artifacts_are_accepted(self) -> None:
        self.assertEqual({"model": self.model, "dataset": self.dataset}, self.manifest())

    def test_changed_artifact_kernel_trainer_schema_or_dataset_bytes_fail(self) -> None:
        for package, path in (
            (self.model, "model.json"), (self.model, "oac_operational_health.py"),
            (self.dataset, "training_source_snapshot.py"), (self.dataset, "schema.json"),
            (self.dataset, "data/train.jsonl"), (self.dataset, "data/test.jsonl"),
            (self.dataset, "data/validation.jsonl"),
        ):
            original = package[path]
            package[path] += b"changed"
            with self.subTest(path=path), self.assertRaises(alignment.AlignmentError):
                self.manifest()
            package[path] = original

    def test_zero_is_not_a_false_privacy_declaration(self) -> None:
        for field in ("contains_phi", "contains_clinical_results"):
            self.dataset_receipt[field] = 0
            self.write_receipts()
            with self.subTest(field=field), self.assertRaises(alignment.AlignmentError):
                self.manifest()
            self.dataset_receipt[field] = False

    def test_rebinding_receipt_hash_does_not_authorize_clinical_or_production_claims(self) -> None:
        for receipt, field, value in (
            (self.dataset_receipt, "generated_data", "clinical_samples"),
            (self.model_receipt, "authority", "clinical_authority"),
            (self.dataset_receipt, "authority", "clinical_authority"),
            (self.model_receipt, "metrics_scope", "production_validated"),
            (self.model_receipt, "purpose", "clinical_result_release"),
        ):
            original = receipt[field]
            receipt[field] = value
            self.write_receipts()
            with self.subTest(field=field, value=value), self.assertRaises(alignment.AlignmentError):
                self.manifest()
            receipt[field] = original
            self.write_receipts()


class GitAndCliFailureTests(unittest.TestCase):
    def args(self) -> list[str]:
        return [
            "--git-revision", REVISION, "--model-revision", REVISION,
            "--dataset-revision", REVISION,
        ]

    def test_git_replacement_objects_cannot_change_the_declared_source_revision(self) -> None:
        result = subprocess.CompletedProcess([], 0, stdout=b"commit\n", stderr=b"")
        with mock.patch.object(alignment.subprocess, "run", return_value=result) as run:
            self.assertEqual(b"commit\n", alignment._git(Path("fixture"), "cat-file", "-t", REVISION))
        argv = run.call_args.args[0]
        self.assertIn("--no-replace-objects", argv)
        self.assertLess(argv.index("--no-replace-objects"), argv.index("cat-file"))
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_structural_and_http_failures_produce_sanitized_incomplete_receipts(self) -> None:
        for failure in (
            alignment.AlignmentError("sensitive-response-body"),
            OSError("sensitive-response-body"),
            http.client.IncompleteRead(b"sensitive-response-body", 100),
            http.client.BadStatusLine("sensitive-response-body"),
        ):
            output = io.StringIO()
            with self.subTest(error=type(failure).__name__):
                with mock.patch.object(alignment, "verify_alignment", side_effect=failure):
                    with mock.patch("sys.stdout", output):
                        result = alignment.main(self.args())
                self.assertEqual(1, result)
                self.assertNotIn("sensitive-response-body", output.getvalue())
                receipt = json.loads(output.getvalue())
                self.assertIs(receipt["complete"], False)
                self.assertIs(receipt["production_promotion_allowed"], False)
                self.assertIs(receipt["clinical_use_authorized"], False)
                self.assertEqual("ALIGNMENT_EVIDENCE_INCOMPLETE", receipt["failure_code"])

    def test_byte_mismatch_receipt_produces_nonzero_exit(self) -> None:
        with mock.patch.object(alignment, "verify_alignment", return_value={"complete": False}):
            with mock.patch("sys.stdout", io.StringIO()):
                self.assertEqual(1, alignment.main(self.args()))

    def test_deep_json_failure_still_emits_an_incomplete_cli_receipt(self) -> None:
        def parse_deep_response(*args):
            return alignment.strict_json(b"[" * 100_000 + b"0" + b"]" * 100_000)

        output = io.StringIO()
        with mock.patch.object(alignment, "verify_alignment", side_effect=parse_deep_response):
            with mock.patch("sys.stdout", output):
                result = alignment.main(self.args())
        self.assertEqual(1, result)
        self.assertIs(json.loads(output.getvalue())["complete"], False)

    def test_existing_receipt_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oac-alignment-tests-") as temporary:
            receipt_path = Path(temporary) / "receipt.json"
            receipt_path.write_bytes(b"retained-original-receipt")
            with mock.patch.object(alignment, "verify_alignment", return_value={"complete": True}):
                with mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
                    result = alignment.main([*self.args(), "--output", str(receipt_path)])
            self.assertEqual(2, result)
            self.assertEqual(b"retained-original-receipt", receipt_path.read_bytes())


class FakeResponse:
    def __init__(self, body: bytes, *, headers: dict | None = None, status: int = 200) -> None:
        self.stream = io.BytesIO(body)
        self.headers = headers if headers is not None else {}
        self.status = status
        self.read_sizes: list[int] = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read1(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.stream.read(size)


class PublicFetchBoundaryTests(unittest.TestCase):
    URL = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/model.json"

    def fetch_response(self, response: FakeResponse) -> tuple[bytes, mock.Mock]:
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(alignment, "build_opener", return_value=opener):
            body = alignment.public_get(self.URL)
        return body, opener

    def test_exact_size_bounded_response_is_read_without_credentials(self) -> None:
        response = FakeResponse(b"12345678", headers={"Content-Length": "8"})
        with mock.patch.object(alignment, "MAX_BYTES", 8):
            body, opener = self.fetch_response(response)
        self.assertEqual(b"12345678", body)
        self.assertTrue(response.closed)
        request = opener.open.call_args.args[0]
        self.assertEqual(self.URL, request.full_url)
        self.assertEqual(15, opener.open.call_args.kwargs["timeout"])
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertNotIn("authorization", headers)
        self.assertNotIn("cookie", headers)
        self.assertEqual("identity", headers["accept-encoding"])

    def test_streaming_size_bound_applies_without_content_length(self) -> None:
        response = FakeResponse(b"123456789")
        with mock.patch.object(alignment, "MAX_BYTES", 8):
            with self.assertRaises(alignment.AlignmentError):
                self.fetch_response(response)
        self.assertTrue(response.closed)
        self.assertEqual([9], response.read_sizes)

    def test_invalid_or_oversize_declared_length_is_rejected_before_reading(self) -> None:
        for length in ("9", "-1", "1.5", "8 ", "unbounded"):
            response = FakeResponse(b"123", headers={"Content-Length": length})
            with self.subTest(length=length), mock.patch.object(alignment, "MAX_BYTES", 8):
                with self.assertRaises(alignment.AlignmentError):
                    self.fetch_response(response)
            self.assertEqual([], response.read_sizes)
            self.assertTrue(response.closed)

    def test_declared_length_must_match_complete_body(self) -> None:
        for length in ("2", "4"):
            response = FakeResponse(b"123", headers={"Content-Length": length})
            with self.subTest(length=length), self.assertRaises(alignment.AlignmentError):
                self.fetch_response(response)
            self.assertTrue(response.closed)

    def test_non_success_status_and_compressed_content_fail_before_reading(self) -> None:
        for response in (
            FakeResponse(b"bad", status=404),
            FakeResponse(b"compressed", headers={"Content-Encoding": "gzip"}),
        ):
            with self.subTest(status=response.status, headers=response.headers):
                with self.assertRaises(alignment.AlignmentError):
                    self.fetch_response(response)
                self.assertEqual([], response.read_sizes)
                self.assertTrue(response.closed)

    def test_whole_read_deadline_terminates_slow_stream(self) -> None:
        response = FakeResponse(b"123")
        with mock.patch.object(alignment.time, "monotonic", side_effect=[0, 46]):
            with self.assertRaises(alignment.AlignmentError):
                self.fetch_response(response)
        self.assertTrue(response.closed)

    def test_non_https_or_different_origins_are_rejected_before_opening(self) -> None:
        for url in (
            "http://huggingface.co/a", "https://another.example/a",
            "https://huggingface.co.another.example/a", "https://user:pass@huggingface.co/a",
            "https://huggingface.co:444/a", "https://huggingface.co/a#fragment",
        ):
            with self.subTest(url=url), mock.patch.object(alignment, "build_opener") as build:
                with self.assertRaises(alignment.AlignmentError):
                    alignment.public_get(url)
                build.assert_not_called()

    def test_redirects_cannot_leave_the_fixed_https_origin(self) -> None:
        handler = alignment._SameOriginRedirect()
        request = alignment.Request(self.URL)
        for url in ("https://another.example/blob", "http://huggingface.co/blob"):
            with self.subTest(url=url), self.assertRaises(alignment.AlignmentError):
                handler.redirect_request(request, None, 302, "Found", {}, url)
        target = "https://huggingface.co/api/resolve-cache/models/fixture"
        redirected = handler.redirect_request(request, None, 302, "Found", {}, target)
        self.assertEqual(target, redirected.full_url)


if __name__ == "__main__":
    unittest.main()
