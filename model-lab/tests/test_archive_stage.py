"""Synthetic, CPU-free checks for the private archive handoff."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from szl_model_lab.archive_stage import archive_candidate, stage_archive, verify_stage
from szl_model_lab.catalog import TRACKS
from szl_model_lab.cli import main


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def small_safetensors() -> bytes:
    shapes = {"network.0.weight": [16, 8], "network.0.bias": [16],
              "network.2.weight": [1, 16], "network.2.bias": [1]}
    cursor = 0
    header = {}
    for name, shape in shapes.items():
        size = 4
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [cursor, cursor + size]}
        cursor += size
    raw_header = json.dumps(header).encode()
    return len(raw_header).to_bytes(8, "little") + raw_header + bytes(cursor)


class ArchiveStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "private-archive"
        self.archive.mkdir()
        self.local = self.root / "plain-local"
        self.local.mkdir()
        self.stage = self.local / "staged"
        self.payload = b'{"synthetic":true,"fixture":"archive-byte-check"}\n'
        (self.archive / "curated").mkdir()
        (self.archive / "curated" / "tiny.jsonl").write_bytes(self.payload)
        self.request = self.root / "request.json"
        self.spec = {
            "schema": "szl.model-lab.archive-request/v1",
            "track": "router",
            "archive_relative_path": "curated/tiny.jsonl",
            "bytes": len(self.payload),
            "sha256": digest(self.payload),
            "source_provenance_sha256": "a" * 64,
            "rights_review_sha256": "b" * 64,
            "leakage_review_sha256": "c" * 64,
            "owner_release_sha256": "d" * 64,
            "rights_status": "APPROVED_FOR_RESEARCH_TRAINING",
            "leakage_status": "PASS",
            "owner_release_status": "RELEASED_FOR_RESEARCH_TRAINING",
        }
        self.write_request()

    def write_request(self):
        self.request.write_text(json.dumps(self.spec), encoding="utf-8")

    def test_stage_binds_local_bytes_and_rejects_drift(self):
        result = stage_archive(self.archive, self.request, self.stage, self.local)
        self.assertEqual(result["state"], "STAGED_LOCAL_BYTES_VERIFIED")
        manifest = self.stage / "stage-manifest.json"
        self.assertEqual(result["manifest_sha256"], digest(manifest.read_bytes()))
        binding = verify_stage(manifest, result["manifest_sha256"], self.stage / "data.jsonl", "router")
        self.assertEqual(binding["data_sha256"], digest(self.payload))
        self.assertNotIn(str(self.archive), manifest.read_text())
        (self.stage / "data.jsonl").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            verify_stage(manifest, result["manifest_sha256"], self.stage / "data.jsonl", "router")

    def test_source_drift_and_missing_release_gate_fail_before_staging(self):
        self.spec["sha256"] = "0" * 64
        self.write_request()
        with self.assertRaises(ValueError):
            stage_archive(self.archive, self.request, self.stage, self.local)
        self.assertFalse(self.stage.exists())
        self.spec["sha256"] = digest(self.payload)
        del self.spec["owner_release_sha256"]
        self.write_request()
        with self.assertRaises(ValueError):
            stage_archive(self.archive, self.request, self.stage, self.local)
        self.assertFalse(self.stage.exists())

    def test_archive_root_inside_checkout_rejected_before_payload_access(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module
        # SIMULATED source placement, with real temp paths and no private data.
        repo = self.root / "source-checkout"
        repo.mkdir()
        nested_archive = repo / "private-archive"
        nested_archive.mkdir()
        fake_module = repo / "model-lab" / "src" / "szl_model_lab" / "archive_stage.py"
        for archive in (repo, nested_archive):
            with self.subTest(archive=archive.name):
                with (patch.object(module, "__file__", str(fake_module)),
                      patch.object(module, "read_regular", side_effect=AssertionError("must not read")),
                      patch.object(module, "_read_archive_regular", side_effect=AssertionError("must not read")),
                      patch.object(module, "write_new", side_effect=AssertionError("must not write"))):
                    with self.assertRaisesRegex(ValueError, "archive_root_overlaps_source_checkout"):
                        stage_archive(archive, self.request, self.stage, self.local)
                    with self.assertRaisesRegex(ValueError, "archive_root_overlaps_source_checkout"):
                        archive_candidate(self.local / "candidate", archive, self.local)
            self.assertFalse((archive / "model-lab-candidates").exists())
        self.assertFalse(self.stage.exists())

    def test_archive_root_ancestor_of_checkout_rejected_before_payload_access(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module
        # SIMULATED layout reproducing the fixed archive destination collision.
        repo = self.archive / "model-lab-candidates"
        repo.mkdir()
        fake_module = repo / "model-lab" / "src" / "szl_model_lab" / "archive_stage.py"
        with (patch.object(module, "__file__", str(fake_module)),
              patch.object(module, "read_regular", side_effect=AssertionError("must not read")),
              patch.object(module, "_read_archive_regular", side_effect=AssertionError("must not read")),
              patch.object(module, "write_new", side_effect=AssertionError("must not write"))):
            with self.assertRaisesRegex(ValueError, "archive_root_overlaps_source_checkout"):
                stage_archive(self.archive, self.request, self.stage, self.local)
            with self.assertRaisesRegex(ValueError, "archive_root_overlaps_source_checkout"):
                archive_candidate(self.local / "candidate", self.archive, self.local)
        self.assertEqual(list(repo.iterdir()), [])
        self.assertFalse(self.stage.exists())

    def test_private_request_inside_checkout_rejected_before_payload_access(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module
        # SIMULATED source placement. The request contains fixture assertions only.
        repo = self.root / "source-checkout"
        repo.mkdir()
        request = repo / "request.json"
        request.write_text(json.dumps(self.spec), encoding="utf-8")
        fake_module = repo / "model-lab" / "src" / "szl_model_lab" / "archive_stage.py"
        with (patch.object(module, "__file__", str(fake_module)),
              patch.object(module, "read_regular", side_effect=AssertionError("must not read")),
              patch.object(module, "_read_archive_regular", side_effect=AssertionError("must not read")),
              patch.object(module, "write_new", side_effect=AssertionError("must not write"))):
            with self.assertRaisesRegex(ValueError, "request_inside_source_checkout"):
                stage_archive(self.archive, request, self.stage, self.local)
        self.assertFalse(self.stage.exists())

    def test_local_root_ancestor_cannot_admit_output_inside_checkout(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module
        repo = self.local / "source-checkout"
        repo.mkdir()
        fake_module = repo / "model-lab" / "src" / "szl_model_lab" / "archive_stage.py"
        with (patch.object(module, "__file__", str(fake_module)),
              patch.object(module, "read_regular", side_effect=AssertionError("must not read")),
              patch.object(module, "_read_archive_regular", side_effect=AssertionError("must not read")),
              patch.object(module, "write_new", side_effect=AssertionError("must not write"))):
            with self.assertRaisesRegex(ValueError, "path_inside_source_checkout"):
                stage_archive(self.archive, self.request, repo / "staged", self.local)
            with self.assertRaisesRegex(ValueError, "path_inside_source_checkout"):
                archive_candidate(repo / "candidate", self.archive, self.local)
        self.assertFalse((repo / "staged").exists())
        self.assertFalse((repo / "candidate").exists())
        self.assertFalse((self.archive / "model-lab-candidates").exists())

    def test_traversal_and_unmaterialized_source_fail_closed(self):
        self.spec["archive_relative_path"] = "../outside.jsonl"
        self.write_request()
        with self.assertRaises(ValueError):
            stage_archive(self.archive, self.request, self.stage, self.local)
        self.spec["archive_relative_path"] = "curated/tiny.jsonl"
        self.write_request()
        source = self.archive / "curated" / "tiny.jsonl"
        if hasattr(source.stat(), "st_file_attributes"):
            from unittest.mock import patch
            import szl_model_lab.archive_stage as module
            original = module._file_attributes
            with patch.object(module, "_file_attributes", side_effect=lambda p: 0x1000 if p == source else original(p)):
                with self.assertRaises(ValueError):
                    stage_archive(self.archive, self.request, self.stage, self.local)

    def test_reparse_ancestor_rejected_before_path_resolution(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module
        alias = self.root / "junction-shaped-parent"
        alias.mkdir()
        original = module._file_attributes
        with patch.object(module, "_file_attributes",
                          side_effect=lambda p: 0x400 if p == alias else original(p)):
            with self.assertRaisesRegex(ValueError, "symlink_or_reparse"):
                stage_archive(self.archive, self.request, alias / "staged", alias)
            with self.assertRaisesRegex(ValueError, "symlink_or_reparse"):
                archive_candidate(alias, self.archive, self.local)
            with self.assertRaisesRegex(ValueError, "symlink_or_reparse"):
                archive_candidate(self.local, alias, self.local)

    def test_hydrated_cloud_tag_admitted_but_unknown_reparse_rejected(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module
        original = module._file_attributes
        with (patch.object(module, "_file_attributes",
                           side_effect=lambda p: 0x400 if p == self.archive else original(p)),
              patch.object(module, "_reparse_tag", return_value=0x9000701a),
              patch.object(module, "_placeholder_state", return_value=0x1)):
            stage_archive(self.archive, self.request, self.stage, self.local)
        with (patch.object(module, "_file_attributes",
                           side_effect=lambda p: 0x400 if p == self.archive else original(p)),
              patch.object(module, "_reparse_tag", return_value=0xa0000003),
              patch.object(module, "_placeholder_state", return_value=0x1)):
            with self.assertRaisesRegex(ValueError, "symlink_or_reparse"):
                module._safe_existing_path(self.archive, allow_cloud=True)

    def test_train_cli_rejects_bad_stage_before_torch_import(self):
        stage_archive(self.archive, self.request, self.stage, self.local)
        from unittest.mock import patch
        with patch.dict(os.environ, {"SZL_ARCHIVE_ROOT": str(self.archive),
                                  "SZL_STAGE_ROOT": str(self.local)}), redirect_stderr(StringIO()):
            code = main(["train", "--track", "router", "--data", str(self.stage / "data.jsonl"),
                         "--output", str(self.local / "candidate"), "--source-revision", "a" * 40,
                         "--acknowledge-research-only", "--archive-stage-manifest",
                         str(self.stage / "stage-manifest.json"), "--archive-stage-sha256", "0" * 64])
        self.assertEqual(code, 2)
        self.assertFalse((self.local / "candidate").exists())

    def test_stage_output_disjoint_before_torch_import_or_writes(self):
        from unittest.mock import patch
        import builtins
        import szl_model_lab.archive_stage as module
        result = stage_archive(self.archive, self.request, self.stage, self.local)
        manifest = self.stage / "stage-manifest.json"
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Torch must not be imported during failed preflight")
            return real_import(name, *args, **kwargs)

        for output in (self.stage, self.stage / "candidate"):
            with self.subTest(output=output.name):
                with self.assertRaisesRegex(ValueError, "candidate_output_inside_input_stage"):
                    module.require_stage_output_disjoint(output, manifest)
                with (patch.dict(os.environ, {"SZL_ARCHIVE_ROOT": str(self.archive),
                                              "SZL_STAGE_ROOT": str(self.local)}),
                      patch.object(builtins, "__import__", side_effect=guarded_import),
                      redirect_stderr(StringIO())):
                    code = main(["train", "--track", "router", "--data", str(self.stage / "data.jsonl"),
                                 "--output", str(output), "--source-revision", "a" * 40,
                                 "--acknowledge-research-only", "--archive-stage-manifest", str(manifest),
                                 "--archive-stage-sha256", result["manifest_sha256"]])
                self.assertEqual(code, 2)
        module.require_stage_output_disjoint(self.local / "sibling-candidate", manifest)
        self.assertEqual({p.name for p in self.stage.iterdir()}, {"data.jsonl", "stage-manifest.json"})
        verify_stage(manifest, result["manifest_sha256"], self.stage / "data.jsonl", "router")

    def test_direct_training_body_rejects_overlap_before_data_read_or_fit(self):
        """Exercise the real preflight body without importing the Torch module.

        Only dependency boundaries are substituted; no data read, fit, model,
        source admission, or training is simulated as a successful operation.
        """
        from unittest.mock import Mock, patch
        result = stage_archive(self.archive, self.request, self.stage, self.local)
        source = Path(__file__).resolve().parents[1] / "src" / "szl_model_lab" / "training.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "train_candidate")
        executable = ast.fix_missing_locations(ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function], type_ignores=[]))
        dataset = Mock()
        dataset.read.side_effect = AssertionError("Dataset.read must not run")
        fit = Mock(side_effect=AssertionError("fit must not run"))
        writer = Mock(side_effect=AssertionError("writer must not run"))
        namespace = {"__name__": "szl_model_lab.training", "__package__": "szl_model_lab",
                     "Path": Path, "verify_source": Mock(return_value={}),
                     "Dataset": dataset, "fit": fit, "write_candidate": writer}
        exec(compile(executable, str(source), "exec"), namespace)
        with patch.dict(os.environ, {"SZL_ARCHIVE_ROOT": str(self.archive),
                                   "SZL_STAGE_ROOT": str(self.local)}):
            with self.assertRaisesRegex(ValueError, "candidate_output_inside_input_stage"):
                namespace["train_candidate"](
                    self.stage / "data.jsonl", "router", self.stage / "candidate", "a" * 40,
                    1, 1, 1, archive_stage_manifest=self.stage / "stage-manifest.json",
                    archive_stage_sha256=result["manifest_sha256"])
        dataset.read.assert_not_called()
        fit.assert_not_called()
        writer.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows native metadata regression")
    def test_native_cloud_metadata_overrides_masked_python_attributes(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module
        self.assertFalse(getattr(self.local.lstat(), "st_file_attributes", 0) & 0x400)
        real_metadata = module._windows_path_metadata
        with patch.object(module, "_windows_path_metadata", side_effect=lambda p:
                          (0x431, 0x9000701a) if p == self.local else real_metadata(p)):
            with self.assertRaisesRegex(ValueError, "symlink_or_reparse"):
                module.require_local_workspace(self.local / "new-stage", self.local, self.archive)
        self.assertFalse((self.local / "new-stage").exists())

    @unittest.skipUnless(os.name == "nt", "Windows native metadata regression")
    def test_native_lookup_errors_do_not_fall_back_to_lstat(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module
        with (patch.object(module, "_windows_path_metadata", side_effect=OSError("native lookup failed")),
              patch.object(Path, "lstat", side_effect=AssertionError("masked fallback forbidden"))):
            for lookup in (module._file_attributes, module._reparse_tag):
                with self.assertRaisesRegex(OSError, "native lookup failed"):
                    lookup(self.archive)

    def test_cloud_partial_unknown_and_recall_rejected_before_payload_open(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module
        source = self.archive / "curated" / "tiny.jsonl"
        real_attributes = module._file_attributes
        cases = [(0x431, 0x9000701a, state) for state in (0, 0x11, 0x31, 0xffffffff)]
        cases += [(0x431, 0xa0000003, 1)]
        cases += [(0x431 | flag, 0x9000701a, 1) for flag in (0x1000, 0x40000, 0x400000)]
        for attributes, tag, state in cases:
            with self.subTest(attributes=attributes, tag=tag, state=state):
                with (patch.object(module, "_file_attributes", side_effect=lambda p:
                                   attributes if p == self.archive else real_attributes(p)),
                      patch.object(module, "_reparse_tag", return_value=tag),
                      patch.object(module, "_placeholder_state", return_value=state),
                      patch.object(module.os, "open", side_effect=AssertionError("payload must not open"))):
                    with self.assertRaises(ValueError):
                        module._read_archive_regular(source, 65536)

    def _small_candidate(self):
        result = stage_archive(self.archive, self.request, self.stage, self.local)
        binding = verify_stage(self.stage / "stage-manifest.json", result["manifest_sha256"],
                               self.stage / "data.jsonl", "router")
        candidate = self.local / "candidate"
        candidate.mkdir()
        bodies = {
            "config.json": json.dumps({"schema": "szl.model-lab.candidate/v1", "state": "TRAINED_UNQUALIFIED",
                                       "track": "router", "features": list(TRACKS["router"].features),
                                       "architecture": "tabular-mlp-8x16x1-v1",
                                       "normalization_id": "synthetic-fixture",
                                       "purpose": "RESEARCH_ADVISORY_ONLY",
                                       "publication_eligible": False, "runtime_qualified": False,
                                       "calibration_validated": False,
                                       "source": {"verification": "GIT_CLEAN", "revision": "a" * 40},
                                       "dataset": {"dataset_sha256": digest(self.payload),
                                                   "normalization_id": "synthetic-fixture",
                                                   "archive_stage": binding},
                                       "recipe": {"device": "cpu"}}).encode(),
            "model.safetensors": small_safetensors(),
            "metrics.json": b'{"split":"validation","test_evaluated":false}',
            "README.md": b"Synthetic local candidate fixture.\n",
        }
        for name, raw in bodies.items():
            (candidate / name).write_bytes(raw)
        (candidate / "manifest.json").write_text(json.dumps({"schema": "szl.model-lab.candidate/v1",
            "files": {name: digest(raw) for name, raw in bodies.items()}, "signed": False,
            "publication_eligible": False}), encoding="utf-8")
        return candidate

    def test_candidate_archive_is_local_copy_only(self):
        candidate = self._small_candidate()
        result = archive_candidate(candidate, self.archive, self.local)
        self.assertEqual(result["state"], "LOCAL_COPY_VERIFIED_REMOTE_UNVERIFIED")
        self.assertFalse(result["remote_restore_verified"])
        self.assertEqual(result["checkpoint_completeness"], "NOT_RESTARTABLE")
        self.assertEqual(len(list((self.archive / "model-lab-candidates").iterdir())), 1)
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module
        original = module._file_attributes
        weight_path = candidate / "model.safetensors"
        with (patch.object(module, "_file_attributes",
                           side_effect=lambda p: 0x1000 if p == weight_path else original(p)),
              patch.object(module, "read_regular", side_effect=AssertionError("must not read"))):
            with self.assertRaisesRegex(ValueError, "candidate_not_fully_materialized"):
                archive_candidate(candidate, self.archive, self.local)
        weight_path.write_bytes(b"invalid safetensors")
        manifest = json.loads((candidate / "manifest.json").read_text())
        manifest["files"]["model.safetensors"] = digest(weight_path.read_bytes())
        (candidate / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "candidate_safetensors_invalid"):
            archive_candidate(candidate, self.archive, self.local)

    def test_stage_and_candidate_reparse_checks_precede_payload_reads(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module

        candidate = self._small_candidate()
        real_attributes = module._file_attributes
        stage_manifest = self.stage / "stage-manifest.json"
        stage_data = self.stage / "data.jsonl"
        stage_digest = digest(stage_manifest.read_bytes())
        for suspect in (stage_manifest, stage_data):
            with self.subTest(path=suspect.name):
                with (patch.object(module, "_file_attributes", side_effect=lambda path:
                                   0x400 if path == suspect else real_attributes(path)),
                      patch.object(module, "_reparse_tag", return_value=0xa0000003),
                      patch.object(module, "read_regular", side_effect=AssertionError("payload read")),
                      patch.object(Path, "iterdir", side_effect=AssertionError("directory enumerated"))):
                    with self.assertRaisesRegex(ValueError, "symlink_or_reparse"):
                        verify_stage(stage_manifest, stage_digest, stage_data, "router")

        suspect = candidate / "model.safetensors"
        with (patch.object(module, "_file_attributes", side_effect=lambda path:
                           0x400 if path == suspect else real_attributes(path)),
              patch.object(module, "_reparse_tag", return_value=0xa0000003),
              patch.object(module, "read_regular", side_effect=AssertionError("payload read"))):
            with self.assertRaisesRegex(ValueError, "symlink_or_reparse"):
                archive_candidate(candidate, self.archive, self.local)

    def test_failed_candidate_copy_stays_pending_and_can_retry(self):
        from unittest.mock import patch
        import szl_model_lab.archive_stage as module

        candidate = self._small_candidate()
        real_write = module.write_new
        writes = 0

        def fail_second_write(path, raw):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("synthetic copy interruption")
            return real_write(path, raw)

        with patch.object(module, "write_new", side_effect=fail_second_write):
            with self.assertRaisesRegex(OSError, "synthetic copy interruption"):
                archive_candidate(candidate, self.archive, self.local)
        parent = self.archive / "model-lab-candidates"
        self.assertFalse(any(p.name.startswith("candidate-") for p in parent.iterdir()))
        self.assertEqual(len([p for p in parent.iterdir() if p.name.startswith(".pending-")]), 1)

        receipt = archive_candidate(candidate, self.archive, self.local)
        finals = [p for p in parent.iterdir() if p.name.startswith("candidate-")]
        self.assertEqual(len(finals), 1)
        self.assertEqual({p.name for p in finals[0].iterdir()},
                         {"config.json", "model.safetensors", "metrics.json", "README.md",
                          "manifest.json", "archive-copy-receipt.json"})
        self.assertEqual(json.loads((finals[0] / "archive-copy-receipt.json").read_text()), receipt)
        self.assertFalse(receipt["remote_restore_verified"])

    def test_candidate_archive_rejects_unlisted_secret_file(self):
        candidate = self.local / "candidate"
        candidate.mkdir()
        (candidate / "token.txt").write_text("synthetic-secret")
        with self.assertRaisesRegex(ValueError, "candidate_file_allowlist_mismatch"):
            archive_candidate(candidate, self.archive, self.local)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                         "Model Lab writer dependencies are not installed")
    def test_actual_writer_candidate_round_trip(self):
        from szl_model_lab.artifacts import load_candidate, write_candidate
        from szl_model_lab.models import AdvisoryMLP
        result = stage_archive(self.archive, self.request, self.stage, self.local)
        binding = verify_stage(self.stage / "stage-manifest.json", result["manifest_sha256"],
                               self.stage / "data.jsonl", "router")
        candidate = self.local / "writer-fixture"
        write_candidate(candidate, AdvisoryMLP("router"),
                        source={"verification": "GIT_CLEAN", "revision": "a" * 40},
                        dataset={"dataset_sha256": digest(self.payload),
                                 "normalization_id": "synthetic-fixture", "archive_stage": binding},
                        validation={}, recipe={"device": "cpu"})
        # Source/review assertions are deliberately synthetic test inputs, not
        # protected-main admission, data approval, or a trained release.
        original_model, config, original_digest = load_candidate(candidate)
        self.assertEqual(config["track"], "router")
        self.assertEqual(archive_candidate(candidate, self.archive, self.local)["local_copy_verified"], True)
        archived = next((self.archive / "model-lab-candidates").iterdir())
        restored_model, restored_config, restored_digest = load_candidate(archived)
        self.assertEqual(restored_config, config)
        self.assertEqual(restored_digest, original_digest)
        self.assertTrue(all(value.equal(restored_model.state_dict()[name])
                            for name, value in original_model.state_dict().items()))
        self.assertIs(json.loads((archived / "archive-copy-receipt.json").read_text())[
            "remote_restore_verified"], False)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors")
                         and importlib.util.find_spec("pydantic"),
                         "Model Lab writer and dataset dependencies are not installed")
    def test_synthetic_trainer_handoff_without_fitting(self):
        """Exercise the real dataset, writer, loader, and archive boundaries."""
        from unittest.mock import patch
        import szl_model_lab.archive_stage as archive_module
        import szl_model_lab.training as training
        from szl_model_lab.artifacts import load_candidate
        from szl_model_lab.models import AdvisoryMLP

        rows = []
        for split in ("train", "validation", "test"):
            for label in (False, True):
                ident = f"synthetic-{split}-{int(label)}"
                rows.append({"example_id": ident, "group_id": ident,
                             "case_sha256": digest(ident.encode()), "split": split,
                             "features": {name: float(0.8 if label else 0.2)
                                          for name in TRACKS["router"].features},
                             "label": label,
                             "provenance": {"source_id": ident, "rights_basis": "test-fixture",
                                            "synthetic": True,
                                            "feature_time": "2026-09-01T00:00:00Z",
                                            "outcome_time": "2026-09-01T00:01:00Z",
                                            "normalization_id": "synthetic-fixture"}})
        self.payload = b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n"
                                for row in rows)
        (self.archive / "curated" / "tiny.jsonl").write_bytes(self.payload)
        self.spec["bytes"] = len(self.payload)
        self.spec["sha256"] = digest(self.payload)
        self.write_request()
        staged = stage_archive(self.archive, self.request, self.stage, self.local)
        source = {"verification": "GIT_CLEAN", "revision": "a" * 40}
        observed = []

        def no_fit(dataset, **budget):
            observed.append((dataset.sha256, dataset.counts, budget))
            self.assertTrue(dataset.summary()["all_synthetic"])
            return AdvisoryMLP("router"), {"fixture_observations": 2}

        candidate = self.local / "trainer-fixture"
        real_verify_stage = archive_module.verify_stage
        with (patch.dict(os.environ, {"SZL_ARCHIVE_ROOT": str(self.archive),
                                   "SZL_STAGE_ROOT": str(self.local)}),
              patch.object(training, "verify_source", return_value=source) as source_check,
              patch.object(training, "fit", side_effect=no_fit) as fit_check,
              patch.object(archive_module, "verify_stage", wraps=real_verify_stage) as stage_check):
            training.train_candidate(self.stage / "data.jsonl", "router", candidate,
                                     "a" * 40, 1, 7, 2,
                                     archive_stage_manifest=self.stage / "stage-manifest.json",
                                     archive_stage_sha256=staged["manifest_sha256"])
        self.assertEqual(source_check.call_count, 2)
        self.assertEqual(stage_check.call_count, 2)
        fit_check.assert_called_once()
        self.assertEqual(observed, [(digest(self.payload),
                                     {"train": 2, "validation": 2, "test": 2},
                                     {"epochs": 1, "seed": 7, "batch_size": 2})])

        original, config, manifest_digest = load_candidate(candidate)
        self.assertEqual(config["dataset"]["archive_stage"]["stage_manifest_sha256"],
                         staged["manifest_sha256"])
        self.assertEqual(config["dataset"]["dataset_sha256"], digest(self.payload))
        receipt = archive_candidate(candidate, self.archive, self.local)
        archived = next(p for p in (self.archive / "model-lab-candidates").iterdir()
                        if p.name.startswith("candidate-"))
        restored, restored_config, restored_digest = load_candidate(archived)
        self.assertEqual(restored_config, config)
        self.assertEqual(restored_digest, manifest_digest)
        self.assertTrue(all(value.equal(restored.state_dict()[name])
                            for name, value in original.state_dict().items()))
        self.assertEqual(receipt["candidate_manifest_sha256"], manifest_digest)
        for name, metadata in receipt["files"].items():
            self.assertEqual(metadata["sha256"], digest((candidate / name).read_bytes()))
            self.assertEqual(metadata["sha256"], digest((archived / name).read_bytes()))
        self.assertEqual(receipt["state"], "LOCAL_COPY_VERIFIED_REMOTE_UNVERIFIED")
        self.assertFalse(receipt["remote_restore_verified"])


if __name__ == "__main__":
    unittest.main()
