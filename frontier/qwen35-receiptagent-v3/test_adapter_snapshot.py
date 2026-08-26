from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent


def load_evaluator():
    here_text = str(HERE)
    if here_text not in sys.path:
        sys.path.insert(0, here_text)
    spec = importlib.util.spec_from_file_location(
        "v3_adapter_snapshot_under_test", HERE / "evaluate_candidate.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluator = load_evaluator()


def simple_hash_adapter(directory: pathlib.Path) -> tuple[str, list[dict[str, object]]]:
    combined = hashlib.sha256()
    files: list[dict[str, object]] = []
    for path in sorted(directory.iterdir()):
        data = path.read_bytes()
        combined.update(path.name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
        files.append(
            {
                "path": path.name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return combined.hexdigest(), files


@unittest.skipUnless(
    os.name == "posix" and all(
        hasattr(os, name) for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    ),
    "adapter snapshotting requires POSIX no-follow descriptor semantics",
)
class AdapterSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_bytes(b"{}")
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weights-v1")

    def test_snapshot_captures_exact_bytes_and_is_removed_after_use(self):
        with mock.patch.object(
            evaluator, "hash_adapter", side_effect=simple_hash_adapter
        ):
            with evaluator.staged_adapter_snapshot(self.adapter) as staged:
                snapshot, digest, files = staged
                snapshot_path = pathlib.Path(snapshot)
                self.assertNotEqual(self.adapter, snapshot_path)
                self.assertEqual(digest, simple_hash_adapter(snapshot_path)[0])
                self.assertEqual(files, simple_hash_adapter(snapshot_path)[1])
                self.assertEqual(0o500, stat.S_IMODE(snapshot_path.stat().st_mode))
                self.assertTrue(
                    all(
                        stat.S_IMODE(path.stat().st_mode) == 0o400
                        for path in snapshot_path.iterdir()
                    )
                )
                (self.adapter / "adapter_model.safetensors").write_bytes(b"weights-v2")
                self.assertEqual(
                    b"weights-v1",
                    (snapshot_path / "adapter_model.safetensors").read_bytes(),
                )
            self.assertFalse(snapshot_path.exists())

    def test_snapshot_rejects_symlinked_required_file(self):
        weights = self.adapter / "adapter_model.safetensors"
        weights.unlink()
        target = self.root / "external.safetensors"
        target.write_bytes(b"external")
        weights.symlink_to(target)
        with self.assertRaisesRegex(
            evaluator.QualificationError, "regular no-follow file"
        ):
            with evaluator.staged_adapter_snapshot(self.adapter):
                self.fail("symlinked adapter unexpectedly staged")

    def test_snapshot_rejects_hardlinked_required_file(self):
        weights = self.adapter / "adapter_model.safetensors"
        weights.unlink()
        target = self.root / "external.safetensors"
        target.write_bytes(b"external")
        os.link(target, weights)
        with self.assertRaisesRegex(
            evaluator.QualificationError, "single-link regular file"
        ):
            with evaluator.staged_adapter_snapshot(self.adapter):
                self.fail("hardlinked adapter unexpectedly staged")

    def test_snapshot_rejects_non_allowlisted_file(self):
        (self.adapter / "surprise.bin").write_bytes(b"unexpected")
        with self.assertRaisesRegex(
            evaluator.QualificationError, "non-allowlisted files"
        ):
            with evaluator.staged_adapter_snapshot(self.adapter):
                self.fail("non-allowlisted adapter file unexpectedly staged")

    def test_stable_descriptor_read_rejects_identity_drift(self):
        source_fd = os.open(
            self.adapter,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        self.addCleanup(os.close, source_fd)
        with mock.patch.object(
            evaluator,
            "stable_file_identity",
            side_effect=[(1,), (2,)],
        ):
            with self.assertRaisesRegex(
                evaluator.QualificationError, "changed during snapshot capture"
            ):
                evaluator.read_stable_adapter_file(source_fd, "adapter_config.json")


class VerifiedAdapterLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = pathlib.Path("/private/adapter-snapshot")
        self.aggregate = "a" * 64
        self.files = [
            {
                "path": "adapter_config.json",
                "bytes": 2,
                "sha256": "b" * 64,
                "jsonKeys": 0,
            }
        ]
        self.files_digest = evaluator.sha256_json(self.files)

    def test_peft_receives_private_snapshot_and_local_only_contract(self):
        loaded = object()
        peft = types.SimpleNamespace(
            from_pretrained=mock.Mock(return_value=loaded)
        )
        with mock.patch.object(
            evaluator,
            "hash_adapter",
            side_effect=[
                (self.aggregate, self.files),
                (self.aggregate, self.files),
            ],
        ):
            observed = evaluator.load_verified_v3_adapter(
                object(),
                self.snapshot,
                expected_sha256=self.aggregate,
                expected_files_sha256=self.files_digest,
                peft_model=peft,
            )
        self.assertIs(loaded, observed)
        call = peft.from_pretrained.call_args
        self.assertEqual(str(self.snapshot), call.args[1])
        self.assertEqual(
            {"is_trainable": False, "local_files_only": True}, call.kwargs
        )

    def test_post_load_snapshot_drift_fails_closed(self):
        peft = types.SimpleNamespace(from_pretrained=mock.Mock(return_value=object()))
        with mock.patch.object(
            evaluator,
            "hash_adapter",
            side_effect=[
                (self.aggregate, self.files),
                ("c" * 64, self.files),
            ],
        ):
            with self.assertRaisesRegex(
                evaluator.QualificationError, "changed during PEFT load"
            ):
                evaluator.load_verified_v3_adapter(
                    object(),
                    self.snapshot,
                    expected_sha256=self.aggregate,
                    expected_files_sha256=self.files_digest,
                    peft_model=peft,
                )

    def test_missing_verified_evidence_fails_before_peft(self):
        peft = types.SimpleNamespace(from_pretrained=mock.Mock())
        with self.assertRaisesRegex(
            evaluator.QualificationError, "requires verified aggregate"
        ):
            evaluator.load_verified_v3_adapter(
                object(),
                self.snapshot,
                expected_sha256=None,
                expected_files_sha256=None,
                peft_model=peft,
            )
        peft.from_pretrained.assert_not_called()


class ExistingLoadModesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fast = types.SimpleNamespace(
            from_pretrained=mock.Mock(return_value=(object(), object())),
            for_inference=mock.Mock(),
        )
        self.peft = types.SimpleNamespace(from_pretrained=mock.Mock(return_value=object()))
        self.unsloth_module = types.ModuleType("unsloth")
        self.unsloth_module.FastVisionModel = self.fast
        self.peft_module = types.ModuleType("peft")
        self.peft_module.PeftModel = self.peft
        self.candidate = {
            "actual_training_base": {
                "repo_id": "local/base",
                "revision": "1" * 40,
                "load_in_4bit": True,
            }
        }

    def modules(self):
        return mock.patch.dict(
            sys.modules,
            {"unsloth": self.unsloth_module, "peft": self.peft_module},
        )

    def test_base_load_does_not_require_adapter_evidence(self):
        with self.modules():
            _model, _processor, identity = evaluator.load_model(
                "base", self.candidate, adapter_dir=None
            )
        self.assertEqual("base", identity["kind"])
        self.peft.from_pretrained.assert_not_called()

    def test_v2_load_contract_is_unchanged(self):
        predecessor = {"adapterRevision": "2" * 40}
        with (
            self.modules(),
            mock.patch.object(
                evaluator,
                "v2_snapshot",
                return_value=(pathlib.Path("/cached/v2"), predecessor),
            ),
        ):
            _model, _processor, identity = evaluator.load_model(
                "v2", self.candidate, adapter_dir=None
            )
        self.assertEqual("2" * 40, identity["adapterRevision"])
        self.peft.from_pretrained.assert_called_once()
        self.assertEqual(
            {"is_trainable": False}, self.peft.from_pretrained.call_args.kwargs
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
