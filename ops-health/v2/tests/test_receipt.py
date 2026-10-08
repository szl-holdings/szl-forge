"""M6: receipt and self-hash integrity of the v2 kernel, and freeze.py's refusal rules.

Every artifact here is a DEV candidate frozen into a temporary directory with the TEST-ONLY
source-state override; nothing is written inside the repository.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from v2.research import freeze, registry
from v2.research.gates import VALID_FEATURES
from v2.tests import _support as S

K = S.kernel()


def _flip_byte(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))


def _rewrite_receipt(path: Path, change) -> None:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    change(receipt)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ReceiptTest(unittest.TestCase):
    def setUp(self):
        self._tmp = S.TempDir()
        self.tmp = self._tmp.__enter__()
        self.info = S.freeze_dev(self.tmp / "art")
        self.model = Path(self.info["model_path"])
        self.receipt = Path(self.info["receipt_path"])

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def assertRefused(self, fragment: str, model=None, receipt=None, module=K):
        with self.assertRaises(module.ModelArtifactError) as ctx:
            module.OpsHealthKernel(model or self.model, receipt or self.receipt)
        self.assertIn(fragment, str(ctx.exception))

    # -- control -------------------------------------------------------------------------
    def test_valid_pair_loads_and_binds_everything(self):
        kernel = K.OpsHealthKernel(self.model, self.receipt)
        receipt = kernel.receipt
        self.assertEqual(receipt["model_sha256"], S.sha256_file(self.model))
        self.assertEqual(receipt["kernel_sha256"], S.sha256_file(registry.KERNEL_PATH))
        self.assertEqual(receipt["kernel_sha256"], K.kernel_file_sha256())
        self.assertEqual(receipt["preregistration_sha256"], registry.PREREG_SHA256)
        self.assertEqual(receipt["data_manifest_sha256"], S.sha256_file(registry.MANIFEST_PATH))
        self.assertEqual(receipt["source"]["commit"], S.FAKE_COMMIT)
        self.assertIs(receipt["source"]["tree_clean"], True)
        self.assertEqual(receipt["attests"], "integrity and origin only; not quality")
        out = kernel.advise({"features": dict(VALID_FEATURES)})
        self.assertIn(out["advisory"], ("ALERT", "NO_ALERT", "ABSTAIN"))

    # -- model ---------------------------------------------------------------------------
    def test_model_byte_flip_is_refused(self):
        for offset in (0, len(self.model.read_bytes()) // 2, len(self.model.read_bytes()) - 2):
            with self.subTest(offset=offset):
                copy = self.tmp / f"model_{offset}.json"
                copy.write_bytes(self.model.read_bytes())
                _flip_byte(copy, offset)
                self.assertRefused("model artifact hash does not match receipt", model=copy)

    def test_consistent_receipt_for_invalid_model_is_still_refused(self):
        artifact = json.loads(self.model.read_text(encoding="utf-8"))
        artifact["authority"] = "full_authority"
        bad = self.tmp / "bad_model.json"
        bad.write_bytes((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        _rewrite_receipt(self.receipt, lambda r: r.update(model_sha256=S.sha256_file(bad)))
        self.assertRefused("model truth-boundary mismatch", model=bad)

    def test_missing_model_and_missing_receipt_are_refused(self):
        self.assertRefused("unable to read model artifact", model=self.tmp / "absent.json")
        self.assertRefused("unable to read artifact receipt", receipt=self.tmp / "absent.json")

    # -- kernel self-hash ------------------------------------------------------------------
    def test_kernel_byte_flip_is_refused(self):
        source = registry.KERNEL_PATH.read_bytes()
        identical, _path = S.load_kernel_copy(self.tmp, source.decode("utf-8"))
        identical.OpsHealthKernel(self.model, self.receipt)  # an exact copy hashes the same
        docstring_at = source.index(b"fail-closed and advisory-only")
        flipped = bytearray(source)
        flipped[docstring_at] ^= 0x01  # 'f' -> 'g' inside the module docstring
        mutant, path = S.load_kernel_copy(self.tmp, bytes(flipped).decode("utf-8"))
        self.assertNotEqual(S.sha256_file(path), S.sha256_file(registry.KERNEL_PATH))
        self.assertRefused("kernel sha256 does not match receipt", module=mutant)

    # -- receipt schema and origin -----------------------------------------------------------
    def test_wrong_schema_is_refused(self):
        _rewrite_receipt(self.receipt, lambda r: r.update(schema="szl-oac/transport-health-artifact-receipt/v1"))
        self.assertRefused("artifact receipt schema mismatch")

    def test_extra_or_missing_receipt_key_is_refused(self):
        _rewrite_receipt(self.receipt, lambda r: r.update(note_quality="high"))
        self.assertRefused("artifact receipt fields mismatch")

    def test_missing_receipt_key_is_refused(self):
        _rewrite_receipt(self.receipt, lambda r: r.pop("data_manifest_sha256"))
        self.assertRefused("artifact receipt fields mismatch")

    def test_duplicate_receipt_key_is_refused(self):
        text = self.receipt.read_text(encoding="utf-8")
        text = text.replace('"attests":', '"attests": "x",\n  "attests":', 1)
        self.receipt.write_text(text, encoding="utf-8")
        self.assertRefused("duplicate JSON key")

    def test_bad_commit_is_refused(self):
        for commit in ("not-a-commit", "0123456789abcdef", S.FAKE_COMMIT[:-1],
                       S.FAKE_COMMIT.upper(), S.FAKE_COMMIT + "0", None, 7):
            with self.subTest(commit=commit):
                _rewrite_receipt(self.receipt, lambda r: r["source"].update(commit=commit))
                self.assertRefused("receipt source commit is not a 40-hex")

    def test_tree_clean_false_via_test_override_is_refused(self):
        info = S.freeze_dev(self.tmp / "dirty", tree_clean=False)
        receipt = json.loads(Path(info["receipt_path"]).read_text(encoding="utf-8"))
        self.assertIs(receipt["source"]["tree_clean"], False)
        self.assertRefused("receipt source tree was not clean", model=Path(info["model_path"]),
                           receipt=Path(info["receipt_path"]))

    def test_bad_digest_format_and_attestation_are_refused(self):
        _rewrite_receipt(self.receipt, lambda r: r.update(generator_sha256="ABC"))
        self.assertRefused("receipt.generator_sha256 is not a valid lower-case hex digest")
        info = S.freeze_dev(self.tmp / "b")
        _rewrite_receipt(Path(info["receipt_path"]), lambda r: r.update(attests="quality verified"))
        self.assertRefused("attestation text mismatch", model=Path(info["model_path"]),
                           receipt=Path(info["receipt_path"]))

    def test_generator_digest_must_match_model(self):
        _rewrite_receipt(self.receipt, lambda r: r.update(generator_sha256="0" * 64))
        self.assertRefused("generator sha256 differs between model and receipt")

    def test_cli_refuses_tampered_model(self):
        _flip_byte(self.model, 10)
        inp = self.tmp / "in.json"
        inp.write_text(json.dumps({"features": VALID_FEATURES}), encoding="utf-8")
        env = dict(os.environ, PYTHONUTF8="1")
        proc = subprocess.run(
            [sys.executable, "-I", "-B", str(registry.KERNEL_PATH), "--model", str(self.model),
             "--receipt", str(self.receipt), "--input", str(inp)],
            capture_output=True, text=True, env=env, timeout=120,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertIn("model artifact hash does not match receipt", proc.stderr)


class FreezeRuleTest(unittest.TestCase):
    """freeze.py refuses a dirty tree for v2/research + kernel and records HEAD (temp repo)."""

    def _git(self, repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=oac-test", "-c", "user.email=oac-test@localhost",
             "-c", "commit.gpgsign=false", *args],
            check=True, capture_output=True, text=True,
        ).stdout

    def test_dirty_tree_refused_clean_tree_records_head(self):
        with S.TempDir() as tmp:
            repo = tmp / "repo"
            (repo / "v2" / "research").mkdir(parents=True)
            (repo / "v2" / "ops-health").mkdir(parents=True)
            (repo / "v2" / "research" / "a.py").write_text("A = 1\n", encoding="utf-8")
            (repo / "v2" / "ops-health" / "ops_health.py").write_text("K = 1\n", encoding="utf-8")
            (repo / "notes.txt").write_text("n\n", encoding="utf-8")
            self._git(repo, "init", "-q")
            self._git(repo, "add", "v2/research/a.py", "v2/ops-health/ops_health.py", "notes.txt")
            self._git(repo, "commit", "-q", "-m", "fixture")
            head = self._git(repo, "rev-parse", "HEAD").strip()
            artifact = S.candidate()
            # dirty research file -> refused, nothing written
            (repo / "v2" / "research" / "a.py").write_text("A = 2\n", encoding="utf-8")
            with self.assertRaises(freeze.FreezeRefused):
                freeze.freeze(artifact, tmp / "out1", selection=S.SELECTION, root=repo)
            self.assertFalse((tmp / "out1").exists())
            # untracked kernel-adjacent file under v2/research -> refused
            (repo / "v2" / "research" / "a.py").write_text("A = 1\n", encoding="utf-8")
            (repo / "v2" / "research" / "new.py").write_text("", encoding="utf-8")
            with self.assertRaises(freeze.FreezeRefused):
                freeze.freeze(artifact, tmp / "out2", selection=S.SELECTION, root=repo)
            (repo / "v2" / "research" / "new.py").unlink()
            # dirty kernel -> refused
            (repo / "v2" / "ops-health" / "ops_health.py").write_text("K = 2\n", encoding="utf-8")
            with self.assertRaises(freeze.FreezeRefused):
                freeze.freeze(artifact, tmp / "out3", selection=S.SELECTION, root=repo)
            (repo / "v2" / "ops-health" / "ops_health.py").write_text("K = 1\n", encoding="utf-8")
            # a dirty file outside the guarded paths does not block; HEAD is recorded
            (repo / "notes.txt").write_text("changed\n", encoding="utf-8")
            info = freeze.freeze(artifact, tmp / "out4", selection=S.SELECTION, root=repo)
            self.assertEqual(info["commit"], head)
            self.assertIs(info["tree_clean"], True)
            receipt = json.loads(Path(info["receipt_path"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["source"]["commit"], head)
            # never overwrite
            with self.assertRaises(freeze.FreezeRefused):
                freeze.freeze(artifact, tmp / "out4", selection=S.SELECTION, root=repo)

    def test_non_canonical_artifact_is_refused(self):
        artifact = S.candidate()
        artifact["intercept"] = artifact["intercept"] + 1e-14
        with S.TempDir() as tmp, self.assertRaises(freeze.FreezeRefused):
            S.freeze_dev(tmp / "x", artifact)


if __name__ == "__main__":
    unittest.main()
