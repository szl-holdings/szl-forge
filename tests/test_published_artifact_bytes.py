"""SAMPLE keys and bytes only; mutations must never become production authority."""
from __future__ import annotations

import base64
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

CHECKOUT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("published_artifact_gate", CHECKOUT / "tools/verify_published_artifact_bytes.py")
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class PublishedArtifactByteGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="szl-SAMPLE-artifact-gate-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "source"
        self.snapshot = Path(self.directory.name) / "snapshot"
        self.root.mkdir()
        self.snapshot.mkdir()
        self.key = Ed25519PrivateKey.generate()  # Ephemeral SAMPLE key, never persisted.
        self.key_doc = self.public_key(self.key)
        self.data = b"SAMPLE opaque adapter bytes; this is not a trained model.\n"
        contract = json.loads((CHECKOUT / "publishing/model-source-bindings.json").read_bytes())
        original = next(item for item in contract["artifacts"] if item["repo_id"] == gate.TARGET)
        self.artifact = copy.deepcopy(original)
        self.revision = self.artifact["hub_revision"]
        sha = hashlib.sha256(self.data).hexdigest()
        self.artifact["expected_weight_sha256"] = {"adapter_model.safetensors": sha}
        self.artifact["adapter_binding"]["adapter_sha256"] = sha
        self.artifact["source_files"] = ["receiptagent/owner_pubkey.json"] + [self.artifact["source_path"] + "/" + path for path in ("candidate.json", "publication.json", "receipts/training_receipt.signed.json", "receipts/eval_receipt.signed.json", "receipts/publication_receipt.signed.json")]
        base = self.artifact["adapter_binding"]
        training = {"trainingImplementation": {"repo_id": base["base_model_repo_id"], "revision": base["base_model_revision"]}, "adapterModelSha256": sha, "sample": True}
        self.training = self.sign(training)
        self.evaluation = self.sign({"trainingReceiptCanonicalSha256": self.canonical_sha(self.training), "evalTotal": 1, "evalContractValid": 1, "sample": True})
        release = {"repository": {"repoId": gate.TARGET, "releaseRevision": base["owner_attested_release_revision"]}, "evidence": {"adapterModelSha256": sha, "trainingReceiptCanonicalSha256": self.canonical_sha(self.training), "evaluationReceiptCanonicalSha256": self.canonical_sha(self.evaluation)}, "immutableGpuInference": {"baseImplementationRevision": base["base_model_revision"]}, "sample": True}
        self.release = self.sign(release)
        files = {"owner_pubkey.json": self.key_doc, "candidate.json": {"sample": True}, "publication.json": {"sample": True}, "adapter_config.json": {"base_model_name_or_path": base["base_model_repo_id"], "revision": base["base_model_revision"]}, "receipts/training_receipt.signed.json": self.training, "receipts/eval_receipt.signed.json": self.evaluation, "receipts/publication_receipt.signed.json": self.release}
        for name, value in files.items():
            self.write(self.snapshot / name, value)
            if name != "adapter_config.json":
                source_name = "receiptagent/owner_pubkey.json" if name == "owner_pubkey.json" else self.artifact["source_path"] + "/" + name
                self.write(self.root / source_name, value)
        (self.snapshot / "adapter_model.safetensors").write_bytes(self.data)
        self.write(self.snapshot / "hub-metadata.json", {"id": gate.TARGET, "private": False, "sha": self.revision, "siblings": [{"rfilename": "adapter_model.safetensors", "size": len(self.data), "lfs": {"size": len(self.data), "sha256": sha}}]})
        self.write(self.root / "publishing/model-source-bindings.json", {"schema": "szl.model-source-bindings/v2", "source_repository": "szl-holdings/szl-forge", "artifacts": [self.artifact]})
        for args in (("init", "--quiet"), ("add", "."), ("commit", "--quiet", "-m", "SAMPLE fixture source only")):
            cp = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", "-c", "core.hooksPath=" + str(self.root / "unused-hooks"), "-c", "user.name=SAMPLE Fixture", "-c", "user.email=sample@example.invalid", "-C", str(self.root), *args], capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr.decode(errors="replace"))
        self.source_revision = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"]).decode().strip()
        self.openssl = shutil.which("openssl")
        git_openssl = Path(r"C:\Program Files\Git\usr\bin\openssl.exe")
        if self.openssl is None and git_openssl.is_file():
            self.openssl = str(git_openssl)
        self.assertIsNotNone(self.openssl, "Required OpenSSL verifier must be installed; absence is not a skipped gate.")
        patcher = mock.patch.object(gate, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    @staticmethod
    def public_key(key: Ed25519PrivateKey) -> dict:
        der = key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        return {"algo": "ed25519", "publicKeySpkiBase64": base64.b64encode(der).decode("ascii"), "keyId": hashlib.sha256(der).hexdigest()[:16]}

    def sign(self, value: dict, key: Ed25519PrivateKey | None = None) -> dict:
        key = key or self.key
        public = self.public_key(key)
        payload = {**value, "keyId": public["keyId"]}
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return {"payload": payload, "canonical": canonical, "signatureBase64": base64.b64encode(key.sign(canonical.encode("utf-8"))).decode("ascii"), "publicKeySpkiBase64": public["publicKeySpkiBase64"], "keyId": public["keyId"]}

    @staticmethod
    def canonical_sha(wrapper: dict) -> str:
        return hashlib.sha256(wrapper["canonical"].encode("utf-8")).hexdigest()

    def run_gate(self, **overrides: str) -> dict:
        parameters = {"snapshot": self.snapshot, "repo_id": gate.TARGET, "revision": self.revision, "source_revision": self.source_revision, "openssl": self.openssl}
        parameters.update(overrides)
        report = gate.verify(**parameters)
        self.assertEqual(report["boundary"]["production_authorization"], "BLOCKED")
        self.assertIs(report["boundary"]["publication_eligible"], False)
        self.assertEqual(report["boundary"]["key_trust"], "REPO_DECLARED")
        self.assertIs(report["boundary"]["whole_hub_repository_verified"], False)
        return report

    def test_public_sample_bytes_and_both_verifiers_pass_without_authority(self) -> None:
        before = {p.relative_to(self.snapshot).as_posix(): p.read_bytes() for p in self.snapshot.rglob("*") if p.is_file()}
        report = self.run_gate()
        self.assertEqual((report["evidence_class"], report["exit_code"]), ("MEASURED", 0))
        self.assertEqual(report["adapter_bytes"]["bytes"], len(self.data))
        self.assertEqual(set(report["external_signatures"]), {"training", "evaluation", "release"})
        self.assertTrue(all(r["accepted"] for r in report["external_signatures"].values()))
        self.assertEqual(before, {p.relative_to(self.snapshot).as_posix(): p.read_bytes() for p in self.snapshot.rglob("*") if p.is_file()})

    def test_mutated_adapter_is_blocked_even_with_correct_metadata(self) -> None:
        (self.snapshot / "adapter_model.safetensors").write_bytes(self.data[:-1] + b"!")
        report = self.run_gate()
        self.assertEqual((report["evidence_class"], report["failure_code"]), ("BLOCKED", "adapter-byte-hash-or-size-mismatch"))

    def test_metadata_without_actual_adapter_is_unavailable(self) -> None:
        (self.snapshot / "adapter_model.safetensors").unlink()
        report = self.run_gate()
        self.assertEqual((report["evidence_class"], report["exit_code"]), ("UNAVAILABLE", 2))

    def test_valid_wrong_key_cannot_replace_source_declared_owner(self) -> None:
        other = Ed25519PrivateKey.generate()
        self.write(self.snapshot / "owner_pubkey.json", self.public_key(other))
        for name, wrapper in (("training", self.training), ("eval", self.evaluation), ("publication", self.release)):
            self.write(self.snapshot / "receipts" / (name + "_receipt.signed.json"), self.sign(wrapper["payload"], other))
        report = self.run_gate()
        self.assertEqual(report["failure_code"], "source-key-pin-mismatch")

    def test_unsigned_receipt_is_blocked(self) -> None:
        wrapper = copy.deepcopy(self.training)
        wrapper["signatureBase64"] = ""
        self.write(self.snapshot / "receipts/training_receipt.signed.json", wrapper)
        self.assertEqual(self.run_gate()["failure_code"], "unsigned-receipt")

    def test_noncanonical_receipt_is_blocked(self) -> None:
        wrapper = copy.deepcopy(self.training)
        wrapper["canonical"] += " "
        self.write(self.snapshot / "receipts/training_receipt.signed.json", wrapper)
        self.assertEqual(self.run_gate()["failure_code"], "maintained-chain-or-config-rejected")

    def test_valid_signed_wrong_evaluation_chain_is_blocked(self) -> None:
        payload = {**self.evaluation["payload"], "trainingReceiptCanonicalSha256": "0" * 64}
        self.write(self.snapshot / "receipts/eval_receipt.signed.json", self.sign(payload))
        self.assertEqual(self.run_gate()["failure_code"], "maintained-chain-or-config-rejected")

    def test_valid_signed_wrong_publication_chain_is_blocked(self) -> None:
        payload = copy.deepcopy(self.release["payload"])
        payload["evidence"]["evaluationReceiptCanonicalSha256"] = "0" * 64
        self.write(self.snapshot / "receipts/publication_receipt.signed.json", self.sign(payload))
        self.assertEqual(self.run_gate()["failure_code"], "publication-chain-mismatch")

    def test_wrong_exact_source_revision_is_blocked(self) -> None:
        self.assertEqual(self.run_gate(source_revision="0" * 40)["failure_code"], "source-revision-mismatch")

    def test_modified_current_source_is_blocked(self) -> None:
        self.write(self.root / "receiptagent/owner_pubkey.json", self.public_key(Ed25519PrivateKey.generate()))
        self.assertEqual(self.run_gate()["failure_code"], "source-working-copy-modified")

    def test_wrong_hub_revision_metadata_is_blocked(self) -> None:
        metadata = json.loads((self.snapshot / "hub-metadata.json").read_bytes())
        metadata["sha"] = "0" * 40
        self.write(self.snapshot / "hub-metadata.json", metadata)
        self.assertEqual(self.run_gate()["failure_code"], "snapshot-metadata-identity-mismatch")

    def test_missing_openssl_is_unavailable(self) -> None:
        report = self.run_gate(openssl=str(self.root / "not-an-openssl-executable"))
        self.assertEqual((report["evidence_class"], report["exit_code"], report["failure_code"]), ("UNAVAILABLE", 2, "openssl-unavailable"))

    def test_external_verifier_rejection_cannot_be_ignored(self) -> None:
        original = subprocess.run

        def reject(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            if "pkeyutl" in command:
                return subprocess.CompletedProcess(command, 1, b"Signature Verification Failure\n", b"")
            return original(command, **kwargs)

        with mock.patch.object(gate.subprocess, "run", side_effect=reject):
            self.assertEqual(self.run_gate()["failure_code"], "openssl-signature-rejected")

    def test_output_hash_names_the_exact_written_utf8_bytes(self) -> None:
        output = Path(self.directory.name) / "report.json"
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = gate.main(["--snapshot", str(self.snapshot), "--repo-id", gate.TARGET, "--revision", self.revision, "--source-revision", self.source_revision, "--openssl", self.openssl, "--output", str(output)])
        raw = output.read_bytes()
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stream.getvalue())["output_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertNotIn(b"\r\n", raw)

    def test_output_cannot_overwrite_snapshot_input(self) -> None:
        path = self.snapshot / "owner_pubkey.json"
        before = path.read_bytes()
        with redirect_stdout(io.StringIO()):
            result = gate.main(["--snapshot", str(self.snapshot), "--repo-id", gate.TARGET, "--revision", self.revision, "--source-revision", self.source_revision, "--output", str(path)])
        self.assertEqual(result, 1)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
