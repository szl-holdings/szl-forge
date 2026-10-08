#!/usr/bin/env python3
"""Verify one immutable local Forge snapshot; never load weights or publish.

Legacy Forge Ed25519 canonical-JSON receipts are not DSSE/ECDSA receipts.
The maintained publisher verifies the existing signed chain; required OpenSSL
checks the same canonical bytes. Snapshot origin remains caller-declared.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_VERIFIER = Path(__file__).with_name("publish_model_source_bindings.py")
TARGET = "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2"
MAX_JSON_BYTES = 2_000_000
MAX_ADAPTER_BYTES = 512_000_000
SHA = re.compile(r"[0-9a-f]{40}\Z")


class GateError(RuntimeError):
    def __init__(self, state: str, code: str):
        super().__init__(code)
        self.state, self.code = state, code


def require(condition: bool, code: str) -> None:
    if not condition:
        raise GateError("BLOCKED", code)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(raw: bytes) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict:
        result: dict = {}
        for key, value in values:
            require(key not in result, "duplicate-json-key")
            result[key] = value
        return result

    def constant(_: str) -> None:
        raise GateError("BLOCKED", "non-finite-json")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def local_file(directory: Path, name: str, limit: int) -> Path:
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe-input-path")
    original = directory / relative
    if not original.exists():
        raise GateError("UNAVAILABLE", "required-file-missing")
    resolved = original.resolve()
    require(resolved.is_relative_to(directory.resolve()) and not original.is_symlink(), "linked-input-path")
    require(resolved.is_file(), "input-is-not-regular-file")
    require(resolved.stat().st_size <= limit, "input-size-bound-exceeded")
    return resolved


def read_json(directory: Path, name: str) -> tuple[dict, bytes]:
    raw = local_file(directory, name, MAX_JSON_BYTES).read_bytes()
    value = _json(raw)
    require(isinstance(value, dict), "json-must-be-object")
    return value, raw


def git(*args: str) -> bytes:
    try:
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GateError("UNAVAILABLE", "source-git-unavailable") from error
    if result.returncode:
        raise GateError("UNAVAILABLE", "source-git-object-unavailable")
    return result.stdout


def maintained_verifier() -> Any:
    try:
        spec = importlib.util.spec_from_file_location("_forge_artifact_binding_verifier", MAINTAINED_VERIFIER)
        if spec is None or spec.loader is None:
            raise ImportError
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (ImportError, OSError) as error:
        raise GateError("UNAVAILABLE", "maintained-verifier-unavailable") from error


def source_contract(source_revision: str, repo_id: str, revision: str) -> tuple[dict, dict]:
    require(bool(SHA.fullmatch(source_revision)), "source-revision-not-exact")
    require(git("rev-parse", "HEAD").decode().strip() == source_revision, "source-revision-mismatch")
    require(repo_id == TARGET, "unsupported-repository")
    require(bool(SHA.fullmatch(revision)), "hub-revision-not-exact")
    path = "publishing/model-source-bindings.json"
    local, _ = read_json(ROOT, path)
    committed = git("show", f"{source_revision}:{path}")
    require(local == _json(committed), "source-contract-working-copy-mismatch")
    require(local.get("schema") == "szl.model-source-bindings/v2" and local.get("source_repository") == "szl-holdings/szl-forge", "source-contract-schema")
    matches = [item for item in local.get("artifacts", []) if item.get("repo_id") == repo_id]
    require(len(matches) == 1, "source-target-not-unique")
    artifact = matches[0]
    require(artifact.get("binding_mode") == "VERIFY_IMMUTABLE_RELEASE" and artifact.get("hub_revision") == revision, "immutable-hub-pin-mismatch")
    require(artifact.get("artifact_class") == "fine_tuned_adapter", "artifact-profile-mismatch")
    source_files = artifact.get("source_files")
    require(isinstance(source_files, list) and bool(source_files) and len(source_files) == len(set(source_files)), "source-file-set-invalid")
    source_hashes = {}
    for name in [path, *source_files]:
        local_file(ROOT, name, MAX_JSON_BYTES)
        require(git("diff", "--name-only", source_revision, "--", name).strip() == b"", "source-working-copy-modified")
        raw = git("show", f"{source_revision}:{name}")
        source_hashes[name] = digest(raw)
    return artifact, {"revision": source_revision, "git_tree": git("rev-parse", source_revision + "^{tree}").decode().strip(), "committed_file_sha256": source_hashes, "comparison_uses_git_blob_bytes": True, "training_source_correctness": "UNKNOWN"}


def find_openssl(value: str | None) -> tuple[str, dict]:
    selected = str(Path(value).resolve()) if value else shutil.which("openssl")
    if not selected or not Path(selected).is_file():
        raise GateError("UNAVAILABLE", "openssl-unavailable")
    try:
        cp = subprocess.run([selected, "version"], capture_output=True, timeout=10, env=public_process_env())
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GateError("UNAVAILABLE", "openssl-version-unavailable") from error
    if cp.returncode:
        raise GateError("UNAVAILABLE", "openssl-version-unavailable")
    return selected, {"version": cp.stdout.decode("utf-8", errors="replace")[:2048].strip(), "executable_sha256": digest(Path(selected).read_bytes())}


def public_process_env() -> dict[str, str]:
    # The verifier subprocess only needs ordinary OS paths, never provider keys.
    return {key: value for key, value in os.environ.items() if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE"}}


def verify_openssl(executable: str, wrappers: dict[str, dict]) -> dict:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    results = {}
    with tempfile.TemporaryDirectory(prefix="szl-public-ed25519-") as directory:
        temp = Path(directory)
        for role, wrapper in wrappers.items():
            payload = wrapper["canonical"].encode("utf-8")
            signature = base64.b64decode(wrapper["signatureBase64"], validate=True)
            der = base64.b64decode(wrapper["publicKeySpkiBase64"], validate=True)
            key = serialization.load_der_public_key(der)
            require(isinstance(key, Ed25519PublicKey), "signature-algorithm-mismatch")
            (temp / "public.pem").write_bytes(key.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
            (temp / "signature.bin").write_bytes(signature)
            (temp / "canonical.bin").write_bytes(payload)
            command = [executable, "pkeyutl", "-verify", "-pubin", "-inkey", "public.pem", "-rawin", "-in", "canonical.bin", "-sigfile", "signature.bin"]
            try:
                cp = subprocess.run(command, cwd=temp, capture_output=True, timeout=10, env=public_process_env())
            except (OSError, subprocess.TimeoutExpired) as error:
                raise GateError("UNAVAILABLE", "openssl-verification-unavailable") from error
            require(cp.returncode == 0, "openssl-signature-rejected")
            results[role] = {"evidence_class": "MEASURED", "accepted": True, "canonical_sha256": digest(payload), "signature_sha256": digest(signature), "spki_der_sha256": digest(der), "argv": command, "cwd_role": "removed_public_only_temporary_directory", "returncode": cp.returncode, "stdout": cp.stdout.decode("utf-8", errors="replace")[:2048], "stderr": cp.stderr.decode("utf-8", errors="replace")[:2048]}
    return results


def verify(snapshot: Path, repo_id: str, revision: str, source_revision: str, openssl: str | None = None) -> dict:
    report: dict = {"schema": "szl.forge-published-artifact-byte-verification/v1", "observed_at": datetime.now(timezone.utc).isoformat(), "repo_id": repo_id if repo_id == TARGET else "UNSUPPORTED", "evidence_class": "UNAVAILABLE", "exit_code": 2, "scope": "ONE_V2_ADAPTER_AND_DECLARED_RECEIPTS_ONLY", "boundary": {"snapshot_provider_origin": "DECLARED", "provider_download_in_this_command": "NOT RUN", "whole_hub_repository_verified": False, "historical_qualification_source_bytes": "NOT RUN", "key_trust": "REPO_DECLARED", "independent_identity_binding": "UNKNOWN", "production_authorization": "BLOCKED", "publication_eligible": False, "model_execution": "NOT RUN", "held_out_evaluation_replay": "NOT RUN", "remote_code_execution": "NOT RUN", "general_dsse_conformance": "NOT RUN", "model_quality": "UNKNOWN", "numerical_equivalence": "UNKNOWN"}}
    try:
        artifact, source = source_contract(source_revision, repo_id, revision)
        report["source"] = source
        executable, runtime = find_openssl(openssl)
        report["openssl_runtime"] = runtime
        metadata, metadata_raw = read_json(snapshot, "hub-metadata.json")
        require(metadata.get("private") is False and metadata.get("id") == repo_id and metadata.get("sha") == revision, "snapshot-metadata-identity-mismatch")
        report["snapshot"] = {"hub_revision": revision, "metadata_sha256": digest(metadata_raw), "origin_trust": "DECLARED"}
        required_files = artifact["required_hub_files"]
        require(isinstance(required_files, list) and bool(required_files), "required-file-list-missing")
        for name in required_files:
            local_file(snapshot, name, MAX_ADAPTER_BYTES if name.endswith(".safetensors") else MAX_JSON_BYTES)
        receipt_paths = artifact["signed_receipts"]
        key, _ = read_json(snapshot, receipt_paths["public_key"])
        source_key_paths = [p for p in artifact["source_files"] if Path(p).name == Path(receipt_paths["public_key"]).name]
        require(len(source_key_paths) == 1, "source-key-pin-not-unique")
        source_key = _json(git("show", f"{source_revision}:{source_key_paths[0]}"))
        require(key == source_key and key.get("algo") == "ed25519", "source-key-pin-mismatch")
        der = base64.b64decode(key["publicKeySpkiBase64"], validate=True)
        require(key.get("keyId") == digest(der)[:16], "source-key-id-mismatch")
        report["signer"] = {"source_key_path": source_key_paths[0], "key_id": key["keyId"], "spki_der_sha256": digest(der), "pin_scope": "CURRENT_CANONICAL_REPOSITORY_DECLARATION", "key_trust": "REPO_DECLARED"}
        wrappers = {role: read_json(snapshot, receipt_paths[role])[0] for role in ("training", "evaluation", "release")}
        for wrapper in wrappers.values():
            require(set(wrapper) == {"payload", "canonical", "signatureBase64", "publicKeySpkiBase64", "keyId"}, "receipt-wrapper-shape")
            require(isinstance(wrapper["signatureBase64"], str) and bool(wrapper["signatureBase64"]), "unsigned-receipt")
            base64.b64decode(wrapper["signatureBase64"], validate=True)
        binding = maintained_verifier()

        def local_downloader(**kwargs: Any) -> str:
            require(kwargs.get("repo_id") == repo_id and kwargs.get("repo_type") == "model" and kwargs.get("revision") == revision and kwargs.get("token") is None, "downloader-target-mismatch")
            require(kwargs.get("filename") in required_files, "downloader-file-not-allowlisted")
            return str(local_file(snapshot, kwargs["filename"], MAX_JSON_BYTES))

        try:
            chain = binding.signed_receipt_evidence(artifact, revision, None, local_downloader)
            config = binding.adapter_config_evidence(artifact, revision, None, local_downloader)
        except Exception as error:
            if isinstance(error, (ImportError, ModuleNotFoundError)):
                raise GateError("UNAVAILABLE", "crypto-verifier-unavailable") from error
            raise GateError("BLOCKED", "maintained-chain-or-config-rejected") from error
        training_hash = digest(wrappers["training"]["canonical"].encode("utf-8"))
        evaluation_hash = digest(wrappers["evaluation"]["canonical"].encode("utf-8"))
        release_evidence = wrappers["release"]["payload"].get("evidence", {})
        require(release_evidence.get("trainingReceiptCanonicalSha256") == training_hash and release_evidence.get("evaluationReceiptCanonicalSha256") == evaluation_hash, "publication-chain-mismatch")
        report["external_signatures"] = verify_openssl(executable, wrappers)
        report["owner_reported_evaluation"] = {"evidence_class": "REPORTED", "values": chain["held_out_evaluation"], "replayed_here": False}
        report["maintained_chain"] = {"evidence_class": "MEASURED", "accepted": True, "training_canonical_sha256": training_hash, "evaluation_canonical_sha256": evaluation_hash, "release_chains_to_training_and_evaluation": True, "adapter_config": config}
        parity = {}
        for name in [receipt_paths["public_key"], receipt_paths["training"], receipt_paths["evaluation"], receipt_paths["release"], "candidate.json", "publication.json"]:
            source_path = source_key_paths[0] if name == receipt_paths["public_key"] else artifact["source_path"] + "/" + name
            require(source_path in artifact["source_files"], "snapshot-source-path-not-bound")
            raw = local_file(snapshot, name, MAX_JSON_BYTES).read_bytes()
            require(raw == git("show", f"{source_revision}:{source_path}"), "snapshot-source-byte-parity-mismatch")
            parity[name] = {"source_path": source_path, "sha256": digest(raw), "exact_git_blob_bytes_match": True}
        report["source_snapshot_parity"] = parity
        adapter = artifact["adapter_binding"]
        filename = adapter["adapter_file"]
        require(set(artifact["expected_weight_sha256"]) == {filename} and artifact["expected_weight_sha256"][filename] == adapter["adapter_sha256"], "adapter-contract-inconsistent")
        entries = [r for r in metadata.get("siblings", []) if r.get("rfilename") == filename]
        require(len(entries) == 1, "adapter-metadata-not-unique")
        entry, path = entries[0], local_file(snapshot, filename, MAX_ADAPTER_BYTES)
        expected_size = entry.get("size")
        require(type(expected_size) is int and expected_size > 0 and entry.get("lfs", {}).get("sha256") == adapter["adapter_sha256"] and entry.get("lfs", {}).get("size") == expected_size, "adapter-metadata-pin-mismatch")
        before = path.stat()
        measured = hashlib.sha256()
        count = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                measured.update(block)
                count += len(block)
                require(count <= MAX_ADAPTER_BYTES, "adapter-stream-too-large")
        after = path.stat()
        require(before.st_size == after.st_size == count == expected_size and before.st_mtime_ns == after.st_mtime_ns and measured.hexdigest() == adapter["adapter_sha256"], "adapter-byte-hash-or-size-mismatch")
        report["adapter_bytes"] = {"evidence_class": "MEASURED", "path": filename, "bytes": count, "sha256": measured.hexdigest(), "matches_current_contract": True, "matches_signed_training_and_publication": True, "loaded_as_model": False}
        report.update(evidence_class="MEASURED", exit_code=0, byte_and_signature_integrity=True)
    except GateError as error:
        report.update(evidence_class=error.state, exit_code=2 if error.state == "UNAVAILABLE" else 1, failure_code=error.code, byte_and_signature_integrity=False)
    except (ImportError, FileNotFoundError) as error:
        report.update(evidence_class="UNAVAILABLE", exit_code=2, failure_code="dependency-or-input-unavailable", error_class=type(error).__name__, byte_and_signature_integrity=False)
    except Exception as error:
        report.update(evidence_class="BLOCKED", exit_code=1, failure_code="malformed-or-rejected-input", error_class=type(error).__name__, byte_and_signature_integrity=False)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--openssl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output in {Path(__file__).resolve(), MAINTAINED_VERIFIER.resolve()} or output.is_relative_to(args.snapshot.resolve()) or (output.is_relative_to(ROOT) and (ROOT / ".git").exists() and subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", output.relative_to(ROOT).as_posix()], capture_output=True).returncode == 0):
        print(json.dumps({"evidence_class": "BLOCKED", "failure_code": "output-cannot-overwrite-input", "exit_code": 1}))
        return 1
    report = verify(args.snapshot, args.repo_id, args.revision, args.source_revision, args.openssl)
    raw = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    print(json.dumps({"evidence_class": report["evidence_class"], "exit_code": report["exit_code"], "output_sha256": digest(raw), "output_bytes": len(raw), "production_authorization": "BLOCKED", "publication_eligible": False}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
