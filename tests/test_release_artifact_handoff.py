# SPDX-License-Identifier: Apache-2.0
"""SIMULATED artifact fixtures test byte handoff, never model or release quality.

No model, curriculum, held-out case, signer, credential, or provider is used.
The helper's MEASURED label covers observed file hashes only; its unsigned
handoff must not grant publication or establish signature verification.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "release_artifact_handoff.py"
SPEC = importlib.util.spec_from_file_location("forge_release_artifact_handoff_tests", HELPER)
assert SPEC is not None and SPEC.loader is not None
handoff = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = handoff
SPEC.loader.exec_module(handoff)

REPOSITORY = "szl-holdings/szl-forge"
SOURCE = "a" * 40
PROFILES = {
    "heldout-eval": ("heldout-eval.json",),
    "model-bom": ("model-bom.cdx.json",),
    "signed-release": (
        "run-manifest.json", "run-manifest.sig", "run-manifest.crt",
        "weights.sha256", "weights.sha256.sig", "weights.sha256.crt",
        "model-bom.cdx.json.sig", "model-bom.cdx.json.crt",
    ),
}


@pytest.fixture
def context():
    return handoff.make_context(REPOSITORY, SOURCE, "123456", "2")


def stage_files(stage: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in stage.iterdir()}


def create_stage(stage: Path, profile: str = "heldout-eval") -> Path:
    handoff.initialize(stage)
    for index, name in enumerate(PROFILES[profile]):
        (stage / name).write_bytes(f"SIMULATED artifact {index}\n".encode("ascii"))
    return stage


def sealed_stage(tmp_path: Path, context, profile="heldout-eval"):
    stage = create_stage(tmp_path / "stage", profile)
    report = handoff.seal(stage, profile, context)
    return stage, report


def write_receipt(stage: Path, report) -> None:
    (stage / "handoff.json").write_text(json.dumps(report), encoding="utf-8")


def require_symlink(link: Path, target: Path, *, directory=False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"host cannot create this real symlink: {exc}")


def test_profiles_are_closed_and_exact():
    assert handoff.PROFILES == PROFILES
    assert handoff.MAX_FILE_BYTES == 16 * 1024 * 1024
    assert handoff.MAX_RECEIPT_BYTES == 16_384
    assert handoff.RECEIPT == "handoff.json"


def test_context_is_exact_and_does_not_call_git(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pure context validation invoked a subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    assert handoff.make_context(REPOSITORY, SOURCE, "123456", "2") == {
        "repository": REPOSITORY, "git_sha": SOURCE, "run_id": "123456",
        "run_attempt": "2", "workflow": "model-publish-gate",
    }


@pytest.mark.parametrize("field,value", [
    ("repository", "other/szl-forge"), ("repository", "SZL-HOLDINGS/szl-forge"),
    ("repository", REPOSITORY + "\n"), ("repository", None),
    ("git_sha", "A" * 40), ("git_sha", "g" * 40),
    ("git_sha", "a" * 39), ("git_sha", "a" * 41),
    ("git_sha", SOURCE + "\n"), ("git_sha", 123),
    ("run_id", "0"), ("run_id", "01"), ("run_id", "1" * 21),
    ("run_id", 123), ("run_attempt", "0"), ("run_attempt", "02"),
    ("run_attempt", "1\n"), ("run_attempt", True),
])
def test_context_rejects_noncanonical_identity(field, value):
    fields = dict(repository=REPOSITORY, git_sha=SOURCE, run_id="123456", run_attempt="2")
    fields[field] = value
    with pytest.raises(handoff.HandoffError):
        handoff.make_context(**fields)


@pytest.mark.parametrize("value", [None, "", "0", "00", "01", "-1", "+1", " 1",
                                  "1 ", "1\n", "1.0", "1e2", "1/2", "١", "１",
                                  "1" * 21, 1, True, False, b"1"])
def test_artifact_id_rejects_missing_and_malformed_values(value):
    with pytest.raises(handoff.HandoffError):
        handoff.validate_artifact_id(value)


@pytest.mark.parametrize("value", ["1", "123456", "9" * 20])
def test_artifact_id_preserves_canonical_decimal(value):
    assert handoff.validate_artifact_id(value) == value


def test_initialize_creates_fresh_stage_and_missing_parents(tmp_path):
    stage = tmp_path / "new-parent" / "new-stage"
    handoff.initialize(stage)
    assert stage.is_dir()
    assert list(stage.iterdir()) == []


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_initialize_never_reuses_or_overwrites_existing_stage(tmp_path, kind):
    stage = tmp_path / "existing"
    if kind == "directory":
        stage.mkdir()
    else:
        stage.write_bytes(b"SIMULATED preserved")
    with pytest.raises(handoff.HandoffError):
        handoff.initialize(stage)
    assert stage.is_dir() if kind == "directory" else stage.read_bytes() == b"SIMULATED preserved"


def test_initialize_rejects_relative_stage():
    with pytest.raises(handoff.HandoffError):
        handoff.initialize(Path("SIMULATED-relative-stage"))


def test_initialize_rejects_checkout_and_descendants(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(handoff, "CHECKOUT", checkout)
    for stage in (checkout, checkout / "nested-stage"):
        with pytest.raises(handoff.HandoffError):
            handoff.initialize(stage)
    assert list(checkout.iterdir()) == []


@pytest.mark.parametrize("profile", PROFILES)
def test_seal_and_verify_exact_artifacts_without_mutation_or_promotion(tmp_path, context, profile):
    stage = create_stage(tmp_path / "stage", profile)
    original = stage_files(stage)
    sealed = handoff.seal(stage, profile, context)
    assert set(sealed) == {"schema", "context", "profile", "evidence_class",
                           "signature_verification", "publication_eligible", "files"}
    assert sealed["schema"] == "szl.release-artifact-handoff/v1"
    assert sealed["context"] == context
    assert sealed["profile"] == profile
    assert sealed["evidence_class"] == "MEASURED"
    assert sealed["signature_verification"] == "UNAVAILABLE"
    assert sealed["publication_eligible"] is False
    assert sealed["files"] == {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        for name, raw in original.items()
    }
    assert set(stage_files(stage)) == set(original) | {"handoff.json"}
    assert all((stage / name).read_bytes() == raw for name, raw in original.items())
    before = stage_files(stage)
    assert handoff.verify(stage, profile, context) == sealed
    assert stage_files(stage) == before


@pytest.mark.parametrize("profile", ["", "unknown", "../heldout-eval", "heldout-eval\n", None])
def test_unknown_profile_is_not_a_filename_or_fallback(tmp_path, context, profile):
    stage = create_stage(tmp_path / "stage")
    before = stage_files(stage)
    for operation in (handoff.seal, handoff.verify):
        with pytest.raises(handoff.HandoffError):
            operation(stage, profile, context)
    assert stage_files(stage) == before


@pytest.mark.parametrize("phase", ["seal", "verify"])
@pytest.mark.parametrize("mutation", ["missing", "empty", "extra", "directory", "nested"])
def test_exact_flat_nonempty_file_set_is_required(tmp_path, context, phase, mutation):
    stage = create_stage(tmp_path / "stage")
    if phase == "verify":
        handoff.seal(stage, "heldout-eval", context)
    artifact = stage / "heldout-eval.json"
    if mutation == "missing":
        artifact.unlink()
    elif mutation == "empty":
        artifact.write_bytes(b"")
    elif mutation == "extra":
        (stage / "unrequested.json").write_bytes(b"SIMULATED extra")
    elif mutation == "directory":
        artifact.unlink()
        artifact.mkdir()
    else:
        (stage / "nested").mkdir()
    with pytest.raises(handoff.HandoffError):
        getattr(handoff, phase)(stage, "heldout-eval", context)


@pytest.mark.parametrize("phase", ["seal", "verify"])
def test_artifact_size_is_bounded(tmp_path, context, phase):
    stage = create_stage(tmp_path / "stage")
    if phase == "verify":
        handoff.seal(stage, "heldout-eval", context)
    # Sparse simulated bytes exercise the real bound without any model fixture.
    with (stage / "heldout-eval.json").open("wb") as stream:
        stream.truncate(handoff.MAX_FILE_BYTES + 1)
    with pytest.raises(handoff.HandoffError):
        getattr(handoff, phase)(stage, "heldout-eval", context)


def test_seal_refuses_existing_receipt_without_replacing_it(tmp_path, context):
    stage, _ = sealed_stage(tmp_path, context)
    before = stage_files(stage)
    with pytest.raises(handoff.HandoffError):
        handoff.seal(stage, "heldout-eval", context)
    assert stage_files(stage) == before


def test_verify_requires_receipt_and_does_not_create_one(tmp_path, context):
    stage = create_stage(tmp_path / "stage")
    before = stage_files(stage)
    with pytest.raises(handoff.HandoffError):
        handoff.verify(stage, "heldout-eval", context)
    assert stage_files(stage) == before


@pytest.mark.parametrize("field,value", [("git_sha", "b" * 40), ("run_id", "123457"),
                                         ("run_attempt", "3")])
def test_previous_source_run_or_attempt_receipt_cannot_be_reused(tmp_path, context, field, value):
    stage, _ = sealed_stage(tmp_path, context)
    different = dict(context, **{field: value})
    before = stage_files(stage)
    with pytest.raises(handoff.HandoffError):
        handoff.verify(stage, "heldout-eval", different)
    assert stage_files(stage) == before


def test_same_size_artifact_mutation_fails_exact_byte_binding(tmp_path, context):
    stage, _ = sealed_stage(tmp_path, context)
    artifact = stage / "heldout-eval.json"
    artifact.write_bytes(b"x" * artifact.stat().st_size)
    before = stage_files(stage)
    with pytest.raises(handoff.HandoffError):
        handoff.verify(stage, "heldout-eval", context)
    assert stage_files(stage) == before


def test_swapping_receipt_from_other_artifact_payload_fails(tmp_path, context):
    first = create_stage(tmp_path / "first")
    second = create_stage(tmp_path / "second")
    (second / "heldout-eval.json").write_bytes(b"SIMULATED other payload")
    handoff.seal(first, "heldout-eval", context)
    handoff.seal(second, "heldout-eval", context)
    (first / "handoff.json").write_bytes((second / "handoff.json").read_bytes())
    with pytest.raises(handoff.HandoffError):
        handoff.verify(first, "heldout-eval", context)


@pytest.mark.parametrize("mutation", ["extra-envelope", "missing-envelope", "schema", "profile",
                                     "evidence", "signature", "promotion", "extra-context",
                                     "missing-context", "workflow", "hash", "size", "extra-file",
                                     "missing-file", "traversal", "absolute", "extra-file-field"])
def test_receipt_envelope_context_and_file_metadata_are_exact(tmp_path, context, mutation):
    stage, sealed = sealed_stage(tmp_path, context)
    report = copy.deepcopy(sealed)
    metadata = report["files"]["heldout-eval.json"]
    if mutation == "extra-envelope":
        report["approval"] = "SIMULATED-NOT-A-PERMIT"
    elif mutation == "missing-envelope":
        del report["evidence_class"]
    elif mutation in {"schema", "profile", "evidence", "signature", "promotion"}:
        key, value = {"schema": ("schema", "wrong/v1"), "profile": ("profile", "model-bom"),
                      "evidence": ("evidence_class", "REPORTED"),
                      "signature": ("signature_verification", "VERIFIED"),
                      "promotion": ("publication_eligible", True)}[mutation]
        report[key] = value
    elif mutation == "extra-context":
        report["context"]["extra"] = "not allowed"
    elif mutation == "missing-context":
        del report["context"]["run_attempt"]
    elif mutation == "workflow":
        report["context"]["workflow"] = "unrelated-workflow"
    elif mutation == "hash":
        metadata["sha256"] = "0" * 64
    elif mutation == "size":
        metadata["size_bytes"] += 1
    elif mutation == "extra-file":
        report["files"]["extra.json"] = dict(metadata)
    elif mutation == "missing-file":
        del report["files"]["heldout-eval.json"]
    elif mutation in {"traversal", "absolute"}:
        name = "../heldout-eval.json" if mutation == "traversal" else str(tmp_path / "outside.json")
        report["files"][name] = report["files"].pop("heldout-eval.json")
    else:
        metadata["extra"] = "not allowed"
    write_receipt(stage, report)
    before = stage_files(stage)
    with pytest.raises(handoff.HandoffError):
        handoff.verify(stage, "heldout-eval", context)
    assert stage_files(stage) == before


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_size_metadata_requires_exact_integer_not_equal_lookalikes(tmp_path, context, value):
    stage = create_stage(tmp_path / "stage")
    (stage / "heldout-eval.json").write_bytes(b"x")
    report = handoff.seal(stage, "heldout-eval", context)
    report["files"]["heldout-eval.json"]["size_bytes"] = value
    write_receipt(stage, report)
    with pytest.raises(handoff.HandoffError):
        handoff.verify(stage, "heldout-eval", context)


@pytest.mark.parametrize("mutation", ["duplicate-top", "duplicate-nested", "invalid-utf8",
                                     "nonfinite", "array", "trailing", "empty", "oversized"])
def test_receipt_json_is_strict_and_bounded(tmp_path, context, mutation):
    stage, report = sealed_stage(tmp_path, context)
    text = json.dumps(report)
    if mutation == "duplicate-top":
        raw = ('{"schema":' + json.dumps(report["schema"]) + "," + text[1:]).encode()
    elif mutation == "duplicate-nested":
        raw = text.replace('"size_bytes":', '"size_bytes":0,"size_bytes":', 1).encode()
    else:
        raw = {"invalid-utf8": b"\xff", "nonfinite": b'{"size_bytes":NaN}', "array": b"[]",
               "trailing": text.encode() + b"{}", "empty": b"",
               "oversized": b" " * (handoff.MAX_RECEIPT_BYTES + 1)}[mutation]
    (stage / "handoff.json").write_bytes(raw)
    before = stage_files(stage)
    with pytest.raises(handoff.HandoffError):
        handoff.verify(stage, "heldout-eval", context)
    assert stage_files(stage) == before


@pytest.mark.parametrize("phase", ["seal", "verify"])
def test_artifact_symlink_is_rejected_with_real_filesystem_control(tmp_path, context, phase):
    stage = create_stage(tmp_path / "stage")
    if phase == "verify":
        handoff.seal(stage, "heldout-eval", context)
    artifact = stage / "heldout-eval.json"
    target = tmp_path / "outside-artifact"
    target.write_bytes(artifact.read_bytes())
    artifact.unlink()
    require_symlink(artifact, target)
    with pytest.raises(handoff.HandoffError):
        getattr(handoff, phase)(stage, "heldout-eval", context)
    assert target.read_bytes() == b"SIMULATED artifact 0\n"


def test_receipt_symlink_is_rejected(tmp_path, context):
    stage, _ = sealed_stage(tmp_path, context)
    receipt = stage / "handoff.json"
    target = tmp_path / "outside-receipt"
    target.write_bytes(receipt.read_bytes())
    receipt.unlink()
    require_symlink(receipt, target)
    with pytest.raises(handoff.HandoffError):
        handoff.verify(stage, "heldout-eval", context)


def test_stage_with_symlink_ancestor_is_rejected(tmp_path):
    parent = tmp_path / "real-parent"
    parent.mkdir()
    alias = tmp_path / "alias-parent"
    require_symlink(alias, parent, directory=True)
    with pytest.raises(handoff.HandoffError):
        handoff.initialize(alias / "stage")
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize("value,expected_success", [("0", False), ("01", False), ("", False),
                                                   ("123456", True)])
def test_artifact_id_cli_remains_fail_closed_under_optimized_python(value, expected_success):
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-O", str(HELPER), "check-artifact-id", "--artifact-id", value],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert (result.returncode == 0) is expected_success


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="real FIFO creation unavailable on this host")
def test_fifo_artifact_is_rejected_before_blocking_open(tmp_path, context):
    stage = create_stage(tmp_path / "stage")
    artifact = stage / "heldout-eval.json"
    artifact.unlink()
    os.mkfifo(artifact)
    # A child timeout makes an accidental blocking open a test failure.
    env = dict(os.environ, GITHUB_REPOSITORY=REPOSITORY,
               GITHUB_SHA=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               GITHUB_RUN_ID="123456", GITHUB_RUN_ATTEMPT="2")
    result = subprocess.run([sys.executable, "-I", "-B", str(HELPER), "seal", "--stage", str(stage),
                             "--profile", "heldout-eval"], env=env, capture_output=True, timeout=10)
    assert result.returncode == 1
    assert b"REGULAR_FILE_REQUIRED" in result.stderr


def test_real_optimized_cli_binds_checkout_and_roundtrips_without_authorization(tmp_path):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    env = dict(os.environ, GITHUB_REPOSITORY=REPOSITORY, GITHUB_SHA=head,
               GITHUB_RUN_ID="123456", GITHUB_RUN_ATTEMPT="2")
    stage = tmp_path / "cli-stage"

    def invoke(command, environment):
        argv = [sys.executable, "-I", "-B", "-O", str(HELPER), command, "--stage", str(stage)]
        if command != "init":
            argv.extend(["--profile", "heldout-eval"])
        return subprocess.run(argv, env=environment, capture_output=True, timeout=15)

    assert invoke("init", dict(env, GITHUB_SHA="b" * 40)).returncode == 1
    assert not stage.exists()
    assert invoke("init", env).returncode == 0
    (stage / "heldout-eval.json").write_bytes(b"SIMULATED CLI evidence")
    assert invoke("seal", env).returncode == 0
    assert invoke("verify", env).returncode == 0
    before = stage_files(stage)
    assert invoke("verify", dict(env, GITHUB_RUN_ATTEMPT="3")).returncode == 1
    assert stage_files(stage) == before
    assert json.loads(before["handoff.json"])["publication_eligible"] is False
