"""SIMULATED and static checks; no dispatch, credentials, or model evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/model-publish-gate.yml"
BODY = WORKFLOW.read_text(encoding="utf-8")
JOBS = yaml.safe_load(BODY)["jobs"]
EDGES = [
    ("model-bom", "heldout-eval", "HELDOUT", "heldout-eval", "Generate Model BOM"),
    ("sign", "heldout-eval", "HELDOUT", "heldout-eval", "Emit run manifest"),
    ("sign", "model-bom", "BOM", "model-bom", "Emit run manifest"),
    ("conformance", "sign", "SIGNED", "signed-release", "Run receipt conformance vectors"),
    ("publish", "heldout-eval", "HELDOUT", "heldout-eval", "Assert publication gate"),
    ("publish", "model-bom", "BOM", "model-bom", "Assert publication gate"),
    ("publish", "sign", "SIGNED", "signed-release", "Assert publication gate"),
]


@pytest.mark.parametrize("consumer,producer,prefix,profile,use", EDGES)
def test_every_consumer_uses_exact_producer_id_then_verifies_before_reading(
        consumer, producer, prefix, profile, use):
    job = JOBS[consumer]
    needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
    assert producer in needs
    expression = "${{ needs." + producer + ".outputs.artifact_id }}"
    assert job["env"][prefix + "_ARTIFACT_ID"] == expression
    steps = job["steps"]
    downloads = [(i, step) for i, step in enumerate(steps)
                 if step.get("uses", "").startswith("actions/download-artifact@")
                 and step["with"].get("artifact-ids") == expression]
    assert len(downloads) == 1
    index, download = downloads[0]
    assert download["with"]["path"] == "${{ env." + prefix + "_STAGE }}"
    assert download["with"]["merge-multiple"] is True
    assert download["with"]["digest-mismatch"] == "error"
    assert not {"name", "pattern", "github-token", "repository", "run-id"} & set(download["with"])
    check = 'check-artifact-id --artifact-id "${' + prefix + '_ARTIFACT_ID}"'
    initialize = 'init --stage "${' + prefix + '_STAGE}"'
    before = [step.get("run", "") for step in steps[:index]]
    assert any(check in script and initialize in script and script.index(check) < script.index(initialize)
               and "set -euo pipefail" in script for script in before)
    consume_index = next(i for i, step in enumerate(steps) if step.get("name") == use)
    verification = 'verify --stage "${' + prefix + '_STAGE}" --profile ' + profile
    assert any(verification in step.get("run", "") for step in steps[index + 1:consume_index])


@pytest.mark.parametrize("producer,prefix,profile", [
    ("heldout-eval", "HELDOUT", "heldout-eval"),
    ("model-bom", "BOM", "model-bom"),
    ("sign", "SIGNED", "signed-release"),
])
def test_producers_seal_exact_stage_and_export_artifact_identity(producer, prefix, profile):
    job = JOBS[producer]
    assert job["outputs"]["artifact_id"] == "${{ steps.upload.outputs.artifact-id }}"
    index, upload = next((i, s) for i, s in enumerate(job["steps"]) if s.get("id") == "upload")
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["path"] == "${{ env." + prefix + "_STAGE }}"
    assert upload["with"]["name"].endswith("-${{ github.run_id }}-${{ github.run_attempt }}")
    assert not upload["with"].get("overwrite", False)
    assert ('seal --stage "${' + prefix + '_STAGE}" --profile ' + profile
            in job["steps"][index - 1]["run"])


def test_no_tracked_snapshots_or_untrusted_expressions_in_shell_source():
    for job in JOBS.values():
        for step in job["steps"]:
            script = step.get("run", "")
            assert "${{" not in script
            assert not re.search(r"(?<![A-Za-z_])(release|bom)/", script)
            for name, value in step.get("env", {}).items():
                if name.endswith("_STAGE"):
                    assert value.startswith("${{ runner.temp }}/")
                    assert "${{ github.run_id }}-${{ github.run_attempt }}" in value
    assert "--signed-manifest" not in BODY
    assert "--conformance-passed" not in BODY


def test_source_pins_permissions_and_protected_publication_are_unchanged():
    workflow = yaml.safe_load(BODY)
    assert workflow["permissions"] == {"contents": "read"}
    assert JOBS["publish"]["environment"] == "hub-production"
    assert set(JOBS["publish"]["needs"]) == {"heldout-eval", "model-bom", "sign", "conformance"}
    credentials = []
    for name, job in JOBS.items():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"] == {"ref": "${{ github.sha }}", "persist-credentials": False}
            if "HF_TOKEN" in step.get("env", {}):
                credentials.append((name, step["name"]))
    assert credentials == [("publish", "Upload to Hub (gated)")]


def test_real_final_assertion_keeps_unsigned_nonconformant_bytes_blocked(tmp_path):
    # Execute only the checked-in assertion's Python heredoc, never workflow jobs.
    script = next(s["run"] for s in JOBS["publish"]["steps"] if s.get("name") == "Assert publication gate")
    code = script.split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    signed = tmp_path / "signed"
    output = tmp_path / "output"
    signed.mkdir()
    output.mkdir()
    manifest = {"eval": {"heldout_passed": True, "refusal_no_regression": True},
                "model_bom_sha256": "a" * 64, "signatures": {"manifest": False},
                "conformance": {"all_vectors_passed": False}, "release": {"kind": "patch"}}
    (signed / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    env = dict(os.environ, SIGNED_STAGE=str(signed), PUBLICATION_STAGE=str(output), ALLOW_MAJOR="no")
    result = subprocess.run([sys.executable, "-I", "-B", "-O", "-c", code], env=env,
                            capture_output=True, timeout=15, check=False)
    assert result.returncode == 1, result.stderr
    verdict = json.loads((output / "publication-gate.json").read_text())
    assert verdict["publication_eligible"] is False
    assert verdict["checks"]["signed"] is False
    assert verdict["checks"]["conformance_passed"] is False
