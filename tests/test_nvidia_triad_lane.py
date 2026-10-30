"""Optional NVIDIA triad lane.

Production Nemo stays pinned to commit 810231a (package 0.4.0). This module
does not edit that pin, publish a package, or deploy a runtime. The contract
checks run only when SZL_FORGE_NEMO_TRIAD_LANE=1, inside a temporary
environment pinned to szl-nemo 116630d2 (package 0.4.2).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_COMMIT = "810231a531188bb569e3faa17396386eb0a5e260"
TRIAD_COMMIT = "116630d2274731998124d7af089a9383074ce476"
TRIAD_URL = (
    "git+https://github.com/szl-holdings/szl-nemo.git@" + TRIAD_COMMIT
)
PRODUCTION_COMMIT_PATHS = (
    ".github/workflows/production-inference-v2.yml",
    "LIVE-GOVERNED-INFERENCE-V2.md",
    "inference/production_control_plane.v2.json",
    "spaces/szl-model-inference-lab/app_governed.py",
    "spaces/szl-model-inference-lab/requirements.txt",
    "spaces/szl-model-inference-lab/tests/test_governed_app.py",
    "spaces/szl-model-inference-lab/verify_governed_live.py",
)
PRODUCTION_VERSION_PATHS = (
    ".github/workflows/live-governed-inference-space-v2.yml",
    ".github/workflows/production-inference-v2.yml",
    "spaces/szl-model-inference-lab/app_governed.py",
    "spaces/szl-model-inference-lab/verify_governed_live.py",
    "tests/test_production_runtime_integration.py",
)

CHECKER = textwrap.dedent(
    """
    import importlib.metadata
    import json
    import math
    import pathlib
    import sys
    import unittest

    from szl_nemo.triad import evaluate_triad

    PIN = "116630d2274731998124d7af089a9383074ce476"
    version = importlib.metadata.version("szl-nemo")
    if version != "0.4.2":
        raise SystemExit("pinned package version is " + version)
    direct = importlib.metadata.distribution("szl-nemo").read_text("direct_url.json")
    if direct is None or PIN not in direct:
        raise SystemExit("installed szl-nemo is not commit " + PIN)

    hex64 = "ab" * 32
    other = "cd" * 32

    def document():
        return {
            "authorization": {
                "principal_id_sha256": hex64,
                "tenant_id_sha256": other,
                "access_decision": "ALLOW",
                "policy_revision": "sha256:" + ("e" * 64),
                "request_sha256": "f" * 64,
            },
            "proposal": {
                "family": "NVIDIA_NEMOTRON",
                "authority": "PROPOSAL_ONLY",
                "output_sha256": "1" * 64,
                "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                "model_revision": "a" * 40,
            },
            "energy": {"joules": None, "label": "UNAVAILABLE", "source": "NVML_ABSENT"},
            "lambda": {"scores": [1.0, 0.25]},
        }

    expected_keys = {
        "schema",
        "witness_identity",
        "authorization",
        "proposal",
        "energy",
        "lambda",
        "nvidia",
        "darpa",
        "dsse",
        "receipt_status",
        "decision",
        "reasons",
        "record_sha256",
    }
    null_energy = evaluate_triad(document())
    null_record = null_energy.record()
    if set(null_record) != expected_keys:
        raise SystemExit("record shape drifted: " + str(sorted(null_record)))
    if null_record["schema"] != "szl.nemo.triad.v1":
        raise SystemExit("schema is not szl.nemo.triad.v1")
    if null_record["proposal"]["family"] != "NVIDIA_NEMOTRON":
        raise SystemExit("proposal family drifted")
    if null_record["proposal"]["authority"] != "PROPOSAL_ONLY":
        raise SystemExit("proposal is not PROPOSAL_ONLY")
    if null_record["nvidia"]["role"] != "PROPOSAL_ONLY":
        raise SystemExit("nvidia role is not PROPOSAL_ONLY")
    if null_record["energy"]["joules"] is not None:
        raise SystemExit("absent NVML did not stay null")
    if null_record["energy"]["label"] != "UNAVAILABLE":
        raise SystemExit("absent NVML label drifted")
    if null_record["energy"]["source"] != "NVML_ABSENT":
        raise SystemExit("absent NVML source drifted")
    if not math.isclose(null_record["lambda"]["score"], 0.5):
        raise SystemExit("modeled geometric mean drifted")
    if null_record["lambda"]["score_label"] != "MODELED":
        raise SystemExit("lambda score is not MODELED")
    if null_record["lambda"]["can_authorize"] is not False:
        raise SystemExit("lambda can authorize")
    if null_record["lambda"]["unconditional_uniqueness"] != "FALSE_AS_STATED":
        raise SystemExit("uniqueness label drifted")
    if null_record["receipt_status"] != "UNSIGNED_HONEST":
        raise SystemExit("receipt is not UNSIGNED_HONEST")
    if null_record["dsse"] is not None:
        raise SystemExit("unsigned record carries a DSSE envelope")

    measured = document()
    measured["energy"] = {"joules": 2, "label": "MEASURED", "source": "NVML"}
    measured_record = evaluate_triad(measured).record()
    if measured_record["energy"]["joules"] != 2.0:
        raise SystemExit("finite NVML joules were not kept")
    if measured_record["energy"]["label"] != "MEASURED":
        raise SystemExit("finite NVML joules were not labeled MEASURED")
    if measured_record["lambda"]["can_authorize"] is not False:
        raise SystemExit("measured energy made lambda authorizing")

    leaked = document()
    leaked["raw_content"] = "do-not-persist-this-prompt"
    blocked = evaluate_triad(leaked)
    if blocked.decision != "BLOCK":
        raise SystemExit("raw content was accepted")
    if not any("raw content" in reason for reason in blocked.reasons):
        raise SystemExit("raw content reason missing")
    if "do-not-persist-this-prompt" in blocked.body_json:
        raise SystemExit("raw content was copied into the record")

    suite = pathlib.Path(sys.argv[1])
    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.discover(str(suite), pattern="test_triad.py")
    )
    if not result.wasSuccessful():
        raise SystemExit("pinned Nemo triad tests failed")
    print("triad-lane-ok")
    """
).strip()


def _head_blob(path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{path}"],
        cwd=ROOT,
        text=True,
    )


def test_production_nemo_pin_unchanged():
    for path in PRODUCTION_COMMIT_PATHS:
        blob = _head_blob(path)
        assert PRODUCTION_COMMIT in blob
        assert TRIAD_COMMIT not in blob
    for path in PRODUCTION_VERSION_PATHS:
        blob = _head_blob(path)
        assert '0.4.0' in blob
        assert TRIAD_COMMIT not in blob
        assert 'version = "0.4.2"' not in blob


@pytest.mark.skipif(
    os.environ.get("SZL_FORGE_NEMO_TRIAD_LANE") != "1",
    reason="optional triad lane; set SZL_FORGE_NEMO_TRIAD_LANE=1",
)
def test_pinned_triad_lane():
    work = Path(tempfile.mkdtemp(prefix="szl-triad-lane-"))
    try:
        venv = work / "venv"
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.check_call(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                TRIAD_URL,
            ]
        )
        suite = work / "nemo_triad_tests"
        suite.mkdir()
        source = "https://raw.githubusercontent.com/szl-holdings/szl-nemo/{}/tests/test_triad.py".format(
            TRIAD_COMMIT
        )
        subprocess.check_call(
            [
                str(python),
                "-c",
                "import pathlib,urllib.request,sys; "
                "pathlib.Path(sys.argv[1]).write_bytes(urllib.request.urlopen(sys.argv[2], timeout=60).read())",
                str(suite / "test_triad.py"),
                source,
            ]
        )
        checker = work / "check_triad_lane.py"
        checker.write_text(CHECKER, encoding="utf-8")
        completed = subprocess.run(
            [str(python), str(checker), str(suite)],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + "\n" + completed.stderr)
        assert "triad-lane-ok" in completed.stdout
    finally:
        shutil.rmtree(work, ignore_errors=True)
