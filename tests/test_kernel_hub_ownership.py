# SPDX-License-Identifier: Apache-2.0
"""Kernel Hub ownership: szl-forge is not a writer of the converged kernel repos.

On 2026-10-13 each kernel source repository's own ``hf-kernel-publish.yml``
became the single canonical writer of its first-class kernel repo
(``huggingface.co/kernels/SZLHOLDINGS/<name>``) and of its same-named model
twin pointer card; szl-formulas followed once its own publisher was live.
Forge's former writers for those targets were retired, and
``publishing/kernel-hub-ownership.json`` records who owns each target now.

These are source assertions over the files tracked in this repository. They do
not observe the Hub and do not prove a hosted run. The executable scan finds a
writer that names a converged target literally; data-driven writers are covered
by the explicit retired-manifest assertions below.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "publishing" / "kernel-hub-ownership.json"

# Independent oracle: Hub repo id -> GitHub repository whose
# hf-kernel-publish.yml owns the kernel repo (and model twin where configured).
KERNEL_REPO_OWNERS = {
    "SZLHOLDINGS/szl-kernels": "szl-holdings/szl-kernels",
    "SZLHOLDINGS/szl-invariants": "szl-holdings/szl-invariants",
    "SZLHOLDINGS/szl-govsign": "szl-holdings/szl-govsign",
    "SZLHOLDINGS/szl-maskmod": "szl-holdings/szl-maskmod",
    "SZLHOLDINGS/szl-receipt-attn": "szl-holdings/szl-receipt-attn",
    "SZLHOLDINGS/szl-blocked": "szl-holdings/szl-blocked",
    "SZLHOLDINGS/szl-ouroboros": "szl-holdings/szl-ouroboros",
    "SZLHOLDINGS/szl-provctl": "szl-holdings/szl-provctl",
    "SZLHOLDINGS/szl-block-kv": "szl-holdings/szl-block-kv",
    "SZLHOLDINGS/YARQA-ATTN": "szl-holdings/YARQA-ATTN",
    "SZLHOLDINGS/szl-formulas": "szl-holdings/szl-formulas",
    "SZLHOLDINGS/szl-lambda-gate": "szl-holdings/szl-lambda-gate",
    "SZLHOLDINGS/szl-governed-norm": "szl-holdings/szl-lambda-gate",
}

# Any of these makes a file a potential Hub writer.
WRITE_MARKERS = re.compile(
    r"\b(?:upload_file|upload_folder|upload_large_folder|create_commit|"
    r"CommitOperation\w*|delete_file|delete_folder|delete_repo|create_branch|"
    r"delete_branch|create_tag|delete_tag|super_squash_history|"
    r"update_repo_settings|push_to_hub|move_repo)\b"
    r"|huggingface-cli\s+upload|\bhf\s+(?:upload|repo|branch|tag)\b"
    r"|git\s+push\b[^\n]*huggingface\.co|huggingface\.co[^\n]*\bgit\s+push\b"
)
TARGET = re.compile(
    r"SZLHOLDINGS/("
    + "|".join(re.escape(repo_id.split("/", 1)[1]) for repo_id in KERNEL_REPO_OWNERS)
    + r")(?![\w.-])"
)
EXECUTABLE_SUFFIXES = {".py", ".yml", ".yaml", ".sh", ".ps1", ".js", ".mjs", ".ts"}
SKIPPED_TOP_LEVEL = {"tests"}

# Retired 2026-10-13; reintroducing any of them would restore a second writer.
RETIRED_WRITERS = (
    ".github/workflows/publish-szl-kernels.yml",
    ".github/workflows/publish-szl-invariants.yml",
    ".github/workflows/reconcile-kernel-provenance.yml",
    ".github/workflows/publish-compatibility-kernels.yml",
    ".github/workflows/kernel-0161-qualification.yml",
    "tools/publish_szl_kernels.py",
    "tools/verify_szl_kernel_runtime.py",
    "tools/kernels_keyless_credentials.py",
    "tools/publish_szl_invariants.py",
    "tools/verify_invariants_hub_bytes.py",
    "tools/invariants_keyless_credentials.py",
    "tools/reconcile_kernel_provenance.py",
    "tools/publish_compatibility_kernels.py",
    "publishing/kernel-provenance-reconciliation.json",
    "publishing/compatibility-kernels.json",
)


def tracked_files() -> list[Path]:
    try:
        listed = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.decode("utf-8")
        candidates = [ROOT / name for name in listed.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        candidates = [
            path for path in ROOT.rglob("*") if ".git" not in path.relative_to(ROOT).parts
        ]
    return sorted(path for path in candidates if path.is_file())


def executable_files() -> list[Path]:
    return [
        path
        for path in tracked_files()
        if path.relative_to(ROOT).parts[0] not in SKIPPED_TOP_LEVEL
        and path.suffix.lower() in EXECUTABLE_SUFFIXES
    ]


def test_registry_is_closed_and_matches_the_independent_oracle() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert payload["schema"] == "szl.kernel-hub-ownership/v1"
    assert payload["kernel_writer"] == ".github/workflows/hf-kernel-publish.yml"
    owners = {target["repo_id"]: target["owner"] for target in payload["targets"]}
    assert len(owners) == len(payload["targets"])
    assert owners == KERNEL_REPO_OWNERS
    for target in payload["targets"]:
        name = target["repo_id"].split("/", 1)[1]
        assert target["kernel_repo"] == f"kernels/SZLHOLDINGS/{name}"
        assert target["owner"] != "szl-holdings/szl-forge"
        assert target["model_repo_writer"].startswith(target["owner"] + " ")
        for writer in target["retired_forge_writers"]:
            path = writer.split(" ", 1)[0]
            if " " not in writer:
                assert not (ROOT / path).exists(), writer


def test_mirror_card_publisher_uses_the_same_registry() -> None:
    from tools.publish_kernel_mirror_cards import KERNEL_REPO_OWNED_TARGETS

    assert KERNEL_REPO_OWNED_TARGETS == KERNEL_REPO_OWNERS


def test_scan_covers_the_publisher_surface() -> None:
    scanned = {path.relative_to(ROOT).as_posix() for path in executable_files()}
    assert ".github/workflows/publish-kernel-mirror-cards.yml" in scanned
    assert "tools/publish_kernel_mirror_cards.py" in scanned
    assert "tools/quarantine_hub_joblib.py" in scanned


def test_marker_and_target_patterns_detect_a_kernel_writer() -> None:
    sample = 'api.create_commit(repo_id="SZLHOLDINGS/szl-blocked", operations=ops)'
    assert WRITE_MARKERS.search(sample)
    assert TARGET.search(sample).group(1) == "szl-blocked"
    assert WRITE_MARKERS.search("huggingface-cli upload SZLHOLDINGS/x .")
    assert WRITE_MARKERS.search("git push https://huggingface.co/SZLHOLDINGS/x main")
    assert WRITE_MARKERS.search("python tools/quarantine_hub_joblib.py") is None
    assert TARGET.search("SZLHOLDINGS/szl-formulas").group(1) == "szl-formulas"
    # Distinct, longer repo ids are not converged targets.
    assert TARGET.search("SZLHOLDINGS/szl-kernels-extra") is None
    # A Forge-owned family card target is not a converged kernel target.
    assert TARGET.search("SZLHOLDINGS/szl-nemo") is None


@pytest.mark.parametrize("relative", RETIRED_WRITERS)
def test_retired_kernel_writer_is_absent(relative: str) -> None:
    assert not (ROOT / relative).exists()


def test_retired_profile_and_entry_writers_stay_out_of_their_files() -> None:
    # Some writers were retired by removing one profile or one target line from
    # a file that stays. The registry names them "<path> profile|entry <name>";
    # each must still be gone from that file.
    from tools.publish_kernel_mirror_cards import PROFILES

    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    checked = 0
    for target in payload["targets"]:
        repo_id = target["repo_id"]
        for writer in target["retired_forge_writers"]:
            if " " not in writer:
                continue
            path, kind, name = writer.split(" ")
            assert repo_id == f"SZLHOLDINGS/{name}", writer
            text = (ROOT / path).read_text(encoding="utf-8")
            assert re.search(re.escape(repo_id) + r"(?![\w.-])", text) is None, writer
            if kind == "profile":
                assert name not in PROFILES, writer
                assert f'"{name}/README.md"' not in text, writer
                assert (
                    re.search(rf"(?m)^ +(?:- profile: )?{re.escape(name)}(?: \\)?$", text)
                    is None
                ), writer
            else:
                assert kind == "entry", writer
            checked += 1
    assert checked


def test_no_forge_hub_writer_names_a_converged_kernel_target() -> None:
    offenders = {}
    for path in executable_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if not WRITE_MARKERS.search(text):
            continue
        targets = sorted({"SZLHOLDINGS/" + match for match in TARGET.findall(text)})
        if targets:
            offenders[path.relative_to(ROOT).as_posix()] = [
                f"{repo_id} (owned by {KERNEL_REPO_OWNERS[repo_id]})" for repo_id in targets
            ]
    assert offenders == {}


@pytest.mark.parametrize(
    "workflow",
    [
        ".github/workflows/hub-joblib-quarantine-via-token1.yml",
        ".github/workflows/hub-honesty-banners-via-token1.yml",
    ],
)
def test_joblib_quarantine_never_targets_a_converged_kernel_twin(workflow: str) -> None:
    # quarantine_hub_joblib.py takes its target from the workflow, so the
    # executable scan alone cannot see these deletes.
    text = (ROOT / workflow).read_text(encoding="utf-8")
    quarantined = re.findall(r"quarantine_hub_joblib\.py --repo-id (\S+)", text)
    assert quarantined
    assert set(quarantined).isdisjoint(KERNEL_REPO_OWNERS)
    assert TARGET.search(text) is None
