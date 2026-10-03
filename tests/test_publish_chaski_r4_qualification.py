"""Offline, fail-closed contracts for the Chaski-R4 qualification sidecar."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools import publish_chaski_card as CARD_PUBLISHER
from tools import publish_chaski_r4_qualification as MODULE


ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_state():
    old_sidecar = b"Publication: UNPUBLISHED\nPromotion: NOT_PROMOTABLE\n"
    new_sidecar = (
        b"Publication: PUBLIC_EXPERIMENTAL_ARTIFACT - published evidence only\n"
        b"Promotion: NOT_PROMOTABLE - owner decision not taken\n"
        b"publication_eligible=false\nreceipts A/B provenance stays unresolved\n"
    )
    base_card = b"base card\n"
    current_card = b"current card\n"
    base_banner = b"base banner\n"
    current_banner = b"current banner\n"
    frozen = {
        ".gitattributes": b"*.safetensors filter=lfs\n",
        "LICENSE": b"Apache License\n",
        "adapter_config.json": b"{}\n",
        "adapter_model.safetensors": b"receipted adapter bytes",
        "evidence/canonical_rerun_20261001_140615.receipt.json": b'{"gate":"measured"}\n',
        "training_receipt.json": b'{"training":"recorded"}\n',
    }
    def binding(card: bytes, banner: bytes, revision: str) -> bytes:
        return CARD_PUBLISHER.build_source_binding(
            profile="chaski-r4",
            source_revision=revision,
            source_assets={
                "README.md": {"sha256": digest(card), "bytes": len(card)},
                "holo-banner.svg": {"sha256": digest(banner), "bytes": len(banner)},
            },
        )

    baseline = {
        **frozen,
        "README.md": base_card,
        "holo-banner.svg": base_banner,
        "szl-source-binding.json": binding(base_card, base_banner, "b" * 40),
        MODULE.SIDECAR: old_sidecar,
    }
    current = {
        **frozen,
        "README.md": current_card,
        "holo-banner.svg": current_banner,
        "szl-source-binding.json": binding(current_card, current_banner, "c" * 40),
        MODULE.SIDECAR: old_sidecar,
    }
    receipt_files = {
        name: {"sha256": digest(baseline[name]), "readback_sha256": digest(baseline[name])}
        for name in MODULE.RECEIPT_FILES
    }
    receipt = {
        "schema": "szl.hf-artifact-publication/v1",
        "repo_id": MODULE.REPO,
        "artifact_state": "PUBLIC_EXPERIMENTAL_ARTIFACT",
        "promotion": "NOT_PROMOTABLE",
        "publication_eligible": False,
        "autonomy_eligible": False,
        "card_revision": "b" * 40,
        "bytes_revision": "a" * 40,
        "receipt_c_sha256": digest(frozen["evidence/canonical_rerun_20261001_140615.receipt.json"]),
        "files": receipt_files,
    }
    receipt_blob = json.dumps(receipt).encode()
    blobs = {
        "chaski_r4/QUALIFICATION_STATE.md": new_sidecar,
        "chaski_r4/evidence/publication_receipt_20261001_171009.json": receipt_blob,
        "chaski_r4/card/README.md": current_card,
        "chaski_r4/card/holo-banner.svg": current_banner,
    }
    contract = MODULE.build_contract(
        lambda name: blobs[name],
        "d" * 40,
        expected_receipt_sha256=digest(receipt_blob),
        expected_bytes_revision="a" * 40,
    )
    return contract, baseline, current


class FakeApi:
    def __init__(self, baseline: dict[str, bytes], current: dict[str, bytes]):
        self.revisions = {"a" * 40: dict(baseline), "e" * 40: dict(current)}
        self.head = "e" * 40
        self.operations = []
        self.parent_commits = []
        self.orgs = [{"name": "SZLHOLDINGS"}]
        self.before_commit = None
        self.live_info_calls = 0
        self.on_live_info = None
        self.simulate_sdk_noop = False
        self.declared_blobs = {
            "b" * 40: {
                MODULE.SOURCE_CARD: baseline["README.md"],
                MODULE.SOURCE_BANNER: baseline["holo-banner.svg"],
            },
            "c" * 40: {
                MODULE.SOURCE_CARD: current["README.md"],
                MODULE.SOURCE_BANNER: current["holo-banner.svg"],
            },
        }

    def whoami(self):
        return {"name": "publisher", "orgs": self.orgs}

    def repo_info(self, repo_id, repo_type, revision=None):
        assert repo_id == MODULE.REPO and repo_type == "model"
        if revision is None:
            self.live_info_calls += 1
            if self.on_live_info:
                self.on_live_info(self)
        sha = revision or self.head
        return SimpleNamespace(
            sha=sha,
            siblings=[SimpleNamespace(rfilename=name) for name in self.revisions[sha]],
        )

    def read(self, revision, name):
        return self.revisions[revision][name]

    def read_source(self, revision, name):
        return self.declared_blobs[revision][name]

    def create_commit(self, **kwargs):
        assert kwargs["revision"] == "main"
        assert kwargs["create_pr"] is False
        if self.before_commit:
            self.before_commit(self)
        self.parent_commits.append(kwargs["parent_commit"])
        if kwargs["parent_commit"] != self.head:
            raise RuntimeError("parent conflict")
        self.operations = kwargs["operations"]
        assert len(self.operations) == 1
        operation = self.operations[0]
        name, content = operation.path_in_repo, operation.path_or_fileobj
        after = dict(self.revisions[self.head])
        after[name] = content
        self.head = "f" * 40
        self.revisions[self.head] = after
        if not self.simulate_sdk_noop:
            operation._is_committed = True
        return SimpleNamespace(oid=self.head)


def publish(contract, api, **kwargs):
    return MODULE.reconcile(
        contract,
        api,
        api.read,
        api.read_source,
        lambda name, content: SimpleNamespace(
            path_in_repo=name, path_or_fileobj=content, _is_committed=False
        ),
        assert_current_source=kwargs.pop("assert_current_source", lambda _revision: None),
        **kwargs,
    )


def test_current_main_guard_rejects_nonmain_branch_with_identical_commit(monkeypatch):
    revision = "d" * 40

    def git(command, **_kwargs):
        if command == ["git", "rev-parse", "--verify", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=f"{revision}\n")
        if command == ["git", "symbolic-ref", "--quiet", "--short", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="szl/other-branch\n")
        raise AssertionError(f"unexpected Git call: {command}")

    monkeypatch.setattr(MODULE.subprocess, "run", git)
    with pytest.raises(MODULE.PublicationError, match="protected main checkout"):
        MODULE.assert_current_main(ROOT, revision)


@pytest.mark.parametrize("detached_actions", [False, True])
def test_current_main_guard_accepts_exact_main_checkout(monkeypatch, detached_actions):
    revision = "d" * 40
    if detached_actions:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_REPOSITORY", "szl-holdings/szl-forge")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        monkeypatch.setenv("GITHUB_SHA", revision)

    def git(command, **_kwargs):
        if command == ["git", "rev-parse", "--verify", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=f"{revision}\n")
        if command == ["git", "symbolic-ref", "--quiet", "--short", "HEAD"]:
            return SimpleNamespace(
                returncode=1 if detached_actions else 0,
                stdout="" if detached_actions else "main\n",
            )
        if command[:2] == ["git", "ls-remote"]:
            return SimpleNamespace(returncode=0, stdout=f"{revision}\trefs/heads/main\n")
        if command[:2] == ["git", "status"]:
            assert command[-2:] == list(MODULE.RUNTIME_MODULES)
            return SimpleNamespace(returncode=0, stdout="")
        if command[:2] == ["git", "hash-object"]:
            return SimpleNamespace(returncode=0, stdout=f"{'a' * 40}\n")
        if command[:3] == ["git", "rev-parse", "--verify"]:
            return SimpleNamespace(returncode=0, stdout=f"{'a' * 40}\n")
        raise AssertionError(f"unexpected Git call: {command}")

    monkeypatch.setattr(MODULE.subprocess, "run", git)
    MODULE.assert_current_main(ROOT, revision)


def test_current_main_guard_rejects_stale_or_unavailable_remote(monkeypatch):
    revision = "d" * 40

    def git(command, **_kwargs):
        if command == ["git", "rev-parse", "--verify", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=f"{revision}\n")
        if command == ["git", "symbolic-ref", "--quiet", "--short", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="main\n")
        if command[:2] == ["git", "ls-remote"]:
            return SimpleNamespace(returncode=0, stdout=f"{'e' * 40}\trefs/heads/main\n")
        raise AssertionError(f"unexpected Git call: {command}")

    monkeypatch.setattr(MODULE.subprocess, "run", git)
    with pytest.raises(MODULE.PublicationError, match="no longer owns current main"):
        MODULE.assert_current_main(ROOT, revision)

    def unavailable(command, **kwargs):
        if command[:2] == ["git", "ls-remote"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return git(command, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", unavailable)
    with pytest.raises(MODULE.PublicationError, match="fresh-main lookup failed"):
        MODULE.assert_current_main(ROOT, revision)


@pytest.mark.parametrize("status", [" M tools/publish_chaski_card.py\n", "M  tools/publish_chaski_card.py\n", "?? tools/publish_chaski_card.py\n"])
def test_current_main_guard_rejects_dirty_runtime_modules(monkeypatch, status):
    revision = "d" * 40

    def git(command, **_kwargs):
        if command == ["git", "rev-parse", "--verify", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=f"{revision}\n")
        if command == ["git", "symbolic-ref", "--quiet", "--short", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="main\n")
        if command[:2] == ["git", "ls-remote"]:
            return SimpleNamespace(returncode=0, stdout=f"{revision}\trefs/heads/main\n")
        if command[:2] == ["git", "status"]:
            assert command[-2:] == list(MODULE.RUNTIME_MODULES)
            return SimpleNamespace(returncode=0, stdout=status)
        raise AssertionError(f"unexpected Git call: {command}")

    monkeypatch.setattr(MODULE.subprocess, "run", git)
    with pytest.raises(MODULE.PublicationError, match="publisher runtime modules"):
        MODULE.assert_current_main(ROOT, revision)


def test_current_main_guard_rejects_runtime_blob_mismatch_even_with_clean_status(monkeypatch):
    revision = "d" * 40

    def git(command, **_kwargs):
        if command == ["git", "rev-parse", "--verify", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=f"{revision}\n")
        if command == ["git", "symbolic-ref", "--quiet", "--short", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="main\n")
        if command[:2] == ["git", "ls-remote"]:
            return SimpleNamespace(returncode=0, stdout=f"{revision}\trefs/heads/main\n")
        if command[:2] == ["git", "status"]:
            return SimpleNamespace(returncode=0, stdout="")
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout=f"{'a' * 40}\n")
        if command[:2] == ["git", "hash-object"]:
            return SimpleNamespace(returncode=0, stdout=f"{'b' * 40}\n")
        raise AssertionError(f"unexpected Git call: {command}")

    monkeypatch.setattr(MODULE.subprocess, "run", git)
    with pytest.raises(MODULE.PublicationError, match="publisher runtime modules"):
        MODULE.assert_current_main(ROOT, revision)


@pytest.mark.parametrize(
    "mutation",
    ["clean_crlf", "unstaged", "staged", "untracked_replacement", "skip_worktree"],
)
def test_current_main_guard_checks_real_git_checkout_bytes(tmp_path, monkeypatch, mutation):
    checkout = tmp_path / "checkout"
    remote = tmp_path / "remote.git"

    def git(*args, cwd=None):
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init", "-q", "-b", "main", str(checkout))
    git("config", "user.name", "Test Publisher", cwd=checkout)
    git("config", "user.email", "publisher@example.invalid", cwd=checkout)
    git("config", "commit.gpgsign", "false", cwd=checkout)
    git("config", "core.autocrlf", "true", cwd=checkout)
    for name in MODULE.RUNTIME_MODULES:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"VALUE = 'reviewed'\r\n")
    git("add", "--", *MODULE.RUNTIME_MODULES, cwd=checkout)
    git("commit", "-qm", "test: reviewed publisher modules", cwd=checkout)
    git("init", "--bare", "-q", str(remote))
    git("remote", "add", "origin", str(remote), cwd=checkout)
    git("push", "--quiet", "origin", "main", cwd=checkout)
    revision = git("rev-parse", "HEAD", cwd=checkout)
    monkeypatch.setattr(MODULE, "CANONICAL_FORGE_REMOTE", str(remote))

    helper = checkout / MODULE.RUNTIME_MODULES[1]
    assert b"\r\n" in helper.read_bytes()
    assert git("status", "--porcelain=v1", cwd=checkout) == ""
    if mutation == "clean_crlf":
        MODULE.assert_current_main(checkout, revision)
        return
    if mutation == "untracked_replacement":
        git("rm", "--cached", "--", MODULE.RUNTIME_MODULES[1], cwd=checkout)
    elif mutation == "skip_worktree":
        git("update-index", "--skip-worktree", "--", MODULE.RUNTIME_MODULES[1], cwd=checkout)
        helper.write_bytes(b"VALUE = 'unreviewed'\r\n")
        assert git("status", "--porcelain=v1", cwd=checkout) == ""
    else:
        helper.write_bytes(b"VALUE = 'unreviewed'\r\n")
        if mutation == "staged":
            git("add", "--", MODULE.RUNTIME_MODULES[1], cwd=checkout)
    with pytest.raises(MODULE.PublicationError, match="publisher runtime modules"):
        MODULE.assert_current_main(checkout, revision)


def test_main_advancing_after_preflight_refuses_noop_success():
    contract, baseline, current = fixture_state()
    current[MODULE.SIDECAR] = contract.sidecar
    api = FakeApi(baseline, current)

    def advanced(_revision):
        raise MODULE.PublicationError("synthetic protected main advanced")

    with pytest.raises(MODULE.PublicationError, match="protected main advanced"):
        publish(contract, api, assert_current_source=advanced)
    assert api.operations == []


def test_main_advancing_after_preflight_refuses_hub_commit():
    contract, baseline, current = fixture_state()
    api = FakeApi(baseline, current)

    def advanced(_revision):
        raise MODULE.PublicationError("synthetic protected main advanced")

    with pytest.raises(MODULE.PublicationError, match="protected main advanced"):
        publish(contract, api, assert_current_source=advanced)
    assert api.operations == []
    assert api.parent_commits == []


def test_main_advancing_during_hub_commit_refuses_published_success():
    contract, baseline, current = fixture_state()
    api = FakeApi(baseline, current)
    main_revision = {"current": contract.source_revision}
    checks = []
    pending = []

    def check_main(revision):
        checks.append(revision)
        if main_revision["current"] != revision:
            raise MODULE.PublicationError("synthetic protected main advanced after commit")

    api.before_commit = lambda _instance: main_revision.update(current="e" * 40)
    with pytest.raises(MODULE.PublicationError, match="advanced after commit"):
        publish(contract, api, assert_current_source=check_main, on_commit=pending.append)
    assert checks == [contract.source_revision, contract.source_revision]
    assert api.parent_commits == ["e" * 40]
    assert api.head == "f" * 40
    assert pending[0]["status"] == "COMMITTED_UNVERIFIED"
    assert pending[0]["hub_revision"] == "f" * 40


def test_real_source_contract_is_bound_to_immutable_git_blob():
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    contract = MODULE.load_source_contract(ROOT, source_revision)
    assert contract.source_revision == source_revision
    assert b"Publication: PUBLIC_EXPERIMENTAL_ARTIFACT" in contract.sidecar
    assert b"Promotion: NOT_PROMOTABLE" in contract.sidecar
    assert contract.receipt["files"][MODULE.SIDECAR]["sha256"] != digest(contract.sidecar)
    assert contract.receipt["bytes_revision"] == MODULE.PINNED_BYTES_REVISION
    assert MODULE.PINNED_RECEIPT_SHA256 == digest(
        subprocess.check_output(
            ["git", "show", f"HEAD:{MODULE.SOURCE_RECEIPT}"], cwd=ROOT
        )
    )


def test_historic_receipt_cannot_be_rebaselined_even_with_new_internal_hashes():
    contract, baseline, current = fixture_state()
    changed = dict(contract.receipt)
    changed["files"] = dict(changed["files"])
    changed["files"]["adapter_model.safetensors"] = {
        "sha256": digest(b"other weights"),
        "readback_sha256": digest(b"other weights"),
    }
    changed_bytes = json.dumps(changed).encode()
    blobs = {
        MODULE.SOURCE_SIDECAR: contract.sidecar,
        MODULE.SOURCE_RECEIPT: changed_bytes,
        MODULE.SOURCE_CARD: current["README.md"],
        MODULE.SOURCE_BANNER: current["holo-banner.svg"],
    }
    with pytest.raises(MODULE.PublicationError, match="pinned source digest"):
        MODULE.build_contract(
            lambda name: blobs[name],
            "d" * 40,
            expected_receipt_sha256=digest(json.dumps(contract.receipt).encode()),
            expected_bytes_revision="a" * 40,
        )


def test_one_sidecar_operation_and_immutable_full_readback():
    contract, baseline, current = fixture_state()
    api = FakeApi(baseline, current)
    pending = []
    result = publish(contract, api, on_commit=pending.append)
    assert result["status"] == "PUBLISHED_AND_READ_BACK"
    assert result["hub_revision"] == "f" * 40
    assert len(api.operations) == 1
    assert api.operations[0].path_in_repo == MODULE.SIDECAR
    assert api.operations[0].path_or_fileobj == contract.sidecar
    assert api.operations[0]._is_committed is True
    assert api.parent_commits == ["e" * 40]
    assert pending[0]["status"] == "COMMITTED_UNVERIFIED"
    assert result["promotion"] == "NOT_PROMOTABLE"
    assert result["publication_eligible"] is False


def test_exact_current_sidecar_is_idempotent_without_a_commit():
    contract, baseline, current = fixture_state()
    current[MODULE.SIDECAR] = contract.sidecar
    api = FakeApi(baseline, current)
    result = publish(contract, api)
    assert result["status"] == "ALREADY_CURRENT"
    assert api.operations == []


@pytest.mark.parametrize("name", ["adapter_model.safetensors", "training_receipt.json", ".gitattributes"])
def test_unrelated_byte_drift_refuses_before_write(name: str):
    contract, baseline, current = fixture_state()
    current[name] += b"changed"
    api = FakeApi(baseline, current)
    with pytest.raises(MODULE.PublicationError, match="drift"):
        publish(contract, api)
    assert api.operations == []


def test_foreign_sidecar_refuses_before_write():
    contract, baseline, current = fixture_state()
    current[MODULE.SIDECAR] = b"unknown owner's note\n"
    api = FakeApi(baseline, current)
    with pytest.raises(MODULE.PublicationError, match="sidecar"):
        publish(contract, api)
    assert api.operations == []


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("qualification", "evaluation_state", "PROMOTED"),
        ("qualification", "release_blocker", "none"),
        ("authority", "weights", True),
        ("authority", "runtime", None),
    ],
)
def test_tampered_card_binding_refuses_before_write(section: str, key: str, value):
    contract, baseline, current = fixture_state()
    binding = json.loads(current["szl-source-binding.json"])
    binding[section][key] = value
    current["szl-source-binding.json"] = (json.dumps(binding, sort_keys=True) + "\n").encode()
    api = FakeApi(baseline, current)
    with pytest.raises(MODULE.PublicationError, match="complete Forge publisher contract"):
        publish(contract, api)
    assert api.operations == []


def test_false_historic_card_source_claim_refuses_before_write():
    contract, baseline, current = fixture_state()
    api = FakeApi(baseline, current)
    api.declared_blobs["c" * 40][MODULE.SOURCE_CARD] = b"not the declared card blob"
    with pytest.raises(MODULE.PublicationError, match="declared immutable Forge source"):
        publish(contract, api)
    assert api.operations == []


def test_unauthorized_identity_refuses_before_write():
    contract, baseline, current = fixture_state()
    api = FakeApi(baseline, current)
    api.orgs = [{"name": "elsewhere"}]
    with pytest.raises(MODULE.PublicationError, match="authorized"):
        publish(contract, api)
    assert api.operations == []


def test_concurrent_hub_parent_refuses_before_write():
    contract, baseline, current = fixture_state()
    api = FakeApi(baseline, current)
    api.before_commit = lambda instance: setattr(instance, "head", "a" * 40)
    with pytest.raises(RuntimeError, match="parent conflict"):
        publish(contract, api)
    assert api.operations == []


def test_sdk_noop_response_cannot_be_claimed_as_our_commit():
    contract, baseline, current = fixture_state()
    api = FakeApi(baseline, current)
    api.simulate_sdk_noop = True
    pending = []
    with pytest.raises(MODULE.PublicationError, match="without an expected-parent sidecar server commit"):
        publish(contract, api, on_commit=pending.append)
    assert len(api.operations) == 1
    assert api.operations[0]._is_committed is False
    assert pending == []


def test_card_writer_advancing_during_preflight_refuses_before_write():
    contract, baseline, current = fixture_state()
    api = FakeApi(baseline, current)

    def advance_card(instance):
        if instance.live_info_calls == 2:
            updated = dict(instance.revisions[instance.head])
            updated["README.md"] = b"another writer's card\n"
            instance.head = "9" * 40
            instance.revisions[instance.head] = updated

    api.on_live_info = advance_card
    with pytest.raises(MODULE.PublicationError, match="parent changed"):
        publish(contract, api)
    assert api.operations == []


def test_source_rejects_promoted_or_unpublished_text():
    contract, baseline, current = fixture_state()
    for text in (
        b"Publication: UNPUBLISHED\nPromotion: NOT_PROMOTABLE\n",
        b"Publication: PUBLIC_EXPERIMENTAL_ARTIFACT\nPromotion: PROMOTED\n",
        contract.sidecar + b"Promotion: PROMOTED\n",
        contract.sidecar + b"publication_eligible=true\n",
    ):
        blobs = {
            "chaski_r4/QUALIFICATION_STATE.md": text,
            "chaski_r4/evidence/publication_receipt_20261001_171009.json": json.dumps(contract.receipt).encode(),
            "chaski_r4/card/README.md": current["README.md"],
            "chaski_r4/card/holo-banner.svg": current["holo-banner.svg"],
        }
        with pytest.raises(MODULE.PublicationError, match="claim"):
            MODULE.build_contract(lambda name: blobs[name], "d" * 40)
