"""Offline publication contract tests; mocks never establish real Hub state."""
import hashlib
import subprocess
from types import SimpleNamespace

import pytest

from szl_model_lab.blueprints import SOURCE_FILES, publish_context, publish_payload, source_payload
from szl_model_lab.safeio import canonical_bytes, strict_json

@pytest.fixture
def payload(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for name in SOURCE_FILES:
        path = root / "model-lab" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source contract fixture\n")
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init"); git("config", "user.name", "Contract Test")
    git("config", "user.email", "fixture@example.invalid")
    git("add", "."); git("commit", "-m", "fixture")
    return source_payload(root, git("rev-parse", "HEAD"), "router")

class FakeHub:
    token = "fixture-not-a-credential"
    def __init__(self, tmp_path, files):
        self.root = tmp_path / "downloads"
        self.root.mkdir()
        self.files = files.copy()
        self.sha = "a" * 40
        self.commits = []
        self.corrupt = False
    def model_info(self, repo, **kwargs):
        return SimpleNamespace(sha=self.sha, siblings=[SimpleNamespace(rfilename=x) for x in self.files])
    def download(self, *, repo_id, filename, revision, **kwargs):
        assert revision == self.sha
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[filename] + (b"tamper" if self.corrupt else b""))
        return str(path)
    def create_commit(self, **kwargs):
        assert kwargs["parent_commit"] == self.sha
        assert kwargs["repo_type"] == "model" and kwargs["revision"] == "main"
        self.commits.append(kwargs)
        self.files.update({op.path_in_repo: op.path_or_fileobj for op in kwargs["operations"]})
        self.sha = "b" * 40
        return SimpleNamespace(oid=self.sha)

def publish(hub, payload, journal=None):
    return publish_payload(hub, "router", payload, download=hub.download,
                           operation=lambda **kw: SimpleNamespace(**kw), journal=journal or (lambda _: None))

def test_export_is_code_only_and_exact(payload):
    binding = strict_json(payload["source-binding.json"])
    assert binding["state"] == "BLUEPRINT_NOT_TRAINED" and binding["weights_present"] is False
    assert not any(name.endswith((".safetensors", ".gguf", ".joblib")) for name in payload)
    assert all(hashlib.sha256(payload[name]).hexdigest() == digest for name, digest in binding["files_sha256"].items())

def test_conditional_commit_and_readback(tmp_path, payload):
    hub = FakeHub(tmp_path, {})
    stages = []
    result = publish(hub, payload, stages.append)
    assert stages == ["COMMIT_REQUEST_STARTED_READBACK_REQUIRED"]
    assert result["state"] == "CODE_ONLY_PUBLICATION_VERIFIED" and result["model_qualified"] is False
    assert result["hf_revision"] == "b" * 40 and len(hub.commits) == 1

def test_identical_source_does_not_commit_again(tmp_path, payload):
    hub = FakeHub(tmp_path, payload)
    assert publish(hub, payload)["hf_revision"] == "a" * 40
    assert not hub.commits

@pytest.mark.parametrize("name", ["model.safetensors", "model.joblib", "foreign.py"])
def test_existing_unknown_artifacts_never_overwritten(tmp_path, payload, name):
    hub = FakeHub(tmp_path, {name: b"existing"})
    with pytest.raises(ValueError, match="preserved"):
        publish(hub, payload)
    assert not hub.commits

def test_existing_unbound_identity_preserved(tmp_path, payload):
    hub = FakeHub(tmp_path, {"README.md": b"existing"})
    with pytest.raises(ValueError, match="preserved"):
        publish(hub, payload)

def test_corrupt_readback_never_verified(tmp_path, payload):
    hub = FakeHub(tmp_path, {})
    hub.corrupt = True
    with pytest.raises(ValueError, match="byte_mismatch"):
        publish(hub, payload)

def test_substituted_binding_rejected_before_network(tmp_path, payload):
    binding = strict_json(payload["source-binding.json"])
    binding["repo_id"] = "SZLHOLDINGS/A11OY-MINI"
    payload["source-binding.json"] = canonical_bytes(binding)
    hub = FakeHub(tmp_path, {})
    with pytest.raises(ValueError, match="invalid_blueprint"):
        publish(hub, payload)
    assert not hub.commits

@pytest.mark.parametrize("key", ["GITHUB_ACTIONS", "GITHUB_REPOSITORY", "GITHUB_REF", "GITHUB_SHA", "GITHUB_EVENT_NAME"])
def test_context_cannot_publish_from_pr(key):
    env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "szl-holdings/szl-forge",
           "GITHUB_REF": "refs/heads/main", "GITHUB_SHA": "a" * 40,
           "GITHUB_EVENT_NAME": "workflow_dispatch"}
    publish_context(env, "a" * 40)
    env[key] = "untrusted"
    with pytest.raises(ValueError):
        publish_context(env, "a" * 40)

def test_missing_repo_creation_is_explicit_and_journaled(tmp_path, payload):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError
    class MissingHub(FakeHub):
        missing = True
        def model_info(self, repo, **kwargs):
            if self.missing:
                response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/fixture"))
                raise RepositoryNotFoundError("fixture not found", response=response)
            return super().model_info(repo, **kwargs)
        def create_repo(self, **kwargs):
            assert kwargs == {"repo_id": "SZLHOLDINGS/A11OY-Router", "repo_type": "model", "private": False, "exist_ok": False}
            self.missing = False
    hub = MissingHub(tmp_path, {})
    stages = []
    assert publish(hub, payload, stages.append)["weights_present"] is False
    assert stages == ["CREATE_REQUEST_STARTED_READBACK_REQUIRED", "COMMIT_REQUEST_STARTED_READBACK_REQUIRED"]

def test_commit_timeout_never_retries(tmp_path, payload):
    class UncertainHub(FakeHub):
        def create_commit(self, **kwargs):
            super().create_commit(**kwargs)
            raise TimeoutError("fixture response lost")
    hub = UncertainHub(tmp_path, {})
    stages = []
    with pytest.raises(TimeoutError):
        publish(hub, payload, stages.append)
    assert len(hub.commits) == 1 and stages == ["COMMIT_REQUEST_STARTED_READBACK_REQUIRED"]

def test_modified_source_digest_rejected(tmp_path, payload):
    payload["source/README.md"] += b"changed"
    with pytest.raises(ValueError, match="digest_mismatch"):
        publish(FakeHub(tmp_path, {}), payload)
