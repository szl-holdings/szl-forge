"""Source projection contracts; synthetic files here are NOT model releases."""
from __future__ import annotations

import hashlib
import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from model_candidates import blueprint as bp
from model_candidates.specs import candidate_spec
from model_candidates.workbench import create_app

SHA = "a" * 40
KEYS = ("router", "invariant-risk", "yarqa-causal")


def sources():
    return {name: ("# synthetic nonexecuted source: " + name + "\n").encode() for name in bp.SOURCE_FILES}


def manifest_hash(bundle):
    return hashlib.sha256(bundle["blueprint-manifest.json"]).hexdigest()


@pytest.mark.parametrize("key", KEYS)
def test_each_projection_uses_canonical_recipe_and_source_bytes(key):
    source = sources()
    bundle = bp.project_source(key, SHA, source)
    result = bp.verify_projection(bundle, manifest_hash(bundle))
    assert result["state"] == "SOURCE_BYTES_MATCH_MANIFEST"
    assert result["source_admission_verified"] is False
    assert result["hub_publication_verified"] is False
    assert result["production_ready"] is False
    assert result["source_revision"] == SHA
    descriptor = json.loads(bundle["architecture.json"])
    assert descriptor["recipe"] == candidate_spec(key)
    assert descriptor["weights_present"] is False
    assert descriptor["recipe"]["benchmark"] is None
    assert all(bundle[name] == source[name] for name in bp.SOURCE_FILES)
    card = bundle["README.md"].decode()
    assert candidate_spec(key)["proposed_hf_id"] in card
    assert "NO TRAINED WEIGHTS" in card
    assert "pipeline_tag" not in card and "library_name: transformers" not in card
    assert SHA in card


@pytest.mark.parametrize("revision", ["main", "v1", "a" * 39, "A" * 40, "a" * 64, None, True])
def test_mutable_or_malformed_revision_rejected(revision):
    with pytest.raises(ValueError):
        bp.project_source("router", revision, sources())


@pytest.mark.parametrize("key", ["unknown", "../router", "A11OY-MINI"])
def test_existing_or_arbitrary_model_not_retargeted(key):
    with pytest.raises(ValueError):
        bp.project_source(key, SHA, sources())


@pytest.mark.parametrize("extra", ["train.jsonl", "model.safetensors", "../key.pem", ".env"])
def test_no_extra_source_or_weight_files(extra):
    original = sources()
    original[extra] = b"not published"
    with pytest.raises(ValueError):
        bp.project_source("router", SHA, original)
    bundle = bp.project_source("router", SHA, sources())
    trusted = manifest_hash(bundle)
    bundle[extra] = b"not published"
    with pytest.raises(ValueError):
        bp.verify_projection(bundle, trusted)


def test_missing_empty_and_non_utf8_sources_rejected():
    for replacement in [None, b"", b"\xff", "not bytes", b"x" * (bp.MAX_FILE + 1)]:
        original = sources()
        if replacement is None:
            original.pop("LICENSE")
        else:
            original["LICENSE"] = replacement
        with pytest.raises(ValueError):
            bp.project_source("router", SHA, original)


def test_manifest_is_externally_anchored_not_self_authenticating():
    bundle = bp.project_source("router", SHA, sources())
    expected = manifest_hash(bundle)
    bundle["model_candidates/networks.py"] = b"tampered source\n"
    m = json.loads(bundle["blueprint-manifest.json"])
    m["files"]["model_candidates/networks.py"] = hashlib.sha256(bundle["model_candidates/networks.py"]).hexdigest()
    bundle["blueprint-manifest.json"] = bp.canonical(m)
    with pytest.raises(ValueError, match="manifest"):
        bp.verify_projection(bundle, expected)


@pytest.mark.parametrize("field,value", [("weights_present", True), ("publication_eligible", True),
                                        ("source_admission", "ADMITTED"), ("hub_revision", "b" * 40)])
def test_authority_cannot_be_inflated_even_with_new_local_hash(field, value):
    bundle = bp.project_source("router", SHA, sources())
    m = json.loads(bundle["blueprint-manifest.json"])
    m[field] = value
    bundle["blueprint-manifest.json"] = bp.canonical(m)
    with pytest.raises(ValueError):
        bp.verify_projection(bundle, manifest_hash(bundle))


def test_duplicate_json_keys_rejected():
    bundle = bp.project_source("router", SHA, sources())
    bundle["blueprint-manifest.json"] = b'{"candidate":"router","candidate":"router"}'
    with pytest.raises(ValueError):
        bp.verify_projection(bundle, manifest_hash(bundle))


def test_plan_ui_and_api_share_recipe_without_io_or_remote_authority():
    with TestClient(create_app(), base_url="http://127.0.0.1:8765", client=("127.0.0.1", 12345)) as client:
        for key in KEYS:
            data = client.get(f"/api/blueprints/{key}").json()
            assert data == bp.blueprint_plan(key)
            assert data["weights_present"] is False
            assert data["observed_at"] is None
            page = client.get("/", params={"candidate": key}).text
            assert "GitHub-first delivery" in page
            assert data["proposed_hf_id"] in page
            assert "NO TRAINED WEIGHTS" in page
            assert "<script" not in page
            assert client.post(f"/api/blueprints/{key}").status_code == 405
        assert client.get("/api/blueprints/unknown").status_code == 404
        assert client.get("/api/blueprints/router", headers={"origin":"null"}).status_code == 403


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


@pytest.fixture
def git_fixture(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "user.name", "Synthetic fixture")
    for name, data in sources().items():
        dest = repo / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    git(repo, "add", ".")
    git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "synthetic source")
    return repo, git(repo, "rev-parse", "HEAD").decode().strip()


def test_git_reader_ignores_worktree_and_never_imports_projection(git_fixture, monkeypatch):
    repo, revision = git_fixture
    (repo / "model_candidates/specs.py").write_text("raise RuntimeError('MUST NOT EXECUTE')\n")
    monkeypatch.setenv("GIT_DIR", str(repo / "missing"))
    observed = bp.read_git_source(repo, revision)
    assert observed == sources()


def test_stale_exporter_rejected(git_fixture):
    repo, revision = git_fixture
    with pytest.raises(ValueError, match="running source"):
        bp.build_blueprint(repo, revision, "router")


def test_git_reader_rejects_symlink_mode(git_fixture):
    repo, _ = git_fixture
    # An index-only symlink fixture works even on Windows without symlink privileges.
    blob = git(repo, "rev-parse", "HEAD:LICENSE").decode().strip()
    git(repo, "update-index", "--add", "--cacheinfo", f"120000,{blob},LICENSE")
    git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "synthetic link")
    revision = git(repo, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="regular"):
        bp.read_git_source(repo, revision)


def test_zip_is_deterministic_non_overwriting_and_confined(tmp_path):
    bundle = bp.project_source("router", SHA, sources())
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    bp.write_archive(first, bundle)
    bp.write_archive(second, bundle)
    assert first.read_bytes() == second.read_bytes()
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        bp.write_archive(first, bundle)
    assert first.read_bytes() == before
    malformed = dict(bundle, **{"../escape": b"bad"})
    with pytest.raises(ValueError):
        bp.write_archive(tmp_path / "bad.zip", malformed)
    assert not (tmp_path / "bad.zip").exists()
