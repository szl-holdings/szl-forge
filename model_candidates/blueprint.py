"""Offline, exact-source model projections; never a Hub writer or release gate.

Reuse the existing candidate recipes. Source packaging, source admission, Hub
publication, trained weights and runtime qualification are separate facts.
Only the explicit CLI writes a NEW local ZIP. HTTP consumers get a plan only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
import zipfile

from .specs import candidate_spec

SCHEMA = "szl.model-candidate-blueprint/v1"
SOURCE_REPOSITORY = "szl-holdings/szl-forge"
SOURCE_FILES = (
    "LICENSE",
    "model_candidates/__init__.py",
    "model_candidates/specs.py",
    "model_candidates/networks.py",
    "model_candidates/workbench.py",
    "model_candidates/blueprint.py",
)
MAX_FILE = 512 * 1024
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _revision(value: str) -> None:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("full lowercase Git commit required")


def blueprint_plan(key: str) -> dict[str, Any]:
    recipe = candidate_spec(key)
    return {
        "schema": SCHEMA, "candidate": key,
        "proposed_hf_id": recipe["proposed_hf_id"], "repo_type": "model",
        "architecture": recipe["architecture"],
        "state": "SOURCE_ONLY_NOT_TRAINED", "observed_at": None,
        "weights_present": False, "publication_eligible": False,
        "source_repository": SOURCE_REPOSITORY,
        "authority_chain": ["GitHub", "Hugging Face", "a-11-oy.com", "a11oy.net"],
        "remaining": ["protected source admission", "existing-writer integration",
                      "immutable Hub readback", "product/proof projection"],
    }


def project_source(key: str, revision: str, sources: dict[str, bytes]) -> dict[str, bytes]:
    """Pure projection over supplied bytes. Does NOT authenticate their origin."""
    _revision(revision)
    recipe = candidate_spec(key)
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_FILES):
        raise ValueError("exact regular source set required; no weights or data")
    for body in sources.values():
        if not isinstance(body, bytes) or not 0 < len(body) <= MAX_FILE or b"\0" in body:
            raise ValueError("bounded nonempty UTF-8 source required")
        try:
            body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("UTF-8 source required") from exc
    files = dict(sources)
    files["architecture.json"] = canonical({
        "schema": SCHEMA, "recipe": recipe, "source_repository": SOURCE_REPOSITORY,
        "source_revision": revision, "weights_present": False,
    })
    files["pyproject.toml"] = (
        '[build-system]\nrequires = ["setuptools>=77", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n\n[project]\n'
        f'name = "szl-candidate-{key}"\nversion = "0.0.0"\n'
        'description = "Source-only SZL research candidate; no trained weights"\n'
        'requires-python = ">=3.11"\nlicense = "Apache-2.0"\n'
        'readme = "README.md"\ndependencies = []\n\n'
        '[tool.setuptools]\npackages = ["model_candidates"]\n'
    ).encode()
    files["README.md"] = f'''---
license: apache-2.0
tags:
- code-only
- untrained
- research
---
# {recipe["proposed_hf_id"]}

**SOURCE ONLY — NO TRAINED WEIGHTS.** This is a proposed model-source projection,
not a trained model, hosted inference service or production release.

Canonical source: https://github.com/{SOURCE_REPOSITORY}/tree/{revision}/model_candidates

{recipe["purpose"]}

`architecture.json` carries the exact recipe, named features or byte vocabulary,
configuration and limitations state from GitHub. The shared `model_candidates`
Python source is included unchanged; the selected candidate is `{key}`. The
original kernel identities and existing signed model releases are untouched.

## Inspect before training

In a separate reviewed environment, the local recipe inspector is
`python -m model_candidates.workbench`. It needs FastAPI and Uvicorn, binds only
loopback, and must not be tunneled or exposed as a multi-user service. It cannot
train, probe hardware, upload, publish, or approve a model. Source factories in
`model_candidates.networks` require PyTorch and initialize RANDOM parameters;
they do not load a checkpoint. No Transformers AutoModel or Ollama compatibility
is asserted. Dependency closure must be qualified separately for the target host.

## Evidence boundary

The manifest is an unsigned content commitment, NOT a signature or independent
source-admission receipt. A verifier must obtain its expected SHA-256 from a
separately trusted publication record. Matching bytes do not establish quality,
rights to future training data, admission to main, or a live Hub publication.
Existing Forge release controls must approve any write and verify the immutable
Hub commit. GitHub and Hugging Face commit IDs are different identities.

Training: NOT STARTED. Weights: ABSENT. Benchmarks: UNAVAILABLE. Publication and
runtime eligibility remain false. Supervised training, admitted data and frozen
held-out evaluation are future steps, not side effects of this repository.
'''.encode()
    manifest = {
        "schema": SCHEMA, "candidate": key, "target_repo_id": recipe["proposed_hf_id"],
        "repo_type": "model", "artifact_class": "CODE_ONLY_MODEL_BLUEPRINT",
        "source_repository": SOURCE_REPOSITORY, "source_revision": revision,
        "source_admission": "NOT_VERIFIED_HERE", "hub_revision": None,
        "weights_present": False, "training_started": False,
        "publication_eligible": False, "signed": False,
        "files": {name: hashlib.sha256(body).hexdigest() for name, body in files.items()},
    }
    files["blueprint-manifest.json"] = canonical(manifest)
    return files


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


def verify_projection(files: dict[str, bytes], expected_manifest_sha256: str) -> dict[str, Any]:
    """Compare complete projected bytes against an externally supplied hash.

    This verifies only the caller-supplied byte map, not hosting or approval.
    Never accept the adjacent manifest as its own source of trust.
    """
    if not isinstance(expected_manifest_sha256, str) or not _DIGEST.fullmatch(expected_manifest_sha256):
        raise ValueError("externally pinned manifest digest required")
    allowed = set(SOURCE_FILES) | {"README.md", "architecture.json", "pyproject.toml", "blueprint-manifest.json"}
    if not isinstance(files, dict) or set(files) != allowed:
        raise ValueError("exact projection file set required")
    if any(not isinstance(v, bytes) or not 0 < len(v) <= MAX_FILE for v in files.values()):
        raise ValueError("bounded file bytes required")
    raw = files["blueprint-manifest.json"]
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("manifest digest mismatch")
    try:
        manifest = json.loads(raw, object_pairs_hook=_unique,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        expected = project_source(manifest["candidate"], manifest["source_revision"],
                                  {name: files[name] for name in SOURCE_FILES})
    except (KeyError, TypeError, UnicodeError) as exc:
        raise ValueError("invalid projection manifest") from exc
    if files != expected:
        raise ValueError("projection bytes, recipe or authority mismatch")
    return {
        "state": "SOURCE_BYTES_MATCH_MANIFEST", "target_repo_id": manifest["target_repo_id"],
        "source_revision": manifest["source_revision"],
        "source_admission_verified": False, "hub_publication_verified": False,
        "weights_present": False, "production_ready": False,
    }


def _git(repository: Path, *args: str) -> bytes:
    # Read-only literal commands, no shell, no hooks/filters or network commands.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_NO_REPLACE_OBJECTS": "1", "GIT_OPTIONAL_LOCKS": "0"})
    return subprocess.run(["git", "--no-pager", "--no-replace-objects", "-C", str(repository), *args],
                          check=True, capture_output=True, timeout=20, env=env).stdout


def read_git_source(repository: Path, revision: str) -> dict[str, bytes]:
    _revision(revision)
    if _git(repository, "rev-parse", "--verify", revision + "^{commit}").decode().strip() != revision:
        raise ValueError("source commit mismatch")
    result = {}
    for path in SOURCE_FILES:
        entry = _git(repository, "ls-tree", "-z", revision, "--", path)
        if not entry.startswith(b"100644 blob ") or entry.count(b"\0") != 1:
            raise ValueError("regular tracked source required")
        metadata, actual_path = entry[:-1].split(b"\t", 1)
        if actual_path.decode() != path:
            raise ValueError("source path mismatch")
        blob = metadata.split()[2].decode()
        size = int(_git(repository, "cat-file", "-s", blob))
        if not 0 < size <= MAX_FILE:
            raise ValueError("source size rejected")
        data = _git(repository, "cat-file", "blob", blob)
        digest = hashlib.sha1(b"blob " + str(size).encode() + b"\0" + data).hexdigest()
        if len(data) != size or digest != blob:
            raise ValueError("source Git blob mismatch")
        result[path] = data
    return result


def build_blueprint(repository: Path, revision: str, key: str) -> dict[str, bytes]:
    sources = read_git_source(repository, revision)
    # Bind the running metadata/exporter to the exact projected package revision.
    for path, body in sources.items():
        if path.startswith("model_candidates/"):
            local = Path(__file__).parent / path.split("/", 1)[1]
            if local.is_symlink() or local.read_bytes() != body:
                raise ValueError("running source differs from projected Git source")
    return project_source(key, revision, sources)


def write_archive(output: Path, files: dict[str, bytes]) -> str:
    """Write an explicit NEW local artifact; never overwrite or extract paths."""
    digest = hashlib.sha256(files.get("blueprint-manifest.json", b"")).hexdigest()
    verify_projection(files, digest)  # Structural self-check, not external trust.
    with output.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, body in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, body)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--candidate", choices=("router", "invariant-risk", "yarqa-causal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = build_blueprint(args.repository, args.source_revision, args.candidate)
    digest = write_archive(args.output, files)
    print(json.dumps({"state": "LOCAL_CODE_ARCHIVE_CREATED", "manifest_sha256": digest,
                      "hub_published": False, "training_started": False}))


if __name__ == "__main__":
    main()
