"""Freeze a v2 candidate: canonical model.json plus artifact_receipt.json.

Canonical model.json: floats rounded to 12 decimals, json.dumps(sort_keys=True, indent=2,
ensure_ascii=True, allow_nan=False) plus a trailing newline.  Candidates built by
pipeline.build_candidate are already at that resolution; freeze refuses an artifact whose
canonical rounding would change any value (downstream research numbers would then not be the
kernel's numbers).

Receipt (schema szl-oac/ops-health-artifact-receipt/v2) binds: model_sha256 (canonical bytes),
kernel_sha256 (v2/ops-health/ops_health.py), generator_sha256 (v2/research/generator.py, which
must equal the data manifest's), data_manifest_sha256 (v2/data/MANIFEST.json),
preregistration_sha256 (must equal the registered digest), source {repository, commit = HEAD,
tree_clean, base_v1_source, v1_hub_revision} and selection.  It attests integrity and origin
only; not quality.

Refusal rule: freeze refuses unless
    git -C ROOT status --porcelain -- v2/research v2/ops-health/ops_health.py
is empty, and records ``git rev-parse HEAD`` as source.commit.  ``source_state_override`` is
TEST ONLY: it replaces the git probe (for example to write tree_clean false, which the kernel
must then refuse).  Output files are created exclusively (never overwritten).

Origin check: ``verify_origin(receipt)`` is the research-side integrity layer for the receipt's
source commit (the standalone kernel cannot run git): the commit must exist and its blobs of the
kernel, generator, data manifest and preregistration must hash to the receipt's digests.  When
the receipt's selection.amendments_sha256 is not null (stage 2), the blob of v2/AMENDMENTS.md at
the source commit must hash to it as well.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from . import generator, kernel_bridge, registry

REPOSITORY = "szl-work/oac-frontier (local git repository; unpublished)"
GUARDED_PATHS = ("v2/research", "v2/ops-health/ops_health.py")
MODEL_NAME = "model.json"
RECEIPT_NAME = "artifact_receipt.json"


class FreezeRefused(RuntimeError):
    pass


def canonicalize(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, Mapping):
        return {str(k): canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(canonicalize(value), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def canonical_model_bytes(artifact: Mapping) -> bytes:
    k = kernel_bridge.kernel()
    validated = k.validate_artifact(artifact)
    if canonicalize(validated) != validated:
        raise FreezeRefused("artifact is not at frozen (12-decimal) resolution; refusing")
    data = canonical_json_bytes(validated)
    if k.validate_artifact(json.loads(data)) != validated:
        raise FreezeRefused("canonical model.json does not round-trip; refusing")
    return data


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def source_state(root: Path = registry.ROOT) -> dict:
    porcelain = _git(root, "status", "--porcelain", "--", *GUARDED_PATHS)
    dirty = [line for line in porcelain.splitlines() if line.strip()]
    return {"commit": _git(root, "rev-parse", "HEAD").strip(), "tree_clean": not dirty,
            "dirty": dirty}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_receipt(model_bytes: bytes, *, commit: str, tree_clean: bool, selection: Mapping) -> dict:
    k = kernel_bridge.kernel()
    prereg = _sha256_file(registry.PREREG_PATH)
    if prereg != registry.PREREG_SHA256:
        raise FreezeRefused("PREREGISTRATION.md differs from the registered digest; refusing")
    manifest_bytes = registry.MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    gen_sha = generator.source_sha256()
    if manifest.get("generator_sha256") != gen_sha:
        raise FreezeRefused("generator.py differs from the data manifest's generator; refusing")
    receipt = {
        "schema": k.RECEIPT_SCHEMA,
        "purpose": k.MODEL_PURPOSE,
        "authority": k.MODEL_AUTHORITY,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "kernel_sha256": _sha256_file(registry.KERNEL_PATH),
        "generator_sha256": gen_sha,
        "data_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "preregistration_sha256": prereg,
        "source": {
            "repository": REPOSITORY,
            "commit": commit,
            "tree_clean": tree_clean,
            "base_v1_source": registry.V1_BASE_SOURCE,
            "v1_hub_revision": registry.V1_HUB_REVISION,
        },
        "selection": dict(selection),
        "attests": k.RECEIPT_ATTESTS,
    }
    return receipt


# Receipt digest -> repository path whose blob at source.commit must hash to it.
ORIGIN_BLOBS = (
    ("kernel_sha256", "v2/ops-health/ops_health.py"),
    ("generator_sha256", "v2/research/generator.py"),
    ("data_manifest_sha256", "v2/data/MANIFEST.json"),
    ("preregistration_sha256", "v2/PREREGISTRATION.md"),
)
# selection.amendments_sha256 (stage 2), when not null, must hash this blob at source.commit.
AMENDMENTS_REL = "v2/AMENDMENTS.md"


class OriginMismatch(RuntimeError):
    """The receipt's source commit does not reproduce the receipt's digests."""


def verify_origin(receipt: Mapping, root: Path = registry.ROOT) -> dict:
    """Integrity layer for the receipt's origin claim (read-only git; stage-1 review A3).

    The standalone kernel can only check that source.commit is a 40-hex string.  This check
    resolves the commit in the repository at ``root`` and requires that the committed blobs of
    the kernel, the generator, the data manifest and the preregistration hash (raw blob bytes,
    ``git cat-file blob``) to the receipt's digests.  An altered source commit, or a receipt
    whose digests were not produced from that commit, raises OriginMismatch.
    """
    k = kernel_bridge.kernel()
    rec = k.validate_receipt(receipt)
    commit = rec["source"]["commit"]
    try:
        kind = _git(root, "cat-file", "-t", commit).strip()
    except subprocess.CalledProcessError as exc:
        raise OriginMismatch(f"source commit {commit} is not in the repository") from exc
    if kind != "commit":
        raise OriginMismatch(f"source commit {commit} is a {kind}, not a commit")
    checked = {}
    for key, rel in ORIGIN_BLOBS:
        try:
            blob = subprocess.run(
                ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{rel}"],
                check=True, capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise OriginMismatch(f"{rel} is absent at source commit {commit}") from exc
        digest = hashlib.sha256(blob).hexdigest()
        if digest != rec[key]:
            raise OriginMismatch(f"receipt {key} does not match {rel} at source commit {commit}")
        checked[key] = digest
    amendments = rec["selection"].get("amendments_sha256")
    if amendments is not None:  # stage 2: the amendments in force at freeze
        try:
            blob = subprocess.run(
                ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{AMENDMENTS_REL}"],
                check=True, capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise OriginMismatch(f"{AMENDMENTS_REL} is absent at source commit {commit}") from exc
        digest = hashlib.sha256(blob).hexdigest()
        if digest != amendments:
            raise OriginMismatch(
                f"receipt selection.amendments_sha256 does not match {AMENDMENTS_REL} "
                f"at source commit {commit}"
            )
        checked["selection.amendments_sha256"] = digest
    return {"commit": commit, "verified": checked}


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def freeze(
    artifact: Mapping,
    out_dir: Path | str,
    *,
    selection: Mapping,
    root: Path = registry.ROOT,
    source_state_override: Mapping | None = None,
) -> dict:
    """Write out_dir/model.json and out_dir/artifact_receipt.json; returns paths and digests."""
    if source_state_override is None:
        state = source_state(root)
        if not state["tree_clean"]:
            raise FreezeRefused(
                "working tree is not clean for " + ", ".join(GUARDED_PATHS)
                + f"; refusing to freeze (dirty: {state['dirty']})"
            )
    else:  # TEST ONLY
        state = {"commit": source_state_override["commit"],
                 "tree_clean": source_state_override["tree_clean"]}
    model_bytes = canonical_model_bytes(artifact)
    receipt = build_receipt(model_bytes, commit=state["commit"], tree_clean=state["tree_clean"],
                            selection=selection)
    receipt_bytes = canonical_json_bytes(receipt)
    out_dir = Path(out_dir)
    model_path = out_dir / MODEL_NAME
    receipt_path = out_dir / RECEIPT_NAME
    for path in (model_path, receipt_path):
        if path.exists():
            raise FreezeRefused(f"refusing to overwrite {path}")
    _write_exclusive(model_path, model_bytes)
    _write_exclusive(receipt_path, receipt_bytes)
    return {
        "model_path": str(model_path),
        "receipt_path": str(receipt_path),
        "model_sha256": receipt["model_sha256"],
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "commit": state["commit"],
        "tree_clean": state["tree_clean"],
    }
