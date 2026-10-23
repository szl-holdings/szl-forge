"""Freeze a v3 candidate: canonical model.json plus artifact_receipt.json (protocol s.6.6, s.6.7).

Canonical model.json (as v2): floats rounded to 12 decimals, ``json.dumps(sort_keys=True,
indent=2, ensure_ascii=True, allow_nan=False)`` plus a trailing newline. ``freeze`` refuses an
artifact whose canonical rounding would change any value, or that the v3 kernel does not accept.

Receipt (schema ``szl-oac/ops-health-artifact-receipt/v3``; s.6.7 item 8, s.14.4) binds:
model_sha256 (canonical bytes), kernel_sha256 (``v3/ops-health/ops_health.py``),
generator_sha256 (``v3/research/generator.py``; must equal the data manifest's when the manifest
records one), data_manifest_sha256, preregistration_sha256 (must be the registered digest),
reregistrations_sha256 (each seal re-registration file, in order: r1, ...), amendments_sha256
(``v3/AMENDMENTS.md`` or null while it does not exist), source {repository, commit, tag,
tree_clean} and fit {segment, declared_fit, trace_sha256, receipts_dir_sha256}. It attests
integrity and origin only; not quality.

Refusal rule: ``freeze`` refuses unless ``git status --porcelain -- v3/research
v3/ops-health/ops_health.py`` (from the research root) is empty, and records ``git rev-parse
HEAD`` as source.commit. ``source_state_override`` is TEST AND DEV ONLY: it replaces the git
probe; DEV receipts carry the fixed fake commit of the test support, which no origin check can
ever accept. Output files are created exclusively (never overwritten).

The research-side origin check of s.14.4 (rules a to e) belongs to the recipe lane and is not
implemented here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from v2.research import freeze as v2_freeze

from . import generator, kernel_bridge, registry

GUARDED_PATHS = ("v3/research", "v3/ops-health/ops_health.py")
MODEL_NAME = "model.json"
RECEIPT_NAME = "artifact_receipt.json"
canonicalize = v2_freeze.canonicalize
canonical_json_bytes = v2_freeze.canonical_json_bytes


class FreezeRefused(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_model_bytes(artifact: Mapping) -> bytes:
    k = kernel_bridge.kernel()
    validated = k.validate_artifact(artifact)
    if canonicalize(validated) != validated:
        raise FreezeRefused("artifact is not at frozen (12-decimal) resolution; refusing")
    data = canonical_json_bytes(validated)
    if k.load_artifact_bytes(data) != validated:
        raise FreezeRefused("canonical model.json does not round-trip; refusing")
    return data


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def source_state(root: Path = registry.ROOT) -> dict:
    porcelain = _git(root, "status", "--porcelain", "--", *GUARDED_PATHS)
    dirty = [line for line in porcelain.splitlines() if line.strip()]
    return {"commit": _git(root, "rev-parse", "HEAD").strip(), "tree_clean": not dirty, "dirty": dirty}


def build_receipt(
    model_bytes: bytes,
    *,
    commit: str,
    tag: str,
    tree_clean: bool,
    fit: Mapping,
    data_manifest_bytes: bytes,
    reregistration_paths: Sequence[Path] = (registry.REREG_R1_PATH,),
    amendments_path: Path = registry.AMENDMENTS_PATH,
) -> dict:
    k = kernel_bridge.kernel()
    prereg = sha256_file(registry.PREREG_PATH)
    if prereg != registry.PREREG_SHA256 or prereg != k.PREREGISTRATION_SHA256:
        raise FreezeRefused("PREREGISTRATION.md differs from the registered digest; refusing")
    gen_sha = generator.source_sha256()
    try:
        manifest = json.loads(data_manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FreezeRefused(f"data manifest is not JSON: {exc}") from exc
    recorded = manifest.get("generator_sha256") if isinstance(manifest, Mapping) else None
    if recorded is not None and recorded != gen_sha:
        raise FreezeRefused("generator.py differs from the data manifest's generator; refusing")
    reregs = []
    for path in reregistration_paths:
        digest = sha256_file(path)
        if Path(path).resolve() == registry.REREG_R1_PATH.resolve() and digest != registry.REREG_R1_SHA256:
            raise FreezeRefused("REREGISTRATION_r1.md differs from its registered digest; refusing")
        reregs.append(digest)
    amendments = sha256_file(amendments_path) if Path(amendments_path).exists() else None
    receipt = {
        "schema": k.RECEIPT_SCHEMA,
        "purpose": k.MODEL_PURPOSE,
        "authority": k.MODEL_AUTHORITY,
        "model_sha256": sha256_bytes(model_bytes),
        "kernel_sha256": sha256_file(registry.KERNEL_PATH),
        "generator_sha256": gen_sha,
        "data_manifest_sha256": sha256_bytes(data_manifest_bytes),
        "preregistration_sha256": prereg,
        "reregistrations_sha256": reregs,
        "amendments_sha256": amendments,
        "source": {"repository": k.REPOSITORY, "commit": commit, "tag": tag, "tree_clean": tree_clean},
        "fit": dict(fit),
        "attests": k.RECEIPT_ATTESTS,
    }
    return receipt


def consistent_receipt(receipt: Mapping, model_bytes: bytes) -> dict:
    """The receipt with model_sha256 regenerated for ``model_bytes`` (every other field kept)."""
    out = json.loads(json.dumps(receipt))
    out["model_sha256"] = sha256_bytes(model_bytes)
    return out


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def write_pair(directory: Path, model_bytes: bytes, receipt: Mapping) -> tuple[Path, Path]:
    directory = Path(directory)
    model = directory / MODEL_NAME
    rec = directory / RECEIPT_NAME
    _write_exclusive(model, model_bytes)
    _write_exclusive(rec, canonical_json_bytes(receipt))
    return model, rec


def freeze(
    artifact: Mapping,
    out_dir: Path | str,
    *,
    tag: str,
    fit: Mapping,
    data_manifest_bytes: bytes,
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
    else:  # TEST AND DEV ONLY
        state = {"commit": source_state_override["commit"], "tree_clean": source_state_override["tree_clean"]}
    model_bytes = canonical_model_bytes(artifact)
    receipt = build_receipt(model_bytes, commit=state["commit"], tag=tag, tree_clean=state["tree_clean"],
                            fit=fit, data_manifest_bytes=data_manifest_bytes)
    out_dir = Path(out_dir)
    for path in (out_dir / MODEL_NAME, out_dir / RECEIPT_NAME):
        if path.exists():
            raise FreezeRefused(f"refusing to overwrite {path}")
    model_path, receipt_path = write_pair(out_dir, model_bytes, receipt)
    return {
        "model_path": str(model_path),
        "receipt_path": str(receipt_path),
        "model_sha256": receipt["model_sha256"],
        "receipt_sha256": sha256_file(receipt_path),
        "commit": state["commit"],
        "tree_clean": state["tree_clean"],
    }


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
