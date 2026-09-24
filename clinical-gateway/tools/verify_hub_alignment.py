"""Read-only, exact-revision Git-to-Hub parity for the OAC research artifacts.

Invariant: success requires BOTH closed packages and every byte to match the
explicit Git commit, including provenance links to source and generated data.
Nothing is uploaded, imported from the Hub, trained, or granted authority.
The caller must separately trust/review the Git commit and this verifier.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


SOURCE_REPOSITORY = "https://github.com/szl-holdings/szl-forge"
MODEL_ID = "SZLHOLDINGS/oac-system-health-v1"
DATASET_ID = "SZLHOLDINGS/oac-clinical-transport-observability-synthetic"
MODEL_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "artifact_receipt.json",
        "example_input.json",
        "model.json",
        "oac_operational_health.py",
    }
)
DATASET_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "data/test.jsonl",
        "data/train.jsonl",
        "data/validation.jsonl",
        "dataset_receipt.json",
        "schema.json",
        "training_source_snapshot.py",
    }
)
PREFIX = "clinical-gateway/"
AUTHORITY = "advisory_only_no_ack_release_or_clinical_authority"
MAX_BYTES = 2 * 1024 * 1024
SCHEMA = "szl-oac/hub-alignment-readback/v1"


class AlignmentError(ValueError):
    """Incomplete, unpinned, or structurally invalid evidence."""


def require_revision(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise AlignmentError(
            "revision must be a full lowercase 40-character commit SHA"
        )
    return value


def strict_json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AlignmentError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise AlignmentError("non-finite JSON number")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise AlignmentError("non-finite JSON number")
        return number

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
            parse_float=finite_float,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise AlignmentError("invalid JSON") from exc


def validate_paths(paths, expected, allow_hub_metadata=False):
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise AlignmentError("file list must contain strings")
    for path in paths:
        if (
            not path
            or "\\" in path
            or any(ord(c) < 32 for c in path)
            or any(part in ("", ".", "..") for part in path.split("/"))
        ):
            raise AlignmentError("unsafe package path")
    if len(paths) != len(set(paths)):
        raise AlignmentError("duplicate package path")
    observed = set(paths)
    if allow_hub_metadata:
        observed.discard(".gitattributes")
    if observed != set(expected):
        raise AlignmentError("package file-set mismatch")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def compare_package(expected, observed):
    validate_paths(list(observed), set(expected))
    return [
        {
            "path": path,
            "expected_sha256": digest(expected[path]),
            "observed_sha256": digest(observed[path]),
            "matches": expected[path] == observed[path],
        }
        for path in sorted(expected)
    ]


def _git(repo_root, *args):
    # Only fixed commands and validated hashes/paths are used, never a shell.
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repo_root), *args],
        capture_output=True,
        timeout=45,
        check=False,
    )
    if result.returncode:
        raise AlignmentError(
            "Git object read failed; required commit/blobs must exist locally"
        )
    return result.stdout


def _blob(repo_root, revision, path):
    entries = _git(repo_root, "ls-tree", "-l", "-z", revision, "--", path).split(b"\0")
    if len(entries) != 2 or not entries[0]:
        raise AlignmentError("expected exactly one Git file")
    header, name = entries[0].split(b"\t", 1)
    mode, kind, oid, size = header.split()
    if (
        mode != b"100644"
        or kind != b"blob"
        or name.decode("utf-8") != path
        or int(size) > MAX_BYTES
    ):
        raise AlignmentError("Git file must be a bounded regular non-executable blob")
    raw = _git(repo_root, "cat-file", "blob", oid.decode("ascii"))
    if len(raw) != int(size):
        raise AlignmentError("Git blob size mismatch")
    return raw


def _package(repo_root, revision, directory, files):
    paths = (
        _git(
            repo_root,
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            revision,
            "--",
            directory + "/",
        )
        .decode("utf-8")
        .split("\0")
    )
    paths = [path.removeprefix(directory + "/") for path in paths if path]
    validate_paths(paths, set(files))
    return {
        path: _blob(repo_root, revision, directory + "/" + path)
        for path in sorted(files)
    }


def manifest_from_git(repo_root, revision):
    require_revision(revision)
    if _git(repo_root, "cat-file", "-t", revision).strip() != b"commit":
        raise AlignmentError("source revision is not a commit")
    model = _package(
        repo_root,
        revision,
        PREFIX + "huggingface/model/oac-system-health-v1",
        MODEL_FILES,
    )
    dataset = _package(
        repo_root,
        revision,
        PREFIX
        + "huggingface/dataset/"
        + "oac-clinical-transport-observability-synthetic",
        DATASET_FILES,
    )
    source = _blob(repo_root, revision, PREFIX + "src/oac_operational_health.py")
    trainer = _blob(
        repo_root, revision, PREFIX + "tools/train_operational_health_model.py"
    )
    mr = strict_json(model["artifact_receipt.json"])
    dr = strict_json(dataset["dataset_receipt.json"])
    if not isinstance(mr, dict) or not isinstance(dr, dict):
        raise AlignmentError("receipt must be an object")
    checks = [
        model["oac_operational_health.py"] == source,
        dataset["training_source_snapshot.py"] == trainer,
        mr.get("schema") == "szl-oac/transport-health-artifact-receipt/v1",
        dr.get("schema") == "szl-oac/transport-health-dataset-receipt/v1",
        mr.get("purpose") == "synthetic_operational_transport_attention_only",
        dr.get("purpose") == "synthetic_operational_transport_observability_only",
        mr.get("authority") == dr.get("authority") == AUTHORITY,
        dr.get("contains_phi") is False,
        dr.get("contains_clinical_results") is False,
        dr.get("generated_data") == "synthetic_only",
        mr.get("metrics_scope")
        == "fixed_seed_synthetic_splits_only_not_production_validation",
        mr.get("model_sha256") == digest(model["model.json"]),
        mr.get("kernel_sha256") == digest(source),
        mr.get("generator_sha256") == dr.get("generator_sha256") == digest(trainer),
        mr.get("dataset_receipt_sha256") == digest(dataset["dataset_receipt.json"]),
        dr.get("schema_sha256") == digest(dataset["schema.json"]),
        dr.get("files")
        == {
            path: digest(dataset[path])
            for path in sorted(DATASET_FILES)
            if path.startswith("data/")
        },
    ]
    if not all(checks):
        raise AlignmentError("canonical source/dataset/model receipt binding mismatch")
    return {"model": model, "dataset": dataset}


def _allowed_url(url):
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.netloc != "huggingface.co"
        or parts.fragment
        or parts.username
        or parts.password
    ):
        raise AlignmentError("only the fixed HTTPS Hugging Face origin is allowed")


class _SameOriginRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _allowed_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def public_get(url):
    """Public, credential-free bounded reads; no Hub code is executed."""
    _allowed_url(url)
    opener = build_opener(_SameOriginRedirect())
    request = Request(
        url,
        headers={"User-Agent": "szl-oac-alignment/1", "Accept-Encoding": "identity"},
    )
    deadline = time.monotonic() + 45
    with opener.open(request, timeout=15) as response:
        if (
            response.status != 200
            or response.headers.get("Content-Encoding", "identity") != "identity"
        ):
            raise AlignmentError("unexpected HTTP status or encoding")
        declared = response.headers.get("Content-Length")
        if declared is not None and (
            not declared.isdecimal() or int(declared) > MAX_BYTES
        ):
            raise AlignmentError("HTTP response size exceeds bound")
        body = bytearray()
        while True:
            chunk = response.read1(min(65536, MAX_BYTES + 1 - len(body)))
            body.extend(chunk)
            if len(body) > MAX_BYTES or time.monotonic() > deadline:
                raise AlignmentError("HTTP response exceeded size/time bound")
            if not chunk:
                break
        if declared is not None and len(body) != int(declared):
            raise AlignmentError("incomplete HTTP response")
    return bytes(body)


def hub_package(kind, repo_id, revision, expected_paths, fetch=public_get):
    require_revision(revision)
    if (kind, repo_id) not in (("model", MODEL_ID), ("dataset", DATASET_ID)):
        raise AlignmentError("destination outside OAC allowlist")
    metadata = strict_json(
        fetch(f"https://huggingface.co/api/{kind}s/{repo_id}/revision/{revision}")
    )
    if not isinstance(metadata, dict) or metadata.get("sha") != revision:
        raise AlignmentError("Hub revision does not match requested immutable commit")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list) or any(
        not isinstance(item, dict) for item in siblings
    ):
        raise AlignmentError("missing Hub file manifest")
    validate_paths([item.get("rfilename") for item in siblings], expected_paths, True)
    prefix = "datasets/" if kind == "dataset" else ""
    return {
        path: fetch(
            f"https://huggingface.co/{prefix}{repo_id}/resolve/{revision}/{quote(path, safe='/')}"
        )
        for path in sorted(expected_paths)
    }


def verify_alignment(
    repo_root, git_revision, model_revision, dataset_revision, fetch=public_get
):
    # Validate all pins before any network access.
    for revision in (git_revision, model_revision, dataset_revision):
        require_revision(revision)
    expected = manifest_from_git(repo_root, git_revision)
    packages = []
    for kind, repo_id, revision in (
        ("model", MODEL_ID, model_revision),
        ("dataset", DATASET_ID, dataset_revision),
    ):
        observed = hub_package(kind, repo_id, revision, set(expected[kind]), fetch)
        files = compare_package(expected[kind], observed)
        packages.append(
            {
                "kind": kind,
                "repo_id": repo_id,
                "revision": revision,
                "matches": all(item["matches"] for item in files),
                "files": files,
            }
        )
    return {
        "schema": SCHEMA,
        "complete": all(package["matches"] for package in packages),
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": git_revision,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "packages": packages,
        "scope": "exact_pinned_source_and_hub_byte_parity_only",
        "excluded_hub_metadata": [".gitattributes"],
        "source_signature_verified": False,
        "training_rerun": False,
        "production_promotion_allowed": False,
        "clinical_use_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional NEW receipt path; existing files are not overwritten",
    )
    args = parser.parse_args(argv)
    try:
        receipt = verify_alignment(
            args.repo_root,
            args.git_revision,
            args.model_revision,
            args.dataset_revision,
        )
    except (
        AlignmentError,
        OSError,
        ValueError,
        HTTPException,
        subprocess.SubprocessError,
    ) as exc:
        # Never reflect remote content, exception URLs, credentials, or tracebacks.
        receipt = {
            "schema": SCHEMA,
            "complete": False,
            "failure_type": type(exc).__name__,
            "failure_code": "ALIGNMENT_EVIDENCE_INCOMPLETE",
            "source_revision": args.git_revision,
            "model_revision": args.model_revision,
            "dataset_revision": args.dataset_revision,
            "production_promotion_allowed": False,
            "clinical_use_authorized": False,
        }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8", newline="\n") as output:
                output.write(encoded)
        except OSError:
            print(
                '{"complete":false,"failure_code":"RECEIPT_WRITE_FAILED"}',
                file=sys.stderr,
            )
            return 2
    print(encoded, end="")
    return 0 if receipt["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
