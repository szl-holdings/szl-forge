"""Stdlib-only, read-only Hub evidence -> local provenance repair proposals.

No publish/apply operation exists. A proposal is not authorization to publish.
The complete-tree assertion is validated against the supplied audit capture;
these unsigned local records are evidence, not independent provider attestation.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

TARGETS = {
    "SZLHOLDINGS/szl-blocked": "model.joblib",
    "SZLHOLDINGS/szl-govsign": "model.joblib",
    "SZLHOLDINGS/szl-provctl": "model.joblib",
    "SZLHOLDINGS/szl-invariants": "model.joblib",
    "SZLHOLDINGS/szl-ouroboros": "model.joblib",
    "SZLHOLDINGS/szl-formulas": "model.joblib",
}
DESTINATION = "MODEL_PROVENANCE.json"
STATUS = "PROPOSED_NOT_APPLIED"
WEIGHT_SUFFIXES = (".joblib", ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf", ".pkl", ".pickle", ".npz")
MAX_DOCUMENT_BYTES = 1_000_000


class InvalidEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def duplicate_guard(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def parse(data):
    return json.loads(data, object_pairs_hook=duplicate_guard,
                      parse_constant=lambda value: (_ for _ in ()).throw(InvalidEvidence("nonfinite JSON value")))


def digest(value, length):
    require(isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None,
            "invalid immutable digest")
    return value


def relative_path(value):
    require(isinstance(value, str) and 0 < len(value) <= 512, "invalid path")
    require(not any(ord(c) < 32 for c in value) and "\\" not in value and ":" not in value,
            "unsafe path characters")
    parts = value.split("/")
    require(all(p not in ("", ".", "..") and not p.endswith((" ", ".")) for p in parts), "unsafe path segment")
    require(not PurePosixPath(value).is_absolute() and str(PurePosixPath(value)) == value, "noncanonical path")
    return value


def validate_snapshot(snapshot):
    require(isinstance(snapshot, dict), "snapshot must be an object")
    repo_id = snapshot["id"]
    require(repo_id in TARGETS and snapshot["repository_type"] == "model", "target outside closed model set")
    revision = digest(snapshot["revision"], 40)
    repo = snapshot["repository_record"]
    require(repo["id"] == repo_id and repo["kind"] == "models" and repo["revision"] == revision,
            "repository identity mismatch")
    require(repo["metadata"]["id"] == repo_id and repo["metadata"]["sha"] == revision, "metadata head mismatch")
    require(repo["tree_coverage"] == "COMPLETE_ACCESSIBLE_RESPONSE", "incomplete tree")
    prefix = f"https://huggingface.co/api/models/{repo_id}/tree/{revision}?"
    pages = repo["tree_pages"]
    require(isinstance(pages, list) and pages, "missing tree page receipts")
    for index, page in enumerate(pages):
        require(page["status"] == 200 and page["requested_url"].startswith(prefix), "invalid tree page receipt")
        digest(page["response_sha256"], 64)
        expected_next = pages[index + 1]["requested_url"] if index + 1 < len(pages) else None
        require(page["next_url"] == expected_next, "incomplete or broken tree pagination")
    files = {}
    rows = snapshot["tree"]
    require(isinstance(rows, list) and len(rows) == repo["tree_entries"], "tree entry count mismatch")
    seen = set()
    for row in rows:
        require(row["kind"] == "models" and row["id"] == repo_id and row["revision"] == revision,
                "tree identity mismatch")
        name = relative_path(row["path"])
        require(name not in seen, "duplicate tree path")
        seen.add(name)
        require(row["type"] in ("file", "directory"), "unsupported tree entry type")
        digest(row["oid"], 40)
        require(type(row["size"]) is int and row["size"] >= 0, "invalid file size")
        if row["type"] == "file":
            files[name] = row
    require(len(files) == repo["file_count"], "file count mismatch")
    siblings = repo["metadata"]["siblings"]
    sibling_paths = [relative_path(row["rfilename"]) for row in siblings]
    require(len(sibling_paths) == len(set(sibling_paths)) and set(sibling_paths) == set(files),
            "metadata siblings do not match complete file tree")
    require(DESTINATION in files, "original provenance missing from tree")
    require(TARGETS[repo_id] not in files, "historical surrogate exists: this absence-only repair is inapplicable")
    require(not any(name.lower().endswith(WEIGHT_SUFFIXES) for name in files), "possible weight artifact requires manual review")
    text = snapshot["text_record"]
    require((text["kind"], text["id"], text["revision"], text["path"]) == ("models", repo_id, revision, DESTINATION),
            "text identity mismatch")
    require(text["receipt"]["status"] == 200, "original provenance read unsuccessful")
    raw = base64.b64decode(snapshot["original_bytes_base64"], validate=True)
    require(0 < len(raw) <= MAX_DOCUMENT_BYTES, "original document size outside bound")
    require(sha(raw) == digest(text["receipt"]["response_sha256"], 64), "original byte hash mismatch")
    require(len(raw) == text["receipt"]["response_bytes"] == files[DESTINATION]["size"], "original byte size mismatch")
    git_oid = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    require(git_oid == files[DESTINATION]["oid"], "original bytes do not match immutable Git blob")
    original = parse(raw)
    require(isinstance(original, dict) and original == text["content"], "original document/content mismatch")
    model = original["model"]
    require(model["id"] == repo_id and model["repository_type"] == "model", "original model identity mismatch")
    require(model["surrogate"]["file"] == TARGETS[repo_id], "unexpected historical artifact")
    require(type(model["trained_weights_present"]) is bool, "malformed original presence assertion")
    return original, files


def proposal_document(snapshot):
    original, files = validate_snapshot(snapshot)
    repo_id, revision = snapshot["id"], snapshot["revision"]
    original_hash = snapshot["text_record"]["receipt"]["response_sha256"]
    receipt_path = original["model"]["surrogate"].get("receipt")
    if receipt_path is not None:
        relative_path(receipt_path)
    return {
        "schema": "szl.model-source-attestation/reconciliation-v2",
        "status": STATUS,
        "model": {
            "id": repo_id, "repository_type": "model",
            "artifact_kind": "kernel-code-and-configuration",
            "trained_weights_present": False,
            "surrogate": {"historical_file": TARGETS[repo_id],
                          "status": "NOT_PRESENT_IN_CURRENT_REVISION",
                          "weights_published_in_this_repo": False},
        },
        "claims": {"weights_present": "NONE", "trained_model": "NOT_CLAIMED",
                   "training_verified": False, "runtime_measured_by_this_reconciliation": False,
                   "reproducible_build": "NOT_CLAIMED"},
        "current_artifact_evidence": {
            "repository_revision": revision,
            "observation_time": snapshot["repository_record"]["metadata_receipt"]["observed_at"],
            "tree_coverage": "COMPLETE_ACCESSIBLE_RESPONSE",
            "tree_entries": len(snapshot["tree"]), "file_count": len(files),
            "tree_records_sha256": sha(canonical(snapshot["tree"])),
            "original_provenance_sha256": original_hash,
            "training_receipt_path": receipt_path,
            "training_receipt_present": receipt_path in files if receipt_path else False,
            "training_receipt_contents_or_metrics_verified": False,
            "scan_weight_suffixes": list(WEIGHT_SUFFIXES),
        },
        "historical_evidence": {
            "status": "PRESERVED_NOT_REVERIFIED",
            "document": copy.deepcopy(original),
            "original_bytes_base64": snapshot["original_bytes_base64"],
            "original_sha256": original_hash,
            "scope": "Historical claims, metrics, dependencies and source-of-record assertions are preserved verbatim, not adopted as current verification.",
        },
        "limits": [
            "This is a proposed metadata correction, not an applied Hub commit or training result.",
            "Current means the exact observed revision above; publication requires a fresh expected-parent check and reviewed source approval.",
            "Artifact absence is checked against complete captured tree paths and recognized weight suffixes, not semantic analysis of every file.",
            "Tree receipts are unsigned audit records; this package does not claim independent third-party attestation.",
            "Training-receipt file presence is not verification of training, rights, metrics, independent human identity or served runtime.",
            "No dependency, stdlib-only, cryptographic, theorem, source-authority or model-quality claim is inferred from absent weights.",
        ],
    }


def make_plan(snapshots):
    require(isinstance(snapshots, list) and len(snapshots) == len(TARGETS), "all six exact targets required")
    require({item["id"] for item in snapshots} == set(TARGETS), "duplicate, missing or unapproved target")
    entries = []
    for snapshot in sorted(snapshots, key=lambda item: item["id"]):
        proposed = proposal_document(snapshot)
        entries.append({
            "id": snapshot["id"], "repository_type": "model", "status": STATUS,
            "expected_parent": snapshot["revision"], "destination": DESTINATION,
            "operation": "PROPOSE_METADATA_REPLACEMENT_ONLY",
            "source_snapshot_sha256": sha(canonical(snapshot)),
            "original_sha256": snapshot["text_record"]["receipt"]["response_sha256"],
            "proposed_sha256": sha(canonical(proposed)), "proposed_document": proposed,
        })
    return {"schema": "szl.provenance-repair-plan/v1", "status": STATUS,
            "provider_mutations": 0, "publish_supported": False, "entries": entries}


def verify_plan(plan, snapshots):
    expected = make_plan(snapshots)
    require(canonical(plan) == canonical(expected), "proposal differs from deterministic evidence-bound plan")
    return {"status": "VERIFIED_PROPOSALS_NOT_APPLIED", "targets": len(TARGETS),
            "plan_sha256": sha(canonical(plan)), "provider_mutations": 0}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield parse(line)


def collect_snapshots(audit):
    repos, trees, texts = {}, {repo: [] for repo in TARGETS}, {}
    for row in read_jsonl(audit / "repositories.jsonl"):
        if row.get("kind") == "models" and row.get("id") in TARGETS:
            require(row["id"] not in repos, "duplicate repository capture")
            repos[row["id"]] = row
    for row in read_jsonl(audit / "files.jsonl"):
        if row.get("kind") == "models" and row.get("id") in TARGETS:
            trees[row["id"]].append(row)
    for row in read_jsonl(audit / "text-evidence.jsonl"):
        if row.get("kind") == "models" and row.get("id") in TARGETS and row.get("path") == DESTINATION:
            require(row["id"] not in texts, "duplicate provenance text capture")
            texts[row["id"]] = row
    require(set(repos) == set(texts) == set(TARGETS), "incomplete target audit capture")
    snapshots = []
    for repo_id in sorted(TARGETS):
        revision = digest(repos[repo_id]["revision"], 40)
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{DESTINATION}"
        # Public immutable GET only. No token discovery, auth header, model load,
        # arbitrary URL input, upload method, or provider-write capability.
        request = Request(url, headers={"User-Agent": "SZL-provenance-proposal/1"})
        with urlopen(request, timeout=45) as response:
            require(response.status == 200, "pinned original download failed")
            raw = response.read(MAX_DOCUMENT_BYTES + 1)
        snapshot = {"id": repo_id, "repository_type": "model", "revision": revision,
                    "repository_record": repos[repo_id], "tree": sorted(trees[repo_id], key=lambda row: row["path"]),
                    "text_record": texts[repo_id], "original_bytes_base64": base64.b64encode(raw).decode("ascii")}
        validate_snapshot(snapshot)
        snapshots.append(snapshot)
    return snapshots


def write_json(path, document):
    with path.open("xb") as handle:
        handle.write(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    propose = sub.add_parser("propose", help="read pinned public originals; create a NEW local proposal directory")
    propose.add_argument("--audit", type=Path, required=True)
    propose.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="offline exact validation; never publishes")
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "propose":
            require(not args.output.exists(), "output already exists; no overwrite permitted")
            snapshots = collect_snapshots(args.audit)
            plan = make_plan(snapshots)
            receipt = verify_plan(plan, snapshots)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "snapshots.json", snapshots)
            write_json(args.output / "plan.json", plan)
            for entry in plan["entries"]:
                target_dir = args.output / entry["id"].split("/")[1]
                target_dir.mkdir()
                write_json(target_dir / DESTINATION, entry["proposed_document"])
            write_json(args.output / "verification.json", receipt)
        else:
            snapshots = parse((args.bundle / "snapshots.json").read_bytes())
            plan = parse((args.bundle / "plan.json").read_bytes())
            receipt = verify_plan(plan, snapshots)
            expected_names = {"snapshots.json", "plan.json", "verification.json"} | {key.split("/")[1] for key in TARGETS}
            require({p.name for p in args.bundle.iterdir()} == expected_names, "unexpected or missing bundle files")
            for entry in plan["entries"]:
                target_dir = args.bundle / entry["id"].split("/")[1]
                require({p.name for p in target_dir.iterdir()} == {DESTINATION}, "unexpected proposal destination")
                actual = parse((target_dir / DESTINATION).read_bytes())
                require(canonical(actual) == canonical(entry["proposed_document"]), "extracted proposal differs")
            require(parse((args.bundle / "verification.json").read_bytes()) == receipt, "verification receipt mismatch")
        print(json.dumps(receipt, indent=2))
        return 0
    except (InvalidEvidence, KeyError, TypeError, ValueError, OSError) as exc:
        print("REFUSED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
