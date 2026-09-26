"""The A11OY-MINI binding cites exact published bytes and grants no authority."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
RECORD = ROOT / "frontier/evaluation/identity_bindings/evals/A11OY-MINI_0619dd65b92a_evaluation_binding_20260926.json"


def minus_text(rel):
    """True when a '* -text' .gitattributes between the file and ROOT stores it verbatim."""
    for parent in (ROOT / rel).parents:
        attrs = parent / ".gitattributes"
        if attrs.is_file() and any(line.split() == [b"*", b"-text"] for line in attrs.read_bytes().splitlines()):
            return True
        if parent == ROOT:
            return False
    raise AssertionError(f"{rel} is outside the repository")


def blob(rel, declared):
    """Canonical Git blob bytes on a Windows autocrlf or a Linux checkout."""
    path = (ROOT / rel).resolve()
    assert path.is_relative_to(ROOT.resolve())
    assert declared == ("-text" if minus_text(rel) else "text"), rel
    raw = path.read_bytes()
    if declared == "-text":
        return raw
    # Autocrlf text is an LF Git blob; a Windows checkout adds CR before LF.
    raw = raw.replace(b"\r\n", b"\n")
    assert b"\r" not in raw, rel
    return raw


def check(rel, declared, sha256, size):
    raw = blob(rel, declared)
    assert hashlib.sha256(raw).hexdigest() == sha256, rel
    assert len(raw) == size, rel
    return raw


def load():
    record = json.loads(RECORD.read_bytes())
    receipt = json.loads(check(record["evaluation_receipt"], record["evaluation_receipt_git_attributes"],
                               record["evaluation_receipt_sha256"], record["evaluation_receipt_bytes"]))
    manifest = json.loads(check(record["identity_manifest_path"], record["identity_manifest_git_attributes"],
                                record["identity_manifest_sha256"], record["identity_manifest_bytes"]))
    review = record["semantic_review"]
    review_doc = json.loads(check(review["path"], review["git_attributes"], review["sha256"], review["bytes"]))
    return record, receipt, manifest, review_doc


def test_record_is_lf_json_and_every_repository_hash_recomputes():
    raw = RECORD.read_bytes().replace(b"\r\n", b"\n")
    assert b"\r" not in raw and raw.endswith(b"}\n")
    record, receipt, _, _ = load()
    evaluator = record["evaluator"]
    package = record["evidence_publication"]["package_manifest"]
    listed = json.loads(check(package["path"], package["git_attributes"], package["sha256"], package["bytes"]))
    listed = {record["evidence_publication"]["evidence_directory"] + "/" + item["path"]: item for item in listed["files"]}
    for item in (evaluator["runner"], evaluator["scoring_helper"], evaluator["canonical_scorer_source"],
                 *evaluator["fixtures"]):
        check(item["path"], item["git_attributes"], item["sha256"], item["bytes"])
        assert listed[item["path"]]["sha256"] == item["sha256"]
    for item in evaluator["fixtures"]:
        check(item["canonical_path"], item["canonical_path_git_attributes"], item["sha256"], item["bytes"])
    for rel, sha in ((record["evaluation_receipt"], record["evaluation_receipt_sha256"]),
                     (record["semantic_review"]["path"], record["semantic_review"]["sha256"])):
        assert listed[rel]["sha256"] == sha
    # The -text evidence subtree keeps the witnessed CRLF receipt bytes.
    assert record["evaluation_receipt_git_attributes"] == "-text"
    assert b"\r\n" in (ROOT / record["evaluation_receipt"]).read_bytes()
    assert record["identity_manifest_git_attributes"] == "text"
    assert receipt["runner_sha256"] == evaluator["runner"]["sha256"]
    assert receipt["scoring"]["helper_sha256"] == evaluator["scoring_helper"]["sha256"]
    assert receipt["scoring"]["canonical_source_sha256"] == evaluator["canonical_scorer_source"]["sha256"]


def test_evaluated_file_matches_receipt_artifact_binding_and_identity_manifest():
    record, receipt, manifest, _ = load()
    binding = receipt["artifact_binding"]
    assert record["repository"] == manifest["repository"] == binding["repo"]
    assert record["pinned_revision"] == manifest["pinned_revision"] == binding["revision"]
    assert record["identity_manifest"] == Path(record["identity_manifest_path"]).name
    assert record["evaluated_files"] == [{"path": binding["file"], "sha256": binding["sha256"], "bytes": binding["bytes"]}]
    assert binding["hub_lfs_sha256"] == binding["rehash_before"] == binding["rehash_after"] == binding["sha256"]
    assert binding["matches_hub_lfs"] is True
    files = {item["path"]: item for item in manifest["files"]}
    for item in record["evaluated_files"] + record["not_evaluated"]:
        entry = files[item["path"]]
        assert entry["hub_lfs_sha256"] == entry["local_sha256"] == item["sha256"]
        assert entry["size"] == item["bytes"]
        assert entry["status"] == "LOCAL_REHASH_MATCHES_HUB_LFS"
    for item in record["identity_manifest_non_weight_files"]:
        assert files[item["path"]]["local_sha256"] == item["sha256"]
        assert files[item["path"]]["size"] == item["bytes"]
    accounted = [item["path"] for key in ("evaluated_files", "not_evaluated", "identity_manifest_non_weight_files")
                 for item in record[key]]
    assert sorted(accounted) == sorted(files)
    assert record["not_evaluated"][0]["path"] == "a11oy-mini-r2-BF16-mmproj.gguf"
    assert not any("mmproj" in part for part in receipt["runtime"]["command"])
    assert receipt["runtime"]["properties"]["modalities"]["vision"] is False
    assert record["runtime"]["mmproj_argument_present"] is False
    for item in record["later_hub_revision_note"]["files"]:
        assert item["identical"] is (item["pinned_revision_lfs_sha256"] == item["later_revision_lfs_sha256"]
                                     and item["pinned_revision_bytes"] == item["later_revision_bytes"])
        assert item["identical"] is True
        assert item["pinned_revision_lfs_sha256"] == files[item["path"]]["hub_lfs_sha256"]


def test_counts_runtime_and_semantic_review_are_copied_not_improved():
    record, receipt, _, review = load()
    assert record["observed_counts"] == receipt["counts"]
    assert record["truncated_cases"] == receipt["truncated_cases"] == []
    assert record["evaluation_receipt_kind"] == receipt["schema"]
    assert record["evaluation_receipt_status"] == receipt["status"] == "MEASURED_BOUNDED_PASS"
    runtime = receipt["runtime"]
    assert record["runtime"]["kind"] == runtime["kind"]
    assert record["runtime"]["version"] == runtime["version"] == ""
    build = runtime["properties"]["build_info"]
    assert record["runtime"]["build_fingerprint"] == build
    assert all(case["response"]["system_fingerprint"] == build for case in receipt["cases"])
    assert record["runtime"]["generation"] == runtime["generation"]
    assert record["runtime"]["context"] == runtime["context"]
    fixtures = {item["canonical_path"]: item for item in record["evaluator"]["fixtures"]}
    for item in receipt["fixture_identity"]:
        assert fixtures[item["path"]]["sha256"] == item["sha256"]
        assert fixtures[item["path"]]["count"] == item["count"]
    assert record["evaluator"]["held_out"] is False
    assert record["evaluator"]["scoring_kind"] == "OUTPUT_ENVELOPE_ONLY"
    summary = record["semantic_review"]
    assert summary["status"] == review["status"] == "SEMANTIC_FAILURES_OBSERVED"
    assert summary["reviewed_receipt_sha256"] == review["receipt_sha256"] == record["evaluation_receipt_sha256"]
    failed = [item["case_id"] for item in review["findings"] if item["status"] == "FAIL"]
    assert summary["failed_case_ids"] == failed and summary["fail_count"] == len(failed) == 4
    assert summary["semantic_pass_rate"] is review["semantic_pass_rate"] is None


def test_binding_grants_no_authority_and_discloses_what_was_not_run():
    record = json.loads(RECORD.read_bytes())
    assert record["schema"] == "szl.evaluation-binding/v1"
    assert record["publication_eligible"] is False
    assert record["autonomy_eligible"] is False
    assert record["production_disposition"] == "HOLD"
    assert record["promotion_effect"] == "NONE"
    assert record["model_output_authority"] == "PROPOSAL_ONLY"
    assert record["signature_status"] == "UNSIGNED_HONEST"
    assert record["held_out_l3"]["status"] == "NOT_RUN"
    assert record["new_inference_performed_for_this_binding"] is False
    assert record["gguf_bytes_downloaded_for_this_binding"] is False
    boundary = record["boundary"].lower()
    assert "no new inference" in boundary and "not qualification" in boundary

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                assert not (key in ("artifactBinding", "artifact_binding") and value is False), key
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(record)
