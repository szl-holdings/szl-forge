"""Synthetic evidence contracts; never accesses a provider or loads model code."""
import base64
import copy
import hashlib
from pathlib import Path
import unittest
from tools import kernel_provenance_evidence as repair


def fixture(repo_id):
    revision = "a" * 40
    original = {"model": {"id": repo_id, "repository_type": "model", "trained_weights_present": True,
                          "surrogate": {"file": repair.TARGETS[repo_id], "receipt": "training.json"}},
                "claims": {"weights_present": "NONE", "trained_model": "SURROGATE_PRESENT_MEASURED_FIDELITY"},
                "historical_metric": {"accuracy": 0.91}}
    raw = repair.canonical(original)
    oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    rows = [{"id": repo_id, "kind": "models", "revision": revision, "path": repair.DESTINATION,
             "type": "file", "oid": oid, "size": len(raw)}]
    record = {"id": repo_id, "kind": "models", "revision": revision,
              "metadata": {"id": repo_id, "sha": revision, "siblings": [{"rfilename": repair.DESTINATION}]},
              "metadata_receipt": {"observed_at": "SYNTHETIC"}, "tree_entries": 1, "file_count": 1,
              "tree_coverage": "COMPLETE_ACCESSIBLE_RESPONSE",
              "tree_pages": [{"status": 200, "requested_url": f"https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true",
                              "response_sha256": "b" * 64, "next_url": None}]}
    text = {"id": repo_id, "kind": "models", "revision": revision, "path": repair.DESTINATION,
            "content": original, "receipt": {"status": 200, "response_sha256": repair.sha(raw), "response_bytes": len(raw)}}
    return {"id": repo_id, "repository_type": "model", "revision": revision, "repository_record": record,
            "tree": rows, "text_record": text, "original_bytes_base64": base64.b64encode(raw).decode()}


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.snapshots = [fixture(repo) for repo in repair.TARGETS]

    def test_workflow_receipts_are_external_and_publication_is_explicit(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/reconcile-kernel-provenance.yml").read_text()
        self.assertIn("EVIDENCE_DIR: ${{ runner.temp }}/kernel-provenance-", workflow)
        self.assertIn('run: mkdir "$EVIDENCE_DIR"', workflow)
        self.assertIn('--receipt "$EVIDENCE_DIR/provenance-publication.jsonl"', workflow)
        self.assertIn('path: ${{ env.EVIDENCE_DIR }}/provenance-*', workflow)
        self.assertNotIn('--receipt reports/', workflow)
        self.assertIn("default: check", workflow)
        self.assertIn("github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'", workflow)
        self.assertNotRegex(workflow, r'(?m)^\s+(push|pull_request_target):')

    def test_eight_bound_proposals_preserve_original_bytes_and_metrics(self):
        plan = repair.make_plan(self.snapshots)
        self.assertEqual(repair.verify_plan(plan, self.snapshots)["targets"], 8)
        self.assertFalse(plan["publish_supported"])
        self.assertEqual(plan["provider_mutations"], 0)
        for row in plan["entries"]:
            self.assertEqual(row["status"], "PROPOSED_NOT_APPLIED")
            self.assertEqual(row["expected_parent"], "a" * 40)
            proposed = row["proposed_document"]
            self.assertFalse(proposed["model"]["trained_weights_present"])
            self.assertNotIn("file", proposed["model"]["surrogate"])
            self.assertEqual(proposed["claims"]["trained_model"], "NOT_CLAIMED")
            history = proposed["historical_evidence"]
            self.assertEqual(repair.parse(base64.b64decode(history["original_bytes_base64"])), history["document"])
            self.assertEqual(history["document"]["historical_metric"]["accuracy"], 0.91)

    def test_missing_duplicate_and_extra_targets_fail(self):
        for values in (self.snapshots[:-1], self.snapshots + [self.snapshots[0]], [self.snapshots[0]] * 8):
            with self.assertRaises(repair.InvalidEvidence):
                repair.make_plan(values)

    def test_unsafe_paths_and_mutable_heads_fail(self):
        for path in ("../x", "/absolute", "a//b", "a/./b", "C:/x", "a\\b", "a\x00b", "a./x"):
            with self.subTest(path=path), self.assertRaises(repair.InvalidEvidence):
                repair.relative_path(path)
        for revision in ("main", "A" * 40, "a" * 39):
            snapshot = copy.deepcopy(self.snapshots[0])
            snapshot["revision"] = revision
            with self.subTest(revision=revision), self.assertRaises(repair.InvalidEvidence):
                repair.validate_snapshot(snapshot)

    def test_original_hash_content_and_blob_oid_must_agree(self):
        for mutation in ("raw", "text", "oid"):
            snapshot = copy.deepcopy(self.snapshots[0])
            if mutation == "raw":
                snapshot["original_bytes_base64"] = base64.b64encode(b"{}").decode()
            elif mutation == "text":
                snapshot["text_record"]["content"]["historical_metric"]["accuracy"] = 1
            else:
                snapshot["tree"][0]["oid"] = "c" * 40
            with self.subTest(mutation=mutation), self.assertRaises(repair.InvalidEvidence):
                repair.validate_snapshot(snapshot)

    def test_tree_coverage_pagination_and_siblings_must_agree(self):
        for mutation in ("coverage", "pagination", "siblings", "identity"):
            snapshot = copy.deepcopy(self.snapshots[0])
            record = snapshot["repository_record"]
            if mutation == "coverage":
                record["tree_coverage"] = "PARTIAL"
            elif mutation == "pagination":
                record["tree_pages"][0]["next_url"] = "missing-page"
            elif mutation == "siblings":
                record["metadata"]["siblings"].append({"rfilename": "unobserved"})
            else:
                record["metadata"]["sha"] = "b" * 40
            with self.subTest(mutation=mutation), self.assertRaises(repair.InvalidEvidence):
                repair.validate_snapshot(snapshot)

    def test_existing_weight_artifact_requires_manual_review(self):
        for path in (repair.TARGETS[self.snapshots[0]["id"]], "new.safetensors"):
            snapshot = copy.deepcopy(self.snapshots[0])
            snapshot["tree"].append({**snapshot["tree"][0], "path": path})
            snapshot["repository_record"]["tree_entries"] = 2
            snapshot["repository_record"]["file_count"] = 2
            snapshot["repository_record"]["metadata"]["siblings"].append({"rfilename": path})
            with self.subTest(path=path), self.assertRaises(repair.InvalidEvidence):
                repair.validate_snapshot(snapshot)

    def test_proposal_cannot_invent_weights_or_approval_or_erase_history(self):
        plan = repair.make_plan(self.snapshots)
        for mutation in ("weights", "history", "approval", "parent", "destination"):
            changed = copy.deepcopy(plan)
            entry = changed["entries"][0]
            if mutation == "weights":
                entry["proposed_document"]["model"]["trained_weights_present"] = True
            elif mutation == "history":
                entry["proposed_document"]["historical_evidence"] = {}
            elif mutation == "approval":
                changed["status"] = "APPROVED"
            elif mutation == "parent":
                entry["expected_parent"] = "b" * 40
            else:
                entry["destination"] = "README.md"
            with self.subTest(mutation=mutation), self.assertRaises(repair.InvalidEvidence):
                repair.verify_plan(changed, self.snapshots)

    def test_duplicate_json_and_nonfinite_values_fail(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(repair.InvalidEvidence):
                repair.parse(raw)


if __name__ == "__main__":
    unittest.main()
