"""The additive correction binds published Git text, preserving original claims."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
RECORD = ROOT / "frontier/evaluation/identity_bindings/evals/chaski-r2_661f8ee9ff6a_source_publication_binding_20260925.json"


def test_correction_explains_original_crlf_hashes_without_rewriting_records():
    correction = json.loads(RECORD.read_text(encoding="utf-8"))
    original = correction["original_evaluation_binding"]
    # Repository text may have Windows checkout line endings. The correction
    # explicitly identifies the canonical LF Git blobs, not working-tree bytes.
    raw = (ROOT / original["path"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == original["sha256"]
    old = json.loads(raw)
    for item in correction["source_publication_identities"]:
        raw = (ROOT / item["path"]).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        assert len(raw) == item["bytes"]
        assert old[item["original_sidecar_field"]] == item["original_declared_sha256"]
        assert hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest() == item["original_declared_sha256"]
        assert item["sha256"] != item["original_declared_sha256"]
    assert correction["production_disposition"] == "HOLD"
    assert correction["publication_eligible"] is False
    assert correction["autonomy_eligible"] is False
    assert correction["merged_model_evaluation_binding"] == "NOT_ESTABLISHED_BY_THIS_ADAPTER_RUN"
