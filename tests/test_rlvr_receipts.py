"""Receipt RLVR reward: schema validation and DSSE, with keys minted in the test."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from training.rlvr.receipts import VERIFIER, dsse_pae, load_schemas, receipt_reward, score_receipt

VENDOR = Path(__file__).resolve().parents[1] / "training" / "rlvr" / "vendor" / "governed-receipt-spec"


def _keys():
    private = ec.generate_private_key(ec.SECP256R1())
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private, public


def _sign(payload, private, payload_type="application/vnd.szl.khipu+json"):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    signature = private.sign(dsse_pae(payload_type, body), ec.ECDSA(hashes.SHA256()))
    return {
        "payloadType": payload_type,
        "payload": base64.b64encode(body).decode(),
        "signatures": [{"keyid": "test-only", "sig": base64.b64encode(signature).decode()}],
        "signed": True,
    }


def _decision():
    return {
        "action": "inference",
        "ns": "rlvr-test",
        "seq": 0,
        "prev": "0" * 64,
        "digest": "ab" * 32,
        "payload_digest": "cd" * 32,
        "ts": 1700000000,
        "decision": "allow",
    }


def _action():
    return {
        "action": {"type": "rlvr.check", "description": "test action"},
        "principal": {
            "type": "human",
            "id": "test-principal",
            "is_service_account": False,
            "auth_method": "hardware_key",
        },
        "authority": {
            "outcome": "ALLOW",
            "evaluated_before_execution": True,
            "policy_ref": "policy:test",
        },
        "evidence": {
            "obligations": [{"id": "observed", "satisfied": True}],
            "completeness": "COMPLETE",
        },
        "timestamp": {"utc": "2026-11-27T00:00:00Z", "ntp_synced": True},
        "side_effect_class": "READ_ONLY",
    }


def test_vendored_schemas_match_the_pinned_commit():
    pin = json.loads((VENDOR / "PIN.json").read_text(encoding="utf-8"))
    assert pin["repository"] == "szl-holdings/governed-receipt-spec"
    assert pin["commit"] == "e9c30c0b4a15312dd888afd0cf614df30057e328"
    for relative, digest in pin["files"].items():
        payload = (VENDOR / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
    loaded = load_schemas()
    assert "seq" in loaded["receipt"]["required"]
    assert loaded["action"]["properties"]["principal"]["properties"]["is_service_account"]["const"] is False


def test_dsse_pae_matches_the_spec_vector():
    assert dsse_pae("ab", b"c") == b"DSSEv1 2 ab 1 c"


def test_signed_receipt_scores_one():
    private, public = _keys()
    envelope = _sign(_decision(), private)
    outcome = score_receipt(json.dumps(envelope), verify_key_pem=public)
    assert outcome.reward == 1.0
    assert outcome.receipt["verifier"] == VERIFIER
    assert outcome.receipt["verdict"] == "pass"
    assert outcome.receipt["schema"] == "governed-receipt"
    assert outcome.receipt["schema_valid"] is True
    assert len(outcome.receipt["input_sha256"]) == 64


def test_schema_valid_unsigned_receipt_scores_zero():
    body = json.dumps(_decision(), sort_keys=True, separators=(",", ":")).encode()
    envelope = {
        "payloadType": "application/vnd.szl.khipu+json",
        "payload": base64.b64encode(body).decode(),
        "signatures": [],
        "signed": False,
    }
    _private, public = _keys()
    outcome = score_receipt(envelope, verify_key_pem=public)
    assert outcome.reward == 0.0
    assert outcome.receipt["schema_valid"] is True
    assert "unsigned" in outcome.receipt["reason"]


def test_tampered_payload_and_wrong_key_score_zero():
    private, public = _keys()
    envelope = _sign(_decision(), private)
    payload = json.loads(base64.b64decode(envelope["payload"]))
    payload["ns"] = "changed"
    envelope["payload"] = base64.b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    tampered = score_receipt(envelope, verify_key_pem=public)
    assert tampered.reward == 0.0
    assert "mismatch" in tampered.receipt["reason"]

    _other, other_public = _keys()
    signed = _sign(_decision(), private)
    wrong_key = score_receipt(signed, verify_key_pem=other_public)
    assert wrong_key.reward == 0.0
    assert wrong_key.receipt["schema_valid"] is True


def test_schema_invalid_and_malformed_score_zero():
    private, public = _keys()
    broken = _decision()
    del broken["ns"]
    missing = score_receipt(_sign(broken, private), verify_key_pem=public)
    assert missing.reward == 0.0
    assert missing.receipt["schema_valid"] is False

    malformed = score_receipt("{", verify_key_pem=public)
    assert malformed.reward == 0.0
    assert "malformed" in malformed.receipt["reason"]

    not_object = score_receipt("[]", verify_key_pem=public)
    assert not_object.reward == 0.0


def test_signed_governed_action_scores_one_and_service_account_fails():
    private, public = _keys()
    ok = score_receipt(_sign(_action(), private), verify_key_pem=public)
    assert ok.reward == 1.0
    assert ok.receipt["schema"] == "governed-action"

    impersonated = _action()
    impersonated["principal"]["is_service_account"] = True
    rejected = score_receipt(_sign(impersonated, private), verify_key_pem=public)
    assert rejected.reward == 0.0
    assert rejected.receipt["schema_valid"] is False


def test_trl_batch_mixes_pass_and_fail():
    private, public = _keys()
    signed = json.dumps(_sign(_decision(), private))
    unsigned = json.dumps({
        "payloadType": "application/vnd.szl.khipu+json",
        "payload": _sign(_decision(), private)["payload"],
        "signatures": [],
        "signed": False,
    })
    batch = receipt_reward([signed, unsigned], verify_key_pem=public)
    assert list(batch) == [1.0, 0.0]
    assert batch.receipts[0]["verdict"] == "pass"
    assert "unsigned" in batch.receipts[1]["reason"]
