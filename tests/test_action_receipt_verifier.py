from __future__ import annotations

import base64
import builtins
import datetime as dt
import hashlib

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

import test_production_inference_v2 as base
from inference.action_receipt_verifier import (
    ACTION_PAYLOAD_TYPE,
    ACTION_SCHEMA,
    canonical_bytes,
    keyid_for_public_pem,
    make_dsse_action_receipt_verifier,
    pae,
)
from inference.production import production_infer, verify_external_execution

NOW = dt.datetime(2026, 11, 3, 12, 0, 0, tzinfo=dt.timezone.utc)
ADMISSION = {
    "authority": "A11OY",
    "human_approval": "APPROVED",
    "signed_receipt_required": True,
}
TOOL = {"tool": "repo.write", "target": "example"}


def _key():
    private = ec.generate_private_key(ec.SECP256R1())
    public_pem = (
        private.public_key()
        .public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        .decode("ascii")
    )
    return private, public_pem


def _payload(admission=ADMISSION, **overrides):
    value = {
        "schema": ACTION_SCHEMA,
        "authority": "A11OY",
        "human_approval": "APPROVED",
        "admission_sha256": hashlib.sha256(canonical_bytes(admission)).hexdigest(),
        "not_after": "2026-11-03T13:00:00Z",
    }
    value.update(overrides)
    return value


def _receipt(private, public_pem, payload=None, *, body=None, payload_type=ACTION_PAYLOAD_TYPE, keyid=None, raw_sign=False):
    body = canonical_bytes(payload) if body is None else body
    message = body if raw_sign else pae(payload_type, body)
    sig = private.sign(message, ec.ECDSA(hashes.SHA256()))
    return {
        "signature_status": "SIGNED_VERIFIED",
        "sha256": hashlib.sha256(body).hexdigest(),
        "dsse": {
            "payloadType": payload_type,
            "payload": base64.b64encode(body).decode("ascii"),
            "signatures": [
                {
                    "sig": base64.b64encode(sig).decode("ascii"),
                    "keyid": keyid or keyid_for_public_pem(public_pem),
                }
            ],
        },
    }


@pytest.fixture
def signer():
    private, public_pem = _key()
    return private, public_pem, make_dsse_action_receipt_verifier(public_pem, now=lambda: NOW)


def test_valid_signed_admission_verifies(signer):
    private, public_pem, verify = signer
    assert verify(dict(ADMISSION), _receipt(private, public_pem, _payload())) is True


def test_signature_from_another_key_is_rejected(signer):
    _, public_pem, verify = signer
    other, other_pem = _key()
    spoofed = _receipt(other, other_pem, _payload(), keyid=keyid_for_public_pem(public_pem))
    assert verify(dict(ADMISSION), spoofed) is False
    assert verify(dict(ADMISSION), _receipt(other, other_pem, _payload())) is False


def test_payload_changed_after_signing_is_rejected(signer):
    private, public_pem, verify = signer
    receipt = _receipt(private, public_pem, _payload())
    forged = canonical_bytes(_payload(not_after="2099-01-01T00:00:00Z"))
    receipt["dsse"]["payload"] = base64.b64encode(forged).decode("ascii")
    receipt["sha256"] = hashlib.sha256(forged).hexdigest()
    assert verify(dict(ADMISSION), receipt) is False


def test_receipt_digest_must_match_payload(signer):
    private, public_pem, verify = signer
    receipt = _receipt(private, public_pem, _payload())
    receipt["sha256"] = "e" * 64
    assert verify(dict(ADMISSION), receipt) is False


def test_wrong_payload_type_is_rejected(signer):
    private, public_pem, verify = signer
    receipt = _receipt(private, public_pem, _payload(), payload_type="application/vnd.szl.khipu+json")
    assert verify(dict(ADMISSION), receipt) is False


def test_signature_must_cover_the_pae_not_the_raw_body(signer):
    private, public_pem, verify = signer
    assert verify(dict(ADMISSION), _receipt(private, public_pem, _payload(), raw_sign=True)) is False


def test_receipt_for_a_different_admission_is_rejected(signer):
    private, public_pem, verify = signer
    other = dict(ADMISSION, scope="other-target")
    assert verify(dict(ADMISSION), _receipt(private, public_pem, _payload(other))) is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "szl.a11oy.action-admission/v0"},
        {"authority": "FORGE"},
        {"human_approval": "PENDING"},
        {"not_after": "2026-11-03T11:59:59Z"},
        {"not_after": "2026-11-03T13:00:00+00:00"},
        {"not_after": None},
        {"not_before": "2026-11-03T12:00:01Z"},
    ],
    ids=["schema", "authority", "pending", "expired", "non-z-time", "no-expiry", "not-yet-valid"],
)
def test_payload_claims_are_enforced(signer, overrides):
    private, public_pem, verify = signer
    assert verify(dict(ADMISSION), _receipt(private, public_pem, _payload(**overrides))) is False


def test_strict_json_rejects_duplicate_keys_and_nan(signer):
    private, public_pem, verify = signer
    good = canonical_bytes(_payload())
    duplicate = good[:-1] + b',"human_approval":"APPROVED"}'
    nan = good[:-1] + b',"weight":NaN}'
    assert verify(dict(ADMISSION), _receipt(private, public_pem, body=duplicate)) is False
    assert verify(dict(ADMISSION), _receipt(private, public_pem, body=nan)) is False


def test_status_labels_alone_are_rejected(signer):
    _, _, verify = signer
    assert verify(dict(ADMISSION), {"signature_status": "SIGNED_VERIFIED", "sha256": "e" * 64}) is False


def test_naive_clock_is_rejected():
    private, public_pem = _key()
    verify = make_dsse_action_receipt_verifier(public_pem, now=lambda: NOW.replace(tzinfo=None))
    assert verify(dict(ADMISSION), _receipt(private, public_pem, _payload())) is False


def test_key_pin_and_key_type_are_checked_at_construction():
    _, public_pem = _key()
    keyid = keyid_for_public_pem(public_pem)
    assert make_dsse_action_receipt_verifier(public_pem, expected_keyid=keyid).keyid == keyid
    with pytest.raises(ValueError):
        make_dsse_action_receipt_verifier(public_pem, expected_keyid="0" * 64)
    ed_pem = (
        ed25519.Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )
    with pytest.raises(ValueError):
        make_dsse_action_receipt_verifier(ed_pem)


def test_missing_cryptography_fails_closed(signer, monkeypatch):
    private, public_pem, verify = signer
    receipt = _receipt(private, public_pem, _payload())
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert verify(dict(ADMISSION), receipt) is False


def _infer(admission, receipt, verify):
    return production_infer(
        base.request(tool_intent=dict(TOOL), action_admission=admission, action_receipt=receipt),
        retriever=base.retriever,
        hydrator=base.hydrator,
        generator=base.Generator(),
        witness=base.witness,
        contract_path=base.CONTRACT,
        receipt_verifier=verify,
    )


def test_production_path_needs_a_real_signature(signer):
    private, public_pem, verify = signer
    admission = dict(ADMISSION)
    receipt = _receipt(private, public_pem, _payload(admission))
    result = _infer(admission, receipt, verify)
    assert result["state"] == "ACTION_READY_EXTERNAL_EXECUTION"
    assert result["authority_state"] == "A11OY_SIGNED_ADMISSION_VERIFIED"
    verified = verify_external_execution(
        result,
        tool_result_sha256="f" * 64,
        postcondition_status="PASS",
        postcondition_details_sha256="1" * 64,
        action_admission=admission,
        action_receipt=receipt,
        witness=base.witness,
        receipt_verifier=verify,
    )
    assert verified["state"] == "VERIFIED"

    unsigned = dict(receipt, dsse=dict(receipt["dsse"], signatures=[]))
    blocked = _infer(admission, unsigned, verify)
    assert blocked["state"] == "BLOCKED"
    assert blocked["reason_codes"][0] == "A11OY_ACTION_CONTRACT_INVALID"


def test_envelope_type_label_must_match_even_if_signature_covers_action_type(signer):
    private, public_pem, verify = signer
    receipt = _receipt(private, public_pem, _payload())
    receipt["dsse"]["payloadType"] = "application/vnd.szl.khipu+json"
    assert verify(dict(ADMISSION), receipt) is False


def test_legacy_or_wrong_keyid_label_is_rejected_even_with_a_valid_signature(signer):
    private, public_pem, verify = signer
    receipt = _receipt(private, public_pem, _payload(), keyid="szlholdings-cosign")
    assert verify(dict(ADMISSION), receipt) is False


def test_non_p256_ecdsa_key_is_refused_at_construction():
    p384 = (
        ec.generate_private_key(ec.SECP384R1())
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode("ascii")
    )
    with pytest.raises(ValueError):
        make_dsse_action_receipt_verifier(p384)


def test_timestamps_must_be_zero_padded_utc(signer):
    private, public_pem, verify = signer
    assert verify(dict(ADMISSION), _receipt(private, public_pem, _payload(not_after="2026-11-3T13:00:00Z"))) is False
