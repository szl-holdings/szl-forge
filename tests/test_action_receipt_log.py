"""Tests for action_receipt_log (synthetic data only; throwaway in-process keys)."""
from __future__ import annotations

import base64
import datetime as dt
import hashlib

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

try:  # canonical home: szl-forge inference/
    from inference import action_receipt_log as arl
    from inference.action_receipt_verifier import (
        ACTION_PAYLOAD_TYPE, ACTION_SCHEMA, canonical_bytes, keyid_for_public_pem,
        make_dsse_action_receipt_verifier, pae)
except ImportError:  # flat prototype layout
    import action_receipt_log as arl
    from action_receipt_verifier import (
        ACTION_PAYLOAD_TYPE, ACTION_SCHEMA, canonical_bytes, keyid_for_public_pem,
        make_dsse_action_receipt_verifier, pae)

NOW = dt.datetime(2026, 11, 3, 12, 0, 0, tzinfo=dt.timezone.utc)
ORIGIN = "example.invalid/szl-c3-test-log"
KEY_NAME = "example.invalid/szl-c3-test-log"

# RFC 6962 roots for the transparency-dev/merkle testonly LeafInputs (Apache-2.0), sizes 1..8.
LEAF_INPUTS = ["", "00", "10", "2021", "3031", "40414243", "5051525354555657", "606162636465666768696a6b6c6d6e6f"]
ROOTS = [
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
    "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
    "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
    "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
    "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
    "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
    "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
]


# ------------------------------------------------------------ independent recursive reference (RFC 9162 2.1)
def ref_mth(leaves):
    if len(leaves) == 0:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return hashlib.sha256(b"\x00" + leaves[0]).digest()
    k = 1
    while k * 2 < len(leaves):
        k *= 2
    return hashlib.sha256(b"\x01" + ref_mth(leaves[:k]) + ref_mth(leaves[k:])).digest()


def ref_path(m, leaves):
    if len(leaves) <= 1:
        return []
    k = 1
    while k * 2 < len(leaves):
        k *= 2
    if m < k:
        return ref_path(m, leaves[:k]) + [ref_mth(leaves[k:])]
    return ref_path(m - k, leaves[k:]) + [ref_mth(leaves[:k])]


def ref_subproof(m, leaves, b):
    n = len(leaves)
    if m == n:
        return [] if b else [ref_mth(leaves)]
    k = 1
    while k * 2 < n:
        k *= 2
    if m <= k:
        return ref_subproof(m, leaves[:k], b) + [ref_mth(leaves[k:])]
    return ref_subproof(m - k, leaves[k:], False) + [ref_mth(leaves[:k])]


def _tree(leaves):
    t = arl.MerkleTree()
    for d in leaves:
        t.append_leaf_hash(arl.leaf_hash(d))
    return t


def _leaves(n):
    return [bytes.fromhex(LEAF_INPUTS[i]) if i < 8 else b"leaf-%d" % i for i in range(n)]


# ------------------------------------------------------------ hashing and proofs
def test_leaf_and_node_hashes_use_domain_prefixes():
    assert arl.leaf_hash(b"").hex() == ROOTS[0]
    assert arl.leaf_hash(b"") != hashlib.sha256(b"").digest()
    a, b = arl.leaf_hash(b""), arl.leaf_hash(b"\x00")
    assert arl.node_hash(a, b).hex() == ROOTS[1]


def test_roots_match_rfc6962_reference_vectors():
    leaves = [bytes.fromhex(h) for h in LEAF_INPUTS]
    t = _tree(leaves)
    assert t.root(0) == hashlib.sha256(b"").digest()
    for n in range(1, 9):
        assert t.root(n).hex() == ROOTS[n - 1]


def test_tree_matches_recursive_reference_for_small_sizes():
    leaves = _leaves(33)
    t = _tree(leaves)
    for n in range(1, 34):
        assert t.root(n) == ref_mth(leaves[:n])
        for m in range(n):
            proof = t.inclusion_proof(m, n)
            assert proof == ref_path(m, leaves[:n])
            assert arl.verify_inclusion(arl.leaf_hash(leaves[m]), m, n, proof, t.root(n))
        for m in range(1, n + 1):
            proof = t.consistency_proof(m, n)
            assert proof == (ref_subproof(m, leaves[:n], True) if m < n else [])
            assert arl.verify_consistency(m, n, t.root(m), t.root(n), proof)


def test_inclusion_refusals():
    leaves = _leaves(13)
    t = _tree(leaves)
    n, m = 13, 5
    root, proof, lh = t.root(n), t.inclusion_proof(m, n), arl.leaf_hash(leaves[m])
    assert arl.verify_inclusion(lh, m, n, proof, root)
    assert not arl.verify_inclusion(lh, m + 1, n, proof, root)
    assert not arl.verify_inclusion(lh, n, n, proof, root)
    assert not arl.verify_inclusion(lh, -1, n, proof, root)
    assert not arl.verify_inclusion(lh, True, n, proof, root)
    assert not arl.verify_inclusion(lh, m, n, proof[:-1], root)
    assert not arl.verify_inclusion(lh, m, n, proof + [root], root)
    assert not arl.verify_inclusion(lh, m, n, [proof[1], proof[0]] + proof[2:], root)
    assert not arl.verify_inclusion(lh, m, n, proof, bytes(32))
    assert not arl.verify_inclusion(hashlib.sha256(leaves[m]).digest(), m, n, proof, root)
    assert not arl.verify_inclusion(lh, m, n, [p[:31] for p in proof], root)
    assert not arl.verify_inclusion(lh, m, 0, [], root)


def test_inclusion_refuses_leaf_hash_that_is_not_32_bytes():
    assert not arl.verify_inclusion(b"", 0, 1, [], b"")
    assert not arl.verify_inclusion(b"x" * 31, 0, 1, [], b"x" * 31)
    assert arl.verify_inclusion(b"x" * 32, 0, 1, [], b"x" * 32)


def test_inclusion_with_too_many_nodes_for_size_one_is_refused():
    t = _tree(_leaves(1))
    assert arl.verify_inclusion(arl.leaf_hash(_leaves(1)[0]), 0, 1, [], t.root(1))
    assert not arl.verify_inclusion(arl.leaf_hash(_leaves(1)[0]), 0, 1, [bytes(32)], t.root(1))


def test_consistency_refusals():
    leaves = _leaves(21)
    t = _tree(leaves)
    m, n = 6, 21
    proof = t.consistency_proof(m, n)
    r1, r2 = t.root(m), t.root(n)
    assert arl.verify_consistency(m, n, r1, r2, proof)
    assert not arl.verify_consistency(0, n, arl.EMPTY_ROOT, r2, [])
    assert not arl.verify_consistency(n, m, r2, r1, proof)
    assert not arl.verify_consistency(m, n, r1, r2, proof[:-1])
    assert not arl.verify_consistency(m, n, r1, r2, proof + [r1])
    assert not arl.verify_consistency(m, n, r1, r2, [])
    assert not arl.verify_consistency(m, n, r2, r2, proof)
    assert not arl.verify_consistency(m, n, r1, r1, proof)
    assert not arl.verify_consistency(n, n, r2, r2, [r1])
    assert not arl.verify_consistency(n, n, r2, r1, [])
    assert arl.verify_consistency(n, n, r2, r2, [])
    assert not arl.verify_consistency(0, n, arl.EMPTY_ROOT, r2, proof)
    p4 = t.consistency_proof(4, 8)
    assert arl.verify_consistency(4, 8, t.root(4), t.root(8), p4)
    assert not arl.verify_consistency(4, 8, t.root(4), t.root(8), p4[:-1])
    assert not arl.verify_consistency(4, 8, bytes(32), t.root(8), p4)


def test_consistency_proof_alone_does_not_bind_second_size():
    """Documents the known pitfall: some other size verifies with the same proof shape."""
    leaves = _leaves(64)
    t = _tree(leaves)
    aliased = 0
    for m in range(1, 16):
        for n in range(m + 1, 33):
            proof = t.consistency_proof(m, n)
            for other in range(m + 1, 64):
                if other != n and arl.verify_consistency(m, other, t.root(m), t.root(n), proof):
                    aliased += 1
    assert aliased > 0


# ------------------------------------------------------------ checkpoints
@pytest.fixture
def log_keys():
    private = ec.generate_private_key(ec.SECP256R1())
    signer = arl.CheckpointSigner(ORIGIN, KEY_NAME, private)
    verifier = arl.CheckpointVerifier(ORIGIN, KEY_NAME, private.public_key())
    return signer, verifier


def _resign_body(note, new_text):
    sep = note.rfind("\n\n")
    return new_text + note[sep + 1 :]


def test_checkpoint_roundtrip_and_tampering(log_keys):
    signer, verifier = log_keys
    root = hashlib.sha256(b"r").digest()
    note = signer.sign(7, root)
    cp = verifier.verify(note)
    assert (cp.origin, cp.tree_size, cp.root) == (ORIGIN, 7, root)
    assert note.splitlines()[4].startswith("\u2014 " + KEY_NAME + " ")
    raw = base64.b64decode(note.splitlines()[4].split(" ")[2])
    der = signer.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    assert raw[:4] == hashlib.sha256(der).digest()[:4]
    flipped = bytearray(root)
    flipped[0] ^= 1
    with pytest.raises(arl.LogRefusal):
        verifier.verify(_resign_body(note, arl.checkpoint_text(ORIGIN, 7, bytes(flipped))))
    with pytest.raises(arl.LogRefusal):
        verifier.verify(_resign_body(note, arl.checkpoint_text(ORIGIN, 8, root)))
    with pytest.raises(arl.LogRefusal):
        verifier.verify(note.replace(ORIGIN + "\n", "example.invalid/other\n", 1))
    with pytest.raises(arl.LogRefusal):
        verifier.verify(note.replace("\n\n", "\nextension\n\n", 1))
    with pytest.raises(arl.LogRefusal):
        verifier.verify(note.replace("\n7\n", "\n07\n", 1))
    with pytest.raises(arl.LogRefusal):
        verifier.verify(note[:-1])
    with pytest.raises(arl.LogRefusal):
        verifier.verify(7)


def test_signed_checkpoint_with_extension_line_is_refused(log_keys):
    signer, verifier = log_keys
    text = arl.checkpoint_text(ORIGIN, 4, bytes(32)) + "extension\n"
    sig = signer._key.sign(text.encode("utf-8"), ec.ECDSA(hashes.SHA256()))
    note = text + "\n" + "\u2014 %s %s\n" % (KEY_NAME, base64.b64encode(signer.key_id + sig).decode())
    with pytest.raises(arl.LogRefusal):
        verifier.verify(note)


def test_checkpoint_from_other_key_is_refused(log_keys):
    signer, verifier = log_keys
    other = arl.CheckpointSigner(ORIGIN, KEY_NAME, ec.generate_private_key(ec.SECP256R1()))
    with pytest.raises(arl.LogRefusal):
        verifier.verify(other.sign(3, bytes(32)))
    note = signer.sign(3, bytes(32))
    sep = note.rfind("\n\n")
    text, good_line = note[: sep + 1], note[sep + 2 :]
    bad_sig = base64.b64encode(signer.key_id + b"\x30" * 70).decode()
    with pytest.raises(arl.LogRefusal):
        verifier.verify(text + "\n" + "\u2014 %s %s\n" % (KEY_NAME, bad_sig) + good_line)
    unknown = "\u2014 example.invalid/unknown %s\n" % base64.b64encode(b"abcd" + b"x" * 64).decode()
    assert verifier.verify(text + "\n" + unknown + good_line).tree_size == 3


def test_checkpoint_from_same_key_but_other_origin_is_refused(log_keys):
    signer, verifier = log_keys
    other_log = arl.CheckpointSigner("example.invalid/other-log", KEY_NAME, signer._key)
    with pytest.raises(arl.LogRefusal, match="origin"):
        verifier.verify(other_log.sign(3, bytes(32)))


def test_signer_refuses_non_p256_key():
    with pytest.raises(arl.LogRefusal):
        arl.CheckpointSigner(ORIGIN, KEY_NAME, ec.generate_private_key(ec.SECP384R1()))


# ------------------------------------------------------------ log, gate and integration
def _admission(i):
    return {"authority": "A11OY", "human_approval": "APPROVED", "signed_receipt_required": True,
            "synthetic": True, "action_id": "syn-%05d" % i, "tool": "synthetic.noop"}


class Env:
    def __init__(self, gate_verifier=None):
        self.receipt_key = ec.generate_private_key(ec.SECP256R1())
        self.pem = self.receipt_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")
        self.dsse = make_dsse_action_receipt_verifier(self.pem, now=lambda: NOW)
        log_key = ec.generate_private_key(ec.SECP256R1())
        self.signer = arl.CheckpointSigner(ORIGIN, KEY_NAME, log_key)
        self.cpv = arl.CheckpointVerifier(ORIGIN, KEY_NAME, log_key.public_key())
        self.log = arl.ActionLog(self.signer, self.dsse, now=lambda: NOW)
        self.gate = arl.AdmissionGate(gate_verifier or self.dsse, self.cpv, now=lambda: NOW)

    def receipt(self, admission, not_after="2026-11-03T13:00:00Z"):
        payload = {"schema": ACTION_SCHEMA, "authority": "A11OY", "human_approval": "APPROVED",
                   "admission_sha256": hashlib.sha256(canonical_bytes(admission)).hexdigest(),
                   "not_after": not_after}
        body = canonical_bytes(payload)
        sig = self.receipt_key.sign(pae(ACTION_PAYLOAD_TYPE, body), ec.ECDSA(hashes.SHA256()))
        return {"sha256": hashlib.sha256(body).hexdigest(),
                "dsse": {"payloadType": ACTION_PAYLOAD_TYPE, "payload": base64.b64encode(body).decode("ascii"),
                         "signatures": [{"sig": base64.b64encode(sig).decode("ascii"),
                                         "keyid": keyid_for_public_pem(self.pem)}]}}

    def sync(self):
        start = self.gate._tree.size
        proof = [] if self.gate.trusted is None else self.log.consistency_proof(self.gate.trusted.tree_size)
        return self.gate.sync(self.log.checkpoint(), proof, self.log.records(start))


def _filled(n=10, **kw):
    env = Env(**kw)
    items = []
    for i in range(n):
        a = _admission(i)
        r = env.receipt(a)
        items.append((a, r, env.log.append(a, r)))
    return env, items


def test_canonical_bytes_match_verifier():
    a = _admission(3)
    assert arl.canonical_bytes(a) == canonical_bytes(a)


def test_log_appends_and_refuses_replay():
    env, items = _filled(5)
    a, r, i = items[2]
    assert env.log.index_of(a) == i == 2
    with pytest.raises(arl.LogRefusal, match="replay"):
        env.log.append(a, r)
    with pytest.raises(arl.LogRefusal, match="replay"):
        env.log.append(a, env.receipt(a))
    assert env.log.size == 5


def test_log_refuses_expired_unsigned_and_mismatched():
    env = Env()
    a = _admission(1)
    with pytest.raises(arl.LogRefusal):
        env.log.append(a, env.receipt(a, not_after="2026-11-03T11:59:59Z"))
    permissive = arl.ActionLog(env.signer, lambda *_: True, now=lambda: NOW)
    with pytest.raises(arl.LogRefusal, match="expired"):
        permissive.append(a, env.receipt(a, not_after="2026-11-03T11:59:59Z"))
    with pytest.raises(arl.LogRefusal, match="does not name"):
        permissive.append(_admission(2), env.receipt(a))
    with pytest.raises(arl.LogRefusal):
        env.log.append(a, {"sha256": "0" * 64, "dsse": {}})

    def boom(*_):
        raise RuntimeError("verifier crashed")

    with pytest.raises(arl.LogRefusal):
        arl.ActionLog(env.signer, boom, now=lambda: NOW).append(a, env.receipt(a))
    with pytest.raises(arl.LogRefusal):
        arl.ActionLog(env.signer, lambda *_: 1, now=lambda: NOW).append(a, env.receipt(a))
    with pytest.raises(arl.LogRefusal):
        arl.ActionLog(env.signer, None)
    assert env.log.size == 0 and permissive.size == 0


def test_log_refuses_empty_checkpoint():
    env = Env()
    with pytest.raises(arl.LogRefusal):
        env.log.checkpoint()


def test_gate_admits_once_and_refuses_replay():
    env, items = _filled(9)
    assert env.sync()
    for a, r, i in items:
        assert env.gate.admit(a, r, i, env.log.inclusion_proof(i)), env.gate.last_reason
    a, r, i = items[4]
    assert not env.gate.admit(a, r, i, env.log.inclusion_proof(i))
    assert "replay" in env.gate.last_reason


def test_gate_refuses_before_sync_and_bad_proofs():
    env, items = _filled(9)
    a, r, i = items[3]
    assert not env.gate.admit(a, r, i, env.log.inclusion_proof(i))
    assert env.sync()
    proof = env.log.inclusion_proof(i)
    assert not env.gate.admit(a, r, i, proof[:-1])
    assert not env.gate.admit(a, r, i + 1, env.log.inclusion_proof(i + 1))
    assert not env.gate.admit(a, r, i, [proof[1], proof[0]] + proof[2:])
    assert not env.gate.admit(_admission(99), env.receipt(_admission(99)), i, proof)
    assert env.gate.admit(a, r, i, proof)


def test_gate_refuses_signature_failures_even_if_logged():
    env, items = _filled(4)
    assert env.sync()
    a, r, i = items[1]
    forged = dict(r)
    forged["dsse"] = dict(r["dsse"], signatures=[{"sig": base64.b64encode(b"x" * 64).decode(), "keyid": "0" * 64}])
    assert not env.gate.admit(a, forged, i, env.log.inclusion_proof(i))
    assert env.gate.admit(a, r, i, env.log.inclusion_proof(i))


def test_gate_refuses_admission_logged_twice():
    env, items = _filled(6)
    a, r, i = items[2]
    record = env.log._records[i]
    env.log._tree.append_leaf_hash(arl.leaf_hash(record))
    env.log._records.append(record)
    assert env.sync()
    assert env.gate.duplicates() == {arl.admission_digest(a): [2, 6]}
    assert not env.gate.admit(a, r, 2, env.log.inclusion_proof(2))
    assert "exactly once" in env.gate.last_reason
    assert not env.gate.admit(a, r, 6, env.log.inclusion_proof(6))
    b, rb, ib = items[3]
    assert env.gate.admit(b, rb, ib, env.log.inclusion_proof(ib))


def test_gate_refuses_expired_even_with_permissive_receipt_verifier():
    env = Env(gate_verifier=lambda *_: True)
    permissive = arl.ActionLog(env.signer, lambda *_: True, now=lambda: dt.datetime(2026, 11, 3, 10, 0, tzinfo=dt.timezone.utc))
    env.log = permissive
    a = _admission(1)
    r = env.receipt(a, not_after="2026-11-03T11:00:00Z")
    i = permissive.append(a, r)
    assert env.sync()
    assert not env.gate.admit(a, r, i, permissive.inclusion_proof(i))
    assert "expired" in env.gate.last_reason


def test_gate_requires_strict_true_and_fails_closed_on_exception():
    env, items = _filled(3, gate_verifier=lambda *_: "yes")
    assert env.sync()
    a, r, i = items[0]
    assert not env.gate.admit(a, r, i, env.log.inclusion_proof(i))

    def boom(*_):
        raise RuntimeError("crash")

    env2, items2 = _filled(3, gate_verifier=boom)
    assert env2.sync()
    a, r, i = items2[0]
    assert env2.gate.admit(a, r, i, env2.log.inclusion_proof(i)) is False


def test_gate_refuses_receipt_with_mismatched_leaf_record():
    env, items = _filled(3, gate_verifier=lambda *_: True)
    assert env.sync()
    a, r, i = items[1]
    other = env.receipt(a, not_after="2026-11-03T12:30:00Z")
    assert not env.gate.admit(a, other, i, env.log.inclusion_proof(i))
    assert "does not match" in env.gate.last_reason


def test_gate_sync_advances_with_consistency_and_refuses_fork_rollback_tamper():
    env, items = _filled(5)
    assert env.sync()
    old_note = env.log.checkpoint()
    for k in range(5, 12):
        a = _admission(k)
        env.log.append(a, env.receipt(a))
    assert env.sync() and env.gate.trusted.tree_size == 12
    assert not env.gate.sync(old_note, [], [])
    assert "rollback" in env.gate.last_reason
    fork = arl.ActionLog(env.signer, env.dsse, now=lambda: NOW)
    for k in range(12):
        a = _admission(k if k != 7 else 700)
        fork.append(a, env.receipt(a))
    a = _admission(12)
    fork.append(a, env.receipt(a))
    assert not env.gate.sync(fork.checkpoint(), fork.consistency_proof(12), fork.records(12))
    honest_next = _admission(12)
    env.log.append(honest_next, env.receipt(honest_next))
    tampered = [arl.leaf_record("1" * 64, "2" * 64, "2026-11-03T13:00:00Z")]
    assert not env.gate.sync(env.log.checkpoint(), env.log.consistency_proof(12), tampered)
    assert not env.gate.sync(env.log.checkpoint(), env.log.consistency_proof(12), [])
    assert env.gate.sync(env.log.checkpoint(), env.log.consistency_proof(12), env.log.records(12))


def test_first_sync_takes_no_consistency_proof_and_needs_all_leaves():
    env, items = _filled(4)
    assert not env.gate.sync(env.log.checkpoint(), [bytes(32)], env.log.records(0))
    assert not env.gate.sync(env.log.checkpoint(), [], env.log.records(1))
    assert env.gate.sync(env.log.checkpoint(), [], env.log.records(0))


def test_gate_sync_refuses_changed_size_with_same_proof():
    env, items = _filled(5)
    assert env.sync()
    for k in range(5, 7):
        a = _admission(k)
        env.log.append(a, env.receipt(a))
    note = env.log.checkpoint()
    proof = env.log.consistency_proof(5)
    for other in (6, 8):
        changed = _resign_body(note, arl.checkpoint_text(ORIGIN, other, env.log.root()))
        assert not env.gate.sync(changed, proof, env.log.records(5))


def test_detect_fork():
    env, items = _filled(8)
    a_note = env.log.checkpoint(5)
    b_note = env.log.checkpoint(8)
    assert arl.detect_fork(env.cpv, a_note, b_note, env.log.consistency_proof(5, 8)) == "CONSISTENT"
    assert arl.detect_fork(env.cpv, b_note, a_note, env.log.consistency_proof(5, 8)) == "CONSISTENT"
    fork = arl.ActionLog(env.signer, env.dsse, now=lambda: NOW)
    for k in range(8):
        adm = _admission(k if k != 3 else 300)
        fork.append(adm, env.receipt(adm))
    assert arl.detect_fork(env.cpv, b_note, fork.checkpoint(8), []) == "FORK_PROVEN"
    assert arl.detect_fork(env.cpv, a_note, fork.checkpoint(8), fork.consistency_proof(5, 8)) == "UNPROVEN"
    assert arl.detect_fork(env.cpv, a_note, b_note, env.log.consistency_proof(5, 8)[:-1]) == "UNPROVEN"
    assert arl.detect_fork(env.cpv, a_note, a_note, [bytes(32)]) == "UNPROVEN"
    assert arl.detect_fork(env.cpv, a_note, a_note[:-1], []) == "INVALID_CHECKPOINT"


def test_logged_receipt_verifier_hook():
    env, items = _filled(6)
    assert env.sync()
    hook = arl.make_logged_receipt_verifier(env.gate)
    a, r, i = items[3]
    wrapped = dict(r, transparency={"leaf_index": i, "inclusion_proof": [p.hex() for p in env.log.inclusion_proof(i)]})
    assert env.dsse(a, wrapped) is True
    assert hook(a, r) is False
    assert hook(a, dict(r, transparency={"leaf_index": True, "inclusion_proof": []})) is False
    assert hook(a, dict(r, transparency={"leaf_index": i, "inclusion_proof": ["zz"]})) is False
    assert hook(a, wrapped) is True
    assert hook(a, wrapped) is False


def test_leaf_record_parser_is_strict():
    rec = arl.leaf_record("a" * 64, "b" * 64, "2026-11-03T13:00:00Z")
    assert arl.parse_leaf_record(rec)["admission_sha256"] == "a" * 64
    with pytest.raises(Exception):
        arl.parse_leaf_record(rec.replace(b'"schema"', b' "schema"'))
    with pytest.raises(Exception):
        arl.parse_leaf_record(rec[:-1] + b',"x":1}')
    with pytest.raises(Exception):
        arl.leaf_record("A" * 64, "b" * 64, "2026-11-03T13:00:00Z")
    with pytest.raises(Exception):
        arl.leaf_record("a" * 64, "b" * 64, "2026-11-03 13:00:00")
