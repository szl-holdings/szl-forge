"""Append-only transparency log for A11oy action admissions (RFC 6962 / RFC 9162 sec 2.1).

Each admission that passes the DSSE action-receipt verifier is appended once to a local
Merkle log:

    leaf hash = SHA-256(0x00 || canonical leaf record)
    node hash = SHA-256(0x01 || left || right)

The leaf record is canonical JSON with schema LEAF_SCHEMA, admission_sha256,
receipt_sha256 and not_after. The log signs checkpoints as C2SP signed notes
(tlog-checkpoint note text: origin, tree size, base64 root; signature type 0x02,
ECDSA P-256 over the note text, key ID = first 4 bytes of SHA-256 of the SPKI DER).

``AdmissionGate`` wraps a receipt verifier. ``admit`` returns True only when all of
these hold:

1. The receipt verifier returns True.
2. The admission's canonical digest appears in exactly one leaf of the synced log and
   has not been admitted before by this gate.
3. The leaf record matches the admission digest, the receipt digest and the signed
   not_after, and not_after has not passed.
4. The presented inclusion proof verifies against the gate's trusted checkpoint.

The gate advances its trusted checkpoint only through ``sync``. Sync checks the
checkpoint signature and a consistency proof from the previous trusted checkpoint,
and requires that the new leaves recompute the signed root.

A consistency proof alone does not bind the second tree size (sizes with the same
proof shape alias). Only the checkpoint signature binds size and root, so the gate
only accepts sizes taken from a verified signed checkpoint.

Limits: exactly-once needs the full leaf sequence (monitor model). A Merkle log
cannot prove that a digest is absent. There are no external witnesses or gossip, so
split views are detected only between checkpoints that the same verifier sees. Any
exception is a refusal.
"""
from __future__ import annotations

import base64
import binascii
import datetime as _dt
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

LEAF_SCHEMA = "szl.a11oy.action-log-leaf/v1"
LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"
HASH_SIZE = 32
SIG_TYPE_ECDSA = 0x02
MAX_SIGNATURES = 16
MAX_TREE_SIZE = 2**53 - 1
EMPTY_ROOT = hashlib.sha256(b"").digest()
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_ORIGIN = re.compile(r"^[\x21-\x7e]{1,255}$")
_SIZE = re.compile(r"^(0|[1-9][0-9]{0,15})$")
_KEY_NAME = re.compile(r"^[\x21-\x2a\x2c-\x7e]{1,255}$")
_SIG_LINE = re.compile(r"^\u2014 (\S+) ([A-Za-z0-9+/=]+)$")
_LEAF_KEYS = frozenset({"schema", "admission_sha256", "receipt_sha256", "not_after"})


class LogRefusal(Exception):
    """Raised by log and checkpoint operations that refuse their input."""


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def canonical_bytes(value: Any) -> bytes:
    """Sorted-key compact UTF-8 JSON (identical to action_receipt_verifier.canonical_bytes)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _reject_constant(token: str) -> Any:
    raise ValueError("non-finite number " + token)


def _no_duplicate_keys(pairs: list) -> dict:
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key " + key)
        out[key] = value
    return out


def strict_json(body: bytes) -> Any:
    return json.loads(body.decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys)


def parse_utc(value: Any) -> _dt.datetime:
    if not isinstance(value, str) or not _UTC.fullmatch(value):
        raise LogRefusal("timestamp must be YYYY-MM-DDTHH:MM:SSZ")
    return _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)


def admission_digest(admission: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(dict(admission))).hexdigest()


def hex32(value: Any) -> bytes:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise LogRefusal("hash must be 64 lowercase hex digits")
    return bytes.fromhex(value)


# ---------------------------------------------------------------- hashing

def leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def _split(n: int) -> int:
    """Largest power of two strictly less than n (n >= 2)."""
    return 1 << ((n - 1).bit_length() - 1)


class MerkleTree:
    """Dense RFC 6962 tree: levels[k][i] is the root of the complete subtree of 2**k leaves at i."""

    def __init__(self) -> None:
        self._levels: list[list[bytes]] = [[]]

    @property
    def size(self) -> int:
        return len(self._levels[0])

    def copy(self) -> "MerkleTree":
        clone = MerkleTree()
        clone._levels = [list(level) for level in self._levels]
        return clone

    def append_leaf_hash(self, digest: bytes) -> int:
        if not isinstance(digest, bytes) or len(digest) != HASH_SIZE:
            raise LogRefusal("leaf hash must be 32 bytes")
        self._levels[0].append(digest)
        index, level = self.size - 1, 0
        while index & 1:
            parent = node_hash(self._levels[level][index - 1], self._levels[level][index])
            level += 1
            if len(self._levels) == level:
                self._levels.append([])
            self._levels[level].append(parent)
            index >>= 1
        return self.size - 1

    def _hash(self, start: int, size: int) -> bytes:
        if size == 0:
            return EMPTY_ROOT
        if size & (size - 1) == 0 and start % size == 0:
            level = size.bit_length() - 1
            return self._levels[level][start >> level]
        k = _split(size)
        return node_hash(self._hash(start, k), self._hash(start + k, size - k))

    def _check_size(self, size: int) -> None:
        if not _is_int(size) or not 0 <= size <= self.size:
            raise LogRefusal("tree size out of range")

    def root(self, size: int | None = None) -> bytes:
        size = self.size if size is None else size
        self._check_size(size)
        return self._hash(0, size)

    def inclusion_proof(self, index: int, size: int) -> list[bytes]:
        self._check_size(size)
        if not _is_int(index) or not 0 <= index < size:
            raise LogRefusal("leaf index out of range")
        return self._path(index, 0, size)

    def _path(self, m: int, start: int, n: int) -> list[bytes]:
        if n == 1:
            return []
        k = _split(n)
        if m < k:
            return self._path(m, start, k) + [self._hash(start + k, n - k)]
        return self._path(m - k, start + k, n - k) + [self._hash(start, k)]

    def consistency_proof(self, first: int, second: int) -> list[bytes]:
        self._check_size(second)
        if not _is_int(first) or not 1 <= first <= second:
            raise LogRefusal("consistency sizes out of range")
        if first == second:
            return []
        return self._subproof(first, 0, second, True)

    def _subproof(self, m: int, start: int, n: int, complete: bool) -> list[bytes]:
        if m == n:
            return [] if complete else [self._hash(start, n)]
        k = _split(n)
        if m <= k:
            return self._subproof(m, start, k, complete) + [self._hash(start + k, n - k)]
        return self._subproof(m - k, start + k, n - k, False) + [self._hash(start, k)]


# ---------------------------------------------------------------- verification (RFC 9162 2.1.3.2, 2.1.4.2)

def _proof_ok(proof: Any) -> bool:
    return isinstance(proof, (list, tuple)) and all(isinstance(p, bytes) and len(p) == HASH_SIZE for p in proof)


def verify_inclusion(leaf_digest: bytes, index: int, size: int, proof: Sequence[bytes], root: bytes) -> bool:
    try:
        if not (_is_int(index) and _is_int(size)) or not 0 <= index < size or size > MAX_TREE_SIZE:
            return False
        if not isinstance(leaf_digest, bytes) or len(leaf_digest) != HASH_SIZE:
            return False
        if not isinstance(root, bytes) or not _proof_ok(proof):
            return False
        fn, sn, r = index, size - 1, leaf_digest
        for p in proof:
            if sn == 0:
                return False
            if fn & 1 or fn == sn:
                r = node_hash(p, r)
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
            else:
                r = node_hash(r, p)
            fn >>= 1
            sn >>= 1
        return sn == 0 and r == root
    except Exception:
        return False


def verify_consistency(first: int, second: int, first_root: bytes, second_root: bytes, proof: Sequence[bytes]) -> bool:
    """Proof check only. It does NOT bind ``second``; bind sizes with a signed checkpoint."""
    try:
        if not (_is_int(first) and _is_int(second)) or first < 1 or second < first or second > MAX_TREE_SIZE:
            return False
        if not isinstance(first_root, bytes) or not isinstance(second_root, bytes) or not _proof_ok(proof):
            return False
        if first == second:
            return len(proof) == 0 and first_root == second_root
        path = list(proof)
        if not path:
            return False
        if first & (first - 1) == 0:
            path = [first_root] + path
        fn, sn = first - 1, second - 1
        while fn & 1:
            fn >>= 1
            sn >>= 1
        fr = sr = path[0]
        for c in path[1:]:
            if sn == 0:
                return False
            if fn & 1 or fn == sn:
                fr = node_hash(c, fr)
                sr = node_hash(c, sr)
                while not fn & 1 and fn != 0:
                    fn >>= 1
                    sn >>= 1
            else:
                sr = node_hash(sr, c)
            fn >>= 1
            sn >>= 1
        return fr == first_root and sr == second_root and sn == 0
    except Exception:
        return False


# ---------------------------------------------------------------- signed checkpoints (C2SP signed note)

class Checkpoint:
    __slots__ = ("origin", "tree_size", "root", "note")

    def __init__(self, origin: str, tree_size: int, root: bytes, note: str) -> None:
        self.origin, self.tree_size, self.root, self.note = origin, tree_size, root, note


def checkpoint_text(origin: str, tree_size: int, root: bytes) -> str:
    if not isinstance(origin, str) or not _ORIGIN.fullmatch(origin):
        raise LogRefusal("malformed origin")
    if not _is_int(tree_size) or not 0 <= tree_size <= MAX_TREE_SIZE or not isinstance(root, bytes) or len(root) != HASH_SIZE:
        raise LogRefusal("malformed checkpoint fields")
    return "%s\n%d\n%s\n" % (origin, tree_size, base64.b64encode(root).decode("ascii"))


def _spki_der(public_key: Any) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)


def _require_p256(key: Any) -> None:
    from cryptography.hazmat.primitives.asymmetric import ec

    if not isinstance(key, (ec.EllipticCurvePublicKey, ec.EllipticCurvePrivateKey)) or key.curve.name != "secp256r1":
        raise LogRefusal("checkpoint key must be ECDSA P-256")


def ecdsa_key_id(public_key: Any) -> bytes:
    """C2SP signed-note type 0x02: truncated SHA-256 of the SPKI DER public key."""
    return hashlib.sha256(_spki_der(public_key)).digest()[:4]


class CheckpointSigner:
    """Signs checkpoints with an in-memory ECDSA P-256 private key (never serialized here)."""

    def __init__(self, origin: str, key_name: str, private_key: Any) -> None:
        _require_p256(private_key)
        if not isinstance(key_name, str) or not _KEY_NAME.fullmatch(key_name):
            raise LogRefusal("malformed key name")
        checkpoint_text(origin, 0, EMPTY_ROOT)
        self.origin, self.key_name, self._key = origin, key_name, private_key
        self.key_id = ecdsa_key_id(private_key.public_key())

    def public_key(self) -> Any:
        return self._key.public_key()

    def sign(self, tree_size: int, root: bytes) -> str:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        text = checkpoint_text(self.origin, tree_size, root)
        sig = self._key.sign(text.encode("utf-8"), ec.ECDSA(hashes.SHA256()))
        line = "\u2014 %s %s\n" % (self.key_name, base64.b64encode(self.key_id + sig).decode("ascii"))
        return text + "\n" + line


class CheckpointVerifier:
    def __init__(self, origin: str, key_name: str, public_key: Any) -> None:
        _require_p256(public_key)
        checkpoint_text(origin, 0, EMPTY_ROOT)
        self.origin, self.key_name, self._key = origin, key_name, public_key
        self.key_id = ecdsa_key_id(public_key)

    def verify(self, note: Any) -> Checkpoint:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        if not isinstance(note, str) or not note.endswith("\n"):
            raise LogRefusal("checkpoint must be a newline-terminated string")
        sep = note.rfind("\n\n")
        if sep < 0:
            raise LogRefusal("checkpoint has no signature block")
        text, block = note[: sep + 1], note[sep + 2 :]
        lines = text.split("\n")
        if len(lines) != 4 or lines[3] != "":
            raise LogRefusal("checkpoint must have exactly origin, size and root lines")
        origin, size_line, root_line = lines[0], lines[1], lines[2]
        if origin != self.origin:
            raise LogRefusal("checkpoint origin mismatch")
        if not _SIZE.fullmatch(size_line):
            raise LogRefusal("malformed tree size")
        tree_size = int(size_line)
        if tree_size > MAX_TREE_SIZE:
            raise LogRefusal("tree size out of range")
        try:
            root = base64.b64decode(root_line, validate=True)
        except (binascii.Error, ValueError):
            raise LogRefusal("malformed root") from None
        if len(root) != HASH_SIZE or base64.b64encode(root).decode("ascii") != root_line:
            raise LogRefusal("root must be canonical base64 of 32 bytes")
        sig_lines = block.split("\n")
        if sig_lines[-1] != "" or len(sig_lines) - 1 > MAX_SIGNATURES or len(sig_lines) < 2:
            raise LogRefusal("malformed signature block")
        verified = False
        for line in sig_lines[:-1]:
            match = _SIG_LINE.fullmatch(line)
            if not match:
                raise LogRefusal("malformed signature line")
            try:
                raw = base64.b64decode(match.group(2), validate=True)
            except (binascii.Error, ValueError):
                raise LogRefusal("malformed signature encoding") from None
            if len(raw) < 5:
                raise LogRefusal("signature too short")
            if match.group(1) != self.key_name or raw[:4] != self.key_id:
                continue
            try:
                self._key.verify(raw[4:], text.encode("utf-8"), ec.ECDSA(hashes.SHA256()))
            except (InvalidSignature, ValueError):
                raise LogRefusal("checkpoint signature from the known key does not verify") from None
            verified = True
        if not verified:
            raise LogRefusal("no valid signature from the known key")
        return Checkpoint(origin, tree_size, root, note)


# ---------------------------------------------------------------- leaf records

def leaf_record(admission_sha256: str, receipt_sha256: str, not_after: str) -> bytes:
    hex32(admission_sha256)
    hex32(receipt_sha256)
    parse_utc(not_after)
    return canonical_bytes({"schema": LEAF_SCHEMA, "admission_sha256": admission_sha256,
                            "receipt_sha256": receipt_sha256, "not_after": not_after})


def parse_leaf_record(data: Any) -> dict:
    if not isinstance(data, bytes):
        raise LogRefusal("leaf record must be bytes")
    value = strict_json(data)
    if not isinstance(value, dict) or set(value) != _LEAF_KEYS or value["schema"] != LEAF_SCHEMA:
        raise LogRefusal("leaf record has the wrong shape")
    hex32(value["admission_sha256"])
    hex32(value["receipt_sha256"])
    parse_utc(value["not_after"])
    if canonical_bytes(value) != data:
        raise LogRefusal("leaf record is not canonical")
    return value


def receipt_payload(receipt: Mapping[str, Any]) -> dict:
    envelope = receipt.get("dsse")
    if not isinstance(envelope, Mapping) or not isinstance(envelope.get("payload"), str):
        raise LogRefusal("receipt has no DSSE payload")
    payload = strict_json(base64.b64decode(envelope["payload"], validate=True))
    if not isinstance(payload, dict):
        raise LogRefusal("receipt payload is not an object")
    return payload


# ---------------------------------------------------------------- the log

class ActionLog:
    """Append-only log. Appends only verified, unexpired, never-seen admissions."""

    def __init__(self, signer: CheckpointSigner, receipt_verifier: Callable[[Mapping, Mapping], bool],
                 *, now: Callable[[], _dt.datetime] = _utcnow) -> None:
        if not callable(receipt_verifier):
            raise LogRefusal("a receipt verifier is required")
        self._signer, self._verify, self._now = signer, receipt_verifier, now
        self._tree = MerkleTree()
        self._records: list[bytes] = []
        self._index: dict[str, int] = {}

    @property
    def size(self) -> int:
        return self._tree.size

    def append(self, admission: Mapping[str, Any], receipt: Mapping[str, Any]) -> int:
        try:
            ok = self._verify(admission, receipt)
        except Exception:
            ok = False
        if ok is not True:
            raise LogRefusal("receipt verifier refused the admission")
        payload = receipt_payload(receipt)
        digest = admission_digest(admission)
        if payload.get("admission_sha256") != digest:
            raise LogRefusal("receipt does not name this admission")
        not_after = payload.get("not_after")
        if self._now() > parse_utc(not_after):
            raise LogRefusal("admission expired (not_after passed)")
        if digest in self._index:
            raise LogRefusal("replay: admission already logged at leaf %d" % self._index[digest])
        record = leaf_record(digest, str(receipt.get("sha256")), not_after)
        index = self._tree.append_leaf_hash(leaf_hash(record))
        self._records.append(record)
        self._index[digest] = index
        return index

    def checkpoint(self, size: int | None = None) -> str:
        size = self.size if size is None else size
        if not _is_int(size) or not 1 <= size <= self.size:
            raise LogRefusal("cannot sign an empty or unknown tree size")
        return self._signer.sign(size, self._tree.root(size))

    def root(self, size: int | None = None) -> bytes:
        return self._tree.root(size)

    def inclusion_proof(self, index: int, size: int | None = None) -> list[bytes]:
        return self._tree.inclusion_proof(index, self.size if size is None else size)

    def consistency_proof(self, first: int, second: int | None = None) -> list[bytes]:
        return self._tree.consistency_proof(first, self.size if second is None else second)

    def records(self, start: int, end: int | None = None) -> list[bytes]:
        return list(self._records[start:end])

    def index_of(self, admission: Mapping[str, Any]) -> int:
        return self._index[admission_digest(admission)]


# ---------------------------------------------------------------- fork detection

def detect_fork(verifier: CheckpointVerifier, note_a: str, note_b: str, proof: Sequence[bytes]) -> str:
    """CONSISTENT, FORK_PROVEN (two signed roots for one size), UNPROVEN (refuse) or INVALID_CHECKPOINT."""
    try:
        a, b = verifier.verify(note_a), verifier.verify(note_b)
    except Exception:
        return "INVALID_CHECKPOINT"
    if a.tree_size > b.tree_size:
        a, b = b, a
    if a.tree_size == b.tree_size:
        if a.root != b.root:
            return "FORK_PROVEN"
        return "CONSISTENT" if len(proof) == 0 else "UNPROVEN"
    return "CONSISTENT" if verify_consistency(a.tree_size, b.tree_size, a.root, b.root, proof) else "UNPROVEN"


# ---------------------------------------------------------------- the gate

class AdmissionGate:
    """Monitor-style verifier: admits an action only if it is signed, logged exactly once and proven."""

    def __init__(self, receipt_verifier: Callable[[Mapping, Mapping], bool], checkpoint_verifier: CheckpointVerifier,
                 *, now: Callable[[], _dt.datetime] = _utcnow) -> None:
        if not callable(receipt_verifier) or not isinstance(checkpoint_verifier, CheckpointVerifier):
            raise LogRefusal("gate needs a receipt verifier and a checkpoint verifier")
        self._verify, self._cpv, self._now = receipt_verifier, checkpoint_verifier, now
        self.trusted: Checkpoint | None = None
        self._tree = MerkleTree()
        self._records: list[bytes] = []
        self._positions: dict[str, list[int]] = {}
        self._consumed: set[str] = set()
        self.last_reason = ""

    def duplicates(self) -> dict[str, list[int]]:
        return {d: list(p) for d, p in self._positions.items() if len(p) > 1}

    def sync(self, note: str, consistency: Sequence[bytes], new_records: Sequence[bytes]) -> bool:
        try:
            self._sync(note, consistency, new_records)
            self.last_reason = "synced"
            return True
        except Exception as exc:
            self.last_reason = "sync refused: %s" % exc
            return False

    def _sync(self, note: str, consistency: Sequence[bytes], new_records: Sequence[bytes]) -> None:
        cp = self._cpv.verify(note)
        if cp.tree_size < 1:
            raise LogRefusal("empty checkpoint")
        if self.trusted is None:
            if len(consistency) != 0:
                raise LogRefusal("first sync takes no consistency proof")
        else:
            if cp.tree_size < self.trusted.tree_size:
                raise LogRefusal("rollback: checkpoint is older than the trusted one")
            if not verify_consistency(self.trusted.tree_size, cp.tree_size, self.trusted.root, cp.root, consistency):
                raise LogRefusal("checkpoint is not consistent with the trusted checkpoint (fork or rollback)")
        if len(new_records) != cp.tree_size - self._tree.size:
            raise LogRefusal("wrong number of new leaf records")
        tree = self._tree.copy()
        parsed = []
        for record in new_records:
            parsed.append(parse_leaf_record(record)["admission_sha256"])
            tree.append_leaf_hash(leaf_hash(record))
        if tree.root() != cp.root:
            raise LogRefusal("leaf records do not reproduce the signed root")
        start = self._tree.size
        self._tree = tree
        self._records.extend(new_records)
        for offset, digest in enumerate(parsed):
            self._positions.setdefault(digest, []).append(start + offset)
        self.trusted = cp

    def admit(self, admission: Mapping[str, Any], receipt: Mapping[str, Any], leaf_index: int,
              inclusion_proof: Sequence[bytes]) -> bool:
        try:
            self._admit(admission, receipt, leaf_index, inclusion_proof)
            self.last_reason = "admitted"
            return True
        except Exception as exc:
            self.last_reason = "refused: %s" % exc
            return False

    def _admit(self, admission, receipt, leaf_index, inclusion_proof) -> None:
        cp = self.trusted
        if cp is None:
            raise LogRefusal("no trusted checkpoint")
        if self._verify(admission, receipt) is not True:
            raise LogRefusal("receipt verifier refused")
        digest = admission_digest(admission)
        payload = receipt_payload(receipt)
        if payload.get("admission_sha256") != digest:
            raise LogRefusal("receipt does not name this admission")
        not_after = payload.get("not_after")
        if self._now() > parse_utc(not_after):
            raise LogRefusal("admission expired (not_after passed)")
        positions = self._positions.get(digest, [])
        if len(positions) != 1:
            raise LogRefusal("admission is logged %d times, not exactly once" % len(positions))
        if positions[0] != leaf_index:
            raise LogRefusal("leaf index does not hold this admission")
        if digest in self._consumed:
            raise LogRefusal("replay: admission already admitted")
        record = self._records[leaf_index]
        fields = parse_leaf_record(record)
        if fields["receipt_sha256"] != receipt.get("sha256") or fields["not_after"] != not_after:
            raise LogRefusal("leaf record does not match the receipt")
        if not verify_inclusion(leaf_hash(record), leaf_index, cp.tree_size, inclusion_proof, cp.root):
            raise LogRefusal("inclusion proof does not verify against the trusted checkpoint")
        self._consumed.add(digest)


def make_logged_receipt_verifier(gate: AdmissionGate) -> Callable[[Mapping, Mapping], bool]:
    """``receipt_verifier=`` hook: the receipt must carry transparency.leaf_index and inclusion_proof (hex)."""

    def verify(admission: Mapping[str, Any], receipt: Mapping[str, Any]) -> bool:
        try:
            entry = receipt.get("transparency")
            if not isinstance(entry, Mapping) or not _is_int(entry.get("leaf_index")):
                return False
            proof_hex = entry.get("inclusion_proof")
            if not isinstance(proof_hex, list):
                return False
            proof = [hex32(h) for h in proof_hex]
            return gate.admit(admission, receipt, entry["leaf_index"], proof) is True
        except Exception:
            return False

    return verify
