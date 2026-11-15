#!/usr/bin/env python3
# RBCG v1 prototype -- SZL Holdings. Licence: UNDECIDED (owner choice). See NOTICE.
"""rbcg1_log.py -- RFC 9162-style Merkle transparency log for RBCG receipts (v1).

IN-SILICO RESEARCH ONLY. SYNTHETIC DATA ONLY. NOT FOR CLINICAL USE. Standard library only.
Implemented from the text of RFC 9162 section 2.1 (MTH 2.1.1, PATH 2.1.3.1, inclusion
verification 2.1.3.2, PROOF/SUBPROOF 2.1.4.1, consistency verification 2.1.4.2); no code was
copied from any implementation. Leaf hash = SHA-256(0x00 || data); node hash =
SHA-256(0x01 || left || right); the empty tree hashes to SHA-256(""). Cross-checked against the
transparency-dev/merkle test data (Apache-2.0) at a pinned commit, see tests/fixtures.

A signed tree head (STH) is a DSSE envelope (ECDSA P-256 / RFC 6979, same code as the receipts)
over {schema, log_id, tree_size, root_hash, timestamp_utc}; log_id is the log key's keyid.
"""
import hashlib

import rbcg1

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"
STH_SCHEMA = "szl.rbcg.sth/v1"
ANCHOR_SCHEMA = "szl.rbcg.anchor/v1"


def leaf_hash(data):
    if type(data) is not bytes:
        raise TypeError("leaf data must be bytes")
    return hashlib.sha256(LEAF_PREFIX + data).digest()


def node_hash(left, right):
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def empty_root():
    return hashlib.sha256(b"").digest()


def _split(n):
    """Largest power of two strictly smaller than n (n >= 2)."""
    return 1 << ((n - 1).bit_length() - 1)


def mth(leaf_hashes):
    """Reference Merkle Tree Hash over a list of leaf hashes (RFC 9162 2.1.1), recursive."""
    n = len(leaf_hashes)
    if n == 0:
        return empty_root()
    if n == 1:
        return leaf_hashes[0]
    k = _split(n)
    return node_hash(mth(leaf_hashes[:k]), mth(leaf_hashes[k:]))


class MerkleLog:
    """Append-only log of leaf hashes. Hashes of complete (power-of-two) subtrees are cached; they
    never change once their leaves exist, so roots and proofs for any past size cost O(log^2 n)."""

    def __init__(self):
        self._leaves = []
        self._perfect = {}

    @property
    def size(self):
        return len(self._leaves)

    def append(self, data):
        self._leaves.append(leaf_hash(data))
        return len(self._leaves) - 1

    def leaf(self, i):
        return self._leaves[i]

    def _hash(self, start, n):
        if n == 1:
            return self._leaves[start]
        if n & (n - 1) == 0:
            key = (start, n)
            h = self._perfect.get(key)
            if h is None:
                half = n >> 1
                h = node_hash(self._hash(start, half), self._hash(start + half, half))
                self._perfect[key] = h
            return h
        k = _split(n)
        return node_hash(self._hash(start, k), self._hash(start + k, n - k))

    def root(self, n=None):
        n = self.size if n is None else n
        if not 0 <= n <= self.size:
            raise ValueError("tree size out of range")
        return empty_root() if n == 0 else self._hash(0, n)

    def inclusion_proof(self, m, n=None):
        """PATH(m, D_n) of RFC 9162 2.1.3.1, as a list of 32-byte node hashes (leaf level first)."""
        n = self.size if n is None else n
        if not (0 <= m < n <= self.size):
            raise ValueError("need 0 <= m < n <= size")
        return self._path(m, 0, n)

    def _path(self, m, start, n):
        if n == 1:
            return []
        k = _split(n)
        if m < k:
            return self._path(m, start, k) + [self._hash(start + k, n - k)]
        return self._path(m - k, start + k, n - k) + [self._hash(start, k)]

    def consistency_proof(self, m, n=None):
        """PROOF(m, D_n) = SUBPROOF(m, D_n, true) of RFC 9162 2.1.4.1 for 0 < m < n; [] when m == n."""
        n = self.size if n is None else n
        if not (0 < m <= n <= self.size):
            raise ValueError("need 0 < m <= n <= size")
        if m == n:
            return []
        return self._subproof(m, 0, n, True)

    def _subproof(self, m, start, n, b):
        if m == n:
            return [] if b else [self._hash(start, n)]
        k = _split(n)
        if m <= k:
            return self._subproof(m, start, k, b) + [self._hash(start + k, n - k)]
        return self._subproof(m - k, start + k, n - k, False) + [self._hash(start, k)]


def _is_hash(h):
    return type(h) is bytes and len(h) == 32


def verify_inclusion(leaf_index, tree_size, leaf_h, path, root):
    """RFC 9162 2.1.3.2. True iff `path` proves `leaf_h` is leaf `leaf_index` of the tree of size
    `tree_size` with root hash `root`. Never raises on malformed input (returns False)."""
    if type(leaf_index) is not int or type(tree_size) is not int or leaf_index < 0 or tree_size < 0:
        return False
    if not (_is_hash(leaf_h) and _is_hash(root)) or type(path) is not list or not all(_is_hash(p) for p in path):
        return False
    if leaf_index >= tree_size:
        return False
    fn, sn = leaf_index, tree_size - 1
    r = leaf_h
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = node_hash(p, r)
            if not fn & 1:
                while fn != 0 and not fn & 1:
                    fn >>= 1
                    sn >>= 1
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def verify_consistency(first, second, first_hash, second_hash, path):
    """RFC 9162 2.1.4.2 for 0 < first < second. Outside that range we follow the conventions encoded
    in the transparency-dev/merkle test data: first == second > 0 needs an empty proof and equal roots
    (compared as opaque bytes); first == 0 (a consistency check from the empty tree proves nothing)
    and first > second fail. Never raises on malformed input."""
    if type(first) is not int or type(second) is not int or first <= 0 or second < first:
        return False
    if type(first_hash) is not bytes or type(second_hash) is not bytes or type(path) is not list:
        return False
    if first == second:
        return len(path) == 0 and first_hash == second_hash
    if not (_is_hash(first_hash) and _is_hash(second_hash)) or not all(_is_hash(p) for p in path):
        return False
    if len(path) == 0:
        return False
    path = list(path)
    if first & (first - 1) == 0:
        path = [first_hash] + path
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
            if not fn & 1:
                while fn != 0 and not fn & 1:
                    fn >>= 1
                    sn >>= 1
        else:
            sr = node_hash(sr, c)
        fn >>= 1
        sn >>= 1
    return fr == first_hash and sr == second_hash and sn == 0


# ------------------------------------------------------------------ signed tree heads, anchors
def sth_payload(log_id, tree_size, root, timestamp_utc):
    return {"schema": STH_SCHEMA, "log_id": log_id, "tree_size": tree_size, "root_hash": root.hex(),
            "timestamp_utc": timestamp_utc}


def sign_sth(log, key, timestamp_utc, n=None):
    n = log.size if n is None else n
    return rbcg1.dsse_envelope(sth_payload(key.public.keyid(), n, log.root(n), timestamp_utc),
                               rbcg1.PT_STH, key)


def make_anchor(log, guarantee_index, sth_issue, head_index=None, sth_final=None):
    """Anchor bundle for the verifier: the issuance STH with the guarantee's inclusion proof and, when
    a monitor chain exists, the final STH, the chain head's inclusion proof and the consistency proof
    issuance -> final."""
    import base64
    import json
    s1 = json.loads(base64.b64decode(sth_issue["payload"]))["tree_size"]
    out = {"schema": ANCHOR_SCHEMA, "sth_issue": sth_issue,
           "guarantee_inclusion": {"leaf_index": guarantee_index, "tree_size": s1,
                                   "path": [h.hex() for h in log.inclusion_proof(guarantee_index, s1)]},
           "sth_final": None, "head_inclusion": None, "consistency": None}
    if sth_final is not None:
        s2 = json.loads(base64.b64decode(sth_final["payload"]))["tree_size"]
        out["sth_final"] = sth_final
        out["head_inclusion"] = {"leaf_index": head_index, "tree_size": s2,
                                 "path": [h.hex() for h in log.inclusion_proof(head_index, s2)]}
        out["consistency"] = {"first_size": s1, "second_size": s2,
                              "path": [h.hex() for h in log.consistency_proof(s1, s2)]}
    return out


def envelope_leaf(env):
    """The leaf data for a receipt: its canonical envelope bytes."""
    return rbcg1.envelope_bytes(env)
