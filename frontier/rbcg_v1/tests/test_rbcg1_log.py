#!/usr/bin/env python3
"""test_rbcg1_log.py -- RFC 9162-style Merkle log tests: published test vectors (transparency-dev/merkle
at a pinned commit, Apache-2.0, data only), the RFC 9162 section 2.1.5 worked example, exhaustive round
trips and tampering. SYNTHETIC DATA ONLY."""
import json
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import rbcg1  # noqa: E402
import rbcg1_log as L  # noqa: E402
import rbcg1_verify as V  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "merkle_vectors.json")
H = bytes.fromhex


def load_vectors():
    with open(FIX, encoding="utf-8") as fh:
        return json.load(fh)


class TestPublishedVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = load_vectors()          # a missing fixture is an error, never a skip

    def test_fixture_provenance(self):
        v = self.v
        self.assertEqual(v["commit"], "c39fbf350a405f68571bf4c42a6a518eb0a46a4b")
        self.assertEqual(v["tarball_sha256"], "ed0c1017ea66fc7e88e67998f7d6b9cc1eef66d30b97b58a1dbfc1aa3a1edd09")
        self.assertIn("Apache-2.0", v["licence"])
        self.assertGreater(len(v["inclusion"]), 50)
        self.assertGreater(len(v["consistency"]), 50)
        self.assertEqual(len(v["root_hashes_hex_by_size"]), len(v["leaf_inputs_hex"]) + 1)

    def test_root_hashes_for_every_size(self):
        lg = L.MerkleLog()
        self.assertEqual(lg.root(0).hex(), self.v["root_hashes_hex_by_size"][0])
        leaves = []
        for i, d in enumerate(self.v["leaf_inputs_hex"]):
            lg.append(H(d))
            leaves.append(L.leaf_hash(H(d)))
            self.assertEqual(lg.root().hex(), self.v["root_hashes_hex_by_size"][i + 1], i + 1)
            self.assertEqual(L.mth(leaves).hex(), self.v["root_hashes_hex_by_size"][i + 1])

    def test_leaf_and_node_domain_separation(self):
        r = self.v["root_hashes_hex_by_size"]
        d0, d1 = (H(x) for x in self.v["leaf_inputs_hex"][:2])
        self.assertEqual(L.leaf_hash(d0).hex(), r[1])
        self.assertEqual(L.node_hash(L.leaf_hash(d0), L.leaf_hash(d1)).hex(), r[2])
        self.assertNotEqual(L.leaf_hash(L.leaf_hash(d0) + L.leaf_hash(d1)).hex(), r[2])

    def test_inclusion_probes(self):
        bad = []
        for p in self.v["inclusion"]:
            got = L.verify_inclusion(p["leaf_index"], p["tree_size"], H(p["leaf_hash"] or ""),
                                     [H(x or "") for x in p["proof"]], H(p["root"] or ""))
            if got != (not p["want_error"]):
                bad.append(p["file"])
        self.assertEqual(bad, [])

    def test_consistency_probes(self):
        bad = []
        for p in self.v["consistency"]:
            got = L.verify_consistency(p["size1"], p["size2"], H(p["root1"] or ""), H(p["root2"] or ""),
                                       [H(x or "") for x in p["proof"]])
            if got != (not p["want_error"]):
                bad.append(p["file"])
        self.assertEqual(bad, [])

    def test_our_generated_proofs_match_published_happy_paths(self):
        lg = L.MerkleLog()
        for d in self.v["leaf_inputs_hex"]:
            lg.append(H(d))
        roots = self.v["root_hashes_hex_by_size"]
        matched = 0
        for p in self.v["inclusion"]:
            n = p["tree_size"]
            if not p["want_error"] and 0 < n < len(roots) and p["root"] == roots[n] and p["leaf_index"] < n \
                    and p["leaf_hash"] == lg.leaf(p["leaf_index"]).hex():
                self.assertEqual([h.hex() for h in lg.inclusion_proof(p["leaf_index"], n)], p["proof"], p["file"])
                matched += 1
        for p in self.v["consistency"]:
            m, n = p["size1"], p["size2"]
            if not p["want_error"] and 0 < m < n < len(roots) and p["root1"] == roots[m] and p["root2"] == roots[n]:
                self.assertEqual([h.hex() for h in lg.consistency_proof(m, n)], p["proof"], p["file"])
                matched += 1
        self.assertGreater(matched, 0)


class TestRFC9162Example(unittest.TestCase):
    """RFC 9162 section 2.1.5: the 7-leaf tree and its stated inclusion and consistency proofs."""

    def test_example(self):
        d = [b"d%d" % i for i in range(7)]
        a, b, c, dd, e, f, j = (L.leaf_hash(x) for x in d)
        g, h, i = L.node_hash(a, b), L.node_hash(c, dd), L.node_hash(e, f)
        k, l = L.node_hash(g, h), L.node_hash(i, j)
        lg = L.MerkleLog()
        for x in d:
            lg.append(x)
        self.assertEqual(lg.root(7), L.node_hash(k, l))
        self.assertEqual(lg.root(3), L.node_hash(g, c))       # hash0
        self.assertEqual(lg.root(4), k)                       # hash1 = k
        self.assertEqual(lg.root(6), L.node_hash(k, i))       # hash2
        self.assertEqual(lg.inclusion_proof(0, 7), [b, h, l])
        self.assertEqual(lg.inclusion_proof(3, 7), [c, g, l])
        self.assertEqual(lg.inclusion_proof(4, 7), [f, j, k])
        self.assertEqual(lg.inclusion_proof(6, 7), [i, k])
        self.assertEqual(lg.consistency_proof(3, 7), [c, dd, g, l])
        self.assertEqual(lg.consistency_proof(4, 7), [l])
        self.assertEqual(lg.consistency_proof(6, 7), [i, j, k])
        for m, n in ((3, 7), (4, 7), (6, 7)):
            self.assertTrue(L.verify_consistency(m, n, lg.root(m), lg.root(n), lg.consistency_proof(m, n)))


class TestRoundTrip(unittest.TestCase):
    def test_all_sizes_and_off_by_one(self):
        lg = L.MerkleLog()
        for n in range(1, 49):
            lg.append(b"leaf-%d" % n)
            rn = lg.root(n)
            for m in range(n):
                pth = lg.inclusion_proof(m, n)
                self.assertTrue(L.verify_inclusion(m, n, lg.leaf(m), pth, rn), (m, n))
                if m + 1 < n:
                    self.assertFalse(L.verify_inclusion(m + 1, n, lg.leaf(m), pth, rn), (m, n))
                self.assertFalse(L.verify_inclusion(m, n, lg.leaf(m), pth, lg.root(n - 1) if n > 1 else bytes(32)))
            for m in range(1, n + 1):
                cp = lg.consistency_proof(m, n)
                self.assertTrue(L.verify_consistency(m, n, lg.root(m), rn, cp), (m, n))
                if m < n:
                    self.assertFalse(L.verify_consistency(m, n, rn, lg.root(m), cp))
                    if m + 1 < n:
                        self.assertFalse(L.verify_consistency(m + 1, n, lg.root(m), rn, cp))

    def test_tampering(self):
        lg = L.MerkleLog()
        for n in range(37):
            lg.append(b"x%d" % n)
        pth = lg.inclusion_proof(13, 37)
        for idx in range(len(pth)):
            t = list(pth)
            t[idx] = bytes([t[idx][0] ^ 1]) + t[idx][1:]
            self.assertFalse(L.verify_inclusion(13, 37, lg.leaf(13), t, lg.root(37)))
        self.assertFalse(L.verify_inclusion(13, 37, lg.leaf(13), pth[:-1], lg.root(37)))
        self.assertFalse(L.verify_inclusion(13, 37, lg.leaf(13), pth + [pth[0]], lg.root(37)))
        cp = lg.consistency_proof(20, 37)
        for idx in range(len(cp)):
            t = list(cp)
            t[idx] = bytes([t[idx][0] ^ 1]) + t[idx][1:]
            self.assertFalse(L.verify_consistency(20, 37, lg.root(20), lg.root(37), t))

    def test_malformed_never_raises(self):
        for args in ((None, 1, b"", [], b""), (0, 1, b"x" * 32, "nope", b"y" * 32), (-1, 5, bytes(32), [], bytes(32)),
                     (0, 1, bytes(31), [], bytes(32))):
            self.assertFalse(L.verify_inclusion(*args))
        self.assertFalse(L.verify_consistency(2, 1, bytes(32), bytes(32), []))
        self.assertFalse(L.verify_consistency(1, 2, bytes(32), bytes(32), []))
        self.assertFalse(L.verify_consistency(1, 2, bytes(32), bytes(32), [b"short"]))
        self.assertFalse(L.verify_consistency(0, 5, bytes(32), bytes(32), []))
        self.assertFalse(L.verify_consistency(0, 0, bytes(32), bytes(32), []))
        self.assertTrue(L.verify_consistency(3, 3, bytes(32), bytes(32), []))
        self.assertFalse(L.verify_consistency(3, 3, bytes(32), bytes(32), [bytes(32)]))

    def test_reference_mth_random_sizes(self):
        rng = random.Random(8)
        lg = L.MerkleLog()
        leaves = []
        for _ in range(300):
            d = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 40)))
            lg.append(d)
            leaves.append(L.leaf_hash(d))
        for n in (1, 2, 3, 5, 64, 65, 127, 128, 129, 255, 300):
            self.assertEqual(lg.root(n), L.mth(leaves[:n]))


class TestSTH(unittest.TestCase):
    def test_sign_verify_and_anchor(self):
        key = rbcg1.SigningKey.generate()
        lg = L.MerkleLog()
        for n in range(5):
            lg.append(b"r%d" % n)
        s1 = L.sign_sth(lg, key, "2026-11-03T00:00:00Z")
        for n in range(5, 23):
            lg.append(b"r%d" % n)
        s2 = L.sign_sth(lg, key, "2026-11-03T00:00:01Z")
        _, p1 = V.check_envelope_obj(s1, rbcg1.PT_STH, key.public, "sth")
        _, p2 = V.check_envelope_obj(s2, rbcg1.PT_STH, key.public, "sth")
        self.assertEqual((p1["tree_size"], p2["tree_size"]), (5, 23))
        self.assertEqual(p1["log_id"], key.public.keyid())
        anc = L.make_anchor(lg, 2, s1, 20, s2)
        self.assertTrue(L.verify_inclusion(2, 5, lg.leaf(2), [H(x) for x in anc["guarantee_inclusion"]["path"]],
                                           H(p1["root_hash"])))
        self.assertTrue(L.verify_inclusion(20, 23, lg.leaf(20), [H(x) for x in anc["head_inclusion"]["path"]],
                                           H(p2["root_hash"])))
        self.assertTrue(L.verify_consistency(5, 23, H(p1["root_hash"]), H(p2["root_hash"]),
                                             [H(x) for x in anc["consistency"]["path"]]))
        other = rbcg1.SigningKey.generate()
        with self.assertRaises(V.Refuse):
            V.check_envelope_obj(s1, rbcg1.PT_STH, other.public, "sth")


if __name__ == "__main__":
    unittest.main()
