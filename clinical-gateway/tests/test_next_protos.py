import json, math, os, random, sys, tempfile, unittest
from pathlib import Path
T = Path(__file__).resolve().parents[1] / "tools"; sys.path.insert(0, str(T))
import oac_drift as D, oac_attest as A, oac_slice_audit as S

class Drift(unittest.TestCase):
    def feats(self, n, shift=0.0, seed=1):
        r = random.Random(seed); return [{"q": r.random() + shift, "b": int(r.random() < 0.5)} for _ in range(n)]
    def test_identical_is_ok(self):
        f = self.feats(300); self.assertEqual(D.report(f, f)["overall"], "OK"); self.assertAlmostEqual(D.psi([1, 2, 3, 4] * 20, [1, 2, 3, 4] * 20), 0, places=6)
    def test_shift_detected(self):
        self.assertEqual(D.report(self.feats(300), self.feats(300, shift=0.5, seed=2))["features"]["q"]["status"], "DRIFT")
    def test_empty_refused(self):
        with self.assertRaises(ValueError): D.psi([], [1.0])
    def test_unseen_category_penalised(self):
        self.assertGreater(D.psi([0, 1] * 50, [0, 1, 2] * 30), 0.25)

class Attest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp()); (self.d / "a.json").write_text("{}"); (self.d / "b.py").write_text("x=1")
        self.env = A.build([self.d / "a.json", self.d / "b.py"], self.d, {"alpha": 0.2})
    def test_ok_is_unsigned_only(self): self.assertEqual(A.verify(self.env, self.d)["verdict"], "UNSIGNED:DIGESTS_OK")
    def test_tamper(self):
        (self.d / "b.py").write_text("x=2"); self.assertEqual(A.verify(self.env, self.d)["verdict"], "REFUSE:DIGEST_MISMATCH:b.py")
    def test_missing(self):
        (self.d / "a.json").unlink(); self.assertTrue(A.verify(self.env, self.d)["verdict"].startswith("REFUSE:MISSING"))
    def test_malformed(self): self.assertTrue(A.verify({"payloadType": A.PAYLOAD_TYPE, "payload": "!!"}, self.d)["verdict"].startswith("REFUSE:MALFORMED"))
    def test_fake_signature_not_accepted_as_signed(self):
        e = dict(self.env); e["signatures"] = [{"sig": "AAAA"}]; self.assertTrue(A.verify(e, self.d)["verdict"].startswith("UNVERIFIED_SIGNATURE"))
    def test_path_escape(self):
        st = json.loads(__import__("base64").b64decode(self.env["payload"])); st["subject"][0]["name"] = "../../etc/passwd"
        e = dict(self.env, payload=__import__("base64").b64encode(json.dumps(st).encode()).decode()); self.assertTrue(A.verify(e, self.d)["verdict"].startswith("REFUSE:PATH_ESCAPE"))
    def test_pae_known_answer(self): self.assertEqual(A.pae("t", b"ab"), b"DSSEv1 1 t 2 ab")

class Slice(unittest.TestCase):
    def test_cp_known(self):  # 0 misses of 10 at delta .1: 1-0.1**(1/10)
        self.assertAlmostEqual(S.cp_upper(0, 10, 0.1), 1 - 0.1 ** 0.1, places=6)
    def test_cp_matches_ltt_boundary(self):  # 3/33 upper bound is just above/at alpha=.2 boundary used by LTT (cdf(3)=.0808<=.1)
        self.assertLessEqual(S.cp_upper(3, 33, 0.1), 0.2)
        self.assertGreater(S.cp_upper(4, 33, 0.1), 0.2)
    def test_underpowered_never_ok(self):
        rows = [{"features": {"b": i % 2}, "label": {"operator_attention_required": True}} for i in range(6)]
        r = S.audit(rows, [0.0] * 6, 0.5); self.assertTrue(all(v["status"] == "UNDERPOWERED" for v in r.values()))
if __name__ == "__main__": unittest.main()
