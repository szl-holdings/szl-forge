import contextlib, io, json, math, os, random, shutil, sys, tempfile, unittest
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
    def _cli(self, env, *extra):
        ef = self.d / "env.json"; ef.write_text(env if isinstance(env, str) else json.dumps(env))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = A.main(["verify", "--base", str(self.d), *extra, str(ef)])
        return rc, out.getvalue()
    def test_cli_unsigned_is_not_a_pass_by_default(self):
        rc, out = self._cli(self.env); self.assertEqual(rc, 3); self.assertIn("UNSIGNED:DIGESTS_OK", out); self.assertIn("not bound to a receipt", out)
    def test_cli_exit_0_only_with_explicit_digest_flag(self):
        self.assertEqual(self._cli(self.env, "--accept-unsigned-digests")[0], 0)
        (self.d / "b.py").write_text("x=2"); self.assertEqual(self._cli(self.env, "--accept-unsigned-digests")[0], 2)  # refusal beats the flag
        (self.d / "b.py").write_text("x=1")
        e = dict(self.env); e["signatures"] = [{"sig": "AAAA"}]; self.assertEqual(self._cli(e, "--accept-unsigned-digests")[0], 3)  # a signature is never accepted
    def test_cli_malformed_envelope_refuses_without_traceback(self):
        for bad in ("{not json", "[" * 100000 + "]" * 100000, "[]"):
            rc, out = self._cli(bad); self.assertEqual(rc, 2); self.assertIn("REFUSE:MALFORMED", out)

class Slice(unittest.TestCase):
    def test_cp_known(self):  # 0 misses of 10 at delta .1: 1-0.1**(1/10)
        self.assertAlmostEqual(S.cp_upper(0, 10, 0.1), 1 - 0.1 ** 0.1, places=6)
    def test_cp_matches_ltt_boundary(self):  # 3/33 upper bound is just above/at alpha=.2 boundary used by LTT (cdf(3)=.0808<=.1)
        self.assertLessEqual(S.cp_upper(3, 33, 0.1), 0.2)
        self.assertGreater(S.cp_upper(4, 33, 0.1), 0.2)
    def test_underpowered_never_ok(self):
        rows = [{"features": {"b": i % 2}, "label": {"operator_attention_required": True}} for i in range(6)]
        r = S.audit(rows, [0.0] * 6, 0.5); self.assertTrue(all(v["status"] == "UNDERPOWERED" for v in r.values()))
class SliceKernelPin(unittest.TestCase):
    MD = Path(__file__).resolve().parents[1] / "huggingface" / "model" / "oac-system-health-v1"
    def setUp(self):
        self.d = Path(tempfile.mkdtemp()); self.evil = self.d / "evil"; shutil.copytree(self.MD, self.evil); self.marker = self.d / "MARKER"
        with open(self.evil / "oac_operational_health.py", "a") as f: f.write(f"\nopen({str(self.marker)!r}, 'w').write('executed')\n")
        self.rows = [json.loads(l) for l in (Path(__file__).resolve().parents[2] / "ops-health/v2/data/legacy_v1/validation.jsonl").read_text().splitlines()[:5]]
    def tearDown(self): shutil.rmtree(self.d, ignore_errors=True)
    def test_pinned_kernel_scores(self):
        sc = S.score_rows(self.MD, self.rows); self.assertEqual(len(sc), 5); self.assertTrue(all(0 <= x <= 1 for x in sc))
    def test_unpinned_kernel_is_refused_before_execution(self):
        with self.assertRaises(S.Refuse): S.score_rows(self.evil, self.rows)
        self.assertFalse(self.marker.exists())
    def test_cli_refuses_unpinned_kernel_exit_3_and_never_runs_it(self):
        sp = self.d / "data"; sp.mkdir(); (sp / "test.jsonl").write_text("\n".join(json.dumps(r) for r in self.rows) + "\n")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = S.main(["--model-dir", str(self.evil), "--data-dir", str(sp), "--lambda", "0.137721"])
        self.assertEqual((rc, out.getvalue().strip()), (3, "REFUSE:UNPINNED_KERNEL")); self.assertFalse(self.marker.exists())
    def test_override_is_deliberate_and_runs_the_kernel(self):
        try: S.score_rows(self.evil, self.rows, allow_unpinned=True)
        except Exception: pass  # the real kernel may reject an edited file; the marker shows it was executed
        self.assertTrue(self.marker.exists())
    def test_pin_set_is_exact(self):
        self.assertEqual(S.PINNED_KERNEL_SHA256, {"5e06ddffce9615a12c351e9926e24fefa142c920017b9dc844c584c58517202e", "e2aa63991b94596a16d69a288510b4b10af88932e075c055d97d1dac4d9a930d"})
if __name__ == "__main__": unittest.main()
