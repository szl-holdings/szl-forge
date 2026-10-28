"""Integration tests: envelope adapter + bridge inside Forge's real ``production_infer``.

Stdlib unittest; synthetic corpus; local spy backend and local spy witness; no network,
no provider, no real model.  Loads ``inference/production.py`` from the repository root
(two levels up) or from ``$E2D_FORGE_ROOT``; the whole class is SKIPPED if it is absent.
"""
import importlib.util
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a11oy_forge_envelope_adapter as ad  # noqa: E402
import forge_seam_bridge as br  # noqa: E402

ROOT = os.environ.get("E2D_FORGE_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
PROD = os.path.join(ROOT, "inference", "production.py")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]+-[A-Z][A-Za-z0-9]*\b")
REV = "1" * 40
POLICY = "2" * 40
PINS = {"tokenizer_revision": "3" * 40, "template_revision": "4" * 40}
RUNTIME = {"engine": "local-spy", "version": "0", "hardware_fingerprint": "synthetic-none"}
CRED = {"cred-aurora": ("principal.aurora.analyst", "tenant-aurora"),
        "cred-borealis": ("principal.borealis.analyst", "tenant-borealis")}


def load_production():
    spec = importlib.util.spec_from_file_location("forge_inference_production", PROD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def corpus():
    with open(os.path.join(HERE, "corpus_manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    docs = {}
    for e in man["docs"]:
        with open(os.path.join(HERE, e["file"]), encoding="utf-8", newline="") as f:
            docs[e["doc_id"]] = dict(e, text=f.read())
    return docs


class SeamGenerator:
    """Pairs an identity callable with a generation callable (Forge Generator protocol)."""

    def __init__(self, identity_fn, call_fn):
        self._identity, self._call = identity_fn, call_fn

    def identity(self):
        return self._identity()

    def __call__(self, context):
        return self._call(context)


class SpyBackend:
    """Local deterministic stand-in for the single fixed model destination."""

    def __init__(self):
        self.calls = 0

    def __call__(self, ctx):
        self.calls += 1
        ev = ctx["evidence"][0]
        body = [ln for ln in ev["content"].splitlines()
                if not ln.startswith(("DOC ", "TENANT ", "REV ", "Marker "))][0]
        return {"text": body.split(". ")[0].rstrip(".") + ". [cite:" + ev["node_id"] + "]"}


@unittest.skipUnless(os.path.isfile(PROD), "Forge inference/production.py not present")
class ForgeProductionSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prod = load_production()
        cls.docs = corpus()

    def setUp(self):
        self.reg = ad.InMemoryRegistry(epoch=1)
        for cred, (pid, tid) in CRED.items():
            self.reg.principals[cred] = ad.TrustedPrincipal(pid, tid, "controller")
        self.dest = ad.FixedDestination("local", "spy-model-0", "5" * 40, "INTERNAL", "ep-spy")
        self.chunks, self.content = {}, {}
        for did, d in self.docs.items():
            pid, tid = CRED["cred-aurora"] if d["tenant"] == "tenant-aurora" else CRED["cred-borealis"]
            self.chunks[did] = ad.Chunk(did, "src-" + did, "g1", REV, d["sha256"], ad.text_sha256(d["text"]))
            self.content[did] = d["text"]
            g = ad.SourceGrant("grant-" + did, "src-" + did, pid, tid, "owner-synthetic", "PUBLIC", REV,
                               d["sha256"], frozenset({"read", "hydrate", "egress"}), 1, 10 ** 9)
            self.reg.grants[(pid, "src-" + did)] = g
            self.reg.transfers.add((g.grant_id, "ep-spy"))
            self.reg.generations["src-" + did] = "g1"
        self.backend = SpyBackend()
        self.adapter = ad.EnvelopeAdapter(self.reg, ad.StoreProfile("store-1", "PUBLIC_DEFAULT"), self.dest,
                                          self.backend, self.content, lambda: 1000, POLICY)

    def index(self, query):
        ents = set(ENTITY_RE.findall(query))           # deliberately tenant-blind
        return [self.chunks[d] for d in sorted(self.docs) if ents & set(ENTITY_RE.findall(self.docs[d]["text"]))]

    def run_case(self, prompt, *, witness=None, generator=None, retriever=None, cred="cred-aurora"):
        s = self.adapter.open(cred, {"prompt": prompt})
        seams = br.ForgeSeams(s, self.index, model_pins=PINS, runtime=RUNTIME)
        w = witness or br.LocalSpyWitness()
        res = self.prod.production_infer(seams.build_request(prompt), retriever=retriever or seams.retriever,
                                         hydrator=seams.hydrator, generator=generator or seams, witness=w)
        return s, seams, w, res

    def assert_blocked_before_model(self, res, code, w, stages, s=None, dag_code=None):
        self.assertEqual(res["state"], "BLOCKED")
        self.assertEqual(w.stages, stages)
        self.assertEqual(res["reason_codes"][0], code)
        self.assertEqual(self.backend.calls, 0)
        self.assertIsNone(res["output"])
        self.assertFalse(res["executed"])
        if dag_code is not None:
            self.assertEqual(s.dag.export()[-1]["code"], dag_code)
            self.assertTrue(ad.verify_dag(s.dag.export()))

    def test_bridged_request_yields_unsigned_proposal(self):
        s, seams, w, res = self.run_case("How often does Lumen-Gate rotate its session keys?")
        self.assertEqual((res["state"], res["authority_state"]), ("PROPOSAL", "NO_ACTION_AUTHORITY"))
        self.assertFalse(res["executed"])
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(w.stages, ["PRE_GENERATION", "POST_GENERATION"])
        self.assertEqual(res["citations"], ["TA-D01"])
        self.assertEqual([h["nodeId"] for h in res["evidence_handles"]], ["TA-D01"])
        self.assertNotIn("BOR-", json.dumps(res))
        self.assertNotIn("borealis deployment", json.dumps(res))
        self.assertEqual(res["receipt"]["signature"]["status"], "UNSIGNED_LOCAL")
        dag = s.dag.export()
        self.assertTrue(ad.verify_dag(dag))
        self.assertEqual([n["decision"] for n in dag if n["kind"] == "chunk"], ["ADMITTED_READ", "DENIED"])
        self.assertEqual(dag[-1], dict(dag[-1], kind="human_review", state="PENDING"))

    def test_adapter_identity_without_bridge_pins_is_blocked(self):
        s = self.adapter.open("cred-aurora", {"prompt": "How often does Lumen-Gate rotate its session keys?"})
        seams = br.ForgeSeams(s, self.index, model_pins=PINS, runtime=RUNTIME)
        w = br.LocalSpyWitness()
        res = self.prod.production_infer(seams.build_request(s._prompt), retriever=seams.retriever,
                                         hydrator=seams.hydrator, generator=SeamGenerator(s.identity, s.generator),
                                         witness=w)
        self.assert_blocked_before_model(res, "GENERATOR_IDENTITY_INVALID", w, [])

    def test_adapter_text_output_without_bridge_shape_is_rejected(self):
        s = self.adapter.open("cred-aurora", {"prompt": "How often does Lumen-Gate rotate its session keys?"})
        seams = br.ForgeSeams(s, self.index, model_pins=PINS, runtime=RUNTIME)
        res = self.prod.production_infer(seams.build_request(s._prompt), retriever=seams.retriever,
                                         hydrator=seams.hydrator, generator=SeamGenerator(seams.identity, s.generator),
                                         witness=br.LocalSpyWitness())
        self.assertEqual((res["state"], res["reason_codes"][0]), ("BLOCKED", "GENERATOR_OUTPUT_INVALID"))
        self.assertEqual(self.backend.calls, 1)          # the one call happened; its output is discarded
        self.assertIsNone(res["output"])

    def test_revocation_between_retrieval_and_hydration(self):
        holder = {}

        def retr(q, k=6):
            out = holder["seams"].retriever(q, k)
            self.reg.revoked.add("grant-TA-D01")
            return out
        s = self.adapter.open("cred-aurora", {"prompt": "How often does Lumen-Gate rotate its session keys?"})
        holder["seams"] = seams = br.ForgeSeams(s, self.index, model_pins=PINS, runtime=RUNTIME)
        w = br.LocalSpyWitness()
        res = self.prod.production_infer(seams.build_request(s._prompt), retriever=retr, hydrator=seams.hydrator,
                                         generator=seams, witness=w)
        self.assert_blocked_before_model(res, "SECOND_BRAIN_AUTHORIZATION_OR_HYDRATION_INVALID", w, [],
                                         s, "GRANT_REVOKED")

    def test_revocation_after_hydration_refused_at_egress(self):
        w = br.LocalSpyWitness(on_stage={"PRE_GENERATION": lambda: self.reg.revoked.add("grant-TA-D01")})
        s, _, w, res = self.run_case("How often does Lumen-Gate rotate its session keys?", witness=w)
        self.assert_blocked_before_model(res, "GENERATOR_OUTPUT_INVALID", w, ["PRE_GENERATION"], s, "GRANT_REVOKED")
        self.assertEqual(s.dag.export()[-1]["stage"], "egress")

    def test_policy_epoch_change_after_hydration_refused_at_egress(self):
        def bump():
            self.reg.epoch += 1
        w = br.LocalSpyWitness(on_stage={"PRE_GENERATION": bump})
        s, _, w, res = self.run_case("How often does Lumen-Gate rotate its session keys?", witness=w)
        self.assert_blocked_before_model(res, "GENERATOR_OUTPUT_INVALID", w, ["PRE_GENERATION"], s,
                                         "POLICY_EPOCH_STALE")

    def test_other_tenant_evidence_only_abstains(self):
        s, _, w, res = self.run_case("How many shards per replica does Meridian-Index store?")
        self.assertEqual((res["state"], res["reason_codes"]), ("ABSTAIN", ["NO_GROUNDED_EVIDENCE"]))
        self.assertEqual(self.backend.calls, 0)
        self.assertEqual(w.stages, [])

    def test_witness_block_before_generation(self):
        s, _, w, res = self.run_case("How often does Lumen-Gate rotate its session keys?",
                                     witness=br.LocalSpyWitness("BLOCK"))
        self.assert_blocked_before_model(res, "NEMO_PRE_GENERATION_BLOCK", w, ["PRE_GENERATION"])

    def test_tampered_body_blocks_hydration(self):
        self.content["TA-D01"] = self.content["TA-D01"].replace("36", "5")
        s, _, w, res = self.run_case("How often does Lumen-Gate rotate its session keys?")
        self.assert_blocked_before_model(res, "SECOND_BRAIN_AUTHORIZATION_OR_HYDRATION_INVALID", w, [],
                                         s, "BODY_DIGEST_MISMATCH")


if __name__ == "__main__":
    unittest.main()
