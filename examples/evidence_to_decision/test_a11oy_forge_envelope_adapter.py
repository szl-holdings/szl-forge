import unittest
from a11oy_forge_envelope_adapter import (Chunk, EnvelopeAdapter, FixedDestination,
    InMemoryRegistry, Refusal, SourceGrant, StoreProfile, TrustedPrincipal,
    canonical_sha256, text_sha256, verify_dag)

REV = "a" * 40
DIG = "b" * 64
POL = "c" * 40
BODY_PUB = "synthetic public chunk body"
BODY_PRV = "synthetic private chunk body"


class CountingStore(dict):
    def __init__(self, *a):
        super().__init__(*a)
        self.reads = 0

    def get(self, k, d=None):
        self.reads += 1
        return super().get(k, d)


class Backend:
    def __init__(self, reply=None, fail=False):
        self.calls = 0
        self.reply = reply or {"text": "synthetic proposal"}
        self.fail = fail

    def __call__(self, ctx):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic outage")
        return dict(self.reply)


class Clock:
    def __init__(self, t=1000):
        self.t = t

    def __call__(self):
        return self.t


def grant(gid="g1", src="S1", pid="alice", tid="t1", vis="PUBLIC",
          purposes=("read", "hydrate", "egress"), epoch=1, exp=2000, rev=REV, dig=DIG):
    return SourceGrant(gid, src, pid, tid, "owner-a", vis, rev, dig, frozenset(purposes),
                       epoch, exp)


class Base(unittest.TestCase):
    def setUp(self):
        self.reg = InMemoryRegistry(epoch=1)
        self.reg.principals = {"cred-alice": TrustedPrincipal("alice", "t1", "controller"),
                               "cred-bob": TrustedPrincipal("bob", "t2", "controller")}
        self.reg.grants[("alice", "S1")] = grant()
        self.reg.grants[("alice", "S2")] = grant("g2", "S2", vis="PRIVATE")
        self.reg.generations = {"S1": "gen-1", "S2": "gen-1"}
        self.reg.transfers = {("g1", "ep-ext"), ("g1", "ep-int"), ("g2", "ep-int"), ("g2", "ep-ext")}
        self.c1 = Chunk("c1", "S1", "gen-1", REV, DIG, text_sha256(BODY_PUB))
        self.c2 = Chunk("c2", "S2", "gen-1", REV, DIG, text_sha256(BODY_PRV))
        self.store = CountingStore({"c1": BODY_PUB, "c2": BODY_PRV})
        self.backend = Backend()
        self.clock = Clock()
        self.ext = FixedDestination("prov-x", "model-x", "d" * 40, "EXTERNAL", "ep-ext")
        self.int = FixedDestination("prov-i", "model-i", "e" * 40, "INTERNAL", "ep-int")

    def adapter(self, kind="PUBLIC_DEFAULT", dest=None, backend=None):
        return EnvelopeAdapter(self.reg, StoreProfile("st", kind), dest or self.ext,
                               backend or self.backend, self.store, self.clock, POL)

    def session(self, chunks=None, **kw):
        s = self.adapter(**kw).open("cred-alice", {"prompt": "q", "request_id": "r1"})
        res = s.retrieve(chunks or [self.c1])
        return s, res

    def hydrate(self, s, res, pid="alice", tid="t1", pol=POL):
        return s.hydrator(res["handles"], principal_id=pid, tenant_id=tid, policy_revision=pol)

    def ctx(self, s, hyd):
        return {"request_id": s.request_id, "evidence": hyd["documents"]}

    def refuse(self, code, fn, *a, **kw):
        with self.assertRaises(Refusal) as cm:
            fn(*a, **kw)
        self.assertEqual(cm.exception.code, code)
        self.assertEqual(self.backend.calls, 0, "model was called despite refusal")


class HappyPath(Base):
    def test_full_path_single_call_proposal_only(self):
        s, res = self.session()
        hyd = self.hydrate(s, res)
        out = s.generator(self.ctx(s, hyd))
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(out["authority"], "PROPOSAL_ONLY")
        self.assertEqual(out["signature_status"], "UNSIGNED")
        self.assertEqual(out["human_review"], "PENDING")
        self.assertIsNone(out["human_approval"])
        dag = s.dag.export()
        self.assertTrue(verify_dag(dag))
        self.assertEqual([n["kind"] for n in dag],
                         ["request", "chunk", "hydration_decision", "model_call", "output",
                          "human_review"])

    def test_forge_shapes(self):
        s, res = self.session()
        self.assertEqual(res["content_access"], "HANDLES_ONLY")
        self.assertEqual(res["evidence_set_sha256"], canonical_sha256(res["evidence"]))
        self.assertEqual([h["nodeId"] for h in res["handles"]],
                         [e["node_id"] for e in res["evidence"]])
        hyd = self.hydrate(s, res)
        self.assertEqual(hyd["content_access"], "CONTROLLER_ONLY")
        self.assertEqual(hyd["state"], "AUTHORIZED_CONTENT_READY")
        self.assertEqual(hyd["principal_id_sha256"], text_sha256("alice"))
        self.assertEqual(hyd["tenant_id_sha256"], text_sha256("t1"))
        self.assertEqual(hyd["evidence_set_sha256"], res["evidence_set_sha256"])
        self.assertEqual(hyd["raw_graph_nodes_admitted_to_gradients"], 0)
        self.assertEqual(s.identity()["model"]["id"], "model-x")

    def test_private_in_scoped_store_internal_ok(self):
        s, res = self.session([self.c2], kind="SCOPED_PRIVATE", dest=self.int)
        s.generator(self.ctx(s, self.hydrate(s, res)))
        self.assertEqual(self.backend.calls, 1)


class Refusals(Base):
    def test_wrong_tenant_denied_at_read(self):
        self.reg.grants[("alice", "S1")] = grant(tid="t2")
        s, res = self.session()
        self.assertEqual(res["handles"], [])
        self.assertEqual(s.dag.export()[-1]["code"], "TENANT_MISMATCH")
        self.refuse("CHUNK_UNKNOWN", s.hydrator, [{"nodeId": "c1"}], principal_id="alice",
                    tenant_id="t1", policy_revision=POL)

    def test_principal_mismatch_in_grant(self):
        self.reg.grants[("alice", "S1")] = grant(pid="bob")
        s, res = self.session()
        self.assertEqual(s.dag.export()[-1]["code"], "PRINCIPAL_MISMATCH")

    def test_grant_source_mismatch(self):
        self.reg.grants[("alice", "S1")] = grant(src="S9")
        s, res = self.session()
        self.assertEqual(s.dag.export()[-1]["code"], "GRANT_SOURCE_MISMATCH")

    def test_revoked_before_hydration_no_body_read(self):
        s, res = self.session()
        self.reg.revoked.add("g1")
        self.refuse("GRANT_REVOKED", self.hydrate, s, res)
        self.assertEqual(self.store.reads, 0)

    def test_revoked_between_hydration_and_egress(self):
        s, res = self.session()
        hyd = self.hydrate(s, res)
        self.reg.revoked.add("g1")
        self.refuse("GRANT_REVOKED", s.generator, self.ctx(s, hyd))

    def test_expired_before_egress_boundary_inclusive(self):
        s, res = self.session()
        hyd = self.hydrate(s, res)
        self.clock.t = 2000
        self.refuse("GRANT_EXPIRED", s.generator, self.ctx(s, hyd))

    def test_policy_epoch_bump_before_egress(self):
        s, res = self.session()
        hyd = self.hydrate(s, res)
        self.reg.epoch = 2
        self.refuse("POLICY_EPOCH_STALE", s.generator, self.ctx(s, hyd))

    def test_stale_generation_before_hydration(self):
        s, res = self.session()
        self.reg.generations["S1"] = "gen-2"
        self.refuse("GENERATION_STALE", self.hydrate, s, res)

    def test_unknown_generation(self):
        del self.reg.generations["S1"]
        s, res = self.session()
        self.assertEqual(s.dag.export()[-1]["code"], "GENERATION_STALE")

    def test_source_digest_mismatch(self):
        bad = Chunk("c1", "S1", "gen-1", REV, "f" * 64, self.c1.body_sha256)
        s, res = self.session([bad])
        self.assertEqual(s.dag.export()[-1]["code"], "SOURCE_DIGEST_MISMATCH")

    def test_source_revision_mismatch(self):
        bad = Chunk("c1", "S1", "gen-1", "0" * 40, DIG, self.c1.body_sha256)
        s, res = self.session([bad])
        self.assertEqual(s.dag.export()[-1]["code"], "SOURCE_REVISION_MISMATCH")

    def test_body_digest_mismatch(self):
        self.store["c1"] = "tampered body"
        s, res = self.session()
        self.refuse("BODY_DIGEST_MISMATCH", self.hydrate, s, res)

    def test_unknown_principal(self):
        self.refuse("PRINCIPAL_UNKNOWN", self.adapter().open, "cred-mallory", {"prompt": "q"})

    def test_unknown_purpose(self):
        s, _ = self.session()
        self.refuse("PURPOSE_UNKNOWN", s._check, self.c1, "train")

    def test_unknown_grant(self):
        del self.reg.grants[("alice", "S1")]
        s, res = self.session()
        self.assertEqual(s.dag.export()[-1]["code"], "GRANT_UNKNOWN")

    def test_registry_unavailable_open(self):
        self.reg.available = False
        self.refuse("REGISTRY_UNAVAILABLE", self.adapter().open, "cred-alice", {"prompt": "q"})

    def test_registry_unavailable_before_egress(self):
        s, res = self.session()
        hyd = self.hydrate(s, res)
        self.reg.available = False
        self.refuse("REGISTRY_UNAVAILABLE", s.generator, self.ctx(s, hyd))

    def test_purpose_hydrate_missing(self):
        self.reg.grants[("alice", "S1")] = grant(purposes=("read",))
        s, res = self.session()
        self.refuse("PURPOSE_NOT_GRANTED", self.hydrate, s, res)

    def test_purpose_egress_missing(self):
        self.reg.grants[("alice", "S1")] = grant(purposes=("read", "hydrate"))
        s, res = self.session()
        hyd = self.hydrate(s, res)
        self.refuse("PURPOSE_NOT_GRANTED", s.generator, self.ctx(s, hyd))

    def test_private_refused_in_public_default_store_even_for_owner(self):
        s, res = self.session([self.c2])
        self.assertEqual(res["handles"], [])
        self.assertEqual(s.dag.export()[-1]["code"], "PRIVATE_SOURCE_IN_PUBLIC_STORE")

    def test_private_to_external_refused(self):
        s, res = self.session([self.c2], kind="SCOPED_PRIVATE", dest=self.ext)
        hyd = self.hydrate(s, res)
        self.refuse("PRIVATE_TO_EXTERNAL", s.generator, self.ctx(s, hyd))

    def test_transfer_not_admitted(self):
        self.reg.transfers = set()
        s, res = self.session()
        hyd = self.hydrate(s, res)
        self.refuse("TRANSFER_NOT_ADMITTED", s.generator, self.ctx(s, hyd))

    def test_hydrator_principal_from_request_mismatch(self):
        s, res = self.session()
        self.refuse("HYDRATION_PRINCIPAL_MISMATCH", self.hydrate, s, res, pid="bob")
        s2, res2 = self.session()
        self.refuse("HYDRATION_PRINCIPAL_MISMATCH", self.hydrate, s2, res2, tid="t2")

    def test_hydrator_policy_mismatch(self):
        s, res = self.session()
        self.refuse("HYDRATION_POLICY_MISMATCH", self.hydrate, s, res, pol="9" * 40)

    def test_context_tamper(self):
        s, res = self.session()
        hyd = self.hydrate(s, res)
        ctx = self.ctx(s, hyd)
        ctx["evidence"] = [dict(ctx["evidence"][0], content="injected")]
        self.refuse("CONTEXT_DIGEST_MISMATCH", s.generator, ctx)

    def test_context_request_mismatch(self):
        s, res = self.session()
        hyd = self.hydrate(s, res)
        self.refuse("CONTEXT_REQUEST_MISMATCH", s.generator,
                    {"request_id": "other", "evidence": hyd["documents"]})

    def test_context_extra_evidence(self):
        s, res = self.session()
        hyd = self.hydrate(s, res)
        ev = hyd["documents"] + [{"node_id": "c9", "sha256": text_sha256("x"), "content": "x"}]
        self.refuse("CONTEXT_EVIDENCE_SET_MISMATCH", s.generator,
                    {"request_id": s.request_id, "evidence": ev})

    def test_generator_before_hydration(self):
        s, res = self.session()
        self.refuse("SESSION_STATE", s.generator, {"request_id": s.request_id, "evidence": []})


class Authority(Base):
    def test_request_authority_fields_refused(self):
        for f in ("human_approval", "signature_status", "grant_ids", "grant_id",
                  "tenant_id", "principal_id", "policy_epoch", "visibility"):
            with self.subTest(f=f):
                self.refuse("UNTRUSTED_AUTHORITY_FIELD", self.adapter().open, "cred-alice",
                            {"prompt": "q", f: True})

    def test_unknown_request_field_refused(self):
        self.refuse("REQUEST_FIELD_UNKNOWN", self.adapter().open, "cred-alice",
                    {"prompt": "q", "extra": 1})

    def test_backend_cannot_set_authority(self):
        b = Backend({"text": "t", "signature_status": "SIGNED", "human_approval": True,
                     "authority": "EXECUTE"})
        s, res = self.session(backend=b)
        out = s.generator(self.ctx(s, self.hydrate(s, res)))
        self.assertEqual(out["signature_status"], "UNSIGNED")
        self.assertIsNone(out["human_approval"])
        self.assertEqual(out["authority"], "PROPOSAL_ONLY")
        self.assertIn("not cryptographic authority", out["authority_note"])

    def test_no_fallback_and_single_use(self):
        b = Backend(fail=True)
        s, res = self.session(backend=b)
        ctx = self.ctx(s, self.hydrate(s, res))
        with self.assertRaises(Refusal) as cm:
            s.generator(ctx)
        self.assertEqual(cm.exception.code, "BACKEND_FAILED_NO_FALLBACK")
        self.assertEqual(b.calls, 1)
        with self.assertRaises(Refusal):
            s.generator(ctx)
        self.assertEqual(b.calls, 1)

    def test_single_use_after_success(self):
        s, res = self.session()
        hyd = self.hydrate(s, res)
        s.generator(self.ctx(s, hyd))
        s.state = "HYDRATED"   # simulate a caller forcing state back
        with self.assertRaises(Refusal) as cm:
            s.generator(self.ctx(s, hyd))
        self.assertEqual(cm.exception.code, "EGRESS_ALREADY_USED")
        self.assertEqual(self.backend.calls, 1)


class DagHashesOnly(Base):
    def test_denied_source_leaves_hashes_only(self):
        self.reg.revoked.add("g2")
        s, res = self.session([self.c1, self.c2], kind="SCOPED_PRIVATE", dest=self.int)
        dag = s.dag.export()
        denied = [n for n in dag if n.get("decision") == "DENIED"]
        self.assertEqual(len(denied), 1)
        self.assertNotIn("body_sha256", denied[0])
        self.assertNotIn("source_digest", denied[0])
        blob = repr(dag)
        self.assertNotIn(BODY_PRV, blob)
        self.assertNotIn(BODY_PUB, blob)
        s.generator(self.ctx(s, self.hydrate(s, res)))
        self.assertNotIn(BODY_PUB, repr(s.dag.export()))

    def test_dag_tamper_detected_and_bodies_forbidden(self):
        s, res = self.session()
        dag = s.dag.export()
        self.assertTrue(verify_dag(dag))
        dag[1]["decision"] = "ADMITTED_EVERYTHING"
        self.assertFalse(verify_dag(dag))
        with self.assertRaises(Refusal):
            s.dag.add("chunk", [], content="x")
        export = s.dag.export()
        export.clear()
        self.assertTrue(s.dag.export())


if __name__ == "__main__":
    unittest.main()
