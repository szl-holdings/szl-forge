#!/usr/bin/env python3
"""test_rbcg1_verify.py -- verdict tests for rbcg1_verify.py, one or more per REFUSED code, including the
v1 transparency-log codes NOT_INCLUDED and INCONSISTENT_STH. SYNTHETIC DATA ONLY."""
import base64
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import rbcg1  # noqa: E402
import rbcg1_log as L  # noqa: E402
import rbcg1_verify as V  # noqa: E402
from synth import synth_rows, crafted_units, BINDINGS, PROTOCOL_SHA  # noqa: E402

S = {}
N_DEP = 8


def b64(b):
    return base64.b64encode(b).decode("ascii")


def resign(env, key, mutate=None, body=None):
    if body is None:
        obj = json.loads(base64.b64decode(env["payload"]))
        mutate(obj)
        body = rbcg1.canonical_json(obj)
    sig = key.sign(rbcg1.pae(env["payloadType"], body))
    return {"payload": b64(body), "payloadType": env["payloadType"],
            "signatures": [{"keyid": key.public.keyid(), "sig": b64(sig)}]}


def chain_for(genv, cal_rows, rows, key, source):
    ch = rbcg1.MonitorChain(genv, cal_rows, key)
    return [ch.add_unit(uid, ur, source)[0] for uid, ur in rbcg1.group_units(rows)]


def eb(envs):
    return [rbcg1.envelope_bytes(e) for e in envs]


def anchored(genv, envs, lkey, pre=(b"protocol",), mid=(), fork_after_issue=False):
    """Log the guarantee and a chain; return (anchor, log)."""
    lg = L.MerkleLog()
    for x in pre:
        lg.append(x)
    gi = lg.append(L.envelope_leaf(genv))
    s1 = L.sign_sth(lg, lkey, "2026-11-03T00:00:01Z")
    for x in mid:
        lg.append(x)
    hi = None
    for e in envs:
        hi = lg.append(L.envelope_leaf(e))
    s2 = L.sign_sth(lg, lkey, "2026-11-03T00:00:02Z")
    return L.make_anchor(lg, gi, s1, hi, s2), lg


def setUpModule():
    S["key"], S["other"] = rbcg1.SigningKey.generate(), rbcg1.SigningKey.generate()
    S["lkey"], S["lother"] = rbcg1.SigningKey.generate(), rbcg1.SigningKey.generate()
    S["pem"], S["lpem"] = S["key"].public.pem(), S["lkey"].public.pem()
    S["cal"], _ = synth_rows(3, 120)
    S["genv"], S["ginfo"] = rbcg1.issue_guarantee(S["cal"], key=S["key"], bindings=BINDINGS, seed_family="synth-cal",
                                                  env_profile="synthetic", log_id=S["lkey"].public.keyid(),
                                                  protocol_sha256=PROTOCOL_SHA, issued_at_utc="2026-11-03T00:00:00Z")
    S["g"] = rbcg1.envelope_bytes(S["genv"])
    S["src"] = "synth-dep-" + S["ginfo"]["digest"][:16]
    S["dep"], S["dep_lab"] = synth_rows(4, N_DEP, prefix=S["src"], with_label=False)
    S["chain"] = chain_for(S["genv"], S["cal"], S["dep"], S["key"], S["src"])
    S["anchor"], S["log"] = anchored(S["genv"], S["chain"], S["lkey"])


def run(g=None, pk="__default__", cal="__default__", chain=None, dep=None, labels=None, policy=None,
        anchor=None, lpk="__default__"):
    return V.verify(S["g"] if g is None else g, S["pem"] if pk == "__default__" else pk,
                    S["cal"] if cal == "__default__" else cal, chain, dep, labels, policy, anchor,
                    (S["lpem"] if anchor is not None else None) if lpk == "__default__" else lpk)


def run_chain(chain_envs=None, dep=None, labels=None, policy=None, g=None, anchor="__default__", lpk="__default__"):
    envs = S["chain"] if chain_envs is None else chain_envs
    if anchor == "__default__":
        anchor = S["anchor"] if chain_envs is None else (
            anchored(S["genv"] if g is None else json.loads(g), envs, S["lkey"])[0])
    return run(g=g, chain=eb(envs), dep=S["dep"] if dep is None else dep, labels=labels, policy=policy,
               anchor=anchor, lpk=lpk)


class TestValid(unittest.TestCase):
    def test_issuance_only(self):
        r = run()
        self.assertEqual(r["verdict"], "VALID", r["detail"])
        self.assertEqual(r["tau_star_c"], S["ginfo"]["payload"]["procedure"]["tau_star_c"])

    def test_issuance_with_chainless_anchor(self):
        a = dict(S["anchor"])
        a.update({"sth_final": None, "head_inclusion": None, "consistency": None})
        r = run(anchor=a)
        self.assertEqual(r["verdict"], "VALID", r["detail"])
        self.assertEqual(r["anchor"]["guarantee_index"], 1)

    def test_in_distribution_chain_with_labels(self):
        r = run_chain(labels=S["dep_lab"])
        self.assertEqual(r["verdict"], "VALID", r["detail"])
        self.assertEqual(r["n_batches"], N_DEP)
        self.assertEqual([t["verdict"] for t in r["trace"]], ["VALID"] * N_DEP)
        self.assertIsNone(r["alarm_batch"])
        self.assertIsNone(r["violation_batch"])
        self.assertEqual(r["anchor"]["head_index"], N_DEP + 1)
        self.assertEqual(r["audit"]["n_scenes_audited"], N_DEP)

    def test_anchor_from_cli_bytes(self):
        r = run_chain(anchor=rbcg1.canonical_json(S["anchor"]))
        self.assertEqual(r["verdict"], "VALID", r["detail"])


class TestMissingInput(unittest.TestCase):
    def test_missing(self):
        cases = [run(pk=None), run(cal=None), run(g=b""), run(chain=eb(S["chain"])), run(dep=S["dep"]),
                 run(labels=S["dep_lab"]), run_chain(anchor=None), run_chain(lpk=None)]
        for r in cases:
            self.assertEqual(r["verdict"], "REFUSED:MISSING_INPUT", r["detail"])
        a = dict(S["anchor"])
        a["consistency"] = None
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:MISSING_INPUT")
        self.assertEqual(run_chain(anchor=None, policy={"require_anchor": False})["verdict"], "VALID")


class TestSchema(unittest.TestCase):
    def _env(self):
        return json.loads(S["g"])

    def test_envelope_extra_missing_and_multi_sig(self):
        e = self._env()
        e["verified"] = True
        self.assertEqual(run(g=json.dumps(e))["verdict"], "REFUSED:SCHEMA")
        e = self._env()
        del e["payloadType"]
        self.assertEqual(run(g=json.dumps(e))["verdict"], "REFUSED:SCHEMA")
        e = self._env()
        e["signatures"] = e["signatures"] * 2
        self.assertEqual(run(g=json.dumps(e))["verdict"], "REFUSED:SCHEMA")
        e = self._env()
        e["payload"] = e["payload"][:-2] + "!!"
        self.assertEqual(run(g=json.dumps(e))["verdict"], "REFUSED:SCHEMA")

    def test_payload_float_nan_dupes_noncanonical_extra(self):
        k = S["key"]
        canon = rbcg1.canonical_json(json.loads(base64.b64decode(S["genv"]["payload"])))

        def setf(o):
            o["procedure"]["alpha_e6"] = 100000.0

        def extra(o):
            o["note"] = "x"
        envs = [resign(S["genv"], k, setf), resign(S["genv"], k, extra),
                resign(S["genv"], k, body=canon.replace(b'"alpha_e6":100000', b'"alpha_e6":NaN')),
                resign(S["genv"], k, body=canon.replace(b'"alpha_e6":100000', b'"alpha_e6":100000,"alpha_e6":100000')),
                resign(S["genv"], k, body=json.dumps(json.loads(canon), sort_keys=True).encode())]
        for e in envs:
            self.assertEqual(run(g=rbcg1.envelope_bytes(e))["verdict"], "REFUSED:SCHEMA")

    def test_rows_and_labels(self):
        bad = [dict(r) for r in S["cal"]]
        bad[0]["extra"] = 1
        self.assertEqual(run(cal=bad)["verdict"], "REFUSED:SCHEMA")
        bad = [dict(r) for r in S["cal"]]
        bad[0]["pmax_e12"] = 0.5
        self.assertEqual(run(cal=bad)["verdict"], "REFUSED:SCHEMA")
        bad = [dict(r) for r in S["cal"]]
        bad[1]["cid"] = bad[0]["cid"]
        self.assertEqual(run(cal=bad)["verdict"], "REFUSED:SCHEMA")
        lab = dict(S["dep_lab"])
        lab["nope/U9/C1"] = "none"
        self.assertEqual(run_chain(labels=lab)["verdict"], "REFUSED:SCHEMA")
        self.assertEqual(run(policy={"unknown": 1})["verdict"], "REFUSED:SCHEMA")

    def test_loads_strict(self):
        for raw in ('{"a":1,"a":2}', '{"a":1.5}', '{"a":NaN}', '[Infinity]', b"\xff\xfe", "{"):
            with self.assertRaises(V.Refuse):
                V.loads_strict(raw, "x")

    def test_anchor_and_sth_schema(self):
        a = json.loads(rbcg1.canonical_json(S["anchor"]))
        a["extra"] = 1
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:SCHEMA")
        a = json.loads(rbcg1.canonical_json(S["anchor"]))
        a["head_inclusion"]["path"] = ["zz" * 32]
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:SCHEMA")
        a = json.loads(rbcg1.canonical_json(S["anchor"]))

        def ex(o):
            o["note"] = "x"
        a["sth_final"] = resign(a["sth_final"], S["lkey"], ex)
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:SCHEMA")


class TestBadSignature(unittest.TestCase):
    def test_tampered_signature(self):
        e = json.loads(S["g"])
        sig = bytearray(base64.b64decode(e["signatures"][0]["sig"]))
        sig[-1] ^= 1
        e["signatures"][0]["sig"] = b64(bytes(sig))
        self.assertIn(run(g=json.dumps(e))["verdict"], ("REFUSED:BAD_SIGNATURE",))

    def test_wrong_key_and_swapped_payload(self):
        self.assertEqual(run(pk=S["other"].public.pem())["verdict"], "REFUSED:BAD_SIGNATURE")
        e = json.loads(S["g"])
        e["signatures"][0]["keyid"] = S["other"].public.keyid()
        self.assertEqual(run(g=json.dumps(e), pk=S["other"].public.pem())["verdict"], "REFUSED:BAD_SIGNATURE")

    def test_monitor_signed_by_other_key(self):
        envs = list(S["chain"])
        envs[2] = resign(envs[2], S["other"], lambda o: None)
        self.assertEqual(run_chain(chain_envs=envs)["verdict"], "REFUSED:BAD_SIGNATURE")

    def test_sth_signed_by_other_log_key(self):
        a, _ = anchored(S["genv"], S["chain"], S["lother"])
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:BAD_SIGNATURE")


class TestHashMismatch(unittest.TestCase):
    def test_calibration_rows_changed(self):
        bad = [dict(r) for r in S["cal"]]
        bad[5]["pred"] = "none" if bad[5]["pred"] != "none" else "opioid"
        self.assertEqual(run(cal=bad)["verdict"], "REFUSED:HASH_MISMATCH")

    def test_code_binding_and_protocol_pins(self):
        self.assertEqual(run(policy={"trusted_rbcg_sha256": ["0" * 64]})["verdict"], "REFUSED:HASH_MISMATCH")
        self.assertEqual(run(policy={"expected_bindings": {"model_sha256": "9" * 64}})["verdict"],
                         "REFUSED:HASH_MISMATCH")
        self.assertEqual(run(policy={"expected_protocol_sha256": "1" * 64})["verdict"], "REFUSED:HASH_MISMATCH")
        self.assertEqual(run(policy={"expected_protocol_sha256": PROTOCOL_SHA})["verdict"], "VALID")

    def test_deployment_row_changed(self):
        dep = [dict(r) for r in S["dep"]]
        dep[30]["pmax_e12"] -= 1
        self.assertEqual(run_chain(dep=dep)["verdict"], "REFUSED:HASH_MISMATCH")


class TestRecompute(unittest.TestCase):
    def _claimfix(self, o):
        p, c, m = o["procedure"], o["calibration"], o["monitor"]
        o["claim"] = rbcg1.claim_text(o["bindings"]["model_id"], p["tau_star_c"], p["selector"]["ood_floor_c"],
                                      p["alpha_e6"], p["delta_e6"], c["n_units"], c["n_rows"], c["seed_family"],
                                      m["primary"]["delta_s_e6"], m["secondary"]["delta_s_e6"])

    def test_lied_tau_star(self):
        def lie(o):
            o["procedure"]["tau_star_c"] -= 5
            self._claimfix(o)
        e = resign(S["genv"], S["key"], lie)
        self.assertEqual(run(g=rbcg1.envelope_bytes(e))["verdict"], "REFUSED:RECOMPUTE_MISMATCH")

    def test_lied_table_theta_and_secondary(self):
        for f in (lambda o: o["procedure"].__setitem__("table_sha256", "0" * 64),
                  lambda o: o["monitor"]["primary"].__setitem__("theta_seed", "1" * 64),
                  lambda o: o["procedure"].__setitem__("secondary_tau_star_c", 90),
                  lambda o: o["procedure"].__setitem__("bet_c_e3", 500)):
            e = resign(S["genv"], S["key"], f)
            self.assertEqual(run(g=rbcg1.envelope_bytes(e))["verdict"], "REFUSED:RECOMPUTE_MISMATCH")

    def test_monitor_values(self):
        for f in (lambda o: o["primary"].__setitem__("p_e12", o["primary"]["p_e12"] + 1000),
                  lambda o: o["secondary"].__setitem__("log_m_e9", o["secondary"]["log_m_e9"] + 1000),
                  lambda o: o["primary"].__setitem__("score", o["primary"]["score"] + 1),
                  lambda o: o.__setitem__("alarm", True)):
            envs = list(S["chain"])
            envs[3] = resign(envs[3], S["key"], f)
            self.assertEqual(run_chain(chain_envs=envs)["verdict"], "REFUSED:RECOMPUTE_MISMATCH")


class TestClaim(unittest.TestCase):
    def test_alpha_delta_and_policy(self):
        self.assertEqual(run(policy={"alpha_e6": 50000})["verdict"], "REFUSED:CLAIM_UNSUPPORTED")
        self.assertEqual(run(policy={"delta_max_e6": 50000})["verdict"], "REFUSED:CLAIM_UNSUPPORTED")
        self.assertEqual(run(policy={"delta_s_max_e6": 9999})["verdict"], "REFUSED:CLAIM_UNSUPPORTED")
        self.assertEqual(run(policy={"unit_rows_max": 30})["verdict"], "REFUSED:CLAIM_UNSUPPORTED")

    def test_claim_grid_procedure_monitor_and_flags(self):
        muts = [lambda o: o["procedure"].__setitem__("grid_c", list(reversed(o["procedure"]["grid_c"]))),
                lambda o: o["procedure"].__setitem__("id", "other"),
                lambda o: o["procedure"].__setitem__("loss", "row_level"),
                lambda o: o["procedure"].__setitem__("p_value", "binom_cdf"),
                lambda o: o["monitor"]["primary"].__setitem__("score_id", "unit_mean_one_minus_pmax_e12_v0"),
                lambda o: o["monitor"].__setitem__("combination", "mixture"),
                lambda o: o["monitor"]["secondary"].__setitem__("delta_s_e6", 5000),
                lambda o: o["audit"].__setitem__("id", "clopper_pearson_union"),
                lambda o: o["anchoring"].__setitem__("leaf_rule", "sha256_raw"),
                lambda o: o.__setitem__("synthetic_only", False),
                lambda o: o.__setitem__("claim", o["claim"].replace("at most 0.1", "at most 0.05"))]
        for f in muts:
            e = resign(S["genv"], S["key"], f)
            self.assertEqual(run(g=rbcg1.envelope_bytes(e))["verdict"], "REFUSED:CLAIM_UNSUPPORTED")


class TestChain(unittest.TestCase):
    def test_prev_seq_binding_count(self):
        envs = list(S["chain"])
        envs[2], envs[3] = envs[3], envs[2]
        self.assertEqual(run_chain(chain_envs=envs)["verdict"], "REFUSED:CHAIN_BROKEN")
        envs = list(S["chain"])
        envs[4] = resign(envs[4], S["key"], lambda o: o.__setitem__("seq", 9))
        self.assertEqual(run_chain(chain_envs=envs)["verdict"], "REFUSED:CHAIN_BROKEN")
        self.assertEqual(run_chain(chain_envs=S["chain"][:-1])["verdict"], "REFUSED:CHAIN_BROKEN")

    def test_prev_link_alone_is_checked(self):
        # correct seq and guarantee binding, only prev_sha256 broken: only the hash-chain link can catch it
        envs = list(S["chain"])
        envs[4] = resign(envs[4], S["key"], lambda o: o.__setitem__("prev_sha256", "3" * 64))
        r = run_chain(chain_envs=envs)
        self.assertEqual(r["verdict"], "REFUSED:CHAIN_BROKEN", r["detail"])
        self.assertIn("prev_sha256", r["detail"])

    def test_chain_bound_to_other_guarantee(self):
        envs = list(S["chain"])
        envs[0] = resign(envs[0], S["key"], lambda o: o.__setitem__("guarantee_sha256", "2" * 64))
        self.assertEqual(run_chain(chain_envs=envs)["verdict"], "REFUSED:CHAIN_BROKEN")


class TestOrderLeakage(unittest.TestCase):
    def test_order_commitment(self):
        dep, lab = synth_rows(4, N_DEP, prefix="no-commit", with_label=False)
        envs = chain_for(S["genv"], S["cal"], dep, S["key"], "no-commit")
        a, _ = anchored(S["genv"], envs, S["lkey"])
        self.assertEqual(run(chain=eb(envs), dep=dep, anchor=a)["verdict"], "REFUSED:ORDER")
        self.assertEqual(run(chain=eb(envs), dep=dep, anchor=a, policy={"require_order_commitment": False})["verdict"],
                         "VALID")

    def test_leakage(self):
        leak = [dict(r) for r in S["cal"][:24 * N_DEP]]
        for r in leak:
            del r["label"]
        envs = chain_for(S["genv"], S["cal"], leak, S["key"], S["src"])
        a, _ = anchored(S["genv"], envs, S["lkey"])
        self.assertEqual(run(chain=eb(envs), dep=leak, anchor=a)["verdict"], "REFUSED:LEAKAGE")


class TestAnchor(unittest.TestCase):
    def _a(self):
        return json.loads(rbcg1.canonical_json(S["anchor"]))

    def test_guarantee_not_included(self):
        a = self._a()
        a["guarantee_inclusion"]["leaf_index"] = 0
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:NOT_INCLUDED")
        other_g, _ = rbcg1.issue_guarantee(S["cal"], key=S["key"], bindings=BINDINGS, seed_family="synth-cal",
                                           env_profile="synthetic", log_id=S["lkey"].public.keyid(),
                                           protocol_sha256=PROTOCOL_SHA, issued_at_utc="2026-11-03T09:09:09Z")
        a2, _ = anchored(other_g, S["chain"], S["lkey"])            # log holds a different guarantee
        self.assertEqual(run_chain(anchor=a2)["verdict"], "REFUSED:NOT_INCLUDED")

    def test_head_not_included(self):
        a = self._a()
        a["head_inclusion"]["leaf_index"] -= 1
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:NOT_INCLUDED")
        a2, _ = anchored(S["genv"], S["chain"][:-1], S["lkey"])      # chain head never logged
        self.assertEqual(run_chain(anchor=a2)["verdict"], "REFUSED:NOT_INCLUDED")

    def test_wrong_log(self):
        a, _ = anchored(S["genv"], S["chain"], S["lother"])
        r = run_chain(anchor=a, lpk=S["lother"].public.pem())
        self.assertEqual(r["verdict"], "REFUSED:NOT_INCLUDED")

    def test_inconsistent_sth_fork(self):
        # the same log key signs two histories: the issuance STH from one, the final STH from a fork
        a1, _ = anchored(S["genv"], S["chain"], S["lkey"], pre=(b"protocol",))
        a2, _ = anchored(S["genv"], S["chain"], S["lkey"], pre=(b"protocol-forked",))
        a = json.loads(rbcg1.canonical_json(a1))
        b = json.loads(rbcg1.canonical_json(a2))
        a["sth_final"], a["head_inclusion"] = b["sth_final"], b["head_inclusion"]
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:INCONSISTENT_STH")

    def test_inconsistent_sizes_and_tampered_consistency(self):
        a = self._a()
        a["consistency"]["first_size"] += 1
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:INCONSISTENT_STH")
        a = self._a()
        p = a["consistency"]["path"]
        p[0] = ("0" if p[0][0] != "0" else "1") + p[0][1:]
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:INCONSISTENT_STH")
        a = self._a()
        a["sth_issue"], a["sth_final"] = a["sth_final"], a["sth_issue"]
        self.assertIn(run_chain(anchor=a)["verdict"], ("REFUSED:INCONSISTENT_STH", "REFUSED:NOT_INCLUDED"))

    def test_head_logged_before_issuance_sth(self):
        lg = L.MerkleLog()
        lg.append(b"protocol")
        gi = lg.append(L.envelope_leaf(S["genv"]))
        hi = None
        for e in S["chain"]:
            hi = lg.append(L.envelope_leaf(e))
        s1 = L.sign_sth(lg, S["lkey"], "t1")
        s2 = L.sign_sth(lg, S["lkey"], "t2")
        a = L.make_anchor(lg, gi, s1, hi, s2)
        self.assertEqual(run_chain(anchor=a)["verdict"], "REFUSED:ORDER")


class TestShiftAndViolation(unittest.TestCase):
    def _go(self, dep, lab, src=None):
        src = src or S["src"]
        envs = chain_for(S["genv"], S["cal"], dep, S["key"], src)
        a, _ = anchored(S["genv"], envs, S["lkey"])
        return run(chain=eb(envs), dep=dep, labels=lab, anchor=a)

    def test_shift_alarm_primary(self):
        dep, lab = synth_rows(31, 20, prefix=S["src"] + "-ood", ood_rate=0.9, with_label=False)
        r = self._go(dep, lab)
        self.assertEqual(r["verdict"], "REFUSED:SHIFT_ALARM", r["detail"])
        self.assertIn(r["alarm_component"], ("primary", "both"))

    def test_shift_alarm_secondary(self):
        dep, lab = synth_rows(32, 30, prefix=S["src"] + "-conf", b_a=2.0, b_b=4.0, with_label=False)
        r = self._go(dep, lab)
        self.assertEqual(r["verdict"], "REFUSED:SHIFT_ALARM", r["detail"])
        self.assertIn(r["alarm_component"], ("secondary", "both"))

    def test_violation_detected_by_labels(self):
        dep, lab = crafted_units(S["src"] + "-viol", 16, 0.97, 7, ood=False)
        for i, r in enumerate(dep):
            r["ood"] = (i % 24) < 4
        r = self._go(dep, lab)
        self.assertEqual(r["verdict"], "VIOLATED", r["detail"])
        self.assertIsNone(r["alarm_batch"])
        self.assertGreater(r["audit"]["cs_lower_bound_mean_loss_e9"], 0)

    def test_small_excess_is_not_a_violation(self):
        dep, lab = crafted_units(S["src"] + "-small", 8, 0.97, 3, ood=False)
        for i, r in enumerate(dep):
            r["ood"] = (i % 24) < 4
        r = self._go(dep, lab)
        self.assertEqual(r["verdict"], "VALID", r["detail"])

    def test_partial_labels_skip_scenes(self):
        dep, lab = crafted_units(S["src"] + "-part", 16, 0.97, 7, ood=False)
        for i, r in enumerate(dep):
            r["ood"] = (i % 24) < 4
        lab = {k: v for k, v in lab.items() if not k.endswith("/C00")}
        r = self._go(dep, lab)
        self.assertEqual(r["verdict"], "VALID", r["detail"])
        self.assertIsNone(r["audit"])

    def test_precedence_follows_first_event(self):
        dep1, lab1 = crafted_units(S["src"] + "-p", 16, 0.97, 7, ood=False)
        dep2, _ = synth_rows(31, 20, prefix=S["src"] + "-q", ood_rate=0.9, with_label=False)
        for i, r in enumerate(dep1):
            r["ood"] = (i % 24) < 4
        r = self._go(dep2 + dep1, lab1)
        self.assertEqual(r["verdict"], "REFUSED:SHIFT_ALARM")
        self.assertTrue(r["alarm_batch"] < (r["violation_batch"] or 999))


class TestFastPathEqualsDirect(unittest.TestCase):
    def test_issuer_fast_path_equals_verifier_direct_path(self):
        for seed, n in ((5, 40), (6, 90), (7, 130)):
            rows, _ = synth_rows(seed, n, ood_unit_rate=0.4)
            units = rbcg1.group_units(rows)
            for d in (100000, 50000):
                fast = rbcg1.ltt_table(units, rbcg1.GRID_C_DEFAULT, 100000, d, 81)
                direct = V.recompute_table(units, list(rbcg1.GRID_C_DEFAULT), 100000, d, 81, 24, 750)
                self.assertEqual(rbcg1.canonical_json(fast), rbcg1.canonical_json(direct))


class TestFailClosed(unittest.TestCase):
    def test_internal_errors_refuse(self):
        orig = V.recompute_table
        try:
            def boom(*a, **k):
                raise RuntimeError("boom")
            V.recompute_table = boom
            self.assertEqual(run()["verdict"], "REFUSED:INTERNAL")
        finally:
            V.recompute_table = orig

    def test_garbage_never_raises_never_valid(self):
        rng = random.Random(1)
        for g in (None, b"", b"{}", b"[]", "x" * 50, 12, {"a": 1}):
            for pk in (None, "", "zz", S["pem"]):
                r = V.verify(g, pk, S["cal"])
                self.assertNotEqual(r["verdict"], "VALID")
        for _ in range(20):
            junk = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 200)))
            self.assertNotEqual(V.verify(junk, S["pem"], S["cal"])["verdict"], "VALID")
            self.assertNotEqual(run_chain(anchor=junk)["verdict"], "VALID")

    def test_byte_flip_fuzz(self):
        rng = random.Random(2)
        g = bytearray(S["g"])
        for _ in range(25):
            h = bytearray(g)
            i = rng.randrange(len(h))
            h[i] = (h[i] + 1 + rng.randrange(255)) % 256
            self.assertNotEqual(V.verify(bytes(h), S["pem"], S["cal"])["verdict"], "VALID")
        a = bytearray(rbcg1.canonical_json(S["anchor"]))
        for _ in range(25):
            h = bytearray(a)
            i = rng.randrange(len(h))
            h[i] = (h[i] + 1 + rng.randrange(255)) % 256
            self.assertNotEqual(run_chain(anchor=bytes(h))["verdict"], "VALID")


class TestCLI(unittest.TestCase):
    def test_cli_exit_codes(self):
        d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR") or None)
        try:
            def w(name, data):
                p = os.path.join(d, name)
                with open(p, "wb") as fh:
                    fh.write(data if isinstance(data, bytes) else data.encode())
                return p
            g = w("g.json", S["g"])
            pk = w("pk.pem", S["pem"])
            lpk = w("lpk.pem", S["lpem"])
            cal = w("cal.jsonl", b"\n".join(rbcg1.canonical_json(r) for r in S["cal"]))
            dep = w("dep.jsonl", b"\n".join(rbcg1.canonical_json(r) for r in S["dep"]))
            ch = w("chain.jsonl", b"\n".join(eb(S["chain"])))
            an = w("anchor.json", rbcg1.canonical_json(S["anchor"]))
            exe = [sys.executable, "-B", os.path.join(PKG, "rbcg1_verify.py")]
            base = ["--guarantee", g, "--pubkey", pk, "--cal-rows", cal]
            p = subprocess.run(exe + base, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            full = base + ["--monitor-chain", ch, "--deploy-rows", dep, "--anchor", an, "--log-pubkey", lpk]
            p = subprocess.run(exe + full, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual(json.loads(p.stdout)["verdict"], "VALID")
            p = subprocess.run(exe + base + ["--monitor-chain", ch, "--deploy-rows", dep], capture_output=True,
                               text=True)
            self.assertEqual(p.returncode, 2)
            self.assertEqual(json.loads(p.stdout)["verdict"], "REFUSED:MISSING_INPUT")
            p = subprocess.run(exe + ["--guarantee", os.path.join(d, "nope"), "--pubkey", pk, "--cal-rows", cal],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
