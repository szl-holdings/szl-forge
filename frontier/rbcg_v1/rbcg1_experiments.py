#!/usr/bin/env python3
# RBCG v1 prototype -- SZL Holdings. Licence: UNDECIDED (owner choice). See NOTICE.
"""rbcg1_experiments.py -- pre-registered E1 (validity), E2 (shift) and E3 (re-verification of the stored
v0 E3 rows) for RBCG v1 on the SZL TRIAGE-X in-silico stack, with v0 run side by side on the SAME
fresh data. SYNTHETIC DATA ONLY. NOT FOR CLINICAL USE.

Pre-registration discipline: every command except `pilot` refuses to run without a signed protocol
receipt (rbcg1_protocol.py) whose code hashes match the files that are running. All evaluation seed
families embed the first 16 hex of the protocol digest, so no evaluation datum can exist before the
protocol was signed. Families of v0 (szl-rbcg-v0-*) and the frozen splits (szl-triagex-v0-*) are
refused. Each rep signs its v1 and v0 guarantees BEFORE its deployment data exist; deployment families
embed both guarantee digests. v1 receipts are appended to a per-shard RFC 9162-style log whose leaf 0
is the protocol receipt.

  pilot  --stack S --out O --reps K               (design only; families szl-rbcg-v1-pilot-*)
  shard  --stack S --out O --protocol P --shard i --nshards 3
  e3     --out O --protocol P --e3dir D           (re-verifies stored v0 E3 rows; no model run)
  merge  --out O --protocol P --nshards 3 --v0-summary F
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "v0_ref"))
import rbcg1  # noqa: E402
import rbcg1_log  # noqa: E402
import rbcg1_verify  # noqa: E402
import rbcg1_protocol  # noqa: E402
import rbcg as rbcg0  # noqa: E402  (v0 reference, exact v0 bytes; sha pinned below)
import rbcg_verify as rbcg0_verify  # noqa: E402

V0_REF_SHA256 = {"rbcg.py": "1a34ee7c8e73b19835dcb6a6ccc9fd4a954799c70cdf54f73757fb00870f6523", "rbcg_verify.py": "d76b39e30682c2b5ea413f6575bac4e3c5421103005e46112a9974ecf80bdffa"}
ALPHA_E6, D10, D05, DELTA_AUDIT = 100000, 100000, 50000, 50000
GRID, FLOOR = rbcg1.GRID_C_DEFAULT, rbcg1.OOD_FLOOR_C_DEFAULT
CASUALTIES_PER_SCENE = 24
SPENT_PREFIXES = ("szl-rbcg-v0-", "szl-triagex-v0-")
LOG_THRESHOLDS = {"100": math.log(100.0), "133": math.log(1e6 / 7500), "400": math.log(1e6 / 2500)}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def dump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, sort_keys=True, separators=(",", ":"))


def fam_ok(f):
    if any(f.startswith(p) for p in SPENT_PREFIXES):
        raise SystemExit(f"REFUSED: seed family {f!r} belongs to v0 or to the spent frozen splits")
    return f


def check_v0_ref():
    for name, want in V0_REF_SHA256.items():
        got = rbcg0.sha256_file(os.path.join(HERE, "v0_ref", name))
        if got != want:
            raise SystemExit(f"REFUSED: v0_ref/{name} sha256 {got} != pinned {want}")


def load_protocol(path):
    """(protocol payload, protocol digest). Refuses unless the signature verifies and the bound code
    hashes equal the files that are running."""
    with open(path, "rb") as fh:
        raw = fh.read()
    with open(os.path.join(os.path.dirname(os.path.abspath(path)), "protocol_public_key.pem"), "r",
              encoding="ascii") as fh:
        pem = fh.read()
    res = rbcg1_protocol.verify_protocol(raw, pem, HERE)
    if res["verdict"] != "VALID":
        raise SystemExit(f"REFUSED: protocol receipt {res['verdict']}: {res['detail']}")
    return res["payload"], res["protocol_sha256"]


def accept2(r, tau_c, tau_ood_c):
    return r["inf"] and r["pmax_e12"] >= (tau_ood_c if r["ood"] else tau_c) * 10 ** 10


def realized(rows, labels, th):
    if th is None:
        return None
    n_acc = k = 0
    for r in rows:
        if accept2(r, th[0], th[1]):
            n_acc += 1
            k += r["pred"] != labels[r["cid"]]
    return {"n_acc": n_acc, "k": k, "coverage": n_acc / len(rows), "sel_err": (k / n_acc) if n_acc else None}


def audit_first_v0(units, labels, th):
    """v0 rule: Clopper-Pearson via BinomSF at delta_audit * 6/(pi^2 j^2) per look (union bound)."""
    if th is None:
        return None
    n = k = 0
    for j, (_, rows) in enumerate(units, 1):
        for r in rows:
            if accept2(r, th[0], th[1]):
                n += 1
                k += r["pred"] != labels[r["cid"]]
        d = DELTA_AUDIT / 1e6 * 6.0 / (math.pi ** 2 * j * j)
        if n > 0 and rbcg0.binom_sf(k, n, ALPHA_E6 / 1e6) < d:
            return j
    return None


def audit_first_v1(units, labels, th, c_e3):
    """v1 rule: anytime-valid betting CS on the scene-level linearized loss (VIOLATED when the lower
    bound on E[L] exceeds 0). th = (tau_c, tau_ood_c) with tau_ood_c = max(tau_c, FLOOR)."""
    if th is None:
        return None
    a = rbcg1.BettingAudit(ALPHA_E6, DELTA_AUDIT, c_e3)
    for j, (_, rows) in enumerate(units, 1):
        x = 0
        n_acc = e = 0
        for r in rows:
            if accept2(r, th[0], th[1]):
                n_acc += 1
                e += r["pred"] != labels[r["cid"]]
        x = rbcg1.loss_num(n_acc, e, ALPHA_E6, rbcg1.UNIT_ROWS_MAX_DEFAULT) / (rbcg1.UNIT_ROWS_MAX_DEFAULT * 10 ** 6)
        if a.add(x):
            return j
    return None


def nominal_from_table(v0_table):
    """The nominal rule (what the frozen advisory does): least conservative tau with n >= 30 and
    k/n <= alpha, read off v0's row-level (n, k) table."""
    for e in sorted(v0_table, key=lambda e: e["tau_c"]):
        if e["n"] >= rbcg0.MIN_ACCEPTED_NOMINAL and e["k"] * 10 ** 6 <= ALPHA_E6 * e["n"]:
            return e["tau_c"]
    return None


def th_of(t):
    return (t, max(t, FLOOR)) if t else None


def transitions(trace):
    out, last = [], None
    for t in trace:
        if t["verdict"] != last:
            out.append([t["batch"], t["verdict"]])
            last = t["verdict"]
    return out


def first_crossings(logms):
    out = {}
    for k, thr in LOG_THRESHOLDS.items():
        out[k] = next((j for j, v in enumerate(logms, 1) if v >= thr), None)
    return out


# ------------------------------------------------------------------------------------------ pilot
def cmd_pilot(a):
    """DESIGN ONLY, run before the protocol is signed, on families szl-rbcg-v1-pilot-<i:02d> that are
    never reused: scene-level LTT power versus calibration size and truncation c, plus timing."""
    from rbcg1_triagex import Stack
    st = Stack(a.stack)
    out = {"started_utc": utc(), "purpose": "design only (n_cal, c); never reused", "reps": []}
    sizes = [int(x) for x in a.sizes.split(",")]
    for i in range(a.reps):
        fam = fam_ok(f"szl-rbcg-v1-pilot-{i:02d}")
        t0 = time.process_time()
        rows, _, _, _ = st.family(fam, max(sizes), "standard")
        t_gen = time.process_time() - t0
        rec = {"family": fam, "gen_cpu_s_per_scene": t_gen / max(sizes), "by_n": {}}
        for n in sizes:
            sub = rows[: n * CASUALTIES_PER_SCENE]
            for c_e3 in (500, 750):
                cal = rbcg1.calibrate(sub, ALPHA_E6, D10, c_e3=c_e3)
                rec["by_n"][f"{n}_c{c_e3}"] = {"tau_d10": cal["tau_star_c"], "n_rej": cal["n_rejected"],
                                               "hb_tau_d10": cal["secondary_tau_star_c"],
                                               "p_bet_e9_tau99": cal["table"][0]["p_bet_e9"]}
            v0c = rbcg0.calibrate(sub, ALPHA_E6, D10, GRID, FLOOR)
            rec["by_n"][f"{n}_v0"] = {"tau_d10": v0c["tau_star_c"]}
        out["reps"].append(rec)
        print(json.dumps({"family": fam, "by_n": {k: v["tau_d10"] for k, v in rec["by_n"].items()}}), flush=True)
    # timing of one full v1 + v0 rep on a pilot family (no results kept)
    t0 = time.process_time()
    key, lkey, key0 = rbcg1.SigningKey.generate(), rbcg1.SigningKey.generate(), rbcg0.SigningKey.generate()
    fam = fam_ok("szl-rbcg-v1-pilot-timing")
    rows, _, _, _ = st.family(fam, a.timing_ncal, "standard")
    t1 = time.process_time()
    cal = rbcg1.calibrate(rows, ALPHA_E6, D10)
    try:
        genv, info = rbcg1.issue_guarantee(rows, key=key, bindings=st.bindings, seed_family=fam,
                                           env_profile="standard", log_id=lkey.public.keyid(),
                                           protocol_sha256="0" * 64, cal=cal)
    except rbcg1.NoGuarantee:
        genv = None
    g0, i0 = rbcg0.issue_guarantee(rows, key=key0, bindings=st.bindings, seed_family=fam, env_profile="standard")
    t2 = time.process_time()
    drows, dlab, _, _ = st.family(fam + "-shf", 80, "shifted", with_label=False)
    t3 = time.process_time()
    timing = {"gen_cal_s": t1 - t0, "ltt_issue_s": t2 - t1, "gen_80_dep_s": t3 - t2}
    if genv is not None:
        lg = rbcg1_log.MerkleLog()
        gi = lg.append(rbcg1_log.envelope_leaf(genv))
        s1 = rbcg1_log.sign_sth(lg, lkey, utc())
        ch = rbcg1.MonitorChain(genv, rows, key)
        envs = []
        for uid, ur in rbcg1.group_units(drows):
            e, _ = ch.add_unit(uid, ur, fam + "-" + info["digest"][:16])
            envs.append(rbcg1.envelope_bytes(e))
            hi = lg.append(rbcg1_log.envelope_leaf(e))
        s2 = rbcg1_log.sign_sth(lg, lkey, utc())
        anc = rbcg1_log.make_anchor(lg, gi, s1, hi, s2)
        t4 = time.process_time()
        vr = rbcg1_verify.verify(rbcg1.envelope_bytes(genv), key.public.pem(), rows, envs, drows, dlab,
                                 {"expected_bindings": st.bindings}, anchor=anc, log_public_key=lkey.public.pem())
        t5 = time.process_time()
        timing.update({"v1_chain80_log_s": t4 - t3, "v1_verify_s": t5 - t4, "v1_verdict": vr["verdict"],
                       "v1_alarm_batch": vr["alarm_batch"], "v1_alarm_component": vr["alarm_component"]})
    t5 = time.process_time()
    ch0 = rbcg0.MonitorChain(g0, rows, key0)
    envs0 = [rbcg0.envelope_bytes(ch0.add_unit(uid, ur, fam + "-" + i0["digest"][:16])[0])
             for uid, ur in rbcg0.group_units(drows)]
    t6 = time.process_time()
    vr0 = rbcg0_verify.verify(rbcg0.envelope_bytes(g0), key0.public.pem(), rows, envs0, drows, dlab,
                              {"expected_bindings": st.bindings})
    t7 = time.process_time()
    timing.update({"v0_chain80_s": t6 - t5, "v0_verify_s": t7 - t6, "v0_verdict": vr0["verdict"],
                   "v0_alarm_batch": vr0["alarm_batch"]})
    out["timing_one_stream"] = timing
    out["finished_utc"] = utc()
    dump(os.path.join(a.out, "pilot_v1.json"), out)
    print(json.dumps(timing, sort_keys=True), flush=True)

# ------------------------------------------------------------------------------------------ shard
def method_thresholds(rec):
    return {"v1_d10": th_of(rec["tau"]["v1_d10"]), "v1_d05": th_of(rec["tau"]["v1_d05"]),
            "v1_hb_d10": th_of(rec["tau"]["v1_hb_d10"]), "v1_d10_n80": th_of(rec["tau"]["v1_d10_n80"]),
            "v0_d10": th_of(rec["tau"]["v0_d10"]), "v0_d05": th_of(rec["tau"]["v0_d05"]),
            "v0_d10_n80": th_of(rec["tau"]["v0_d10_n80"]), "nominal_1d": th_of(rec["tau"]["nominal_1d"]),
            "frozen_tau064": (64, 81)}


def run_rep(st, ctx, r, example_dir=None):
    P16, n_cal, n_dep, c_e3 = ctx["P16"], ctx["n_cal"], ctx["n_dep"], ctx["c_e3"]
    c0 = time.process_time()
    cal_family = fam_ok(f"szl-rbcg-v1-{P16}-cal-{r:04d}")
    cal_rows, _, _, _ = st.family(cal_family, n_cal, "standard")
    units = rbcg1.group_units(cal_rows)
    assert len(units) == n_cal and all(len(u) == CASUALTIES_PER_SCENE for _, u in units)
    pre = cal_rows[: ctx["n_prefix"] * CASUALTIES_PER_SCENE]
    cal10 = rbcg1.calibrate(cal_rows, ALPHA_E6, D10, c_e3=c_e3, units=units)
    cal05 = rbcg1.calibrate(cal_rows, ALPHA_E6, D05, c_e3=c_e3, units=units)
    cal10_80 = rbcg1.calibrate(pre, ALPHA_E6, D10, c_e3=c_e3)
    v0cal = rbcg0.calibrate(cal_rows, ALPHA_E6, D10, GRID, FLOOR)
    v0cal80 = rbcg0.calibrate(pre, ALPHA_E6, D10, GRID, FLOOR)
    rec = {"rep": r, "cal_family": cal_family, "cal_rows_sha256": rbcg1.rows_sha256(cal_rows),
           "n_cal_rows": len(cal_rows), "n_cal_units": len(units),
           "tau": {"v1_d10": cal10["tau_star_c"], "v1_d05": cal05["tau_star_c"],
                   "v1_hb_d10": cal10["secondary_tau_star_c"], "v1_d10_n80": cal10_80["tau_star_c"],
                   "v0_d10": v0cal["tau_star_c"], "v0_d05": rbcg0.ltt_fixed_sequence(v0cal["table"], D05)[0],
                   "v0_d10_n80": v0cal80["tau_star_c"], "nominal_1d": nominal_from_table(v0cal["table"])},
           "v1_p_bet_e9_tau99": cal10["table"][0]["p_bet_e9"],
           "cal_v0_table_at": {str(e["tau_c"]): [e["n"], e["k"]] for e in v0cal["table"]
                               if e["tau_c"] in (64, 70, 75, 80, 85, 90)},
           "cal_scene_err_at064": None, "issued": {}, "issuance_verdict": {}, "streams": {}}
    # scene-level error counts at tau 0.64 for the design effect (v0-comparable)
    cnt = rbcg1.unit_counts_by_tau(units, (64,), FLOOR)[0]
    rec["cal_scene_err_at064"] = [sum(e for _, e in cnt), sum(a for a, _ in cnt),
                                  sum(e * e for _, e in cnt), sum(a * a for a, _ in cnt), sum(a * e for a, e in cnt)]
    # ---- issue v1 and v0 guarantees BEFORE any deployment data exist
    g1 = g0 = None
    try:
        g1, i1 = rbcg1.issue_guarantee(cal_rows, key=ctx["k1"], bindings=st.bindings, seed_family=cal_family,
                                       env_profile="standard", log_id=ctx["lkey"].public.keyid(),
                                       protocol_sha256=ctx["P"], alpha_e6=ALPHA_E6, delta_e6=D10, c_e3=c_e3,
                                       issued_at_utc=utc(), cal=cal10)
        rec["issued"]["v1_d10"] = {"digest": i1["digest"], "utc": i1["payload"]["issued_at_utc"]}
        vr = rbcg1_verify.verify(rbcg1.envelope_bytes(g1), ctx["pem1"], cal_rows, policy=ctx["pol1"])
        rec["issuance_verdict"]["v1_d10"] = vr["verdict"]
        g_idx = ctx["log"].append(rbcg1_log.envelope_leaf(g1))
        sth_issue = rbcg1_log.sign_sth(ctx["log"], ctx["lkey"], utc())
    except rbcg1.NoGuarantee:
        rec["issued"]["v1_d10"] = None
        rec["issuance_verdict"]["v1_d10"] = "NO_GUARANTEE_ISSUED"
    try:
        g0, i0 = rbcg0.issue_guarantee(cal_rows, key=ctx["k0"], bindings=st.bindings, seed_family=cal_family,
                                       env_profile="standard", alpha_e6=ALPHA_E6, delta_e6=D10,
                                       issued_at_utc=utc())
        rec["issued"]["v0_d10"] = {"digest": i0["digest"], "utc": i0["payload"]["issued_at_utc"]}
    except rbcg0.NoGuarantee:
        rec["issued"]["v0_d10"] = None
    d1 = rec["issued"]["v1_d10"]["digest"][:16] if g1 is not None else \
        rbcg1.sha256_hex(("NO_GUARANTEE_V1|" + rec["cal_rows_sha256"]).encode())[:16]
    d0 = rec["issued"]["v0_d10"]["digest"][:16] if g0 is not None else \
        rbcg1.sha256_hex(("NO_GUARANTEE_V0|" + rec["cal_rows_sha256"]).encode())[:16]
    commit = f"{d1}-{d0}"
    rec["order"] = {"commit": commit, "deploy_generation_started_utc": utc()}
    rec["order"]["issued_before_deploy"] = all(v is None or v["utc"] < rec["order"]["deploy_generation_started_utc"]
                                              for v in rec["issued"].values())
    ths = method_thresholds(rec)
    for sname, prof, tag in (("in_dist", "standard", "dep"), ("shifted", "shifted", "shf")):
        fam = fam_ok(f"szl-rbcg-v1-{P16}-{tag}-{r:04d}-{commit}")
        rows, labels, _, _ = st.family(fam, n_dep, prof, with_label=False)
        dunits = rbcg1.group_units(rows)
        srec = {"family": fam, "rows_sha256": rbcg1.rows_sha256(rows), "n_rows": len(rows),
                "realized": {m: realized(rows, labels, th) for m, th in ths.items()},
                "audit_v1_first": {m: audit_first_v1(dunits, labels, ths[m], c_e3)
                                   for m in ("v1_d10", "v0_d10", "frozen_tau064")},
                "audit_v0_first": {m: audit_first_v0(dunits, labels, ths[m])
                                   for m in ("v1_d10", "v0_d10", "frozen_tau064")}}
        if g1 is not None:
            chain = rbcg1.MonitorChain(g1, cal_rows, ctx["k1"])
            menvs, lp, ls = [], [], []
            for uid, ur in dunits:
                env, stp = chain.add_unit(uid, ur, fam)
                menvs.append(rbcg1.envelope_bytes(env))
                lp.append(stp["primary"]["log_m"])
                ls.append(stp["secondary"]["log_m"])
                h_idx = ctx["log"].append(rbcg1_log.envelope_leaf(env))
            sth_final = rbcg1_log.sign_sth(ctx["log"], ctx["lkey"], utc())
            anchor = rbcg1_log.make_anchor(ctx["log"], g_idx, sth_issue, h_idx, sth_final)
            vr = rbcg1_verify.verify(rbcg1.envelope_bytes(g1), ctx["pem1"], cal_rows, menvs, rows, labels,
                                     ctx["pol1"], anchor=anchor, log_public_key=ctx["lpem"])
            srec["v1"] = {"verdict": vr["verdict"], "detail": vr["detail"], "alarm_batch": vr["alarm_batch"],
                          "alarm_component": vr["alarm_component"], "violation_batch": vr["violation_batch"],
                          "n_batches": vr["n_batches"], "transitions": transitions(vr["trace"]),
                          "audit": vr["audit"], "anchor": vr["anchor"],
                          "primary_first": first_crossings(lp), "secondary_first": first_crossings(ls),
                          "max_log_m": [round(max(lp), 4), round(max(ls), 4)]}
            if r == 0:
                srec["v1"]["log_m_primary"] = [round(v, 4) for v in lp]
                srec["v1"]["log_m_secondary"] = [round(v, 4) for v in ls]
            if example_dir is not None:
                write_example(example_dir, sname, g1, ctx, cal_rows, rows, labels, menvs, anchor, vr)
        if g0 is not None:
            ch0 = rbcg0.MonitorChain(g0, cal_rows, ctx["k0"])
            menvs0, l0 = [], []
            for uid, ur in dunits:
                env, stp = ch0.add_unit(uid, ur, fam)
                menvs0.append(rbcg0.envelope_bytes(env))
                l0.append(stp["log_m"])
            vr0 = rbcg0_verify.verify(rbcg0.envelope_bytes(g0), ctx["pem0"], cal_rows, menvs0, rows, labels,
                                      ctx["pol0"])
            srec["v0"] = {"verdict": vr0["verdict"], "alarm_batch": vr0["alarm_batch"],
                          "violation_batch": vr0["violation_batch"], "n_batches": vr0["n_batches"],
                          "transitions": transitions(vr0["trace"]), "conf_first": first_crossings(l0)}
        rec["streams"][sname] = srec
    rec["cpu_s"] = round(time.process_time() - c0, 3)
    return rec


def write_example(d, sname, g1, ctx, cal_rows, rows, labels, menvs, anchor, vr):
    os.makedirs(os.path.join(d, sname), exist_ok=True)
    with open(os.path.join(d, "guarantee.dsse.json"), "wb") as fh:
        fh.write(rbcg1.envelope_bytes(g1))
    with open(os.path.join(d, "cal_rows.jsonl"), "wb") as fh:
        fh.write(b"\n".join(rbcg1.canonical_json(x) for x in cal_rows) + b"\n")
    for name, pem in (("public_key.pem", ctx["pem1"]), ("log_public_key.pem", ctx["lpem"])):
        with open(os.path.join(d, name), "w", encoding="ascii", newline="\n") as fh:
            fh.write(pem)
    with open(os.path.join(d, sname, "deploy_rows.jsonl"), "wb") as fh:
        fh.write(b"\n".join(rbcg1.canonical_json(x) for x in rows) + b"\n")
    with open(os.path.join(d, sname, "labels.json"), "wb") as fh:
        fh.write(rbcg1.canonical_json(labels))
    with open(os.path.join(d, sname, "monitor_chain.jsonl"), "wb") as fh:
        fh.write(b"\n".join(menvs) + b"\n")
    with open(os.path.join(d, sname, "anchor.json"), "wb") as fh:
        fh.write(rbcg1.canonical_json(anchor))
    dump(os.path.join(d, "policy.json"), ctx["pol1"])
    out = dict(vr)
    out["trace"] = out["trace"][-5:]
    dump(os.path.join(d, sname, "verify_output.json"), out)


def make_ctx(st, payload, P):
    k1, k0, lkey = rbcg1.SigningKey.generate(), rbcg0.SigningKey.generate(), rbcg1.SigningKey.generate()
    ctx = {"P": P, "P16": P[:16], "n_cal": payload["design"]["n_cal_scenes"],
           "n_dep": payload["design"]["n_dep_scenes_per_stream"], "n_prefix": payload["design"]["n_prefix_scenes"],
           "c_e3": payload["design"]["bet_c_e3"], "k1": k1, "k0": k0, "lkey": lkey,
           "pem1": k1.public.pem(), "pem0": k0.public.pem(), "lpem": lkey.public.pem(),
           "pol1": {"expected_bindings": dict(st.bindings), "expected_protocol_sha256": P},
           "pol0": {"expected_bindings": dict(st.bindings)}, "log": rbcg1_log.MerkleLog()}
    return ctx


def cmd_shard(a):
    from rbcg1_triagex import Stack
    check_v0_ref()
    payload, P = load_protocol(a.protocol)
    with open(a.protocol, "rb") as fh:
        penv = json.loads(fh.read())
    st = Stack(a.stack)
    ctx = make_ctx(st, payload, P)
    ctx["log"].append(rbcg1_log.envelope_leaf(penv))          # leaf 0 = the protocol receipt
    R, N_REF = payload["design"]["R"], payload["design"]["n_ref_scenes_per_profile"]
    out = {"shard": a.shard, "nshards": a.nshards, "protocol_sha256": P, "started_utc": utc(),
           "protocol_signed_utc": payload["signed_utc"], "public_key_pem_v1": ctx["pem1"],
           "public_key_pem_v0": ctx["pem0"], "log_public_key_pem": ctx["lpem"],
           "keyids": {"v1": ctx["k1"].public.keyid(), "v0": ctx["k0"].public.keyid(),
                      "log": ctx["lkey"].public.keyid()},
           "bindings": st.bindings, "rbcg1_code_sha256": rbcg1.rbcg_code_sha256(), "reps": [], "ref": {}}
    if not out["protocol_signed_utc"] < out["started_utc"]:
        raise SystemExit("REFUSED: protocol must be signed before evaluation data are generated")
    c0 = time.process_time()
    for prof in ("standard", "shifted"):
        fam = fam_ok(f"szl-rbcg-v1-{ctx['P16']}-ref-{prof}")
        h = {"N_non": [0] * 101, "K_non": [0] * 101, "N_ood": [0] * 101, "K_ood": [0] * 101,
             "n_rows": 0, "n_scenes": 0}
        for idx in range(a.shard, N_REF, a.nshards):
            rows, labels, _, _ = st.scene(fam, idx, prof, with_label=False)
            h["n_scenes"] += 1
            for r in rows:
                h["n_rows"] += 1
                if not r["inf"]:
                    continue
                tm = r["pmax_e12"] // 10 ** 10
                err = int(r["pred"] != labels[r["cid"]])
                key = "ood" if r["ood"] else "non"
                h["N_" + key][tm] += 1
                h["K_" + key][tm] += err
        out["ref"][prof] = h
    out["ref_cpu_s"] = round(time.process_time() - c0, 3)
    ex = os.path.join(a.out, "example_log_sth") if a.shard == 0 else None
    for r in range(a.shard, R, a.nshards):
        out["reps"].append(run_rep(st, ctx, r, example_dir=ex if r == 0 else None))
        if len(out["reps"]) % 10 == 0:
            print(f"shard {a.shard}: {len(out['reps'])} reps, cpu {time.process_time() - c0:.1f}s", flush=True)
    out["log"] = {"size": ctx["log"].size, "root": ctx["log"].root().hex(),
                  "final_sth": rbcg1_log.sign_sth(ctx["log"], ctx["lkey"], utc())}
    out["finished_utc"] = utc()
    out["cpu_s"] = round(time.process_time() - c0, 3)
    dump(os.path.join(a.out, f"shard_{a.shard}.json"), out)
    print(f"shard {a.shard} done: {len(out['reps'])} reps, cpu {out['cpu_s']}s, log size {ctx['log'].size}", flush=True)

# ------------------------------------------------------------------------------------------ E3
def _read_rows(path):
    with open(path, "rb") as fh:
        return [rbcg1_verify.loads_strict(ln, os.path.basename(path)) for ln in fh.read().decode("utf-8").splitlines()
                if ln.strip()]


def cmd_e3(a):
    """Re-verification ONLY, from the stored v0 E3 rows (examples/e3 of rbcg_v0.zip). No model is run and
    the frozen split is not read. The v1 lines below are a descriptive re-analysis of the stored rows,
    computed after the protocol was signed and never used for any design decision."""
    check_v0_ref()
    payload, P = load_protocol(a.protocol)
    d = a.e3dir
    names = ("guarantee.dsse.json", "public_key.pem", "cal_rows.jsonl", "deploy_rows.jsonl", "labels.json",
             "monitor_chain.jsonl", "policy_e3.json")
    files = {n: rbcg1.sha256_file(os.path.join(d, n)) for n in names}
    with open(os.path.join(d, "guarantee.dsse.json"), "rb") as fh:
        g_raw = fh.read()
    with open(os.path.join(d, "public_key.pem"), "r", encoding="ascii") as fh:
        pem = fh.read()
    cal = _read_rows(os.path.join(d, "cal_rows.jsonl"))
    dep = _read_rows(os.path.join(d, "deploy_rows.jsonl"))
    with open(os.path.join(d, "labels.json"), "rb") as fh:
        labels = rbcg1_verify.loads_strict(fh.read(), "labels")
    with open(os.path.join(d, "monitor_chain.jsonl"), "rb") as fh:
        chain = [ln for ln in fh.read().splitlines() if ln.strip()]
    with open(os.path.join(d, "policy_e3.json"), "rb") as fh:
        pol = rbcg1_verify.loads_strict(fh.read(), "policy")
    vr = rbcg0_verify.verify(g_raw, pem, cal, chain, dep, labels, pol)
    vr_strict = rbcg0_verify.verify(g_raw, pem, cal, chain, dep, labels,
                                    {"expected_bindings": pol.get("expected_bindings", {})})
    g = json.loads(__import__("base64").b64decode(json.loads(g_raw)["payload"]))
    tau0 = g["procedure"]["tau_star_c"]
    cal_units, dep_units = rbcg1.group_units(cal), rbcg1.group_units(dep)
    c_e3 = payload["design"]["bet_c_e3"]
    v1cal = rbcg1.calibrate(cal, ALPHA_E6, D10, c_e3=c_e3, units=cal_units)
    mon = rbcg1.monitor_spec(rbcg1.rows_sha256(cal), len(cal_units))
    tm = rbcg1.TwoScoreMonitor(cal_units, mon)
    lp, ls, alarm_at, comp = [], [], None, None
    for j, (_, ur) in enumerate(dep_units, 1):
        s = tm.step(ur)
        lp.append(s["primary"]["log_m"])
        ls.append(s["secondary"]["log_m"])
        if s["alarm"] and alarm_at is None:
            alarm_at = j
            comp = ("both" if s["primary"]["alarm"] and s["secondary"]["alarm"] else
                    "primary" if s["primary"]["alarm"] else "secondary")
    out = {"kind": "re-verification of stored v0 E3 rows; no model run; frozen split not read",
           "protocol_sha256": P, "stored_files_sha256": files, "n_cal_rows": len(cal), "n_cal_units": len(cal_units),
           "n_deploy_rows": len(dep), "n_deploy_units": len(dep_units),
           "v0_reverification": {"verdict": vr["verdict"], "detail": vr["detail"], "alarm_batch": vr["alarm_batch"],
                                 "violation_batch": vr["violation_batch"], "tau_star_c": vr["tau_star_c"],
                                 "transitions": transitions(vr["trace"]), "verdict_strict_policy": vr_strict["verdict"]},
           "v1_descriptive": {"note": "descriptive re-analysis of stored rows; not a v1 receipt",
                              "v1_ltt_on_stored_80_scene_cal": {"tau_star_c": v1cal["tau_star_c"],
                                                                "n_rejected": v1cal["n_rejected"],
                                                                "p_bet_e9_tau99": v1cal["table"][0]["p_bet_e9"],
                                                                "hb_tau_star_c": v1cal["secondary_tau_star_c"]},
                              "monitor_alarm_batch": alarm_at, "monitor_alarm_component": comp,
                              "primary_first": first_crossings(lp), "secondary_first": first_crossings(ls),
                              "audit_v1_first_at_v0_tau": audit_first_v1(dep_units, labels, th_of(tau0), c_e3),
                              "audit_v1_first_at_064": audit_first_v1(dep_units, labels, (64, 81), c_e3),
                              "audit_v0_first_at_v0_tau": audit_first_v0(dep_units, labels, th_of(tau0)),
                              "realized_at_v0_tau": realized(dep, labels, th_of(tau0))}}
    dump(os.path.join(a.out, "e3_v1.json"), out)
    print(json.dumps({"v0_verdict": vr["verdict"], "v0_alarm": vr["alarm_batch"], "strict": vr_strict["verdict"],
                      "v1_tau_on_stored_cal": v1cal["tau_star_c"], "v1_monitor_alarm": alarm_at, "comp": comp}))


# ------------------------------------------------------------------------------------------ merge
def wilson(k, n):
    lo, hi = rbcg1.wilson_interval(k, n, 0.95) if n else (0.0, 1.0)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "wilson95": [lo, hi]}


def sign_test(n_pos, n_neg):
    """Exact sign test on discordant pairs: one-sided p for 'positive more often', and two-sided p."""
    n = n_pos + n_neg
    if n == 0:
        return {"n_pos": 0, "n_neg": 0, "p_one_sided": 1.0, "p_two_sided": 1.0}
    p1 = rbcg1.binom_sf(n_pos, n, 0.5)
    p2 = min(1.0, 2.0 * min(rbcg1.binom_cdf(n_pos, n, 0.5), rbcg1.binom_sf(n_pos, n, 0.5)))
    return {"n_pos": n_pos, "n_neg": n_neg, "p_one_sided": p1, "p_two_sided": p2}


def q(v, f):
    v = sorted(v)
    if not v:
        return None
    return statistics.quantiles(v, n=4)[f] if len(v) > 1 else v[0]


def boot_median_diff(a_vals, b_vals, B, seed):
    """Percentile bootstrap (paired, over reps) of median(a) - median(b)."""
    rng = random.Random(seed)
    n = len(a_vals)
    ds = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(statistics.median([a_vals[i] for i in idx]) - statistics.median([b_vals[i] for i in idx]))
    ds.sort()
    return [ds[int(0.025 * B)], ds[int(0.975 * B) - 1]]


def boot_rate_diff(a_bool, b_bool, B, seed):
    rng = random.Random(seed)
    n = len(a_bool)
    ds = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(sum(a_bool[i] for i in idx) / n - sum(b_bool[i] for i in idx) / n)
    ds.sort()
    return [ds[int(0.025 * B)], ds[int(0.975 * B) - 1]]


def cmd_merge(a):
    payload, P = load_protocol(a.protocol)
    des = payload["design"]
    shards = []
    for i in range(a.nshards):
        with open(os.path.join(a.out, f"shard_{i}.json"), encoding="utf-8") as fh:
            shards.append(json.load(fh))
    reps = sorted((r for s in shards for r in s["reps"]), key=lambda r: r["rep"])
    R = des["R"]
    assert [r["rep"] for r in reps] == list(range(R)), "missing reps"
    n_dep, cens = des["n_dep_scenes_per_stream"], des["n_dep_scenes_per_stream"] + 1
    Bboot, bseed = des["bootstrap"]["B"], des["bootstrap"]["seed"]
    ref = {}
    for prof in ("standard", "shifted"):
        h = {k: [0] * 101 for k in ("N_non", "K_non", "N_ood", "K_ood")}
        h["n_rows"] = h["n_scenes"] = 0
        for s in shards:
            for k in ("N_non", "K_non", "N_ood", "K_ood"):
                h[k] = [x + y for x, y in zip(h[k], s["ref"][prof][k])]
            h["n_rows"] += s["ref"][prof]["n_rows"]
            h["n_scenes"] += s["ref"][prof]["n_scenes"]
        ref[prof] = h
    cache = {}

    def true_risk(prof, th):
        key = (prof, th[0], th[1])
        if key not in cache:
            h = ref[prof]
            n = sum(h["N_non"][th[0]:]) + sum(h["N_ood"][th[1]:])
            k = sum(h["K_non"][th[0]:]) + sum(h["K_ood"][th[1]:])
            lo, hi = rbcg1.wilson_interval(k, n, 0.95) if n else (0.0, 1.0)
            cache[key] = {"n_acc": n, "k": k, "risk": k / n if n else None, "wilson95": [lo, hi],
                          "coverage": n / h["n_rows"]}
        return cache[key]

    alpha = ALPHA_E6 / 1e6
    P16 = P[:16]
    fams_ok = all(P16 in r["cal_family"] and all(P16 in r["streams"][s]["family"] for s in r["streams"])
                  for r in reps)
    res = {"generated_utc": utc(), "protocol_sha256": P, "protocol_P16": P16,
           "protocol_signed_utc": payload["signed_utc"], "R": R, "alpha": alpha,
           "deltas": {"ltt": [0.10, 0.05], "monitor_primary": 0.0075, "monitor_secondary": 0.0025,
                      "audit": DELTA_AUDIT / 1e6},
           "design": des, "seed_families": payload["families"],
           "hashes": {"bindings": shards[0]["bindings"], "rbcg1_code_sha256": shards[0]["rbcg1_code_sha256"],
                      "keyids": [s["keyids"] for s in shards], "v0_ref_sha256": V0_REF_SHA256},
           "reference": {prof: {"n_rows": ref[prof]["n_rows"], "n_scenes": ref[prof]["n_scenes"]} for prof in ref},
           "cpu_s": {"shards": [s["cpu_s"] for s in shards], "ref": [s["ref_cpu_s"] for s in shards]}}
    res["order_enforcement"] = {
        "protocol_signed_before_every_shard_start": all(payload["signed_utc"] < s["started_utc"] for s in shards),
        "all_evaluation_families_embed_protocol_P16": fams_ok,
        "all_reps_issued_before_deploy": all(r["order"]["issued_before_deploy"] for r in reps)}
    names = ["v1_d10", "v1_d05", "v1_hb_d10", "v1_d10_n80", "v0_d10", "v0_d05", "v0_d10_n80", "nominal_1d"]
    res["issuance"] = {m: wilson(sum(1 for r in reps if r["tau"][m] is not None), R) for m in names}
    res["issuance_verdicts_v1"] = {v: sum(1 for r in reps if r["issuance_verdict"].get("v1_d10") == v)
                                   for v in sorted({r["issuance_verdict"].get("v1_d10") for r in reps})}
    res["tau_distribution"] = {}
    for m in names:
        v = [r["tau"][m] for r in reps if r["tau"][m] is not None]
        res["tau_distribution"][m] = ({"n": len(v), "none": R - len(v), "min": min(v), "q25": q(v, 0),
                                       "median": statistics.median(v), "q75": q(v, 2), "max": max(v)}
                                      if v else {"n": 0, "none": R})
    pts = [r["cal_v0_table_at"]["64"] for r in reps]
    errs = [k / n for n, k in pts]
    pbar = sum(k for n, k in pts) / sum(n for n, k in pts)
    binv = statistics.mean([pbar * (1 - pbar) / n for n, k in pts])
    res["design_effect_cal_tau064"] = {"observed_var": statistics.pvariance(errs), "binomial_var": binv,
                                       "ratio": statistics.pvariance(errs) / binv, "pooled_err": pbar,
                                       "n_cal_scenes": des["n_cal_scenes"]}
    mths = names + ["frozen_tau064"]
    for ename, sname, prof in (("E1", "in_dist", "standard"), ("E2", "shifted", "shifted")):
        E = {"profile": prof, "methods": {}}
        for m in mths:
            ths = [method_thresholds(r)[m] for r in reps]
            tr = [true_risk(prof, th) if th else None for th in ths]
            viol = [x is not None and x["risk"] is not None and x["risk"] > alpha for x in tr]
            sel = [r["streams"][sname]["realized"][m] for r in reps]
            issued = [x for x in tr if x is not None]
            E["methods"][m] = {
                "n_reps_with_threshold": len(issued),
                "P_true_risk_gt_alpha_unconditional": wilson(sum(viol), R),
                "P_true_risk_gt_alpha_given_issued": wilson(sum(viol), len(issued)),
                "mean_true_risk_given_issued": statistics.mean(x["risk"] for x in issued) if issued else None,
                "mean_true_coverage_given_issued": statistics.mean(x["coverage"] for x in issued) if issued else None,
                "mean_true_coverage_unconditional": sum(x["coverage"] for x in issued) / R,
                "P_realized_sel_err_gt_alpha": wilson(sum(1 for x in sel if x and x["sel_err"] is not None
                                                          and x["sel_err"] > alpha), len(issued)),
                "mean_realized_coverage_given_issued": (statistics.mean(x["coverage"] for x in sel if x)
                                                        if issued else None)}
        for m in ("v1_d10", "v0_d10", "frozen_tau064"):
            ths = [method_thresholds(r)[m] for r in reps]
            tr = [true_risk(prof, th) if th else None for th in ths]
            for rule in ("v1", "v0"):
                f = [r["streams"][sname][f"audit_{rule}_first"][m] for r in reps]
                ok = [x is not None for x, th in zip(f, ths) if th]
                above = [x is not None for x, t in zip(f, tr) if t and t["risk"] > alpha]
                below = [x is not None for x, t in zip(f, tr) if t and t["risk"] <= alpha]
                E["methods"][m][f"label_audit_{rule}_rule"] = {
                    "flagged_within_stream": wilson(sum(ok), len(ok)),
                    "flagged_when_true_risk_gt_alpha": wilson(sum(above), len(above)),
                    "flagged_when_true_risk_le_alpha": wilson(sum(below), len(below)),
                    "median_flag_scene": statistics.median([x for x in f if x is not None]) if any(
                        x is not None for x in f) else None}
        # paired audit comparison at the common frozen threshold (true risk known from the reference)
        fa1 = [r["streams"][sname]["audit_v1_first"]["frozen_tau064"] is not None for r in reps]
        fa0 = [r["streams"][sname]["audit_v0_first"]["frozen_tau064"] is not None for r in reps]
        E["label_audit_paired_at_tau064"] = {
            "true_risk_at_064": true_risk(prof, (64, 81)), "v1_flag_rate": wilson(sum(fa1), R),
            "v0_flag_rate": wilson(sum(fa0), R),
            "mcnemar_exact": sign_test(sum(1 for x, y in zip(fa1, fa0) if x and not y),
                                       sum(1 for x, y in zip(fa1, fa0) if y and not x)),
            "diff_v1_minus_v0_boot95": boot_rate_diff(fa1, fa0, Bboot, bseed + 7)}
        # monitor and verifier, v1 vs v0 on the same streams
        both = [r for r in reps if "v1" in r["streams"][sname] and "v0" in r["streams"][sname]]
        mv = {"n_reps_with_both": len(both)}
        for ver in ("v1", "v0"):
            vs = [r["streams"][sname][ver] for r in reps if ver in r["streams"][sname]]
            verdicts = {}
            for v in vs:
                verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
            al = [v["alarm_batch"] for v in vs if v["alarm_batch"] is not None]
            mv[ver] = {"n_streams": len(vs), "final_verdicts": verdicts,
                       "non_valid_rate": wilson(sum(1 for v in vs if v["verdict"] != "VALID"), len(vs)),
                       "alarm_rate": wilson(len(al), len(vs)),
                       "alarm_batch": ({"median": statistics.median(al), "q25": q(al, 0), "q75": q(al, 2),
                                        "min": min(al), "max": max(al)} if al else None),
                       "violated_rate": wilson(sum(1 for v in vs if v["verdict"] == "VIOLATED"), len(vs))}
            if ver == "v1":
                comps = {}
                for v in vs:
                    if v["alarm_component"]:
                        comps[v["alarm_component"]] = comps.get(v["alarm_component"], 0) + 1
                mv[ver]["alarm_component_counts"] = comps
                mv[ver]["anchor_checked_streams"] = sum(1 for v in vs if v.get("anchor") and
                                                        v["anchor"].get("head_index") is not None)
                for comp, key in (("primary", "primary_first"), ("secondary", "secondary_first")):
                    for t in LOG_THRESHOLDS:
                        fc = [v[key][t] for v in vs if v[key][t] is not None]
                        mv[ver][f"{comp}_alone_at_{t}"] = {"alarm_rate": wilson(len(fc), len(vs)),
                                                           "median": statistics.median(fc) if fc else None}
        a1 = [r["streams"][sname]["v1"]["alarm_batch"] or cens for r in both]
        a0 = [r["streams"][sname]["v0"]["alarm_batch"] or cens for r in both]
        nv1 = [r["streams"][sname]["v1"]["verdict"] != "VALID" for r in both]
        nv0 = [r["streams"][sname]["v0"]["verdict"] != "VALID" for r in both]
        if both:
            mv["paired"] = {
                "censor_value_no_alarm": cens,
                "median_alarm_v1_censored": statistics.median(a1), "median_alarm_v0_censored": statistics.median(a0),
                "median_diff_v0_minus_v1": statistics.median(a0) - statistics.median(a1),
                "median_diff_v0_minus_v1_boot95": boot_median_diff(a0, a1, Bboot, bseed),
                "sign_test_v1_earlier": sign_test(sum(1 for x, y in zip(a1, a0) if x < y),
                                                  sum(1 for x, y in zip(a1, a0) if x > y)),
                "ties": sum(1 for x, y in zip(a1, a0) if x == y),
                "non_valid_v1_minus_v0_boot95": boot_rate_diff(nv1, nv0, Bboot, bseed + 1),
                "mcnemar_non_valid": sign_test(sum(1 for x, y in zip(nv1, nv0) if x and not y),
                                               sum(1 for x, y in zip(nv1, nv0) if y and not x))}
        E["monitor_and_verifier"] = mv
        res[ename] = E
    with open(os.path.join(a.out, "e3_v1.json"), encoding="utf-8") as fh:
        res["E3"] = json.load(fh)
    # pre-registered decision rules (see protocol 'analysis_plan')
    e1 = res["E1"]["methods"]
    dec = {}
    for m, d in (("v1_d10", 0.10), ("v1_d05", 0.05)):
        w = e1[m]["P_true_risk_gt_alpha_unconditional"]
        dec[f"H1_{m}"] = ("HOLDS" if w["wilson95"][1] < d else "CONSISTENT_NOT_DEMONSTRATED" if w["rate"] <= d
                          else "REFUTED" if w["wilson95"][0] > d else "INCONCLUSIVE_ABOVE")
    pe = res["E2"]["monitor_and_verifier"].get("paired")
    if pe:
        st_ = pe["sign_test_v1_earlier"]
        dec["H2a_v1_alarms_earlier"] = ("SUPPORTED" if st_["p_one_sided"] < 0.05 and
                                        pe["median_alarm_v1_censored"] < pe["median_alarm_v0_censored"]
                                        else "NOT_SUPPORTED")
        dec["H2b_non_valid_v1_vs_v0"] = {"v1": res["E2"]["monitor_and_verifier"]["v1"]["non_valid_rate"],
                                         "v0": res["E2"]["monitor_and_verifier"]["v0"]["non_valid_rate"],
                                         "mcnemar_two_sided_p": pe["mcnemar_non_valid"]["p_two_sided"]}
    fa_rate = res["E1"]["monitor_and_verifier"].get("v1", {}).get("alarm_rate", {}).get("rate")
    dec["S1_false_alarm_v1_le_delta_s"] = None if fa_rate is None else fa_rate <= 0.01
    res["decisions"] = dec
    res["log_anchoring"] = {"per_shard": [{"size": s["log"]["size"], "root": s["log"]["root"]} for s in shards]}
    if a.v0_summary:
        with open(a.v0_summary, encoding="utf-8") as fh:
            v0s = json.load(fh)
        res["v0_archived_REPORTED"] = {
            "source": "rbcg_results_summary.json (v0 run, R=600, n_cal 80 scenes, seeds szl-rbcg-v0-*)",
            "E1_rbcg_d10": v0s["E1"]["methods"]["rbcg_d10"], "E2_rbcg_d10": v0s["E2"]["methods"]["rbcg_d10"],
            "E1_monitor": v0s["E1"]["monitor_and_verifier"], "E2_monitor": v0s["E2"]["monitor_and_verifier"],
            "tau_distribution": v0s["tau_distribution"]}
    res["per_rep"] = [{"rep": r["rep"], "tau": r["tau"], "cpu_s": r["cpu_s"], "cal_family": r["cal_family"],
                       "in_dist": {k: r["streams"]["in_dist"].get(k, {}).get("verdict") for k in ("v1", "v0")},
                       "shifted": {k: [r["streams"]["shifted"].get(k, {}).get("verdict"),
                                       r["streams"]["shifted"].get(k, {}).get("alarm_batch")] for k in ("v1", "v0")}}
                      for r in reps]
    res["rep0_v1_shifted_log_m"] = {k: reps[0]["streams"]["shifted"].get("v1", {}).get(k)
                                    for k in ("log_m_primary", "log_m_secondary")}
    dump(os.path.join(a.out, a.name), res)
    print(json.dumps({"issuance": {m: res["issuance"][m]["k"] for m in names}, "decisions": dec}, sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "shard", "e3", "merge"])
    ap.add_argument("--stack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--sizes", default="80,160,240,320,400,480")
    ap.add_argument("--timing-ncal", type=int, default=320)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--e3dir")
    ap.add_argument("--v0-summary")
    ap.add_argument("--name", default="rbcg_v1_results_2026-11-03.json")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd != "pilot" and not a.protocol:
        raise SystemExit("REFUSED: a signed protocol receipt is required (pre-registration)")
    {"pilot": cmd_pilot, "shard": cmd_shard, "e3": cmd_e3, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
