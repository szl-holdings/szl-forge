#!/usr/bin/env python3
# RBCG v1 prototype -- SZL Holdings. Licence: UNDECIDED (owner choice). See NOTICE.
"""mutate_rbcg1.py -- hand-written mutation harness for rbcg1.py, rbcg1_log.py and rbcg1_verify.py.

Each mutant is an exact single-occurrence source replacement that re-creates a realistic bug. The
harness (1) checks every anchor occurs exactly once, (2) REFUSES to score unless the unmutated suite is
fully green, (3) requires a comment-only no-op mutant to SURVIVE (so a broken runner cannot report
everything as killed), then (4) runs the suite (failfast) against each mutant in an in-package scratch
directory, at most 3 concurrent processes. Exit 0 only if every mutant is killed.
  python3 -B mutate_rbcg1.py [report.json]
"""
import concurrent.futures
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ("rbcg1.py", "rbcg1_log.py", "rbcg1_verify.py", "run_tests.py")
REQUIRED = ("N01_betting_wealth_sign", "N02_missing_ville_threshold", "N03_cs_width_bug",
            "N04_merkle_node_prefix_swap", "N05_inclusion_path_off_by_one", "N06_sth_signature_not_checked")

MUTANTS = [('M02_binom_tail_off_by_one', 'rbcg1.py', 'for j in range(0, k + 1)])))', 'for j in range(0, k)])))', 'binomial CDF sums j < k instead of j <= k'),
    ('M03_fixed_sequence_no_stop', 'rbcg1.py', '            tau_star, n_rej = e["tau_c"], n_rej + 1\n        else:\n            break', '            tau_star, n_rej = e["tau_c"], n_rej + 1\n        else:\n            continue', 'fixed-sequence testing does not stop at the first non-rejection'),
    ('M05_alarm_at_delta', 'rbcg1.py', '    return math.log(10 ** 6 / delta_s_e6)', '    return math.log(delta_s_e6 / 10 ** 6)', 'monitor alarms at M >= delta_s instead of M >= 1/delta_s'),
    ('M06_pae_length_in_chars', 'rbcg1.py', 'str(len(t)).encode("ascii")', 'str(len(payload_type)).encode("ascii")', 'DSSE PAE uses character length of payloadType'),
    ('M07_nonce_reuse', 'rbcg1.py', '    for k in rfc6979_nonces(d, h1):', '    for k in rfc6979_nonces(d, b"\\x00" * 32):', 'ECDSA nonce independent of the message'),
    ('M08_cp_lower_wrong_tail', 'rbcg1.py', '        if binom_sf(k, n, mid) < delta:\n            lo = mid', '        if binom_cdf(k, n, mid) < delta:\n            lo = mid', 'Clopper-Pearson lower bound bisects on the wrong tail'),
    ('M09_pvalue_excludes_self', 'rbcg1.py', '    eq = (hi - lo) + 1', '    eq = (hi - lo)', 'smoothed conformal p-value omits the new point from the tie count'),
    ('M10_ecdsa_no_range_check', 'rbcg1.py', '    if not (1 <= r < P256_N and 1 <= s < P256_N):\n        return False\n    if pub is None', '    if False:\n        return False\n    if pub is None', 'ECDSA verify accepts r or s outside [1, n-1]'),
    ('M11_canonical_unsorted', 'rbcg1.py', '    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,', '    return json.dumps(obj, sort_keys=False, separators=(",", ":"), ensure_ascii=False,', 'canonical JSON does not sort keys'),
    ('M12_der_nonminimal_accepted', 'rbcg1.py', '        if ln > 1 and v[0] == 0 and not (v[1] & 0x80):\n            raise ValueError("non-minimal DER INTEGER")', '        if False:\n            raise ValueError("non-minimal DER INTEGER")', 'DER decoder accepts non-minimal INTEGERs'),
    ('M13_fixed_sequence_strict_lt', 'rbcg1.py', '        if e[key] <= delta_e6 * 1000:', '        if e[key] < delta_e6 * 1000:', 'fixed-sequence rejects only when p < delta'),
    ('M14_ood_floor_ignored', 'rbcg1.py', '    return (max(tau_c, ood_floor_c) if row["ood"] else tau_c) * 10 ** 10', '    return tau_c * 10 ** 10', 'selector ignores the tau_ood floor'),
    ('M14b_fast_path_ood_floor_ignored', 'rbcg1.py', '            if r["ood"] and pm < ood_floor_c * 10 ** 10:\n                continue\n            tm = min(pm // 10 ** 10, 101)', '            tm = min(pm // 10 ** 10, 101)', 'issuer fast path ignores the OOD floor'),
    ('M15_theta_seed_unbound', 'rbcg1.py', '    return hashlib.sha256((domain + "|" + cal_rows_sha256).encode("ascii")).hexdigest()', '    return hashlib.sha256(domain.encode("ascii")).hexdigest()', 'monitor tie-break seeds not bound to the calibration rows'),
    ('M16_martingale_bets_on_p', 'rbcg1.py', '            self.log_comp[i] += math.log(e) + (e - 1.0) * lp', '            self.log_comp[i] += math.log(e) + (e - 1.0) * p', 'power-martingale update uses p instead of log p'),
    ('N01_betting_wealth_sign', 'rbcg1.py', '        logk += math.log1p(min(lam, cap) * (m - x))', '        logk += math.log1p(min(lam, cap) * (x - m))', 'LTT betting wealth uses (x - m): bets on the wrong side'),
    ('N02_missing_ville_threshold', 'rbcg1.py', '        if self.violated_at is None and self.logk >= self.log_thr:', '        if self.violated_at is None and self.logk > 0.0:', 'label audit flags at K > 1 instead of the Ville threshold 1/delta'),
    ('N03_cs_width_bug', 'rbcg1.py', '        self.log_thr = math.log(1.0 / self.delta)', '        self.log_thr = math.log(0.5 / self.delta)', 'confidence-sequence threshold 1/(2 delta): CS too narrow'),
    ('N03b_cs_cap_wrong_side', 'rbcg1.py', '        cap = self.c / m\n        return math.fsum', '        cap = self.c / (1.0 - m)\n        return math.fsum', 'CS lower bound truncates bets with c/(1-m) instead of c/m'),
    ('N04_merkle_node_prefix_swap', 'rbcg1_log.py', 'LEAF_PREFIX = b"\\x00"\nNODE_PREFIX = b"\\x01"', 'LEAF_PREFIX = b"\\x01"\nNODE_PREFIX = b"\\x00"', 'Merkle leaf/node domain-separation prefixes swapped'),
    ('N05_inclusion_path_off_by_one', 'rbcg1_log.py', '        if m < k:\n            return self._path(m, start, k)', '        if m <= k:\n            return self._path(m, start, k)', 'inclusion-path generation splits at m <= k (off by one)'),
    ('N05b_inclusion_verify_off_by_one', 'rbcg1_log.py', '    fn, sn = leaf_index, tree_size - 1\n', '    fn, sn = leaf_index, tree_size\n', 'inclusion verification starts sn at tree_size (off by one)'),
    ('N06_sth_signature_not_checked', 'rbcg1_verify.py', '    _, s = check_envelope_obj(env_obj, rbcg1.PT_STH, log_pub, what)', '    s = json.loads(base64.b64decode(env_obj["payload"]))', 'verifier does not check the STH signature'),
    ('N06b_sth_consistency_not_checked', 'rbcg1_verify.py', '    if not rbcg1_log.verify_consistency(', '    if False and not rbcg1_log.verify_consistency(', 'verifier does not check STH consistency'),
    ('N07_ltt_cap_wrong_side', 'rbcg1.py', '    cap = c / (1.0 - m)\n    logk = best = 0.0', '    cap = c / m\n    logk = best = 0.0', 'LTT bets truncated at c/m (capital can turn negative)'),
    ('N08_hb_floor_not_ceil', 'rbcg1.py', '    kc = -((-s) // den)', '    kc = s // den', 'Bentkus term uses floor instead of ceil'),
    ('N09_secondary_alarm_ignored', 'rbcg1.py', '        self.alarmed = self.alarmed or out["primary"]["alarm"] or out["secondary"]["alarm"]', '        self.alarmed = self.alarmed or out["primary"]["alarm"]', 'issuer monitor ignores the secondary CTM'),
    ('N10_loss_sign', 'rbcg1.py', '    return e * 10 ** 6 - alpha_e6 * a + alpha_e6 * n_max', '    return e * 10 ** 6 + alpha_e6 * a - alpha_e6 * n_max', 'linearized loss sign error'),
    ('N11_prpl_not_predictable', 'rbcg1.py', '        out.append(math.sqrt(two_log / ((var_sum / t) * t * math.log1p(t))))\n        mu_t = (mu_sum + x) / (t + 1)\n        var_sum += (x - mu_t) ** 2', '        mu_t = (mu_sum + x) / (t + 1)\n        var_sum += (x - mu_t) ** 2\n        out.append(math.sqrt(two_log / ((var_sum / t) * t * math.log1p(t))))', 'bet size uses X_t (not predictable)'),
    ('N12_primary_score_wrong', 'rbcg1.py', '    return sum(1 for r in rows if r["ood"])', '    return sum(1 for r in rows if not r["ood"])', 'primary OOD-count score counts non-OOD rows'),
    ('V01_verifier_ignores_monitor', 'rbcg1_verify.py', '            first_event = first_event or ("REFUSED:SHIFT_ALARM", j)', '            pass', 'verifier ignores a monitor alarm'),
    ('V02_cal_hash_not_checked', 'rbcg1_verify.py', '    if rbcg1.sha256_hex(rbcg1.canonical_json(cal_rows)) != c["rows_sha256"]:', '    if False:', 'calibration-rows hash not checked'),
    ('V03_signature_check_skipped', 'rbcg1_verify.py', '    if not pub.verify(rbcg1.pae(expected_type, body), sig):', '    if False:', 'ECDSA signature check skipped'),
    ('V04_alpha_not_compared', 'rbcg1_verify.py', '    if p["alpha_e6"] != pol["alpha_e6"]:', '    if False:', 'claimed alpha not compared with the policy'),
    ('V05_audit_point_estimate', 'rbcg1_verify.py', '                newly = audit.add(x)', '                newly = audit.add(x) or sum(audit.xs) / len(audit.xs) > audit.m0', 'label audit uses the point estimate'),
    ('V06_chain_prev_not_checked', 'rbcg1_verify.py', '            if m["prev_sha256"] != prev_digest:', '            if False:', 'monitor hash chain not checked'),
    ('V07_extra_fields_accepted', 'rbcg1_verify.py', '    if have != want:\n        raise Refuse("SCHEMA"', '    if not want <= have:\n        raise Refuse("SCHEMA"', 'schema check tolerates extra fields'),
    ('V08_floats_and_nan_accepted', 'rbcg1_verify.py', '        return json.loads(raw, object_pairs_hook=_no_dupes, parse_float=_no_float,\n                          parse_constant=_no_constant)', '        return json.loads(raw, object_pairs_hook=_no_dupes)', 'strict loader accepts floats, NaN and Infinity'),
    ('V09_order_not_checked', 'rbcg1_verify.py', '            if pol["require_order_commitment"] and gdig[:16] not in m["deploy_source"]:', '            if False:', 'deployment-after-issuance commitment not checked'),
    ('V10_tau_not_compared', 'rbcg1_verify.py', '    if tau != p["tau_star_c"] or n_rej != p["n_rejected"]:', '    if False:', 'recomputed tau* not compared'),
    ('V11_verifier_alarm_at_delta', 'rbcg1_verify.py', '    thr = {name: math.log(10 ** 6 / mon[name]["delta_s_e6"]) for name in COMPONENTS}', '    thr = {name: math.log(mon[name]["delta_s_e6"] / 10 ** 6) for name in COMPONENTS}', 'verifier alarm at delta instead of 1/delta'),
    ('V12_leakage_not_checked', 'rbcg1_verify.py', '    if any(r["cid"] in cal_cids or r["unit"] in cal_unit_ids for r in deploy_rows):', '    if False:', 'calibration/deployment overlap not checked'),
    ('V13_batch_hash_not_checked', 'rbcg1_verify.py', '                    m["batch_sha256"] != rbcg1.sha256_hex(rbcg1.canonical_json(rows)):', '                    False:', 'monitor batch digest not compared'),
    ('V14_claim_text_not_checked', 'rbcg1_verify.py', '    if g["claim"] != _claim_expected(g):', '    if False:', 'claim text not compared'),
    ('V15_exceptions_escape', 'rbcg1_verify.py', '    except Exception as e:  # fail closed on anything unexpected', '    except ZeroDivisionError as e:  # fail closed on anything unexpected', 'unexpected exceptions escape'),
    ('V16_head_order_not_checked', 'rbcg1_verify.py', '    if hi["leaf_index"] < s1["tree_size"]:', '    if False:', 'monitor head logged before the issuance STH is accepted'),
    ('V17_log_id_not_checked', 'rbcg1_verify.py', '    if s1["log_id"] != log_pub.keyid() or s1["log_id"] != g["anchoring"]["log_id"]:', '    if False:', "STH log id not compared with the guarantee's log"),
    ('V18_guarantee_inclusion_not_checked', 'rbcg1_verify.py', '    if not rbcg1_log.verify_inclusion(gi["leaf_index"], gi["tree_size"], g_leaf, gpath,', '    if False and not rbcg1_log.verify_inclusion(gi["leaf_index"], gi["tree_size"], g_leaf, gpath,', 'guarantee inclusion proof not checked'),
    ('V19_protocol_binding_not_checked', 'rbcg1_verify.py', '    if pol["expected_protocol_sha256"] is not None and g["protocol_sha256"] != pol["expected_protocol_sha256"]:', '    if False:', 'pre-registered protocol binding not checked'),
    ('V20_head_inclusion_not_checked', 'rbcg1_verify.py', '    if not rbcg1_log.verify_inclusion(hi["leaf_index"], hi["tree_size"], h_leaf, hpath,', '    if False and not rbcg1_log.verify_inclusion(hi["leaf_index"], hi["tree_size"], h_leaf, hpath,', 'monitor-chain head inclusion not checked')]
SANITY = ("S00_noop_comment", "rbcg1.py", "# RBCG v1 prototype -- SZL Holdings.",
          "# RBCG v1 prototype -- SZL Holdings (comment-only no-op).", "comment-only change; must SURVIVE")


def make_copy(dst):
    os.makedirs(dst, exist_ok=True)
    for f in FILES:
        shutil.copy2(os.path.join(HERE, f), os.path.join(dst, f))
    shutil.copytree(os.path.join(HERE, "tests"), os.path.join(dst, "tests"),
                    ignore=shutil.ignore_patterns("__pycache__"))


def run_suite(d, timeout, failfast):
    env = dict(os.environ)
    env["TMPDIR"] = os.path.join(d, "tmp")
    env["RBCG_FAILFAST"] = "1" if failfast else "0"
    os.makedirs(env["TMPDIR"], exist_ok=True)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-B", "run_tests.py"], cwd=d, capture_output=True, text=True,
                           timeout=timeout, env=env)
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        rc, out = "timeout", ""
    failing = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\S+) \(([^)]+)\)", out, re.M)))
    line = (re.findall(r"^RBCG_SUITE .*$", out, re.M) or [""])[-1]
    return rc, out, failing, time.time() - t0, line


def apply(d, file, old, new):
    p = os.path.join(d, file)
    with open(p, encoding="utf-8") as fh:
        src = fh.read()
    if src.count(old) != 1:
        raise RuntimeError(f"anchor occurs {src.count(old)} times in {file}")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src.replace(old, new))
    py_compile.compile(p, doraise=True)


def score(m, root, timeout=900, failfast=True):
    mid, file, old, new, desc = m
    d = tempfile.mkdtemp(prefix=mid + "_", dir=root)
    make_copy(d)
    apply(d, file, old, new)
    rc, out, failing, secs, line = run_suite(d, timeout, failfast)
    shutil.rmtree(d, ignore_errors=True)
    return {"id": mid, "file": file, "description": desc, "killed": rc != 0, "exit": rc,
            "seconds": round(secs, 1), "suite_line": line,
            "killing_tests": [f"{a} ({b.split('.')[-1]})" for a, b in failing][:12]}


def main(argv):
    out_path = argv[1] if len(argv) > 1 else "mutation_report.json"
    report = {"harness": "mutate_rbcg1.py", "python": sys.version.split()[0], "n_mutants": len(MUTANTS),
              "required_new_mutants": list(REQUIRED)}
    ids = [m[0] for m in MUTANTS]
    if any(r not in ids for r in REQUIRED) or len(set(ids)) != len(ids):
        report["verdict"] = "REFUSED: required mutant missing or duplicate ids"
    for mid, file, old, new, _ in MUTANTS + [SANITY]:
        if "verdict" in report:
            break
        with open(os.path.join(HERE, file), encoding="utf-8") as fh:
            n = fh.read().count(old)
        if n != 1:
            report["verdict"] = f"REFUSED: anchor of {mid} occurs {n} times in {file}"
    if "verdict" not in report:
        root = os.path.join(HERE, "_mut_scratch")
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root)
        d0 = os.path.join(root, "baseline")
        make_copy(d0)
        rc, out, failing, secs, line = run_suite(d0, 1800, failfast=False)
        report["baseline"] = {"exit": rc, "seconds": round(secs, 1), "suite_line": line,
                              "failing": [f"{a} ({b})" for a, b in failing]}
        if rc != 0:
            report["verdict"] = "REFUSED: the unmutated suite is not fully green; no mutant was scored"
            report["baseline_output_tail"] = out[-3000:]
        else:
            s = score(SANITY, root, failfast=False)
            report["sanity_noop"] = s
            if s["killed"]:
                report["verdict"] = "REFUSED: the no-op mutant was killed; the harness cannot discriminate"
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                    res = list(ex.map(lambda m: score(m, root), MUTANTS))
                report["mutants"] = res
                report["killed"] = sum(r["killed"] for r in res)
                report["total"] = len(res)
                report["required_killed"] = {r["id"]: r["killed"] for r in res if r["id"] in REQUIRED}
                report["verdict"] = "ALL_KILLED" if report["killed"] == len(res) else "SURVIVORS"
        shutil.rmtree(root, ignore_errors=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(json.dumps({k: report.get(k) for k in ("verdict", "killed", "total")}))
    return 0 if report.get("verdict") == "ALL_KILLED" else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
