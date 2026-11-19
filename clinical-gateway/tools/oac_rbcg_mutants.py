#!/usr/bin/env python3
"""Mutation check for oac_rbcg.py (stdlib). Applies each deliberate code mutant to a copy of oac_rbcg.py in a scratch tree that
symlinks the repo's model/data dirs, runs that copy's `selftest`, and requires it to FAIL (non-zero exit). Exit 0 iff every mutant
is killed and every mutation anchor was found (a stale anchor is an error, never a silent pass).  Usage: python3 oac_rbcg_mutants.py"""
import json, os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); F = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = open(f"{HERE}/oac_rbcg.py").read()
POLICY = '    if not policy_ok(rc): return "REFUSE:POLICY"\n'; HASHCK = '    if now != rc["files"]: return "REFUSE:HASH"\n'
M = {
 "M01 policy_ok always true": [("    return (all(_num(x) for x in v) and", "    return (True or")],
 "M02 shift_alpha lower bound removed": [("POLICY_MIN_SHIFT_ALPHA <= v[2] <= 1", "v[2] <= 1")],
 "M03 shift_alpha bound flipped (#723 B1)": [("POLICY_MIN_SHIFT_ALPHA <= v[2] <= 1", "v[2] <= POLICY_MIN_SHIFT_ALPHA")],
 "M04 alpha ceiling removed": [("0 < v[0] <= POLICY_MAX_ALPHA", "0 < v[0] <= 1")],
 "M05 delta ceiling removed": [("0 < v[1] <= POLICY_MAX_DELTA", "0 < v[1] <= 1")],
 "M06 threshold floor removed": [("and 0 <= v[3] <= 1)", "and v[3] <= 1)")],
 "M07 threshold ceiling removed": [("and 0 <= v[3] <= 1)", "and 0 <= v[3])")],
 "M08 bool allowed in _num": [("not isinstance(x, bool) and ", "")],
 "M09 verify-level file pin removed": [("    if not allow_unpinned:\n        why = pin_failure", "    if False:\n        why = pin_failure")],
 "M10 MALFORMED wrapper narrowed": [("    except CAUGHT as e: return", "    except ZeroDivisionError as e: return")],
 "M11 policy check ordered after hash check": [(POLICY, ""), (HASHCK, HASHCK + POLICY)],
 "M12 on-disk kernel pin removed from loader": [('    if not allow_unpinned and h not in pinned_values(KERNEL, pins): raise Refuse("UNPINNED_KERNEL")\n', "")],
 "M13 pinned set widened with an all-zero profile": [("POLICY_MAX_ALPHA, POLICY_MAX_DELTA, POLICY_MIN_SHIFT_ALPHA = ",
     'PINNED_PROFILES["zeros"] = {n: "0" * 64 for n in PINNED_PROFILES["repo-main"]}\nPOLICY_MAX_ALPHA, POLICY_MAX_DELTA, POLICY_MIN_SHIFT_ALPHA = ')],
 "M14 CLI bad-JSON handler removed": [('            try: rc0 = json.load(open(argv[1]))\n            except CAUGHT as e: print(f"REFUSE:MALFORMED:{type(e).__name__}"); return 1\n',
     "            rc0 = json.load(open(argv[1]))\n")],
 "M15 CLI verify always exits 0": [('return 0 if v == "ACCEPT" else 1', "return 0")],
 "M16 CLI verify exits non-zero only for SHIFT": [('return 0 if v == "ACCEPT" else 1', 'return 1 if v == "REFUSE:SHIFT" else 0')],
 "M17 minimum deployment rows ignored": [("    if len(dep) < POLICY_MIN_DEPLOY:", "    if False:")],
 "M18 minimum deployment rows off by one": [("    if len(dep) < POLICY_MIN_DEPLOY:", "    if len(dep) <= POLICY_MIN_DEPLOY:")],
 "M19 model.json pin dropped": [('for n in ("model.json", "artifact_receipt.json", "calibration.jsonl"):', 'for n in ("artifact_receipt.json", "calibration.jsonl"):')],
 "M20 artifact_receipt pin dropped": [('for n in ("model.json", "artifact_receipt.json", "calibration.jsonl"):', 'for n in ("model.json", "calibration.jsonl"):')],
 "M21 calibration pin dropped": [('for n in ("model.json", "artifact_receipt.json", "calibration.jsonl"):', 'for n in ("model.json", "artifact_receipt.json"):')],
 "M22 profile mixing allowed": [('    if not any(files == p for p in pins.values()): return "UNPINNED_FILE:profile"\n', "")],
 "M23 RecursionError not caught": [("ZeroDivisionError, RecursionError, MemoryError)", "ZeroDivisionError, MemoryError)")],
 "M24 MemoryError not caught": [("ZeroDivisionError, RecursionError, MemoryError)", "ZeroDivisionError, RecursionError)")],
 "M25 argv parsed at import time": [("class Refuse(Exception): pass", 'if "--allow-unpinned-kernel" in sys.argv: sys.argv.remove("--allow-unpinned-kernel")\nclass Refuse(Exception): pass')],
 "M26 import-time exit on unpinned kernel": [("class Refuse(Exception): pass",
     'if hashlib.sha256(open(f"{MD}/{KERNEL}", "rb").read()).hexdigest() not in {p[KERNEL] for p in PINNED_PROFILES.values()}: sys.exit(3)\nclass Refuse(Exception): pass')],
 "M27 threshold need not be a calibration score": [(' or rc["threshold"] not in {s for s, _ in cal}', "")],
 "M28 binomial CDF off by one": [("def binom_cdf(k, n, p): return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))", "def binom_cdf(k, n, p): return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k))")],
 "M29 file-hash check removed": [(HASHCK, "")],
 "M30 receipt seal check removed": [('!= rc.get("receipt_sha256"): return "REFUSE:RECEIPT_TAMPERED"', '!= rc.get("receipt_sha256") and False: return "REFUSE:RECEIPT_TAMPERED"')],
 "M31 issue() emits receipts over unpinned files": [("    if why: raise Refuse(why)", "    if False: raise Refuse(why)")],
 "M33 delta ignored in verify (forgery at a calibration-score threshold)": [(' or p > rc["delta"]', "")],
 "M34 kernel cache keyed by hash only (stale __file__)": [("    key = (h, os.path.realpath(path))", "    key = h")],
 "M35 deployment rows must carry labels": [("    dep = scores_only(K, deploy_path)", "    dep = [s for s, _ in scored(K, deploy_path)]")],
 "M36 verify p-value at the policy alpha, not the receipt's": [('p = binom_cdf(k, len(pos), rc["alpha"])', "p = binom_cdf(k, len(pos), POLICY_MAX_ALPHA)")],
 "M32 kernel hash from receipt only (no pin at load, no verify-level kernel pin)": [('    if not allow_unpinned and h not in pinned_values(KERNEL, pins): raise Refuse("UNPINNED_KERNEL")\n', ""),
     ("    if not allow_unpinned:\n        why = pin_failure", "    if False:\n        why = pin_failure")],
}
def main():
    res = {}; W = tempfile.mkdtemp(prefix="_rb_mut_")
    try:
        for name, subs in M.items():
            src = SRC
            for a, b in subs:
                if a not in src: res[name] = {"killed": False, "error": "ANCHOR_NOT_FOUND"}; break
                src = src.replace(a, b, 1)
            else:
                shutil.rmtree(W, ignore_errors=True); os.makedirs(f"{W}/clinical-gateway/tools")
                os.symlink(f"{F}/clinical-gateway/huggingface", f"{W}/clinical-gateway/huggingface"); os.symlink(f"{F}/ops-health", f"{W}/ops-health")
                open(f"{W}/clinical-gateway/tools/oac_rbcg.py", "w").write(src)
                p = subprocess.run([sys.executable, "-B", f"{W}/clinical-gateway/tools/oac_rbcg.py", "selftest"], capture_output=True, text=True, cwd=tempfile.gettempdir())
                try: failed = [k for k, v in json.loads(p.stdout.strip().splitlines()[-1])["cases"].items() if not v]
                except (ValueError, IndexError, KeyError): failed = ["(crash) " + ((p.stderr.strip().splitlines() or ["?"])[-1])[:80]]
                res[name] = {"killed": p.returncode != 0, "exit": p.returncode, "failed_cases": failed}
    finally: shutil.rmtree(W, ignore_errors=True)
    allk = all(r["killed"] for r in res.values())
    print(json.dumps({"all_killed": allk, "killed": sum(r["killed"] for r in res.values()), "total": len(res), "mutants": res}, indent=1)); return 0 if allk else 5
if __name__ == "__main__": sys.exit(main())
