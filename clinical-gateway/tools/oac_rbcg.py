#!/usr/bin/env python3
"""Receipt-bound missed-alert guarantee for oac-system-health-v1 (stdlib). Commands: issue | verify RECEIPT DEPLOY.jsonl | selftest  [--allow-unpinned-kernel].
A receipt certifies, via Learn-then-Test on the calibration split, that P(missed-alert rate > alpha) <= delta at `threshold`.
verify() refuses (fail-closed) on: a receipt policy looser than the pinned policy (alpha <= 0.20, delta <= 0.10, shift_alpha >= 0.05:
a LOWER shift_alpha is more lenient; threshold finite and in [0, 1], the score range), any model.json / artifact_receipt.json / kernel /
calibration hash outside the pinned profiles (REFUSE:UNPINNED_KERNEL, REFUSE:UNPINNED_FILE:<name>) unless --allow-unpinned-kernel,
any bound-file hash change (REFUSE:HASH), a certificate that does not recompute (REFUSE:CERT), fewer than 30 deployment rows
(REFUSE:INSUFFICIENT_DEPLOYMENT), a calibration->deployment score shift (two-sample KS permutation test, p < shift_alpha), and malformed
input (REFUSE:MALFORMED:<type>, never a traceback). The kernel is executable Python: its bytes are hashed and checked against the pin
BEFORE they are compiled and run, and nothing is imported or executed when this module is imported. The CLI `verify` exits 0 only for
ACCEPT. The receipt seal is an integrity hash, not a signature; ACCEPT means "internally consistent and bound to a pinned profile",
not "issued by a trusted party". Synthetic operations telemetry only."""
import hashlib, json, math, os, random, shutil, subprocess, sys, tempfile, types
F = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root; this file lives at clinical-gateway/tools/
MD = f"{F}/clinical-gateway/huggingface/model/oac-system-health-v1"; DD = f"{F}/ops-health/v2/data/legacy_v1"
KERNEL = "oac_operational_health.py"
BOUND = ("model.json", "artifact_receipt.json", KERNEL)  # files hashed from the model dir; calibration.jsonl is the fourth bound file
ALLOW_FLAGS = ("--allow-unpinned-kernel", "--allow-unpinned")  # deliberate escape; disables every file pin below, not just the kernel's
# Pinned profiles: each is one coherent set of file hashes. A receipt must match one profile entirely (no mixing between profiles).
#   repo-main    : in-repo files on main (#722 sync)
#   hf-ed0a3bbd  : HF SZLHOLDINGS/oac-system-health-v1 @ ed0a3bbd model files (artifact_receipt and kernel differ from the in-repo copies)
# calibration.jsonl is validation.jsonl (identical bytes in the repo and in the HF dataset).
PINNED_PROFILES = {
    "repo-main": {"model.json": "f111b7fc65db561c80763b25e538366a19bed18526ca881915777487935d9557",
                  "artifact_receipt.json": "53180b900b984c034a74ea77a20e11d7fae7a8f47f0ce2397499d181ebd2b685",
                  KERNEL: "5e06ddffce9615a12c351e9926e24fefa142c920017b9dc844c584c58517202e",
                  "calibration.jsonl": "4557220a73b7f8efbb59ab41d043785cebaff71eac4d9fef8c59800a4f8f5abf"},
    "hf-ed0a3bbd": {"model.json": "f111b7fc65db561c80763b25e538366a19bed18526ca881915777487935d9557",
                    "artifact_receipt.json": "c5b58aa1f85c67ec6ce64a5516cbe5d3e72f1063eed768b4bfe997f5a18c0a27",
                    KERNEL: "e2aa63991b94596a16d69a288510b4b10af88932e075c055d97d1dac4d9a930d",
                    "calibration.jsonl": "4557220a73b7f8efbb59ab41d043785cebaff71eac4d9fef8c59800a4f8f5abf"}}
POLICY_MAX_ALPHA, POLICY_MAX_DELTA, POLICY_MIN_SHIFT_ALPHA = 0.20, 0.10, 0.05
POLICY_MIN_DEPLOY = 30  # same floor as #723 (8a233c4b); see docs section 9 for the measured justification
CAUGHT = (OverflowError, OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError, RecursionError, MemoryError)
class Refuse(Exception): pass
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def pinned_values(name, pins=None): return {p[name] for p in (PINNED_PROFILES if pins is None else pins).values()}
def pin_failure(files, pins=None):
    """None if `files` (name -> sha256) equals one pinned profile exactly, else the refusal reason."""
    pins = PINNED_PROFILES if pins is None else pins
    if not isinstance(files, dict): return "UNPINNED_FILE:files"
    if files.get(KERNEL) not in pinned_values(KERNEL, pins): return "UNPINNED_KERNEL"
    for n in ("model.json", "artifact_receipt.json", "calibration.jsonl"):
        if files.get(n) not in pinned_values(n, pins): return f"UNPINNED_FILE:{n}"
    if not any(files == p for p in pins.values()): return "UNPINNED_FILE:profile"
    return None
_KERNELS = {}
def load_kernel_class(md, allow_unpinned=False, pins=None):
    """Hash the kernel bytes, require the pin, and compile exactly those bytes (no import by path, no TOCTOU). Nothing runs before the pin."""
    path = f"{md}/{KERNEL}"; raw = open(path, "rb").read(); h = hashlib.sha256(raw).hexdigest()
    if not allow_unpinned and h not in pinned_values(KERNEL, pins): raise Refuse("UNPINNED_KERNEL")
    if h not in _KERNELS:
        mod = types.ModuleType(f"_oac_kernel_{h[:16]}"); mod.__file__ = path; sys.modules[mod.__name__] = mod  # dataclasses needs the module registered
        exec(compile(raw, path, "exec"), mod.__dict__); _KERNELS[h] = mod.OperationalHealthKernel
    return _KERNELS[h]
def kernel(md, allow_unpinned=False, pins=None):
    return load_kernel_class(md, allow_unpinned, pins)(f"{md}/model.json", f"{md}/artifact_receipt.json")
def jsonl_rows(path): return [json.loads(l) for l in open(path) if l.strip()]
def scored(K, path):
    return [(K.score(r["features"])["operator_attention_score"], int(bool(r["label"]["operator_attention_required"]))) for r in jsonl_rows(path)]
def binom_cdf(k, n, p): return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
def ks(a, b):
    A, B = sorted(a), sorted(b); i = j = 0; m = 0.0
    for x in sorted(set(a) | set(b)):
        while i < len(A) and A[i] <= x: i += 1
        while j < len(B) and B[j] <= x: j += 1
        m = max(m, abs(i / len(A) - j / len(B)))
    return m
def shift_p(a, b, B=2000, seed=11):
    rng = random.Random(seed); obs = ks(a, b); pool = a + b; c = 0
    for _ in range(B):
        rng.shuffle(pool); c += ks(pool[:len(a)], pool[len(a):]) >= obs - 1e-12
    return round(obs, 6), (c + 1) / (B + 1)
def ltt(cal, alpha, delta):
    pos = [s for s, y in cal if y]; best = None
    for t in sorted({s for s, _ in cal}):
        k = sum(1 for s in pos if s < t); p = binom_cdf(k, len(pos), alpha)
        if p <= delta: best = (t, k, len(pos), p)
        else: break
    return best
def seal(d):
    b = {k: v for k, v in d.items() if k != "receipt_sha256"}
    return {**b, "receipt_sha256": hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()}
def issue(md=MD, cal_path=f"{DD}/validation.jsonl", alpha=0.20, delta=0.10, shift_alpha=0.05, allow_unpinned=False):
    K = kernel(md, allow_unpinned); cal = scored(K, cal_path); t, k, n1, p = ltt(cal, alpha, delta)
    files = {n: sha(f"{md}/{n}") for n in BOUND}; files["calibration.jsonl"] = sha(cal_path)
    why = None if allow_unpinned else pin_failure(files)
    if why: raise Refuse(why)  # never issue a receipt that verify() would refuse as unpinned
    return seal({"schema": "szl-oac/receipt-bound-missed-alert-guarantee/v0", "files": files, "alpha": alpha, "delta": delta,
            "threshold": t, "cal_misses": k, "cal_positives": n1, "p_value": p, "shift_test": "two-sample KS on model scores, permutation B=2000 seed 11",
            "shift_alpha": shift_alpha, "claim": f"P(missed-alert rate > {alpha}) <= {delta} for deployment batches exchangeable with calibration"})
def _num(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def policy_ok(rc):
    """The seal only proves self-consistency, so the receipt cannot choose its own policy. A lower shift_alpha is MORE lenient.
    Scores are sigmoid probabilities, so a threshold outside [0, 1] can only be a degenerate or forged value."""
    v = [rc.get(k) for k in ("alpha", "delta", "shift_alpha", "threshold")]
    return (all(_num(x) for x in v) and 0 < v[0] <= POLICY_MAX_ALPHA and 0 < v[1] <= POLICY_MAX_DELTA
            and POLICY_MIN_SHIFT_ALPHA <= v[2] <= 1 and 0 <= v[3] <= 1)
def verify(rc, deploy_path, md=MD, cal_path=f"{DD}/validation.jsonl", allow_unpinned=False, pins=None):
    try: return _verify(rc, deploy_path, md, cal_path, allow_unpinned, pins)
    except Refuse as e: return f"REFUSE:{e}"
    except CAUGHT as e: return f"REFUSE:MALFORMED:{type(e).__name__}"
def _verify(rc, deploy_path, md, cal_path, allow_unpinned, pins):
    b = {k: v for k, v in rc.items() if k != "receipt_sha256"}
    if hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest() != rc.get("receipt_sha256"): return "REFUSE:RECEIPT_TAMPERED"
    if not policy_ok(rc): return "REFUSE:POLICY"
    if not allow_unpinned:
        why = pin_failure(rc["files"], pins)
        if why: return f"REFUSE:{why}"
    now = {n: sha(f"{md}/{n}") for n in BOUND}; now["calibration.jsonl"] = sha(cal_path)
    if now != rc["files"]: return "REFUSE:HASH"
    K = kernel(md, allow_unpinned, pins); cal = scored(K, cal_path); pos = [s for s, y in cal if y]
    k = sum(1 for s in pos if s < rc["threshold"]); p = binom_cdf(k, len(pos), rc["alpha"])
    if k != rc["cal_misses"] or p > rc["delta"] or rc["threshold"] not in {s for s, _ in cal}: return "REFUSE:CERT"  # issue() only emits a calibration score
    dep = [s for s, _ in scored(K, deploy_path)]
    if len(dep) < POLICY_MIN_DEPLOY: return "REFUSE:INSUFFICIENT_DEPLOYMENT"
    if shift_p([s for s, _ in cal], dep)[1] < rc["shift_alpha"]: return "REFUSE:SHIFT"
    return "ACCEPT"
def _run_cli(*args, cwd=None):
    p = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), *args], capture_output=True, text=True, cwd=cwd)
    return p.returncode, p.stdout.strip()
def selftest():
    W = tempfile.mkdtemp(prefix="_rb_fx_"); ok = {}; V = f"{DD}/validation.jsonl"; T = f"{DD}/test.jsonl"
    global jsonl_rows
    keep_rows = jsonl_rows
    try:
        rc = issue(); ok["issue_reproduces_ltt_0.2"] = abs(rc["threshold"] - 0.137721) < 1e-6 and rc["cal_misses"] == 3
        ok["kat_binom"] = abs(binom_cdf(0, 33, 0.1) - 0.9 ** 33) < 1e-15 and abs(binom_cdf(1, 33, 0.1) - (0.9 ** 33 + 33 * 0.1 * 0.9 ** 32)) < 1e-15
        ok["accept_exchangeable"] = verify(rc, V) == "ACCEPT"           # deployment == calibration
        ok["refuse_shifted_test"] = verify(rc, T) == "REFUSE:SHIFT"      # real val->test shift (p~0.03)
        # tamper calibration data: flip one label in a copy
        lines = open(V).read().splitlines(True); r0 = json.loads(lines[0])
        r0["label"]["operator_attention_required"] = not r0["label"]["operator_attention_required"]; lines[0] = json.dumps(r0) + "\n"
        open(f"{W}/cal.jsonl", "w").writelines(lines)
        ok["refuse_tampered_calibration"] = verify(rc, V, cal_path=f"{W}/cal.jsonl") == "REFUSE:HASH"
        bad = dict(rc); bad["threshold"] = 0.2; ok["refuse_edited_threshold"] = verify(bad, V) == "REFUSE:RECEIPT_TAMPERED"
        ok["refuse_resealed_uncertifiable_threshold"] = verify(seal(bad), V) == "REFUSE:CERT"
        # internally consistent forgery: true miss count at an uncertifiable threshold, resealed -> only the delta check can refuse
        Kc = kernel(MD); calp = [x for x, y in scored(Kc, V) if y]
        forg = dict(rc); forg["threshold"] = 0.2; forg["cal_misses"] = sum(1 for x in calp if x < 0.2)
        forg["p_value"] = binom_cdf(forg["cal_misses"], len(calp), forg["alpha"])
        ok["refuse_consistent_forgery_failing_delta"] = forg["p_value"] > rc["delta"] and verify(seal(forg), V) == "REFUSE:CERT"
        # synthetic shift: deployment = top-scored half of test
        rows = [l for l in open(T) if l.strip()]
        rows.sort(key=lambda l: -Kc.score(json.loads(l)["features"])["operator_attention_score"])
        open(f"{W}/hot.jsonl", "w").writelines(rows[:120])
        ok["refuse_strong_shift"] = verify(rc, f"{W}/hot.jsonl") == "REFUSE:SHIFT"
        # review hardening: a resealed receipt may not pick a looser policy (lower shift_alpha is MORE lenient)
        ok["control_resealed_policy_still_refuses_shift"] = verify(seal(rc), T) == "REFUSE:SHIFT"
        ok["refuse_shift_alpha_lenient_values"] = all(verify(seal({**rc, "shift_alpha": v}), T) == "REFUSE:POLICY" for v in (0, -1, 1e-300, 0.049, float("nan"), True))
        ok["refuse_relaxed_alpha_delta"] = all(verify(seal(d), V) == "REFUSE:POLICY" for d in
            ({**rc, "alpha": 0.99, "threshold": 0.0, "cal_misses": 0, "p_value": binom_cdf(0, len(calp), 0.99)}, {**rc, "delta": 0.99}, {**rc, "threshold": float("nan")}))
        ok["refuse_malformed_no_traceback"] = (verify(seal({**rc, "alpha": 10 ** 400}), V).startswith("REFUSE:MALFORMED")
            and verify([], V).startswith("REFUSE:MALFORMED") and verify(rc, f"{W}/missing.jsonl").startswith("REFUSE:MALFORMED"))
        # review fix 4: RecursionError and MemoryError are MALFORMED, not tracebacks
        deep = {}; cur = deep
        for _ in range(100000): cur["a"] = {}; cur = cur["a"]
        open(f"{W}/deep.jsonl", "w").write("[" * 100000 + "]" * 100000 + "\n")
        def _mem(*a, **k): raise MemoryError
        jsonl_rows = _mem; mem = verify(rc, V); jsonl_rows = keep_rows
        ok["malformed_recursion_and_memory"] = (verify({**rc, "x": deep}, V).startswith("REFUSE:MALFORMED")
            and verify(rc, f"{W}/deep.jsonl") == "REFUSE:MALFORMED:RecursionError" and mem == "REFUSE:MALFORMED:MemoryError")
        # review fix 2: minimum deployment rows (0 is refused explicitly, not by a ZeroDivisionError)
        drows = [l for l in open(T) if l.strip()]; dv = {}
        for n in (0, 1, 3, 10, 29, 30):
            open(f"{W}/d{n}.jsonl", "w").writelines(drows[:n]); dv[n] = verify(rc, f"{W}/d{n}.jsonl")
        ok["min_deployment_rows_0_1_3_10_29_refused_30_ok"] = (all(dv[n] == "REFUSE:INSUFFICIENT_DEPLOYMENT" for n in (0, 1, 3, 10, 29))
            and dv[30] in ("ACCEPT", "REFUSE:SHIFT"))
        # review fix 5: threshold floor/ceiling (scores are probabilities) and threshold must be a calibration score
        npos = len(calp)
        def thr(t, miss): return seal({**rc, "threshold": t, "cal_misses": miss, "p_value": binom_cdf(miss, npos, rc["alpha"])})
        ok["refuse_threshold_outside_score_range"] = all(verify(thr(t, 0), V) == "REFUSE:POLICY" for t in (-1e9, -0.01, 1.01, 1e9, float("inf")))
        ok["refuse_threshold_not_a_calibration_score"] = verify(thr(1e-9, 0), V) == "REFUSE:CERT"
        # review fix 3: model.json / artifact_receipt / calibration / kernel hashes are pinned (profiles are coherent, zeros never pinned)
        zero = "0" * 64
        pf = lambda name, v: verify(seal({**rc, "files": {**rc["files"], name: v}}), V)
        ok["refuse_unpinned_file_hashes"] = (pf("model.json", zero) == "REFUSE:UNPINNED_FILE:model.json"
            and pf("artifact_receipt.json", zero) == "REFUSE:UNPINNED_FILE:artifact_receipt.json"
            and pf("calibration.jsonl", zero) == "REFUSE:UNPINNED_FILE:calibration.jsonl" and pf(KERNEL, zero) == "REFUSE:UNPINNED_KERNEL"
            and pf(KERNEL, ["x"]).startswith("REFUSE:MALFORMED") and verify(seal({**rc, "files": None}), V) == "REFUSE:UNPINNED_FILE:files")
        # check order is seal -> policy -> pin -> file hashes (a bad policy is reported even when the files also differ)
        zk = {**rc["files"], KERNEL: zero}
        ok["check_order_policy_then_pin_then_hash"] = (verify(seal({**rc, "shift_alpha": 0, "files": zk}), V) == "REFUSE:POLICY"
            and verify(seal({**rc, "files": zk}), V, cal_path=f"{W}/cal.jsonl") == "REFUSE:UNPINNED_KERNEL"
            and verify(rc, V, cal_path=f"{W}/cal.jsonl") == "REFUSE:HASH")
        mix = {**PINNED_PROFILES["repo-main"], KERNEL: PINNED_PROFILES["hf-ed0a3bbd"][KERNEL]}  # each hash pinned, combination is not
        ok["refuse_mixed_profiles"] = pin_failure(mix) == "UNPINNED_FILE:profile" and all(pin_failure(p) is None for p in PINNED_PROFILES.values())
        want = {"model.json": {"f111b7fc65db561c80763b25e538366a19bed18526ca881915777487935d9557"},
                "artifact_receipt.json": {"53180b900b984c034a74ea77a20e11d7fae7a8f47f0ce2397499d181ebd2b685", "c5b58aa1f85c67ec6ce64a5516cbe5d3e72f1063eed768b4bfe997f5a18c0a27"},
                KERNEL: {"5e06ddffce9615a12c351e9926e24fefa142c920017b9dc844c584c58517202e", "e2aa63991b94596a16d69a288510b4b10af88932e075c055d97d1dac4d9a930d"},
                "calibration.jsonl": {"4557220a73b7f8efbb59ab41d043785cebaff71eac4d9fef8c59800a4f8f5abf"}}
        hexok = lambda h: len(h) == 64 and set(h) <= set("0123456789abcdef") and set(h) != {"0"}
        ok["pinned_sets_exact_and_no_zero_hash"] = (len(PINNED_PROFILES) == 2 and all(set(p) == set(want) for p in PINNED_PROFILES.values())
            and all(pinned_values(n) == want[n] and all(hexok(h) for h in pinned_values(n)) for n in want))
        # the override is deliberate: an unpinned receipt is refused by default and ACCEPTed only with allow_unpinned
        unp = verify(rc, V, pins={}); ovr = verify(rc, V, pins={}, allow_unpinned=True)
        ok["refuse_unpinned_kernel_unless_flag"] = unp == "REFUSE:UNPINNED_KERNEL" and ovr == "ACCEPT"
        # issue() never emits a receipt over unpinned files (artifact_receipt.json and the calibration split are edited here without changing meaning)
        md2 = f"{W}/md2"; shutil.copytree(MD, md2); open(f"{md2}/artifact_receipt.json", "a").write("\n")
        open(f"{W}/cal_nl.jsonl", "w").write(open(V).read() + "\n")
        def why_issue(**kw):
            try: issue(**kw); return None
            except Refuse as e: return str(e)
        ok["issue_refuses_unpinned_files_unless_override"] = (why_issue(md=md2) == "UNPINNED_FILE:artifact_receipt.json"
            and why_issue(cal_path=f"{W}/cal_nl.jsonl") == "UNPINNED_FILE:calibration.jsonl"
            and why_issue(md=md2, allow_unpinned=True) is None and why_issue(cal_path=f"{W}/cal_nl.jsonl", allow_unpinned=True) is None)
        # on-disk kernel pin (fix 6a): an edited kernel is never executed. The edit appends a marker-file write.
        evil = f"{W}/evil_model"; shutil.copytree(MD, evil); marker = f"{W}/MARKER"
        open(f"{evil}/{KERNEL}", "a").write(f"\nopen({marker!r}, 'w').write('executed')\n")
        def refused(fn):
            try: fn(); return False
            except Refuse as e: return str(e) == "UNPINNED_KERNEL"
            except Exception: return False  # e.g. the kernel started executing: a failed case, not a crash
        r1 = refused(lambda: load_kernel_class(evil)); r2 = refused(lambda: issue(md=evil)); gone = not os.path.exists(marker)
        fake = seal({**rc, "files": {**rc["files"], KERNEL: sha(f"{evil}/{KERNEL}")}})  # receipt that claims the edited kernel
        v_fake = verify(fake, V, md=evil); gone2 = not os.path.exists(marker)
        ok["unpinned_kernel_never_executed"] = r1 and r2 and gone and gone2 and v_fake == "REFUSE:UNPINNED_KERNEL"
        load_kernel_class(evil, allow_unpinned=True)
        ok["override_executes_unpinned_kernel_control"] = os.path.exists(marker)
        # import-time behaviour (fix 7 and 6a): importing never exits, never reads argv, never executes the kernel
        tree = f"{W}/tree"; os.makedirs(f"{tree}/clinical-gateway/tools"); os.makedirs(f"{tree}/clinical-gateway/huggingface/model")
        shutil.copy(os.path.abspath(__file__), f"{tree}/clinical-gateway/tools/oac_rbcg.py")
        shutil.copytree(evil, f"{tree}/clinical-gateway/huggingface/model/oac-system-health-v1"); os.symlink(f"{F}/ops-health", f"{tree}/ops-health")
        os.remove(marker); tools = f"{tree}/clinical-gateway/tools"
        p = subprocess.run([sys.executable, "-B", "-c", "import sys; sys.argv = ['x', '--allow-unpinned-kernel', 'bogus']; import oac_rbcg; print('IMPORT_OK', sys.argv[1:])"],
                           capture_output=True, text=True, cwd=tools, env={**os.environ, "PYTHONPATH": tools})
        ok["import_has_no_side_effects"] = (p.returncode == 0 and p.stdout.strip() == "IMPORT_OK ['--allow-unpinned-kernel', 'bogus']" and not os.path.exists(marker))
        p = subprocess.run([sys.executable, "-B", f"{tools}/oac_rbcg.py", "issue"], capture_output=True, text=True, cwd=W)
        ok["cli_issue_refuses_unpinned_kernel"] = p.returncode == 3 and "REFUSE:UNPINNED_KERNEL" in p.stdout and not os.path.exists(marker)
        # review fix 1 and 6c: CLI verify exits 0 only for ACCEPT; bad JSON is a MALFORMED verdict with a non-zero exit
        json.dump(rc, open(f"{W}/rc.json", "w")); json.dump(seal({**rc, "shift_alpha": 0}), open(f"{W}/rc_policy.json", "w"))
        open(f"{W}/notjson.json", "w").write("{not json"); open(f"{W}/nested.json", "w").write("[" * 100000 + "]" * 100000)
        cl = {"accept": _run_cli("verify", f"{W}/rc.json", V), "shift": _run_cli("verify", f"{W}/rc.json", T),
              "policy": _run_cli("verify", f"{W}/rc_policy.json", T), "badjson": _run_cli("verify", f"{W}/notjson.json", V),
              "nested": _run_cli("verify", f"{W}/nested.json", V), "missing": _run_cli("verify", f"{W}/none.json", V),
              "short": _run_cli("verify", f"{W}/rc.json"), "baddep": _run_cli("verify", f"{W}/rc.json", f"{W}/none.jsonl")}
        ok["cli_exit_0_only_for_accept"] = (cl["accept"] == (0, "ACCEPT") and cl["shift"] == (1, "REFUSE:SHIFT") and cl["policy"] == (1, "REFUSE:POLICY")
            and cl["baddep"][0] == 1 and cl["baddep"][1].startswith("REFUSE:MALFORMED:") and cl["short"][0] == 2)
        ok["cli_bad_json_is_malformed_nonzero"] = (cl["badjson"][0] == 1 and cl["badjson"][1].startswith("REFUSE:MALFORMED:") and "Traceback" not in cl["badjson"][1]
            and cl["nested"] == (1, "REFUSE:MALFORMED:RecursionError") and cl["missing"][0] == 1 and cl["missing"][1].startswith("REFUSE:MALFORMED:FileNotFoundError"))
    finally:
        jsonl_rows = keep_rows; shutil.rmtree(W, ignore_errors=True)
    res = {"cases": ok, "result": "PASS" if all(ok.values()) else "FAIL"}
    print(json.dumps(res)); return 0 if res["result"] == "PASS" else 3
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv); allow = any(f in argv for f in ALLOW_FLAGS)  # parsed here, never at import time
    argv = [a for a in argv if a not in ALLOW_FLAGS]; cmd = argv[0] if argv else "selftest"
    try:
        if cmd == "issue":
            rc = issue(allow_unpinned=allow); json.dump(rc, open("oac_rbcg_receipt.json", "w"), indent=1)
            print(json.dumps({"threshold": rc["threshold"], "receipt": rc["receipt_sha256"]})); return 0
        if cmd == "verify" and len(argv) == 3:
            try: rc0 = json.load(open(argv[1]))
            except CAUGHT as e: print(f"REFUSE:MALFORMED:{type(e).__name__}"); return 1
            v = verify(rc0, argv[2], allow_unpinned=allow); print(v)
            if allow: print("allow-unpinned-kernel: file pins disabled for this run", file=sys.stderr)
            return 0 if v == "ACCEPT" else 1
        if cmd == "selftest": return selftest()
    except Refuse as e:
        print(f"REFUSE:{e}"); return 3
    print("usage: oac_rbcg.py issue | verify RECEIPT DEPLOY.jsonl | selftest  [--allow-unpinned-kernel]", file=sys.stderr); return 2
if __name__ == "__main__":
    sys.exit(main())
