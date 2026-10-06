#!/usr/bin/env python3
"""Stdlib bounded-loop runner mirroring the ouroboros loop contract (workstream E, ours).

Purpose: drive v2 hyperparameter search as a bounded loop with a hard iteration
cap, a fixed search space, and one governed receipt per iteration (CONTRACT §5.3-5.4).
Standard library only; no network I/O of any kind.

Loop semantics REPORTED from szl-holdings/ouroboros @ 0f030741
(src/loop-kernel.ts, src/types.ts, src/receipt-emitter.ts,
packages/ouroboros/src/loop-kernel.ts); this adapter is OURS:

  MIRRORED
  * Hard iteration budget: the loop runs at most `max_iterations` steps
    (ouroboros `maxSteps`; `for (i = 0; i < maxSteps; i++)`).
  * Budget exit: finishing the budget with work left = CAP_REACHED
    (ouroboros exitReason 'budgetExhausted').
  * Caller-signalled stop: a trial returning {"abort": true} stops the loop
    = ABORTED (ouroboros 'aborted').
  * Errors are never swallowed: an exception from the trial propagates to the
    caller (ouroboros: "Kernel never swallows errors").
  * One loop-summary receipt after exit carrying id/label/exit reason/steps run/
    max steps (ouroboros buildLoopReceipt canonical fields), with Λ and energy
    reported honestly as null / UNAVAILABLE (never fabricated).
  * Trace is the product: a JSON trace with per-step records.

  DIFFERENT (deliberate)
  * Stop reason SEARCH_SPACE_EXHAUSTED (not in ouroboros): every candidate of the
    fixed space was evaluated.  Precedence: if the space is exhausted exactly when
    the cap is hit, SEARCH_SPACE_EXHAUSTED is reported (cap_reached flag = true).
    Ouroboros would report 'budgetExhausted' in that case.
  * Invalid caps FAIL CLOSED with an exception (None, bool, float, NaN, Inf,
    negative, > HARD_MAX_ITERATIONS).  ouroboros src/ instead falls back to the
    default 8 for non-finite values and floors fractions; packages/ouroboros/
    does neither (a fractional/Infinity maxSteps is used as-is there).
  * One receipt PER ITERATION (ouroboros emits one receipt per LOOP), including
    an aborted attempt, chained with prev/digest via math/receipt_adapter.py
    and WRITTEN TO A LOCAL DIRECTORY.  ouroboros POSTs its loop receipt
    fire-and-forget to SZL_RECEIPT_SINK or a default remote URL; this adapter
    never touches the network.
  * Not mirrored: 'converged' (adjacent-state delta), 'consistent' (online
    consistency proxy), retroactive earliestSafeExit, depth allocator, Λ gate,
    GradeVec receipts of runtime/types, dual-witness closure.  A grid/random
    search over a fixed space has no state delta to converge on.
  * Receipts attest integrity and origin of each trial record only -- never the
    quality of a candidate.

Usage:
    python -B math/ouroboros_adapter.py --self-test [--out DIR]
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import receipt_adapter as ra  # noqa: E402

__all__ = ["SEARCH_SPACE_EXHAUSTED", "CAP_REACHED", "ABORTED", "HARD_MAX_ITERATIONS",
           "run_bounded_search", "select_best", "verify_run_dir"]

ADAPTER_VERSION = "1.0.1"
OUROBOROS_REF = "szl-holdings/ouroboros@0f030741"
SEARCH_SPACE_EXHAUSTED = "SEARCH_SPACE_EXHAUSTED"
CAP_REACHED = "CAP_REACHED"
ABORTED = "ABORTED"
HARD_MAX_ITERATIONS = 100_000  # absolute ceiling regardless of caller input


def _check_cap(max_iterations: Any) -> int:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an int (got %r)" % (max_iterations,))
    if max_iterations < 0 or max_iterations > HARD_MAX_ITERATIONS:
        raise ValueError("max_iterations must be in [0, %d]" % HARD_MAX_ITERATIONS)
    return max_iterations


def _freeze_space(search_space: Any) -> List[Any]:
    if not isinstance(search_space, (list, tuple)):
        raise TypeError("search_space must be a list/tuple (fixed and finite), got %s"
                        % type(search_space).__name__)
    # JSON round-trip: deep copy + proof every candidate is canonical-JSON-able
    return json.loads(ra.canonical_json(list(search_space)).decode("utf-8"))


def run_bounded_search(trial_fn: Callable[[Any, int], Dict[str, Any]],
                       search_space: Sequence[Any], *, max_iterations: int,
                       receipts_dir: str, run_id: str, label: str = "oac.bounded-search",
                       ns: str = "oac-frontier", action: str = "hyperparameter-trial",
                       order_seed: Optional[int] = None,
                       ts_fn: Optional[Callable[[], Any]] = None,
                       origin: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate candidates of a fixed search space under a hard cap.

    trial_fn(candidate, iteration) -> JSON-able dict of results (e.g. validation
    metrics), or {"abort": True} to stop.  Candidates are visited in the given
    order, or in a seeded permutation when ``order_seed`` is an int.  Writes one
    receipt per iteration plus one loop-summary receipt into ``receipts_dir``
    (must be absent or empty) and a ``trace.json`` holding every receipted
    payload so payload digests can be re-checked.  Returns the trace dict.
    """
    cap = _check_cap(max_iterations)
    space = _freeze_space(search_space)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if os.path.isdir(receipts_dir) and os.listdir(receipts_dir):
        raise FileExistsError("receipts_dir is not empty (refusing to mix chains): %s"
                              % receipts_dir)
    os.makedirs(receipts_dir, exist_ok=True)
    order = list(range(len(space)))
    if order_seed is not None:
        if isinstance(order_seed, bool) or not isinstance(order_seed, int):
            raise TypeError("order_seed must be an int or None")
        random.Random(order_seed).shuffle(order)
    space_digest = ra.payload_digest(space)
    order_digest = ra.payload_digest(order)
    ts_fn = ts_fn or (lambda: None)
    base_origin = {"runner": "oac-frontier/math/ouroboros_adapter.py",
                   "runner_version": ADAPTER_VERSION, "mirrors": OUROBOROS_REF}
    if origin:
        base_origin.update(copy.deepcopy(origin))

    loop_meta = {"run_id": run_id, "label": label, "max_iterations": cap,
                 "search_space_size": len(space), "search_space_digest": space_digest,
                 "order_seed": order_seed, "order_digest": order_digest}
    steps: List[Dict[str, Any]] = []
    receipts: List[Dict[str, Any]] = []
    aborted = False
    t0 = time.perf_counter()
    n_iter = min(cap, len(space))            # hard bound, fixed before the loop starts
    for i in range(n_iter):
        idx = order[i]
        candidate = copy.deepcopy(space[idx])
        s0 = time.perf_counter()
        result = trial_fn(candidate, i)      # exceptions propagate (never swallowed)
        dur = (time.perf_counter() - s0) * 1000.0
        if not isinstance(result, dict):
            raise TypeError("trial_fn must return a dict, got %s" % type(result).__name__)
        # Freeze the result immediately: a deep, JSON-canonical copy decouples the
        # trace from any object the trial_fn keeps mutating (aliasing), and fails
        # closed on non-JSON values (NaN/Inf, sets, ...) before this iteration's
        # receipt is written.
        result = json.loads(ra.canonical_json(result).decode("utf-8"))
        is_abort = result.get("abort") is True
        payload = {"kind": "trial", "run_id": run_id, "iteration": i,
                   "candidate_index": idx, "candidate": space[idx],
                   "result": result, "search_space_digest": space_digest}
        rec = ra.make_receipt(payload, action=action, ns=ns,
                              prev_receipt=receipts[-1] if receipts else None,
                              ts=ts_fn(), origin=base_origin,
                              extra={"loop": dict(loop_meta, iteration=i)})
        path = ra.write_receipt(rec, receipts_dir)
        receipts.append(rec)
        steps.append({"iteration": i, "candidate_index": idx, "candidate": space[idx],
                      "result": result, "payload": payload,
                      "receipt_file": os.path.basename(path), "receipt_digest": rec["digest"],
                      "duration_ms": dur})
        if is_abort:
            aborted = True
            break
    ran = len(steps)
    if aborted:
        stop = ABORTED
    elif ran == len(space):
        stop = SEARCH_SPACE_EXHAUSTED
    else:
        stop = CAP_REACHED
    summary = {"kind": "loop-summary", "run_id": run_id, "label": label,
               "stop_reason": stop, "iterations_run": ran, "max_iterations": cap,
               "search_space_size": len(space), "search_space_digest": space_digest,
               "order_digest": order_digest,
               "cap_reached": ran >= cap, "space_exhausted": ran == len(space) and not aborted,
               "iteration_receipt_digests": [r["digest"] for r in receipts],
               "lambda": None, "energy": {"joules": None, "label": "UNAVAILABLE"}}
    srec = ra.make_receipt(summary, action="bounded-loop-summary", ns=ns,
                           prev_receipt=receipts[-1] if receipts else None,
                           ts=ts_fn(), origin=base_origin,
                           extra={"loop": dict(loop_meta, stop_reason=stop)})
    spath = ra.write_receipt(srec, receipts_dir)
    trace = {"run_id": run_id, "label": label, "stop_reason": stop,
             "iterations_run": ran, "max_iterations": cap,
             "search_space_size": len(space), "search_space_digest": space_digest,
             "order": order[:n_iter], "order_seed": order_seed,
             "steps": steps, "summary_payload": summary,
             "summary_receipt_file": os.path.basename(spath),
             "summary_receipt_digest": srec["digest"],
             "total_duration_ms": (time.perf_counter() - t0) * 1000.0,
             "semantics": "mirrors %s bounded-loop contract (see module docstring); "
                          "receipts attest integrity/origin only" % OUROBOROS_REF}
    with open(os.path.join(receipts_dir, "trace.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(trace, fh, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        fh.write("\n")
    return trace


def select_best(trace: Dict[str, Any], key: str, mode: str = "max") -> Optional[Dict[str, Any]]:
    """Deterministic best step by result[key] (finite numbers only; ties -> earliest iteration).

    Use ONLY with validation-split metrics (CONTRACT §5.3)."""
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")
    best = None
    for s in trace["steps"]:
        v = s["result"].get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        if best is None or (v > best[0] if mode == "max" else v < best[0]):
            best = (v, s)
    return None if best is None else best[1]


def verify_run_dir(receipts_dir: str) -> List[str]:
    """Re-verify a run directory: every receipt, the chain, and payload digests vs trace.json."""
    errs: List[str] = []
    with open(os.path.join(receipts_dir, "trace.json"), "r", encoding="utf-8") as fh:
        trace = json.load(fh)
    recs = {}
    for name in sorted(os.listdir(receipts_dir)):
        if name == "trace.json" or not name.endswith(".json"):
            continue
        recs[name] = ra.read_receipt(os.path.join(receipts_dir, name))
    for st in trace["steps"]:
        r = recs.get(st["receipt_file"])
        if r is None:
            errs.append("missing receipt %s" % st["receipt_file"])
            continue
        errs += ["%s: %s" % (st["receipt_file"], e)
                 for e in ra.verify_receipt(r, st["payload"], check_payload=True)]
    sr = recs.get(trace["summary_receipt_file"])
    if sr is None:
        errs.append("missing summary receipt")
    else:
        errs += ["summary: %s" % e for e in
                 ra.verify_receipt(sr, trace["summary_payload"], check_payload=True)]
    errs += ["chain: %s" % e for e in ra.verify_chain(list(recs.values()))]
    if len(recs) != trace["iterations_run"] + 1:
        errs.append("receipt count %d != iterations_run+1 (%d)"
                    % (len(recs), trace["iterations_run"] + 1))
    return errs


# --------------------------------------------------------------------------- #
# self-test                                                                   #
# --------------------------------------------------------------------------- #
def _self_test(out_dir: Optional[str]) -> int:
    import shutil
    import tempfile
    base = out_dir or tempfile.mkdtemp(prefix="ouro_selftest_")
    if os.path.isdir(base):
        shutil.rmtree(base)
    os.makedirs(base)
    print("ouroboros_adapter %s self-test | python %s | mirrors %s"
          % (ADAPTER_VERSION, sys.version.split()[0], OUROBOROS_REF))
    print("trials are SYNTHETIC toy functions; results describe loop mechanics only")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print("  %-4s %s %s" % ("PASS" if ok else "FAIL", name, detail))

    fixed = lambda: "2026-09-26T00:00:00Z"  # noqa: E731
    toy = lambda c, i: {"val_score": round(1.0 - abs(c["x"] - 0.3), 6)}  # noqa: E731
    grid = [{"x": round(0.1 * k, 1)} for k in range(10)]

    def run(name, space, cap, fn=toy, run_id=None, **kw):
        d = os.path.join(base, name)
        return d, run_bounded_search(fn, space, max_iterations=cap, receipts_dir=d,
                                     run_id=run_id or name, ts_fn=fixed, **kw)

    for name, space, cap, want_stop, want_n in [
        ("cap_lt_space", grid, 4, CAP_REACHED, 4),
        ("cap_gt_space", grid[:3], 10, SEARCH_SPACE_EXHAUSTED, 3),
        ("cap_eq_space", grid[:5], 5, SEARCH_SPACE_EXHAUSTED, 5),
        ("empty_space", [], 3, SEARCH_SPACE_EXHAUSTED, 0),
        ("cap_zero", grid, 0, CAP_REACHED, 0),
    ]:
        d, tr = run(name, space, cap)
        n_files = len([f for f in os.listdir(d) if f != "trace.json"])
        errs = verify_run_dir(d)
        check("%-13s stop=%s iters=%d receipts=%d" % (name, tr["stop_reason"],
                                                      tr["iterations_run"], n_files),
              tr["stop_reason"] == want_stop and tr["iterations_run"] == want_n
              and n_files == want_n + 1 and not errs, "; ".join(errs[:2]))
    _, tr = run("cap_eq_flags", grid[:5], 5)
    check("cap==space flags (cap_reached & space_exhausted)",
          tr["summary_payload"]["cap_reached"] and tr["summary_payload"]["space_exhausted"])

    def aborting(c, i):
        return {"abort": True} if i == 2 else toy(c, i)
    d, tr = run("abort_at_2", grid, 8, fn=aborting)
    check("abort at iteration 2 -> ABORTED, 3 iteration receipts",
          tr["stop_reason"] == ABORTED and tr["iterations_run"] == 3 and not verify_run_dir(d))
    # hard cap is never exceeded even if trial_fn tries to extend the space
    calls = []

    def greedy(c, i):
        calls.append(i)
        grid_copy.append({"x": 9.9})  # mutating the caller's list must not matter
        return toy(c, i)
    grid_copy = list(grid)
    d, tr = run("mutating_trial", grid_copy, 6, fn=greedy)
    check("cap holds & space frozen under mutation", len(calls) == 6 and
          tr["search_space_size"] == 10 and tr["stop_reason"] == CAP_REACHED)
    # aliasing: a trial_fn that returns the SAME mutable dict every time must not
    # rewrite earlier trace steps (trace/select_best/verify_run_dir must see the
    # per-iteration values that were receipted)
    shared = {}

    def aliased(c, i):
        shared["val_score"] = round(0.1 * i, 6)
        return shared
    d, tr = run("aliased_result", grid[:5], 5, fn=aliased)
    vals = [s["result"]["val_score"] for s in tr["steps"]]
    ab = select_best(tr, "val_score")
    ae = verify_run_dir(d)
    check("aliased result dict frozen per iteration (trace, select_best, verify)",
          vals == [0.0, 0.1, 0.2, 0.3, 0.4] and ab is not None and ab["iteration"] == 4
          and not ae, "(vals=%s best=%s errs=%d)" % (vals, ab and ab["iteration"], len(ae)))
    # determinism
    d1, t1 = run("det_a", grid, 7, order_seed=123, run_id="det")
    d2, t2 = run("det_b", grid, 7, order_seed=123, run_id="det")
    same = all(open(os.path.join(d1, f), "rb").read() == open(os.path.join(d2, f), "rb").read()
               for f in os.listdir(d1) if f != "trace.json")
    check("determinism: same seed+ts -> byte-identical receipts", same
          and t1["order"] == t2["order"] and sorted(os.listdir(d1)) == sorted(os.listdir(d2)))
    _, t3 = run("det_c", grid, 7, order_seed=124, run_id="det")
    check("different order_seed -> different visit order", t3["order"] != t1["order"])
    best = select_best(t1, "val_score")
    top = max(s["result"]["val_score"] for s in t1["steps"])
    first_top = min(s["iteration"] for s in t1["steps"] if s["result"]["val_score"] == top)
    check("select_best picks max val_score (tie -> earliest)",
          best is not None and best["iteration"] == first_top,
          "(iteration %s, val_score %s)" % (best and best["iteration"], top))
    # fail-closed inputs
    for bad in (None, True, 1.5, float("inf"), float("nan"), -1, HARD_MAX_ITERATIONS + 1):
        try:
            run("bad_cap_%s" % str(bad).replace(".", "_"), grid, bad)
            check("rejects cap %r" % (bad,), False)
        except (TypeError, ValueError) as exc:
            check("rejects cap %r" % (bad,), True, "(%s)" % type(exc).__name__)
    try:
        run("gen_space", (g for g in grid), 3)
        check("rejects generator search space", False)
    except TypeError:
        check("rejects generator search space", True)
    try:
        run("cap_lt_space", grid, 2)
        check("refuses non-empty receipts_dir", False)
    except FileExistsError:
        check("refuses non-empty receipts_dir", True)
    try:
        run("nan_space", [{"x": float("nan")}], 1)
        check("rejects NaN candidate", False)
    except ValueError:
        check("rejects NaN candidate", True)

    def boom(c, i):
        if i == 3:
            raise RuntimeError("trial failure")
        return toy(c, i)
    d = os.path.join(base, "error_propagates")
    try:
        run_bounded_search(boom, grid, max_iterations=8, receipts_dir=d, run_id="e", ts_fn=fixed)
        check("trial exception propagates", False)
    except RuntimeError:
        recs = [ra.read_receipt(os.path.join(d, f)) for f in sorted(os.listdir(d))]
        check("trial exception propagates; 3 completed receipts remain & chain verifies",
              len(recs) == 3 and not ra.verify_chain(recs) and
              all(not ra.verify_receipt(r) for r in recs))
    # tamper a receipt on disk
    d = os.path.join(base, "cap_lt_space")
    f = sorted(x for x in os.listdir(d) if x != "trace.json")[1]
    p = os.path.join(d, f)
    r = ra.read_receipt(p)
    r["ns"] = "tampered"
    ra.write_receipt(r, os.path.join(base, "tampered_copy"), name=f)
    for g in os.listdir(d):
        if g != f:
            with open(os.path.join(d, g), "rb") as a, \
                    open(os.path.join(base, "tampered_copy", g), "wb") as b:
                b.write(a.read())
    check("detects tampered receipt in run dir",
          bool(verify_run_dir(os.path.join(base, "tampered_copy"))))
    n_fail = sum(1 for ok in results if not ok)
    print("output dir: %s" % base)
    print("SUMMARY: %d checks, %d failed -> %s" % (len(results), n_fail,
                                                   "PASS" if n_fail == 0 else "FAIL"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        od = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
        raise SystemExit(_self_test(od))
    print(__doc__)
