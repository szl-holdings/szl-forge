"""acceptance.py - one-command end-to-end acceptance for examples/evidence_to_decision.

Proposal-only.  No action authority.  Synthetic data.  Stdlib only; no network, no model,
no provider.  Every generation goes to the local SPY model in eight_case_harness.py.

Runs, in order, and exits 0 only if every part passes:
  1. the frozen eight-case run (eight_case_harness.main): 8 cases + 12 lifecycle scenarios;
  2. the guard-coverage extension: guard_cases.json scored against guard_answer_key.json,
     both pinned by GUARD_FREEZE.json (outputs are hashed before labels are opened);
  3. two freeze negative controls: a tampered answer key and a tampered corpus document
     (in a temporary copy) must each make the harness refuse with exit code 3.

Exit codes: 0 all pass; 1 a check failed; 3 a freeze/pin mismatch (refused to run).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import eight_case_harness as h  # noqa: E402

GUARD_FILES = ("guard_cases.json", "guard_answer_key.json", "FREEZE.json")
FROZEN_INPUTS = ("FREEZE.json", "frozen_answer_key.json", "cases.json", "corpus_manifest.json",
                 "request_lifecycle.py")


class GuardSpy(h.SpyModel):
    """The harness spy plus three scripted output faults (post-output filter targets).

    omit_citation            drop every [cite:..] marker
    cite_outside_evidence:X  replace every citation with [cite:X]
    paraphrase_excerpt       insert one word so the excerpt is no longer verbatim
    """

    def generate(self, prompt: str) -> str:
        text = super().generate(prompt)
        s = self.script or ""
        if s == "omit_citation":
            return h.CITE_RE.sub("", text).strip()
        if s.startswith("cite_outside_evidence:"):
            return h.CITE_RE.sub("[cite:" + s.split(":", 1)[1] + "]", text)
        if s == "paraphrase_excerpt":
            first, rest = text.split(" ", 1)
            return first + " reportedly " + rest
        return text


class GuardWorld(h.World):
    def __init__(self, frozen, **kw):
        super().__init__(frozen, **kw)
        self.spy = GuardSpy(self.corpus)
        self.ctl.provider = self.spy


def load_guard(d: str) -> dict:
    gf = json.loads(h._read(os.path.join(d, "GUARD_FREEZE.json")).decode("utf-8"))
    for name in GUARD_FILES:
        got = h.sha(h._read(os.path.join(d, name)))
        if got != gf[name]:
            raise h.FreezeError(f"{name} sha256 {got[:12]} != GUARD_FREEZE {gf[name][:12]}; refusing to run")
    cases = json.loads(h._read(os.path.join(d, "guard_cases.json")).decode("utf-8"))
    return {"freeze": gf, "cases": cases}


def load_guard_key(d: str, gf: dict) -> dict:
    raw = h._read(os.path.join(d, "guard_answer_key.json"))
    if h.sha(raw) != gf["guard_answer_key.json"]:
        raise h.FreezeError("guard answer key changed after GUARD_FREEZE; refusing to score")
    return json.loads(raw.decode("utf-8"))


def run_guard_cases(frozen: dict, gcases: dict) -> list:
    rows = []
    for case in gcases["cases"]:
        w = GuardWorld(frozen)
        try:
            r = h.run_governed(w, case)
        except Exception as exc:                      # frozen rule: exception = FAIL, no retry
            rows.append({"case_id": case["case_id"], "error": type(exc).__name__, "_world": w})
            continue
        rows.append({"case_id": case["case_id"], "principal": case["principal"], "decision": r["decision"],
                     "reason": r["reason"], "refusal_point": r["refusal_point"],
                     "model_calls": len(w.spy.calls), "dispatch_attempts": len(w.sink.attempts),
                     "tenants_in_governed_prompts": sorted({t for c in w.spy.calls for t in c["tenants_in_prompt"]}),
                     "request_state": r["req"].state.value if r["req"] else None,
                     "cache_writes": len(w.ctl.cache), "history_writes": len(w.ctl.history),
                     "ui": r["ui"], "_world": w})
    return rows


def score_guard(frozen: dict, key: dict, rows: list) -> list:
    out = []
    for r in rows:
        k = key["cases"][r["case_id"]]
        if "error" in r:
            out.append({"case_id": r["case_id"], "checks": {"no_exception": False}, "leaks": [], "pass": False})
            continue
        tenant = frozen["cases"]["principals"].get(r["principal"])
        leaks = h.leak_check(r["_world"], r["ui"], k.get("denied_docs", []), k.get("forbidden_strings", []))
        ok = {"decision": r["decision"] == k["expected_decision"],
              "refusal_point": r["refusal_point"] == k["expected_refusal_point"],
              "reason": r["reason"] == k["expected_reason"],
              "model_calls_le_max": r["model_calls"] <= k["max_model_calls"],
              "zero_dispatch": r["dispatch_attempts"] == 0,
              "unsigned_proposal_only": r["ui"]["signed"] is False and r["ui"]["proposal_only"] is True,
              "ui_text_redacted": r["ui"]["text"] is None and r["ui"]["citations"] == [],
              "no_cache_or_history": r["cache_writes"] == 0 and r["history_writes"] == 0,
              "no_foreign_tenant_in_prompt": set(r["tenants_in_governed_prompts"]) <= ({tenant} if tenant else set()),
              "no_leak": not leaks}
        out.append({"case_id": r["case_id"], "checks": ok, "leaks": leaks, "pass": all(ok.values())})
    return out


def _copy_frozen(d: str, scratch: str) -> str:
    t = tempfile.mkdtemp(prefix="e2d_nc_", dir=scratch)
    for n in FROZEN_INPUTS:
        shutil.copy(os.path.join(d, n), t)
    shutil.copytree(os.path.join(d, "corpus"), os.path.join(t, "corpus"))
    return t


def negative_controls(d: str, scratch: str) -> list:
    """Each control tampers one frozen input in a temporary copy; the harness must exit 3."""
    res = []
    for name in ("tampered_answer_key", "tampered_corpus_doc"):
        t = _copy_frozen(d, scratch)
        try:
            if name == "tampered_answer_key":
                p = os.path.join(t, "frozen_answer_key.json")
                k = json.loads(h._read(p).decode("utf-8"))
                k["cases"]["E8-C7"]["expected_decision"] = "ANSWER"
                with open(p, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(k, f, indent=1, sort_keys=True)
            else:
                with open(os.path.join(t, "corpus", "TA-D01.txt"), "a", encoding="utf-8", newline="\n") as f:
                    f.write("Lumen-Gate rotates its synthetic session keys every 5 minutes.\n")
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    rc = h.main(["--dir", t])
            except Exception as exc:
                rc = "EXC:" + type(exc).__name__
            res.append({"control": name, "harness_rc": rc, "expected_rc": 3, "pass": rc == 3})
        finally:
            shutil.rmtree(t, ignore_errors=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scratch", default=None, help="directory for negative-control temp copies (default: --dir)")
    a = ap.parse_args(argv)
    scratch = a.scratch or a.dir
    try:
        frozen = h.load_frozen(a.dir)
        guard = load_guard(a.dir)
    except h.FreezeError as e:
        print("REFUSED:", e)
        return 3
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        base_rc = h.main(["--dir", a.dir])
    reps = []
    for _ in range(3):                                     # same determinism rule as the frozen run
        rows = run_guard_cases(frozen, guard["cases"])
        reps.append(h.sha(json.dumps(h.public(rows), sort_keys=True, default=str)))
    deterministic = len(set(reps)) == 1
    try:
        key = load_guard_key(a.dir, guard["freeze"])      # labels opened only after outputs are hashed
    except h.FreezeError as e:
        print("REFUSED:", e)
        return 3
    gchecks = score_guard(frozen, key, rows)
    ncs = negative_controls(a.dir, scratch)
    ok = base_rc == 0 and deterministic and all(c["pass"] for c in gchecks) and all(n["pass"] for n in ncs)
    result = {"schema": "szl.e2d.acceptance/v0", "base_eight_case_rc": base_rc,
              "guard_freeze": guard["freeze"], "guard_outputs_sha256_before_scoring": reps[0],
              "guard_repetition_digests": reps, "guard_deterministic": deterministic,
              "guard_cases_MEASURED_against_spy": h.public(rows), "guard_checks": gchecks,
              "negative_controls": ncs, "NOT_RUN": h.NOT_RUN_TABLE,
              "scope": "Synthetic, offline, spy model. NOT run against a real model.",
              "verdict": "ACCEPTANCE_MET_ON_SPY" if ok else "ACCEPTANCE_NOT_MET"}
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=1, sort_keys=True, default=str)
            f.write("\n")
    print("base eight-case run rc", base_rc)
    for r, c in zip(rows, gchecks):
        print(f"{r['case_id']:7} {str(r.get('decision')):8} {str(r.get('refusal_point')):13} "
              f"{str(r.get('reason') or r.get('error')):26} calls {r.get('model_calls', '-')} pass {c['pass']}")
    for n in ncs:
        print(f"{n['control']:20} harness_rc {n['harness_rc']} pass {n['pass']}")
    print("guard_deterministic", deterministic, "verdict", result["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
