"""mutation_sweep.py - remove each tagged harness guard and record which layer notices.

Stdlib only.  For every tag ``# GUARD:Hxx`` in eight_case_harness.py, a temporary copy of
this directory is made in which the condition of every ``if``/``elif`` line carrying that
tag is replaced with ``False``.  Three layers are then run against the copy:

  unit             python -m unittest discover (all test_*.py here)
  base_acceptance  eight_case_harness.py        (the frozen eight-case run only)
  acceptance       acceptance.py                (frozen run + guard cases + negative controls)

A mutant is KILLED only by completed behavioral failures. Timeouts, crashes, newly skipped
or missing tests, and malformed acceptance evidence are INVALID, never kills. The sweep
REFUSES to score (exit 3) unless all clean layers pass. Invalid mutant layers fail the
sweep (exit 1), even if another layer detects the mutation.

Usage: python mutation_sweep.py [--scratch DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "eight_case_harness.py"
TAG_RE = re.compile(r"# GUARD:(H\d+)")
UNIT = ["-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-t", "."]


def unit_outcome(rc, stderr, target_path=None):
    """Recognize assertions or completed test errors traced to mutated code."""
    counts = re.findall(r"^Ran (\d+) tests? in .+$", stderr, re.MULTILINE)
    lines = stderr.strip().splitlines()
    last = lines[-1] if lines else ""
    summary = re.fullmatch(r"(OK|FAILED)(?: \(([^()]*)\))?", last)
    result = {"outcome": "INVALID", "tests": int(counts[0]) if len(counts) == 1 else 0}
    if not summary or result["tests"] == 0:
        return result
    fields = {}
    if summary[2]:
        for field in summary[2].split(", "):
            match = re.fullmatch(r"(failures|errors|skipped|expected failures|unexpected successes)=(\d+)", field)
            if not match or match[1] in fields:
                return result
            fields[match[1]] = int(match[2])
    result["coverage"] = {"tests": result["tests"], "skipped": fields.get("skipped", 0),
                          "expected_failures": fields.get("expected failures", 0)}
    # The example has one deliberately disabled planted-failure test. Preserve
    # its baseline skip count, but never credit unrelated errors or a changed census.
    if result["tests"] <= sum(result["coverage"][k] for k in ("skipped", "expected_failures")):
        return result
    if fields.get("unexpected successes", 0):
        return result
    if fields.get("errors", 0):
        errors = re.findall(r"^ERROR: (.*?)(?=^={10,}|^Ran \d+ tests? in |\Z)",
                            stderr, re.MULTILINE | re.DOTALL)
        frames = [re.findall(r'^  File "([^"]+)", line \d+, in ', error, re.MULTILINE)
                  for error in errors]
        if (not target_path or len(errors) != fields["errors"] or any(
                not stack or stack[-1] != target_path or re.search(
                    r"(?:setUp|tearDown)(?:Class|Module)?|(?:ImportError|ModuleNotFoundError|SyntaxError):", error)
                for error, stack in zip(errors, frames))):
            return result
    if rc == 0 and summary[1] == "OK" and not fields.get("failures", 0):
        result["outcome"] = "PASS"
    elif rc == 1 and summary[1] == "FAILED" and fields.get("failures", 0) + fields.get("errors", 0) > 0:
        result["outcome"] = "KILLED"
    return result


def acceptance_outcome(rc, report, target):
    """Require the harness's structured result, not merely its exit status."""
    if not isinstance(report, dict) or rc not in (0, 1):
        return {"outcome": "INVALID"}
    is_base = target == TARGET
    schema = "szl.e8.results/v0" if is_base else "szl.e2d.acceptance/v0"
    verdicts = ("GOVERNANCE_ACCEPTANCE_MET_ON_SPY", "GOVERNANCE_ACCEPTANCE_NOT_MET") if is_base else (
        "ACCEPTANCE_MET_ON_SPY", "ACCEPTANCE_NOT_MET")
    groups = ("governed_checks", "scenario_checks") if is_base else ("guard_checks", "negative_controls")
    measured = ("governed_MEASURED_against_spy", "scenarios_MEASURED") if is_base else (
        "guard_cases_MEASURED_against_spy",)
    if report.get("schema") != schema or report.get("verdict") != verdicts[rc]:
        return {"outcome": "INVALID"}
    signature = {}
    passes = []
    for key in groups:
        rows = report.get(key)
        if not isinstance(rows, list) or not rows:
            return {"outcome": "INVALID"}
        signature[key] = len(rows)
        for row in rows:
            if not isinstance(row, dict) or type(row.get("pass")) is not bool:
                return {"outcome": "INVALID"}
            if key == "negative_controls" and row.get("harness_rc") not in (0, 1, 3):
                return {"outcome": "INVALID"}
            passes.append(row["pass"])
    for key in measured:
        rows = report.get(key)
        if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
            return {"outcome": "INVALID"}
        signature[key] = len(rows)
    deterministic = report.get("deterministic" if is_base else "guard_deterministic")
    if deterministic is not True:
        return {"outcome": "INVALID"}
    base_rc = 0 if is_base else report.get("base_eight_case_rc")
    if base_rc not in (0, 1):
        return {"outcome": "INVALID"}
    passed = all(passes) and base_rc == 0
    if passed != (rc == 0):
        return {"outcome": "INVALID"}
    return {"outcome": "PASS" if passed else "KILLED", "coverage": signature}


def run(args, cwd, timeout=300):
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="e2d_result_") as result_dir:
        report_path = os.path.join(result_dir, "result.json")
        command = list(args)
        is_unit = args == UNIT
        if not is_unit:
            command += ["--out", report_path]
        try:
            r = subprocess.run([sys.executable] + command, cwd=cwd, capture_output=True,
                               text=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"rc": "TIMEOUT", "last": "TIMEOUT", "outcome": "INVALID"}
        if is_unit:
            outcome = unit_outcome(r.returncode, r.stderr, os.path.join(cwd, TARGET))
        else:
            try:
                with open(report_path, encoding="utf-8") as f:
                    report = json.load(f)
                outcome = acceptance_outcome(r.returncode, report, args[0])
            except (OSError, ValueError, TypeError, AttributeError):
                outcome = {"outcome": "INVALID"}
            if r.stderr.strip():
                outcome = {"outcome": "INVALID"}
    tail = (r.stderr.strip() or r.stdout.strip()).splitlines()
    return {"rc": r.returncode, "last": tail[-1] if tail else "", **outcome}


def layers(cwd):
    return {"unit": run(UNIT, cwd), "base_acceptance": run([TARGET], cwd), "acceptance": run(["acceptance.py"], cwd)}


def mutate(src, tag):
    out, n = [], 0
    for ln in src.splitlines(keepends=True):
        m = re.match(r"^(\s*)(if|elif) (.+):(\s+# GUARD:" + tag + r"\b.*)$", ln.rstrip("\n"))
        if m:
            ln = f"{m.group(1)}{m.group(2)} False:{m.group(4)}\n"
            n += 1
        out.append(ln)
    return "".join(out), n


def copy_tree(dst):
    ignore = shutil.ignore_patterns("__pycache__", "e2d_*", "e8t_*", "*.pyc")
    shutil.copytree(HERE, dst, ignore=ignore)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = tempfile.mkdtemp(prefix="e2d_sweep_", dir=a.scratch)
    try:
        clean = os.path.join(root, "clean")
        copy_tree(clean)
        base = layers(clean)
        green = all(r["outcome"] == "PASS" for r in base.values())
        if not green:
            print("REFUSED: clean suite is not green; mutants not scored", json.dumps(base))
            return 3
        with open(os.path.join(HERE, TARGET), encoding="utf-8") as f:
            src = f.read()
        tags = sorted(set(TAG_RE.findall(src)), key=lambda t: int(t[1:]))
        rows = []
        for tag in tags:
            d = os.path.join(root, tag)
            copy_tree(d)
            mutated, n = mutate(src, tag)
            if n == 0 or mutated == src:
                rows.append({"guard": tag, "sites": 0, "error": "NOT_MUTATED"})
                continue
            compile(mutated, TARGET, "exec")
            with open(os.path.join(d, TARGET), "w", encoding="utf-8", newline="\n") as f:
                f.write(mutated)
            res = layers(d)
            for layer, evidence in res.items():
                if evidence.get("coverage") != base[layer].get("coverage"):
                    evidence["outcome"] = "INVALID"
            rows.append({"guard": tag, "sites": n, **{k + "_rc": v["rc"] for k, v in res.items()},
                         **{"killed_by_" + k: v["outcome"] == "KILLED" for k, v in res.items()},
                         "outcomes": {k: v["outcome"] for k, v in res.items()}})
            shutil.rmtree(d, ignore_errors=True)
        summary = {k: sum(1 for r in rows if r.get("killed_by_" + k)) for k in ("unit", "base_acceptance", "acceptance")}
        result = {"schema": "szl.e2d.mutation-sweep/v0", "clean": base, "mutants_total": len(rows),
                  "killed": summary, "rows": rows,
                  "base_acceptance_survivors": [r["guard"] for r in rows if not r.get("killed_by_base_acceptance")],
                  "acceptance_survivors": [r["guard"] for r in rows if not r.get("killed_by_acceptance")],
                  "invalid": [r["guard"] for r in rows if "error" in r or
                              "INVALID" in r.get("outcomes", {}).values()]}
        if a.out:
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(result, f, indent=1, sort_keys=True)
                f.write("\n")
        for r in rows:
            print(r["guard"], "sites", r.get("sites"), "unit", r.get("killed_by_unit"),
                  "base_acc", r.get("killed_by_base_acceptance"), "acc", r.get("killed_by_acceptance"))
        print("total", len(rows), "killed", json.dumps(summary))
        ok = rows and not result["invalid"] and all(
            r.get("killed_by_unit") and r.get("killed_by_acceptance") for r in rows)
        return 0 if ok else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
