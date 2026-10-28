"""mutation_sweep.py - remove each tagged harness guard and record which layer notices.

Stdlib only.  For every tag ``# GUARD:Hxx`` in eight_case_harness.py, a temporary copy of
this directory is made in which the condition of every ``if``/``elif`` line carrying that
tag is replaced with ``False``.  Three layers are then run against the copy:

  unit             python -m unittest discover (all test_*.py here)
  base_acceptance  eight_case_harness.py        (the frozen eight-case run only)
  acceptance       acceptance.py                (frozen run + guard cases + negative controls)

A mutant is KILLED by a layer when that layer exits non-zero.  The sweep REFUSES to score
(exit 3) unless the clean copy is green first: unit rc 0 with last line starting "OK",
and both acceptance commands rc 0.

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


def run(args, cwd):
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        r = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        return {"rc": "TIMEOUT", "last": "TIMEOUT"}
    tail = (r.stderr.strip() or r.stdout.strip()).splitlines()
    return {"rc": r.returncode, "last": tail[-1] if tail else ""}


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
        green = (base["unit"]["rc"] == 0 and str(base["unit"]["last"]).startswith("OK")
                 and base["base_acceptance"]["rc"] == 0 and base["acceptance"]["rc"] == 0)
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
            rows.append({"guard": tag, "sites": n, **{k + "_rc": v["rc"] for k, v in res.items()},
                         **{"killed_by_" + k: v["rc"] != 0 for k, v in res.items()}})
            shutil.rmtree(d, ignore_errors=True)
        summary = {k: sum(1 for r in rows if r.get("killed_by_" + k)) for k in ("unit", "base_acceptance", "acceptance")}
        result = {"schema": "szl.e2d.mutation-sweep/v0", "clean": base, "mutants_total": len(rows),
                  "killed": summary, "rows": rows,
                  "base_acceptance_survivors": [r["guard"] for r in rows if not r.get("killed_by_base_acceptance")],
                  "acceptance_survivors": [r["guard"] for r in rows if not r.get("killed_by_acceptance")]}
        if a.out:
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(result, f, indent=1, sort_keys=True)
                f.write("\n")
        for r in rows:
            print(r["guard"], "sites", r.get("sites"), "unit", r.get("killed_by_unit"),
                  "base_acc", r.get("killed_by_base_acceptance"), "acc", r.get("killed_by_acceptance"))
        print("total", len(rows), "killed", json.dumps(summary))
        ok = rows and all(r.get("killed_by_unit") and r.get("killed_by_acceptance") for r in rows)
        return 0 if ok else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
