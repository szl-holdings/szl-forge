"""Supplementary, post-run step of the section 10 mutation stage (v2-mutation fix).  SYNTHETIC.

Question: does the committed test layer notice a recorded BLIND_SPOT mutant when that mutant is
put IN PLACE of the frozen v2 pair?  The recorded run (v2/results/mutation_readiness.json)
evaluated every artifact and receipt mutant on a temporary copy and never ran the v2 test suite
with a mutant in place.  This script does that, for the 11 recorded BLIND_SPOT mutants plus an
unmodified control, and records the outcome.  It is NOT part of any registered verdict: the
recorded statuses are unchanged (PREREGISTRATION section 10: pass-2 artifact mutants "must then
be caught by a behavioral gate").

How:
  * The tree is `git archive` of the mutation stage's last commit PIN (929677d), without
    v2/data/sealed/ and without any *.jsonl member (pathspec exclusions, so no sealed byte is
    even streamed).  It is extracted under logs/v2/_inplace_tmp/ (inside this work tree, so
    `git -C <tree>` resolves to this repository for verify_origin and parent_commit) and deleted
    after each variant.  math/lambda_stdlib.py is copied in after its sha256 is checked against
    the recorded run.
  * Mutants are rebuilt with v2.research.mutation.artifact_mutants / receipt_mutants from the
    frozen pair at PIN; every artifact mutant's model sha256 must equal the recorded one.  Pass-2
    pairs use the consistently regenerated receipt, exactly as the recorded run did.
  * Test module: v2.tests.test_mutation_suite, the only test module at PIN that reads the frozen
    pair (the regex hits over v2/tests at PIN are recorded in the output; MODELED from source).
  * A variant's result = its failing test ids minus the control's failing test ids.  A host
    problem (paging file, MemoryError, disk full) in the output triggers one retry after 30 s and
    is recorded.

Usage (repository root):  PYTHONUTF8=1 py -3.12 -B logs/v2/mutation_inplace_tests.py
Writes logs/v2/mutation_inplace_tests.{json,txt} and logs/v2/mutation_inplace_tests/<variant>.txt.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v2.research import freeze, mutation as M  # noqa: E402

PIN = "929677d9f38254d87fa969f611d7cee29341f265"
MODULE = "v2.tests.test_mutation_suite"
OUT_JSON = ROOT / "logs" / "v2" / "mutation_inplace_tests.json"
OUT_TXT = ROOT / "logs" / "v2" / "mutation_inplace_tests.txt"
LOG_DIR = ROOT / "logs" / "v2" / "mutation_inplace_tests"
TMP = ROOT / "logs" / "v2" / "_inplace_tmp"
MARKER = ROOT / "runs" / "TEST_OPENED.json"
HOST_PROBLEM = re.compile(r"WinError 1455|MemoryError|No space left|ENOSPC|WinError 112|paging file")
FAIL_LINE = re.compile(r"^(?:ERROR|FAIL): \S+ \(([\w.]+)\)", re.M)
READS_FROZEN = re.compile(r'registry\.V2\s*/\s*"ops-health"|V2_OPS_DIR|\bM\.V2_DIR\b|mutation\.V2_DIR')


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
        with OUT_TXT.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
    except OSError:  # a full disk must not end the run while it waits for space
        pass


# Each variant writes about 2 MB (a 1.5 MB tree, a log, test temp files), so the floor is 256 MiB
# rather than the 1 GiB used before large writes; below it the script waits 60 s up to 10 times
# and then stops with a blocker.  Completed variants are kept in OUT_JSON, and a rerun resumes.
DISK_FLOOR = 256 * 1024 ** 2


def wait_for_disk() -> None:
    for _ in range(10):
        if shutil.disk_usage(ROOT).free >= DISK_FLOOR:
            return
        log(f"disk free {shutil.disk_usage(ROOT).free} bytes < {DISK_FLOOR} bytes; waiting 60 s")
        time.sleep(60)
    log("BLOCKER: disk free stayed below the floor; stopping (rerun resumes)")
    raise SystemExit("BLOCKER: disk free stayed below the floor")


def extract_tree(dest: Path) -> list[str]:
    data = git_bytes("archive", "--format=tar", PIN, "--", ".",
                     ":(exclude)v2/data/sealed", ":(exclude,glob)**/*.jsonl")
    names = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        for member in tar.getmembers():
            names.append(member.name)
            if "sealed/" in member.name.replace("\\", "/") and member.name.startswith("v2/data"):
                raise SystemExit(f"refusing: sealed member {member.name} in archive")
            if member.name.endswith(".jsonl"):
                raise SystemExit(f"refusing: split member {member.name} in archive")
        tar.extractall(dest, filter="data")
    return names


def run_module(tree: Path, vid: str) -> dict:
    cmd = [sys.executable, "-B", "-m", "unittest", "-v", MODULE]
    env = {**os.environ,"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    started = time.time()
    proc = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=1800)
    out = proc.stdout + proc.stderr
    (LOG_DIR / f"{vid}.txt").write_text(f"$ {' '.join(cmd[1:])}  (cwd = tree at {PIN[:7]})\n"
                                        + out + f"\nexit {proc.returncode}\n",
                                        encoding="utf-8", newline="\n")
    ran = re.search(r"^Ran (\d+) tests? in ([\d.]+)s", out, re.M)
    tail = [line for line in out.splitlines() if line.startswith(("OK", "FAILED"))]
    return {"exit": proc.returncode, "ran": int(ran.group(1)) if ran else None,
            "seconds": round(time.time() - started, 1), "outcome": tail[-1] if tail else None,
            "failing": sorted(set(FAIL_LINE.findall(out))),
            "host_problem": bool(HOST_PROBLEM.search(out)),
            "log": (LOG_DIR / f"{vid}.txt").relative_to(ROOT).as_posix()}


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    recorded_path = ROOT / "v2" / "results" / "mutation_readiness.json"
    recorded_bytes = recorded_path.read_bytes()
    recorded = json.loads(recorded_bytes)
    lam_path = ROOT / "math" / "lambda_stdlib.py"
    lam = lam_path.read_bytes()
    if sha(lam) != recorded["protocol"]["lambda_stdlib_sha256"]:
        raise SystemExit("refusing: math/lambda_stdlib.py differs from the recorded run")
    v2b = git_bytes("show", f"{PIN}:v2/ops-health/model.json")
    v2rb = git_bytes("show", f"{PIN}:v2/ops-health/artifact_receipt.json")
    if sha(v2b) != recorded["frozen_v2"]["model_sha256"] or sha(v2rb) != recorded["frozen_v2"]["receipt_sha256"]:
        raise SystemExit("refusing: frozen pair at PIN differs from the recorded run")
    v2a, v2r = json.loads(v2b), json.loads(v2rb)
    amap = {m["id"]: m for m in M.artifact_mutants(v2a)}
    rmap = {r["id"]: r for r in M.receipt_mutants(v2r)}
    rec_art = {m["id"]: m for m in recorded["artifact_mutants"]}

    variants: dict[str, dict] = {"control": {"kind": "control", "model": v2b, "receipt": v2rb}}
    for b in recorded["blind_spots"]:
        bid = b["id"]
        if bid in amap:
            mbytes = freeze.canonical_model_bytes(amap[bid]["artifact"])
            if sha(mbytes) != rec_art[bid]["model_sha256"]:
                raise SystemExit(f"refusing: {bid} model sha256 differs from the recorded run")
            receipt = dict(copy.deepcopy(v2r), model_sha256=sha(mbytes))
            variants[bid] = {"kind": "artifact, pass 2 (consistent receipt)", "model": mbytes,
                             "receipt": freeze.canonical_json_bytes(receipt)}
        elif bid in rmap:
            variants[bid] = {"kind": "receipt", "model": v2b,
                             "receipt": freeze.canonical_json_bytes(rmap[bid]["receipt"])}
        else:
            raise SystemExit(f"refusing: recorded blind spot {bid} is not an artifact/receipt mutant")

    results = {
        "schema": "szl-oac/ops-health-mutation-inplace-tests/v1", "synthetic": True,
        "supplementary_post_run": True, "part_of_registered_verdict": False,
        "pin": PIN, "module": MODULE, "python": platform.python_version(),
        "recorded_results_sha256": sha(recorded_bytes),
        "lambda_stdlib_sha256": sha(lam),
        "test_opened_marker_before": MARKER.exists(),
        "variants": {},
    }
    previous = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    if (previous.get("pin") == PIN and previous.get("module") == MODULE
            and previous.get("recorded_results_sha256") == results["recorded_results_sha256"]):
        for key in ("frozen_pair_readers_at_pin", "tree_members", "tree_has_sealed_or_jsonl"):
            if key in previous:
                results[key] = previous[key]
        results["test_opened_marker_before"] = previous["test_opened_marker_before"]
        results["variants"] = {vid: r for vid, r in previous.get("variants", {}).items()
                               if vid in variants and not r.get("host_problem")}
        results["resumed_with"] = sorted(results["variants"])
    log(f"start: pin {PIN[:7]}, module {MODULE}, variants {list(variants)}; "
        f"TEST_OPENED.json exists before: {results['test_opened_marker_before']}")

    for vid, v in variants.items():
        if vid in results["variants"]:
            log(f"{vid}: complete in a previous invocation, kept")
            continue
        attempts = []
        for attempt in (1, 2):
            wait_for_disk()
            tree = TMP / vid
            if tree.exists():
                shutil.rmtree(tree)
            tree.mkdir(parents=True)
            names = extract_tree(tree)
            if vid == "control" and "frozen_pair_readers_at_pin" not in results:
                hits = []
                for path in sorted((tree / "v2" / "tests").glob("*.py")):
                    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                        if READS_FROZEN.search(line):
                            hits.append(f"{path.relative_to(tree).as_posix()}:{n}: {line.strip()}")
                results["frozen_pair_readers_at_pin"] = hits
                results["tree_members"] = len(names)
                results["tree_has_sealed_or_jsonl"] = any(
                    n.endswith(".jsonl") or n.startswith("v2/data/sealed") for n in names)
            (tree / "math").mkdir(exist_ok=True)
            shutil.copyfile(lam_path, tree / "math" / "lambda_stdlib.py")
            ops = tree / "v2" / "ops-health"
            (ops / "model.json").write_bytes(v["model"])
            (ops / "artifact_receipt.json").write_bytes(v["receipt"])
            placed = {"model_sha256": sha((ops / "model.json").read_bytes()),
                      "receipt_sha256": sha((ops / "artifact_receipt.json").read_bytes())}
            r = run_module(tree, vid if attempt == 1 else f"{vid}.retry")
            shutil.rmtree(tree)
            attempts.append(r)
            log(f"{vid} (attempt {attempt}): ran {r['ran']} in {r['seconds']}s; {r['outcome']}; "
                f"failing {r['failing']}; host_problem {r['host_problem']}")
            if not r["host_problem"]:
                break
            time.sleep(30)
        final = attempts[-1]
        results["variants"][vid] = {"kind": v["kind"], **placed, **final,
                                    "attempts": len(attempts),
                                    "first_attempt_host_problem": attempts[0]["host_problem"]}
        OUT_JSON.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                            newline="\n")

    control = results["variants"]["control"]
    control_ok = control["outcome"] == "OK" and not control["host_problem"]
    caught, clean = [], []
    for vid, r in results["variants"].items():
        if vid == "control":
            continue
        r["extra_vs_control"] = sorted(set(r["failing"]) - set(control["failing"]))
        (caught if r["extra_vs_control"] else clean).append(vid)
    results["control_ok"] = control_ok
    results["summary"] = {
        "recorded_blind_spots": len(results["variants"]) - 1,
        "in_place_suite_fails": caught,
        "in_place_suite_passes": clean,
        "blind_spots_under_literal_any_layer_reading": len(clean),
        "any_host_problem": any(r["host_problem"] for r in results["variants"].values()),
    }
    results["test_opened_marker_after"] = MARKER.exists()
    if TMP.exists() and not any(TMP.iterdir()):
        TMP.rmdir()
    OUT_JSON.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                        newline="\n")
    log(f"control ok: {control_ok}; in-place suite fails for {caught}; passes for {len(clean)} "
        f"of {len(results['variants']) - 1}; TEST_OPENED.json exists after: "
        f"{results['test_opened_marker_after']}")
    log("DONE")
    return 0 if control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
