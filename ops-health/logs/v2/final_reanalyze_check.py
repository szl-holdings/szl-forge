"""Re-run the registered final analysis and prove that no number changed (post-opening).  SYNTHETIC.

Run from the repository root:
    PYTHONUTF8=1 py -3.12 -B logs/v2/final_reanalyze_check.py

Why: after the single opening, v2/research/final_analyze.py gained rendering-only additions and
its line endings went from CRLF to LF.  This script shows that the analysis itself is untouched and
that a full re-analysis reproduces the committed v2/results/final_evaluation.json.

What it does (it opens no data split; it reads only v2/results/final/, the frozen model files that
final_analyze.analyze() re-hashes, and the committed final_evaluation.json):
  1. Static check: parses the analysis-time script (git commit cbac497, sha256 a5325246...) and the
     current one, and lists which top-level functions and constants are byte-for-byte identical as
     ASTs.  Every function on the analysis path must be identical.
  2. Checks that the modules analyze() imports (bootstrap, metrics, registry, truth) have no change
     since the opening HEAD (30d50b5), in git and in the working tree.
  3. Runs final_analyze.analyze() on the real score directory with the registered protocol
     (2,000 resamples, seed 20260927) and parts_dir=None, so no saved per-set part is reused.  The
     result goes to scratch/v2_final_recheck/ (git-ignored), never to v2/results/.
  4. Compares every leaf of the new evaluation with the committed final_evaluation.json.  Numeric
     leaves must be identical, except wall-clock ``seconds`` fields, which are counted and
     excluded.  Other differing leaves must be run metadata (generated_utc, the analysis script's
     sha256 and git head); anything else is an unexpected difference.
  5. Writes v2/results/final_recheck.json (read by final_analyze --render-only).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v2.research import final_analyze as FA  # noqa: E402

ANALYSIS_TIME_COMMIT = "cbac497"
OPENING_HEAD = "30d50b5"
IMPORTED = ("v2/research/bootstrap.py", "v2/research/metrics.py", "v2/research/registry.py",
            "v2/research/truth.py")
ANALYSIS_PATH_DEFS = ("load_scores", "evaluation_sets", "_validity", "analyze_set",
                      "disaggregate_test", "interval_rule", "win_conditions", "truth_recovery",
                      "validity_summary", "ablations", "_git_head", "analyze", "sha256_bytes",
                      "_utc", "_json_bytes", "AnalysisError")
EXPECTED_METADATA = ("generated_utc", "inputs.analysis_script.sha256",
                     "inputs.analysis_script.git_head")
OUT_DIR = ROOT / "scratch" / "v2_final_recheck"
OUT_JSON = ROOT / "v2" / "results" / "final_recheck.json"
MEM_RETRIES = 20


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RetryingStdout:
    """stdout (the log file) that survives ENOSPC: unwritten bytes are retried, not lost."""

    encoding = "utf-8"

    def write(self, text: str) -> int:
        data = text.encode("utf-8")
        for _ in range(720):
            try:
                while data:
                    data = data[os.write(1, data):]
                return len(text)
            except OSError as exc:
                if exc.errno != 28:
                    raise
                time.sleep(10)
        raise OSError(28, "stdout stayed full")

    def flush(self) -> None:
        pass


def wait_for_disk(minimum: int = 600 * 2**20, tries: int = 240) -> None:
    """The host disk swings to 0 bytes free; wait (up to 2 h) instead of failing."""
    for _ in range(tries):
        free = shutil.disk_usage(ROOT).free
        if free >= minimum:
            return
        print(f"[disk] {free} bytes free < {minimum}; waiting 30 s", flush=True)
        time.sleep(30)
    raise SystemExit("disk stayed full")


def write_retry(path: Path, data: bytes, tries: int = 240) -> None:
    """Write and re-read; on ENOSPC wait and retry (the content is already computed)."""
    for attempt in range(tries):
        try:
            path.write_bytes(data)
            if path.read_bytes() != data:
                raise OSError(28, "re-read bytes differ")
            return
        except OSError as exc:
            if exc.errno != 28 or attempt == tries - 1:
                raise
            print(f"[disk] ENOSPC writing {path.name}; retry in 30 s", flush=True)
            time.sleep(30)


def git(*args: str) -> str:
    for _ in range(12):  # git itself can fail for a few seconds while the disk is at 0 bytes
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout
        time.sleep(5)
    raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr}")


def top_level(tree: ast.Module) -> dict[str, str]:
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = ast.dump(node.value)
    return out


def static_check() -> dict:
    old_bytes = subprocess.run(["git", "-C", str(ROOT), "show",
                                f"{ANALYSIS_TIME_COMMIT}:v2/research/final_analyze.py"],
                               check=True, capture_output=True).stdout
    new_bytes = (ROOT / FA.SCRIPT_REL).read_bytes()
    old = top_level(ast.parse(old_bytes.decode("utf-8")))
    new = top_level(ast.parse(new_bytes.decode("utf-8")))
    same = sorted(k for k in old if k in new and old[k] == new[k])
    changed = sorted(k for k in old if k in new and old[k] != new[k])
    return {
        "analysis_time_script": {"commit": ANALYSIS_TIME_COMMIT, "sha256": sha(old_bytes),
                                 "crlf": b"\r\n" in old_bytes},
        "current_script": {"sha256": sha(new_bytes), "crlf": b"\r\n" in new_bytes},
        "identical": same, "changed": changed,
        "added": sorted(k for k in new if k not in old),
        "removed": sorted(k for k in old if k not in new),
        "analysis_path_identical": all(k in same for k in ANALYSIS_PATH_DEFS),
    }


def imported_check() -> dict:
    committed = git("diff", "--name-only", OPENING_HEAD, "HEAD", "--", *IMPORTED).split()
    worktree = git("diff", "--name-only", "HEAD", "--", *IMPORTED).split()
    return {"since": OPENING_HEAD, "files": list(IMPORTED), "changed_in_commits": committed,
            "changed_in_worktree": worktree, "unchanged": not committed and not worktree}


def leaves(value, path=""):
    if isinstance(value, dict):
        for k in sorted(value):
            yield from leaves(value[k], f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, value


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare(committed: dict, rechecked: dict) -> dict:
    a, b = dict(leaves(committed)), dict(leaves(rechecked))
    compared = identical = timing = 0
    worst = 0.0
    metadata, unexpected = [], []
    for path in sorted(set(a) | set(b)):
        if path not in a or path not in b:
            unexpected.append({"path": path, "why": "present on one side only"})
            continue
        x, y = a[path], b[path]
        last = path.rsplit(".", 1)[-1]
        if is_num(x) and is_num(y):
            if last == "seconds":
                timing += 1
                continue
            compared += 1
            if x == y:
                identical += 1
            else:
                worst = max(worst, abs(float(x) - float(y)))
                unexpected.append({"path": path, "committed": x, "rechecked": y})
        elif x != y:
            (metadata if path in EXPECTED_METADATA else unexpected).append(
                path if path in EXPECTED_METADATA else {"path": path, "committed": x, "rechecked": y})
    return {"numeric_fields_compared": compared, "numeric_fields_identical": identical,
            "max_abs_numeric_diff": worst, "timing_fields_excluded": timing,
            "leaves_committed": len(a), "leaves_rechecked": len(b),
            "expected_metadata_differences": metadata, "unexpected_differences": unexpected,
            "verdict_committed": committed["verdict"]["result"],
            "verdict_rechecked": rechecked["verdict"]["result"]}


def main() -> int:
    sys.stdout = RetryingStdout()
    t0 = time.time()
    wait_for_disk()
    print(f"free disk {shutil.disk_usage(ROOT).free} bytes", flush=True)
    st = static_check()
    print(f"[static] analysis-time script {st['analysis_time_script']['sha256'][:16]}... "
          f"(crlf={st['analysis_time_script']['crlf']}) vs current "
          f"{st['current_script']['sha256'][:16]}... (crlf={st['current_script']['crlf']})",
          flush=True)
    print(f"[static] identical: {', '.join(st['identical'])}", flush=True)
    print(f"[static] changed: {', '.join(st['changed']) or 'none'}; added: "
          f"{', '.join(st['added']) or 'none'}; removed: {', '.join(st['removed']) or 'none'}",
          flush=True)
    print(f"[static] every analysis-path definition identical: {st['analysis_path_identical']}",
          flush=True)
    imp = imported_check()
    print(f"[imports] {', '.join(IMPORTED)} changed since {OPENING_HEAD}: commits "
          f"{imp['changed_in_commits'] or 'none'}, worktree {imp['changed_in_worktree'] or 'none'}",
          flush=True)
    committed_bytes = FA.OUT_JSON.read_bytes()
    committed = json.loads(committed_bytes.decode("utf-8"))
    for attempt in range(MEM_RETRIES + 1):
        try:  # analyze() is a deterministic function of the saved files, so a retry is exact
            evaluation = FA.analyze(FA.SCORE_DIR, parts_dir=None, resamples=FA.RESAMPLES,
                                    log=lambda s: print(f"[analyze] {s}", flush=True))
            break
        except MemoryError:
            if attempt == MEM_RETRIES:
                raise
            print(f"[analyze] MemoryError; retry {attempt + 1}/{MEM_RETRIES} in 30 s", flush=True)
            time.sleep(30)
            wait_for_disk()
    if evaluation["mode"] != "FINAL" or not evaluation["registered_protocol"]:
        raise SystemExit("not a FINAL, registered-protocol analysis")
    scratch_bytes = FA._json_bytes(evaluation)
    cmp_ = compare(committed, evaluation)
    print(f"[compare] leaves {cmp_['leaves_committed']} committed / {cmp_['leaves_rechecked']} "
          f"rechecked; numeric compared {cmp_['numeric_fields_compared']}, identical "
          f"{cmp_['numeric_fields_identical']}, max abs diff {cmp_['max_abs_numeric_diff']:.3e}; "
          f"timing fields excluded {cmp_['timing_fields_excluded']}", flush=True)
    print(f"[compare] run-metadata differences: {cmp_['expected_metadata_differences']}", flush=True)
    print(f"[compare] unexpected differences: {len(cmp_['unexpected_differences'])}", flush=True)
    for d in cmp_["unexpected_differences"][:20]:
        print(f"[compare]   {d}", flush=True)
    print(f"[compare] verdict committed {cmp_['verdict_committed']}, rechecked "
          f"{cmp_['verdict_rechecked']}", flush=True)
    ok = (not cmp_["unexpected_differences"] and st["analysis_path_identical"]
          and imp["unchanged"] and cmp_["verdict_committed"] == cmp_["verdict_rechecked"]
          and cmp_["numeric_fields_compared"] == cmp_["numeric_fields_identical"])
    out = {
        "schema": "szl-oac/ops-health-final-recheck/v2", "synthetic": True, "post_hoc": True,
        "note": "written after the single opening; a full re-run of the registered analysis from "
                "the saved score files, compared field by field with the committed "
                "final_evaluation.json; it opens no data split",
        "inputs": {"script": "logs/v2/final_reanalyze_check.py",
                   "script_sha256": sha(Path(__file__).read_bytes()),
                   "log": "logs/v2/final_reanalyze_check.txt",
                   "analysis_script_sha256": st["current_script"]["sha256"],
                   "python": platform.python_version(),
                   "score_manifest_sha256": evaluation["inputs"]["score_manifest_sha256"],
                   "committed_evaluation_sha256": sha(committed_bytes),
                   "rechecked_evaluation_sha256": sha(scratch_bytes),
                   "rechecked_evaluation_path": "scratch/v2_final_recheck/final_evaluation.json "
                                                "(git-ignored)",
                   "git_head": evaluation["inputs"]["analysis_script"]["git_head"]},
        "static": st, "imports": imp, "comparison": cmp_, "all_checks_pass": ok,
        "seconds": round(time.time() - t0, 1),
    }
    data = (json.dumps(out, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_retry(OUT_DIR / "final_evaluation.json", scratch_bytes)
    write_retry(OUT_JSON, data)
    print(f"wrote v2/results/{OUT_JSON.name}: {len(data)} bytes sha256 {sha(data)}", flush=True)
    print(f"all checks pass: {ok} ({time.time() - t0:.0f} s)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
