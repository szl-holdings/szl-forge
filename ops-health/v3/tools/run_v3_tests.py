"""Run the v3 step-2 test suite as ONE DEV run and write every output (prereg s.15.5). SYNTHETIC.

Usage (from anywhere; the research root is found from this file):

    PYTHONHASHSEED=0 python -I v3/tools/run_v3_tests.py --host <label> [--out DIR] [--require-complete]

One invocation is one DEV run: it generates DEV rows only under the split name 'dev' and within
the DEV cap (2,048 rows per regime; ``guards.DEV_LEDGER``), runs the unittest suite under
``v3/tests``, checks that every test required by s.3.3 and s.6.7 exists and passed, and writes to
``logs/v3/dev/<run_id>/`` (relative to the research root, unless --out is given):

    dev_outputs.json   deterministic DEV outputs, for the cross-host byte comparison: sha256 of
                       every DEV stream, the v1-parity and v2-parity digests, the DEV fit's
                       model.json digest and fitted parameters, the gate plumbing values on the
                       DEV gate rows, the advisories' digest, bootstrap count-vector digests on
                       DEV-sized layouts, the mutation catalogue ids, the DEV ledger
    tests.json         every test id with its outcome (no timings), and the requirement coverage
    dev_fit/           the DEV fit's model.json, artifact_receipt.json, diagnostics.json
    advisories.jsonl   the kernel's advisories on the DEV gate rows
    run.json           host facts: run id, interpreter (version, executable sha256), platform,
                       runtime-pin status, git HEAD and tree state, UTC start and end, exit status
    stdout.txt         the unittest text output (timings vary)

Deterministic files (expected byte-identical across hosts at one commit if the platform math
libraries agree; the cross-host DEV check compares them): dev_outputs.json, tests.json,
advisories.jsonl, dev_fit/model.json, dev_fit/artifact_receipt.json, dev_fit/diagnostics.json.
Host-fact files (differ by host by design): run.json, stdout.txt. run.json lists the sha256 of
every deterministic file.

Nothing here is a P or H statistic, a tolerance, a detection ratio, a power or a WIN-chance
quantity; every gate value is DEV plumbing with hand-made tolerances. Receipts carry the fixed
fake commit of the test support, which no origin check accepts.

Exit status: 0 iff every test passed (a skip counts as a failure, s.14.5) and every implemented
requirement is covered; with --require-complete, also iff no requirement is PENDING.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import secrets
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (requirement id, protocol reference, test id prefixes | None, pending owner | None)
REQUIREMENTS = (
    ("v1-parity", "s.3.3", ["v3.tests.test_generator_parity.V1ParityTest.test_v1_parity_byte_for_byte"], None),
    ("v2-parity-non-sealed", "s.3.3", ["v3.tests.test_generator_parity.V2ParityTest.test_v2_non_sealed_digests"], None),
    ("twin-parity", "s.3.3", [
        "v3.tests.test_generator_parity.TwinParityTest.test_twin_observed_equals_v2_clock_skew_bytes",
        "v3.tests.test_generator_parity.TwinParityTest.test_twin_observed_equals_plain_clock_skew_on_v3_dev_seed"], None),
    ("d-mode-check", "s.3.3", [
        "v3.tests.test_generator_parity.DModeTest.test_arms_identical_except_seconds",
        "v3.tests.test_generator_parity.DModeTest.test_no_d_seconds_equal_s_nom_draws"], None),
    ("guard-refusals", "s.3.3 / s.3.2 item 5", [
        "v3.tests.test_guards.V2SealGuardTest", "v3.tests.test_guards.SealedPathGuardTest"], None),
    ("sealed-salt-only-from-verifier", "s.3.3 (last beacon item)", ["v3.tests.test_generator_parity.SealedModeTest"], None),
    ("dev-rule", "s.15.5", ["v3.tests.test_guards.DevRuleTest"], None),
    ("beacon-client-verifier", "s.3.3 beacon tests", None, "v3b-beacon"),
    ("seal-check", "s.3.3 seal-check tests", None, "v3b-seal"),
    ("probes-76", "s.6.7", [
        "v3.tests.test_kernel_contract.FailClosedProbeTest.test_every_input_probe_is_refused_with_its_message",
        "v3.tests.test_kernel_contract.FailClosedProbeTest.test_every_transform_table_probe_is_refused_at_load",
        "v3.tests.test_kernel_contract.FailClosedProbeTest.test_failclosed_gate_counts_all_76"], None),
    ("duplicate-keys", "s.6.7", ["v3.tests.test_kernel_contract.StrictJsonTest"], None),
    ("authority-map-all-false", "s.6.7", [
        "v3.tests.test_kernel_contract.AuthorityMapTest.test_all_false_on_alert_no_alert_and_abstain"], None),
    ("stdlib-only-import", "s.6.7", ["v3.tests.test_kernel_contract.ImportTest"], None),
    ("determinism", "s.6.7 item 9", ["v3.tests.test_determinism.DeterminismTest.test_train_and_freeze_twice_is_byte_identical"], None),
    ("receipt-mismatch", "s.6.7", ["v3.tests.test_kernel_contract.ReceiptTest"], None),
    ("broken-baselines", "s.6.7 / s.11.1", ["v3.tests.test_gates.BrokenBaselineTest"], None),
    ("mutation-suite-catalogue", "s.6.7 / s.11.4", ["v3.tests.test_mutation_catalogue"], None),
    ("mutation-suite-passes", "s.11.4 passes 1 and 2 (step 7)", None, "v3b-mut / step 7"),
    ("metric-recomputation", "s.6.7", ["v3.tests.test_gates.MetricRecomputationTest"], None),
    ("gates-v3", "s.11.2", ["v3.tests.test_gates.GateFormulaTest", "v3.tests.test_gates.FixtureEvaluationTest"], None),
    ("tolerance-rule", "s.11.3 (hand-made numbers)", ["v3.tests.test_gates.ToleranceRuleTest"], None),
    ("tolerance-digest-refusal", "s.11.3", ["v3.tests.test_mutation_catalogue.ToleranceDigestRefusalTest"], None),
    ("bootstrap-reuse", "s.8", ["v3.tests.test_bootstrap_reuse"], None),
    ("origin-rules-a-e", "s.14.4 / s.14.5", None, "v3b-recipe"),
)


DETERMINISTIC_FILES = ("dev_outputs.json", "tests.json", "advisories.jsonl", "dev_fit/model.json",
                       "dev_fit/artifact_receipt.json", "dev_fit/diagnostics.json")
HOST_FACT_FILES = ("run.json", "stdout.txt")


class _Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes: dict[str, str] = {}
        self.timings: dict[str, float] = {}
        self._start: dict[str, float] = {}

    def startTest(self, test):
        self._start[test.id()] = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        self.timings[test.id()] = round(time.perf_counter() - self._start.get(test.id(), time.perf_counter()), 3)
        super().stopTest(test)

    def _set(self, test, outcome):
        tid = test.id()
        previous = self.outcomes.get(tid)
        if previous in (None, "passed") or outcome in ("error", "failure"):
            self.outcomes[tid] = outcome

    def addSuccess(self, test):
        super().addSuccess(test)
        self._set(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._set(test, "failure")

    def addError(self, test, err):
        super().addError(test, err)
        self._set(test, "error")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._set(test, "skipped")

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._set(test, "expected_failure")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._set(test, "unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._set(test, "failure" if issubclass(err[0], test.failureException) else "error")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def coverage(outcomes: dict[str, str]) -> list[dict]:
    rows = []
    for req_id, ref, prefixes, owner in REQUIREMENTS:
        if prefixes is None:
            rows.append({"id": req_id, "protocol": ref, "status": "PENDING", "owner": owner, "tests": []})
            continue
        matched = sorted(t for t in outcomes if any(t == p or t.startswith(p + ".") for p in prefixes))
        missing = [p for p in prefixes if not any(t == p or t.startswith(p + ".") for t in outcomes)]
        failed = [t for t in matched if outcomes[t] != "passed"]
        status = "COVERED" if matched and not missing and not failed else ("MISSING" if missing else "FAILED")
        rows.append({"id": req_id, "protocol": ref, "status": status, "owner": None, "tests": matched,
                     "missing": missing, "failed": failed})
    return rows


def dev_outputs(out_dir: Path) -> dict:
    """Deterministic DEV outputs of this run (computed from the suite's cached DEV rows)."""
    from v3.research import bootstrap, freeze, gates, generator, guards, mutation, pipeline, registry
    from v3.tests import _support as S

    out: dict = {"schema": "szl-oac/ops-health-dev-outputs/v3", "synthetic": True, "dev": True,
                 "note": "DEV plumbing only (prereg s.15.5): no P or H statistic, tolerance, detection "
                         "ratio, power or WIN-chance quantity; hand-made plumbing tolerances"}
    out["sources_sha256"] = {
        registry.relpath(p): _sha256(p.read_bytes())
        for p in sorted((registry.V3 / "research").glob("*.py")) + [registry.KERNEL_PATH]
    }
    d_arm, nod_arm = S.d_rows()
    observed, latent = S.twin_rows()
    out["dev_streams_sha256"] = {
        "nominal_1024": pipeline.rows_sha256(S.nominal_rows()),
        "D_arm_512": pipeline.rows_sha256(d_arm),
        "noD_arm_512": pipeline.rows_sha256(nod_arm),
        "twin_observed_256": pipeline.rows_sha256(observed),
        "twin_latent_256": pipeline.rows_sha256(latent),
        "clock_skew_256": pipeline.rows_sha256(S.shift_rows("clock_skew")),
        **{f"{r}_1024": pipeline.rows_sha256(S.shift_rows(r))
           for r in ("queue_saturation", "failure_bursts", "missing_tls")},
    }
    out["v1_parity_sha256"] = {s: _sha256(generator.jsonl_bytes(rows))
                               for s, rows in generator.generate_v1_parity().items()}
    out["v2_parity"] = {s: generator.v2_parity_digest(s) for s in registry.V2_PARITY_SPLITS}
    fit_dir = out_dir / "dev_fit"
    info = S.freeze_dev(fit_dir)
    (fit_dir / "diagnostics.json").write_bytes(freeze.canonical_json_bytes(S.diagnostics()))
    model_bytes = Path(info["model_path"]).read_bytes()
    artifact = json.loads(model_bytes)
    out["dev_fit"] = {
        "model_sha256": info["model_sha256"],
        "receipt_sha256": info["receipt_sha256"],
        "train_rows": artifact["training"]["splits"]["train"]["rows"],
        "intercept": artifact["intercept"],
        "weights": artifact["weights"],
        "threshold_index": artifact["threshold_index"],
        "q_hat": artifact["conformal"]["q_hat"],
        "iterations": artifact["training"]["iterations"],
        "stop_reason": artifact["training"]["stop_reason"],
    }
    ctx = gates.GateContext(S.gate_rows(), "dev-gates")
    ref = gates.evaluate_fixture(info["model_path"], info["receipt_path"], ctx, reference=None, tolerances=None)
    details = ref["details"]
    out["gates_dev_plumbing"] = {
        "gates": ref["gates"],
        "balanced_accuracy_exact": details["balanced_accuracy_exact"],
        "ece": details["ece"],
        "brier": details["brier"],
        "monotone": details["monotone"],
        "failclosed": details["failclosed"],
        "baseline_validity": details["baseline_validity"]["status"],
    }
    kernel = S.kernel().OpsHealthKernel(info["model_path"], info["receipt_path"])
    adv = b"".join(json.dumps(kernel.advise({"features": r["features"]}), sort_keys=True,
                              separators=(",", ":")).encode() + b"\n" for r in ctx.rows)
    (out_dir / "advisories.jsonl").write_bytes(adv)
    out["advisories_sha256"] = _sha256(adv)
    layouts = {"test": dict(sizes=[342]), "pooled_shift": dict(sizes=(85, 85, 85, 85)),
               "h3_twins": dict(sizes=[256]), "legacy_v1_test": dict(sizes=[240])}
    out["bootstrap_count_vectors_sha256"] = {
        name: _sha256(_canon(list(bootstrap.resampler(name, resamples=200, **kw)))) for name, kw in layouts.items()
    }
    out["mutation_catalogue"] = mutation.catalogue_summary(artifact, registry.KERNEL_PATH.read_text(encoding="utf-8"))
    out["dev_ledger"] = guards.DEV_LEDGER.snapshot()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True, help="host label, for example local or vm")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args(argv)
    from v3.research import guards, registry

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{stamp}-{args.host}-{secrets.token_hex(4)}"
    out_dir = args.out if args.out is not None else registry.DEV_LOG_DIR / run_id
    out_dir = guards.refuse_sealed_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.perf_counter()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(registry.V3 / "tests"), pattern=args.pattern,
                                                top_level_dir=str(ROOT))
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=_Result)
    result = runner.run(suite)
    outcomes = dict(sorted(result.outcomes.items()))
    cov = coverage(outcomes)
    dev_error = None
    try:
        outputs = dev_outputs(out_dir)
        (out_dir / "dev_outputs.json").write_bytes(
            (json.dumps(outputs, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode())
    except Exception as exc:  # noqa: BLE001 -- recorded, makes the run fail
        dev_error = f"{type(exc).__name__}: {exc}"
    not_passed = sorted(t for t, o in outcomes.items() if o != "passed")
    implemented_ok = all(r["status"] in ("COVERED", "PENDING") for r in cov)
    pending = [r["id"] for r in cov if r["status"] == "PENDING"]
    ok = not not_passed and implemented_ok and dev_error is None and result.testsRun > 0
    if args.require_complete and pending:
        ok = False
    tests_doc = {"schema": "szl-oac/ops-health-dev-tests/v3", "synthetic": True, "dev": True,
                 "tests_run": result.testsRun, "outcomes": outcomes, "not_passed": not_passed,
                 "requirements": cov, "pending_requirements": pending, "complete": not pending}
    (out_dir / "tests.json").write_bytes((json.dumps(tests_doc, sort_keys=True, indent=2) + "\n").encode())
    (out_dir / "stdout.txt").write_text(stream.getvalue(), encoding="utf-8", newline="\n")
    deterministic = {}
    for rel in DETERMINISTIC_FILES:
        path = out_dir / rel
        deterministic[rel] = _sha256(path.read_bytes()) if path.is_file() else None
    exe = Path(sys.executable)
    run_doc = {
        "schema": "szl-oac/ops-health-dev-run/v3", "synthetic": True, "dev": True, "run_id": run_id,
        "host_label": args.host, "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "python": {"version": sys.version, "executable": str(exe), "executable_sha256": _sha256(exe.read_bytes()),
                   "implementation": sys.implementation.name},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                     "libc": list(platform.libc_ver()) if platform.system() == "Linux" else None},
        "runtime_pins": guards.runtime_pin_status(),
        "runtime_pin_note": None if guards.runtime_pin_status()["pinned"] else
        "NOT the runtime pin (CPython 3.12.10, python -I, PYTHONHASHSEED=0); DEV run only",
        "git": {"head": _git("rev-parse", "HEAD"), "status_v3": _git("status", "--porcelain", "--", "v3")},
        "tests_run": result.testsRun, "not_passed": not_passed, "dev_outputs_error": dev_error,
        "pending_requirements": pending, "exit_ok": ok,
        "deterministic_files_sha256": deterministic, "host_fact_files": list(HOST_FACT_FILES),
        "timings_seconds": dict(sorted(result.timings.items())),
    }
    (out_dir / "run.json").write_bytes((json.dumps(run_doc, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps({"run_id": run_id, "out": str(out_dir), "tests_run": result.testsRun,
                      "not_passed": not_passed[:20], "pending": pending, "dev_outputs_error": dev_error,
                      "ok": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
