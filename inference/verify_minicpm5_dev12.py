"""Verify the archived local development comparison without running a model.

Copyright 2026 SZL Holdings. SPDX-License-Identifier: Apache-2.0
This is unsigned evidence integrity and consistency, not independent execution proof.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "frontier/minicpm5/evidence/2026-09-08-rtx5050-dev12"
SOURCE = "3f12d5078fe058ca66c41afb3fedbecd96ee9468"
MANIFEST_SHA256 = "87e1e0832ba2bb82db9084e1a19ec6131fe1ed939d8f6612555dddf2da6d3ac2"
RECORDS = {
    "baseline": "2c964479fc7bf0136d40a0430b994cda991d438632df691e3253f9766694b29b",
    "candidate": "e02ae339d8ef35c361302f59684c71086c0c4303ae4bfe608175afe86ccb33b4",
}
RUNNERS = {
    "baseline": "b1f750253004d43fd44ead3581245920867ee775b9388ff3eec002af80a29af7",
    "candidate": "69274d8c2f0a1c96deac844c61e09662a683df6cbc3fcd374669562f1c4dac88",
}
_SPEC = importlib.util.spec_from_file_location(
    "dev12_report_verifier", Path(__file__).with_name("verify_minicpm5_report.py"))
assert _SPEC is not None and _SPEC.loader is not None
v = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    raw = path.read_bytes()
    v.require(len(raw) <= 1024 * 1024, "evidence file exceeds size bound")
    value = json.loads(raw, object_pairs_hook=v.q._pairs, parse_constant=v.q._constant)
    v.require(isinstance(value, dict), "evidence must be object")
    return value


def syntax(path: Path, function: str | None = None) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if function is not None:
        matches = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == function]
        v.require(len(matches) == 1, "missing or ambiguous source function")
        tree = matches[0]
    return ast.dump(tree, include_attributes=False)


def verify(bundle: Path = BUNDLE) -> dict:
    # Pin every historical artifact through an independently fixed manifest.
    # Updating a file and its adjacent digest must not rewrite the archive.
    v.require(sha(bundle / "manifest.json") == MANIFEST_SHA256,
              "historical manifest identity changed")
    manifest = read_json(bundle / "manifest.json")
    files = manifest.get("files")
    expected_files = {
        "baseline-v1.json", "explicit-lookup-v2.json", "comparison.json",
        "minicpm5-explicit-lookup-v2.patch",
        *(f"source/{lane}/inference/{name}" for lane in RUNNERS for name in
          ("minicpm5_qualification.py", "verify_minicpm5_report.py")),
    }
    v.require(isinstance(files, dict) and set(files) == expected_files,
              "manifest has missing or unexpected evidence paths")
    for name, digest in files.items():
        path = bundle / name
        v.require(path.resolve().is_relative_to(bundle.resolve()) and not path.is_symlink(),
                  "evidence path escapes bundle")
        v.require(digest == sha(path), "evidence file digest mismatch")
    v.require(manifest.get("sourceBaseRevision") == SOURCE
              and manifest.get("executionSourceWasUnpublished") is True,
              "historical source provenance changed")

    baseline = read_json(bundle / "baseline-v1.json")
    candidate = read_json(bundle / "explicit-lookup-v2.json")
    comparison = read_json(bundle / "comparison.json")
    reports = {"baseline": baseline, "candidate": candidate}
    for lane, report in reports.items():
        runner = bundle / f"source/{lane}/inference/minicpm5_qualification.py"
        v.require(sha(runner) == RUNNERS[lane], "executed source snapshot changed")
        v.verify(report, SOURCE, RUNNERS[lane])
        v.require(report["recordSha256"] == RECORDS[lane], "historical record identity changed")
        v.require(report["jobId"] is None, "local run cannot claim a provider job")
        for suffix, key in (("RecordSha256", "recordSha256"), ("RunnerSha256", "runnerSha256"),
                            ("PassedCases", "passedCases"), ("CompletedCases", "completedCases")):
            v.require(comparison.get(lane + suffix) == report[key], "comparison disagrees with record")
    for lane, name in (("baseline", "baseline-v1.json"), ("candidate", "explicit-lookup-v2.json")):
        v.require(comparison.get(lane + "FileSha256") == sha(bundle / name), "comparison file hash mismatch")

    before = bundle / "source/baseline/inference/minicpm5_qualification.py"
    after = bundle / "source/candidate/inference/minicpm5_qualification.py"
    unchanged = {name: syntax(before, name) == syntax(after, name)
                 for name in ("suite", "grade", "strict_json", "verify_files")}
    v.require(all(unchanged.values()) and comparison.get("unchangedFunctions") == unchanged,
              "fixture, grader, parser, or model verification changed")
    v.require(sha(Path(__file__).with_name("minicpm5_qualification.py")) == RUNNERS["baseline"],
              "original baseline runner changed")
    active = Path(__file__).with_name("minicpm5_prompt_qualification.py")
    v.require(syntax(after, "build_messages") == syntax(active, "build_messages"),
              "active request construction differs from measured experiment")
    v.require(v.p.plan("explicit-lookup-v2") == candidate["plan"],
              "active prompt or workload differs from measured experiment")
    for name in unchanged:
        v.require(getattr(v.p, name) is getattr(v.p.base, name),
                  "active prompt module replaced original contract helpers")
    for key, equal in {
        "sameModelArtifactHashes": baseline["artifactSha256"] == candidate["artifactSha256"],
        "sameSuiteHash": baseline["plan"]["suiteSha256"] == candidate["plan"]["suiteSha256"],
        "sameHardware": baseline["hardware"] == candidate["hardware"],
    }.items():
        v.require(equal and comparison.get(key) is True, "comparison inputs differ")
    expected = [{"id": c["id"], "category": c["category"],
                 "baselinePassed": b["passed"], "candidatePassed": c["passed"],
                 "baselineOutputSha256": b["outputSha256"], "candidateOutputSha256": c["outputSha256"],
                 "candidateReasonCodes": c["reasonCodes"]}
                for b, c in zip(baseline["cases"], candidate["cases"], strict=True)]
    v.require(comparison.get("caseComparison") == expected, "case comparison changed")
    fixed = {"suiteUse": "DEVELOPMENT_NOT_HELD_OUT", "sourceBaseRevision": SOURCE,
             "sourcePatchUnpublished": True, "experiment": "explicit-lookup-v2",
             "rawModelOnly": True, "inputPreprocessing": "NONE", "outputRepair": "NONE",
             "constrainedDecoding": False, "weightsChanged": False, "trainingExecuted": False,
             "toolExecuted": False, "providerMutation": False, "independentWitness": False,
             "productionDisposition": "HOLD", "promotionEffect": "NONE", "timedOut": False,
             "exitCode": 0, "elapsedSeconds": 163.906,
             "maxGenerationSeconds": 300, "hardProcessTimeoutSeconds": 480,
             "candidateStatus": candidate["status"], "p50GenerationMs": candidate["p50GenerationMs"],
             "p95GenerationMs": candidate["p95GenerationMs"],
             "totalGenerationMs": round(sum(c["generationMs"] for c in candidate["cases"]), 3)}
    for key, value in fixed.items():
        v.require(type(comparison.get(key)) is type(value) and comparison[key] == value,
                  "comparison claim changed: " + key)
    return {"status": "INTEGRITY_VERIFIED", "baselinePassedCases": baseline["passedCases"],
            "candidatePassedCases": candidate["passedCases"], "caseCount": candidate["completedCases"],
            "suiteUse": "DEVELOPMENT_NOT_HELD_OUT", "independentWitness": False,
            "productionDisposition": "HOLD", "sourceAtExecution": "UNPUBLISHED_PATCH_WITH_EXECUTED_FILE_HASH"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    args = parser.parse_args()
    try:
        result = verify(args.bundle)
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        print(json.dumps({"status": "INVALID", "errorType": type(exc).__name__,
                          "productionDisposition": "HOLD"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
