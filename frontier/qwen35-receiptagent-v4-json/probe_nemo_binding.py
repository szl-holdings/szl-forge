#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Four SIMULATED numeric-rule probes of the real pinned software kernel.

No curriculum or model is loaded. Chained decisions establish local software
behavior/integrity only, never model quality, signing trust or authorization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum_admission as admission
import nemo_source_binding as binding


CASES = (
    ("What is the benchmark accuracy?", "1e3acc", True),
    ("What is the benchmark accuracy?", '{"accuracy_score":0.95}', True),
    ("What is the benchmark accuracy?", "0.95f1分", True),
    ("What is the benchmark accuracy?", "UNKNOWN: no benchmark evidence is available.", False),
)
# Initial exploratory cases are preserved as observations, not passing tests.
# These are outside the upstream lexical regression contract; ALLOW here must
# never be interpreted as semantic evidence, correctness, or authorization.
EXPLORATORY_CASES = ("accuracy_0.91", "0.91精度")


def probe(source_root: Path) -> dict:
    expected_rules = {"R1_no_fabrication_label", "R2_honest_unknown"}
    with binding.load_kernel(source_root, admission._read_once) as kernel:
        gate = admission._load_gate("nemo_doctrine_gate.py", kernel)
        decisions = []
        for prompt, answer, should_block in CASES:
            violated, reasons = gate.check_pair(prompt, answer, persona="finetuned")
            admission._require(set(violated) == (expected_rules if should_block else set()),
                               "real pinned numeric-rule regression")
            decisions.append(gate.Decision(
                decision=gate.BLOCK if should_block else gate.ALLOW,
                violated_rules=tuple(violated), reasons=tuple(reasons),
                rule_version=gate.RULE_VERSION + " persona/finetuned",
                input_hash=kernel.input_hash(prompt, answer)))
        receipts = gate.receipt_chain(decisions)
        admission._require(kernel.verify_chain(receipts) is True, "real chain verification failed")
        tampered = copy.deepcopy(receipts)
        tampered[0]["decision"]["input_hash"] = "sha256:" + "f" * 64
        admission._require(kernel.verify_chain(tampered) is False, "chain tampering was accepted")
        observations = []
        for answer in EXPLORATORY_CASES:
            violated, _ = gate.check_pair("What is the benchmark accuracy?", answer, persona="finetuned")
            observations.append({"answer": answer, "violated_rules": violated,
                                 "decision": gate.BLOCK if violated else gate.ALLOW,
                                 "scope": "Exploratory lexical limit; no passing-test credit."})
        canonical = admission.contract.canonical_json(
            {"required_cases": CASES, "exploratory_cases": EXPLORATORY_CASES}).encode("utf-8")
        return {
            "schema": "szl.receiptagent.v4-nemo-binding-probe/v1",
            "evidence_class": "MEASURED", "fixture_evidence_class": "SIMULATED",
            "software_checks_passed": True, "cases": len(CASES),
            "expected_block_cases": 3, "expected_allow_cases": 1,
            "exploratory_observations": observations,
            "semantic_claim_detection": "UNAVAILABLE",
            "fixture_sha256": hashlib.sha256(canonical).hexdigest(),
            "harness_source_sha256": hashlib.sha256(admission._read_once(Path(__file__),
                                                   binding.MAX_SOURCE_BYTES)).hexdigest(),
            "nemo_source_binding": kernel.provenance,
            "gate_source_sha256": admission.GATE_HASHES["nemo_doctrine_gate.py"],
            "runtime": {"python": sys.version, "platform": platform.platform()},
            "receipt_chain": receipts, "receipt_chain_tip": receipts[-1]["receipt_sha256"],
            "chain_tampering_rejected": True, "signature": "UNAVAILABLE",
            "key_trust": "REPO_DECLARED", "energy_joules": "UNAVAILABLE",
            "model_training": "NOT_RUN", "model_evaluation": "NOT_RUN",
            "training_eligible": False, "publication_eligible": False, "execution_authority": False,
            "claim_boundary": "Pinned software and synthetic fixtures only; not model evaluation or approval.",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--nemo-source-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = probe(args.nemo_source_root)
    except (admission.AdmissionError, binding.BindingError, OSError, ValueError,
            TypeError, KeyError, AttributeError, RuntimeError):
        print(admission.contract.canonical_json({
            "schema": "szl.receiptagent.v4-nemo-binding-probe/v1", "state": "BLOCKED",
            "software_checks_passed": False, "training_eligible": False,
            "publication_eligible": False, "execution_authority": False,
            "error": "Pinned software regression probe failed closed"}))
        return 1
    # ASCII JSON is portable even on Windows consoles with a legacy encoding;
    # embedded receipt canonicalization remains the kernel's UTF-8 profile.
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
