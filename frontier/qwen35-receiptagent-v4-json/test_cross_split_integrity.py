#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent source/reference checks; never a model score or training input.

This test module may inspect the public frozen final gate. Neither curriculum
generator nor admission/training loader may import it. Failures intentionally
report only aggregate labels, not final-gate cases to curriculum authors.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import json_contract as contract

PATHS = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "heldout/test.jsonl"}
COMMITMENTS = {
    "train": "26c3b11c64d0d6201e29716656c1898893fb449249f6a67611a68f477a64699f",
    "dev": "24914bdf060efd4900d9fbbc7058982a8b46ef71977f6d9cd030775e4b6d4608",
    "test": "973e5f5a272ad7c1304acec0db36fae77a6d0c2e0ec0c10c4401361b7889c575",
}
FAMILIES = {
    "train": {"dependency-lock-audit", "build-artifact-lineage",
              "documentation-link-inventory", "ci-cache-key-review"},
    "dev": {"cli-option-manifest", "frontend-static-asset-index"},
    "test": {"package-license-ledger", "schema-migration-compatibility",
             "repository-access-matrix"},
}
KINDS = {"DRAFT", "RECOVERY", "REFUSAL"}
AUTHORITIES = {"APPROVE", "EXECUTE", "FABRICATE", "DISCLOSE_SECRET", "RESEND_QUARANTINED"}


def normalize_text(text: str) -> str:
    """Remove superficial synthetic markers, not semantic family vocabulary.

    Equality here is a conservative textual-duplication check, NOT a semantic
    independence proof. Protocol keys, system prompt and fixed claims are shared
    by design and are deliberately not treated as leaked task/value content.
    """
    text = re.sub(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", "<time>", text)
    text = re.sub(r"[0-9a-f]{12,64}", "<digest>", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\b", "<number>", text)
    text = re.sub(r"\b(?:train|dev|test|heldout|final)\b", "<split>", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip().casefold()


class CrossSplitIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = {split: (HERE / path).read_bytes() for split, path in PATHS.items()}
        cls.rows = {split: [contract.strict_object(line) for line in data.decode("utf-8").splitlines()]
                    for split, data in cls.raw.items()}
        cls.requests = {split: [contract.strict_object(row["messages"][1]["content"])["request"]
                               for row in rows] for split, rows in cls.rows.items()}
        cls.manifest = contract.strict_object((HERE / "curriculum-manifest.json").read_text(encoding="utf-8"))
        cls.heldout_manifest = json.loads((HERE / "heldout" / "manifest.json").read_text(encoding="utf-8"))

    def test_frozen_bytes_match_public_commitments(self) -> None:
        for split, raw in self.raw.items():
            self.assertTrue(hashlib.sha256(raw).hexdigest() == COMMITMENTS[split],
                            f"{split}: frozen byte commitment differs")
            self.assertTrue(self.manifest["splits"][split]["sha256"] == COMMITMENTS[split],
                            f"{split}: curriculum commitment differs")
            self.assertTrue(b"\r" not in raw and raw.endswith(b"\n"), f"{split}: LF policy differs")

    def test_exact_class_and_family_denominators(self) -> None:
        for split, rows in self.rows.items():
            per_kind = {"train": 30, "dev": 15, "test": 20}[split]
            expected = Counter({(family, kind): per_kind for family in FAMILIES[split] for kind in KINDS})
            actual = Counter((row["family"], row["kind"]) for row in rows)
            self.assertTrue(actual == expected, f"{split}: class/family denominator differs")
            self.assertTrue(len(rows) == {"train": 360, "dev": 90, "test": 180}[split],
                            f"{split}: row denominator differs")

    def test_recovery_and_refusal_strata_are_balanced_per_family(self) -> None:
        for split, rows in self.rows.items():
            recoveries, refusals = Counter(), Counter()
            for row, request in zip(rows, self.requests[split]):
                if row["kind"] == "RECOVERY":
                    statuses = [item["status"] for item in request["evidence"] if item["status"] != "OK"]
                    self.assertTrue(len(statuses) == 1, f"{split}: recovery failure count differs")
                    recoveries[(row["family"], statuses[0])] += 1
                elif row["kind"] == "REFUSAL":
                    refusals[(row["family"], request["requestedAuthority"])] += 1
            count = {"train": 6, "dev": 3, "test": 4}[split]
            self.assertTrue(recoveries == Counter({(family, status): count for family in FAMILIES[split]
                                                   for status in contract.RECOVERY_RULES}),
                            f"{split}: recovery coverage differs")
            self.assertTrue(refusals == Counter({(family, authority): count for family in FAMILIES[split]
                                                 for authority in AUTHORITIES}),
                            f"{split}: refusal coverage differs")

    def test_ids_are_unique_and_namespaced_to_full_request(self) -> None:
        seen = set()
        for split, rows in self.rows.items():
            for row, request in zip(rows, self.requests[split]):
                self.assertTrue(row["id"] not in seen, f"{split}: duplicate identity")
                seen.add(row["id"])
                content = {key: value for key, value in request.items() if key != "requestId"}
                expected = contract.namespaced_request_id(split, row["kind"], row["family"], content)
                self.assertTrue(row["id"] == request["requestId"] == expected,
                                f"{split}: request identity binding differs")

    def test_no_cross_split_families_exact_requests_or_normalized_text_overlap(self) -> None:
        exact = {split: {contract.canonical_json(request) for request in requests}
                 for split, requests in self.requests.items()}
        tasks = {split: {normalize_text(request["task"]) for request in requests}
                 for split, requests in self.requests.items()}
        values = {split: {normalize_text(item["value"]) for request in requests for item in request["evidence"]}
                  for split, requests in self.requests.items()}
        for left, right in itertools.combinations(PATHS, 2):
            for label, mapping in (("family", FAMILIES), ("exact request", exact),
                                   ("normalized task", tasks), ("normalized evidence value", values)):
                self.assertTrue(not mapping[left].intersection(mapping[right]),
                                f"{left}/{right}: {label} overlap")

    def test_reference_pairs_are_valid_but_never_authorized(self) -> None:
        for split, rows in self.rows.items():
            for row, request in zip(rows, self.requests[split]):
                messages = row["messages"]
                self.assertTrue([item["role"] for item in messages] == ["system", "user", "assistant"],
                                f"{split}: message topology differs")
                self.assertTrue(messages[0]["content"] == contract.SYSTEM_PROMPT,
                                f"{split}: shared public system prompt differs")
                request_text = contract.canonical_json(request)
                self.assertTrue(messages[1]["content"] == contract.prompt_envelope(request_text),
                                f"{split}: canonical envelope binding differs")
                try:
                    result = contract.validate_pair(request_text, messages[2]["content"])
                except contract.ContractError:
                    self.fail(f"{split}: synthetic reference conformance differs")
                self.assertTrue(result["conforms"] is True and result["execution_authority"] is False,
                                f"{split}: reference or authority boundary differs")
                self.assertTrue(all(item["label"] == "SIMULATED" for item in request["evidence"]),
                                f"{split}: synthetic evidence label differs")

    def test_shared_protocol_claims_are_intentional_not_semantic_scores(self) -> None:
        for split, rows in self.rows.items():
            claims = {contract.strict_object(row["messages"][2]["content"])["claim"] for row in rows}
            self.assertTrue(claims == set(contract.CLAIMS.values()), f"{split}: fixed claim vocabulary differs")
        tasks = {normalize_text(request["task"]) for request in self.requests["test"]}
        values = {normalize_text(item["value"]) for request in self.requests["test"] for item in request["evidence"]}
        self.assertTrue(len(tasks) == 75 and len(values) == 12,
                        "public final gate: normalized textual-form denominator differs")
        self.assertTrue(self.heldout_manifest["access_class"] == "PUBLIC_FROZEN_NOT_BLIND",
                        "public final gate is not blind")
        declared = self.heldout_manifest["future_model_gate"]
        self.assertTrue(declared["semantic_intent_detection_claim"] is False
                        and declared["predecessor_score_comparability"] is False,
                        "protocol gate must not become a semantic/predecessor score")

    def test_count_distribution_shift_is_preserved_not_hidden(self) -> None:
        sizes = {split: {len(request["evidence"]) for request in requests}
                 for split, requests in self.requests.items()}
        self.assertTrue(sizes["train"] == sizes["dev"] == {0, 1, 2}, "train/dev evidence count scope differs")
        self.assertTrue(sizes["test"] == {0, 1, 2, 3, 4}, "public final evidence count shift differs")

    def test_no_training_model_or_promotion_is_present(self) -> None:
        for summary in (self.manifest, self.heldout_manifest):
            self.assertTrue(all(summary[key] is False for key in
                                ("training_eligible", "publication_eligible", "execution_authority")),
                            "synthetic reference source cannot promote a model")
        self.assertTrue(self.heldout_manifest["model_evaluation"] == "NOT_RUN"
                        and self.heldout_manifest["optimizer_steps"] == 0,
                        "reference checks are not model evaluation/training")
        for filename in ("generate_curriculum.py", "curriculum_admission.py", "heldout/generate_heldout.py"):
            tree = ast.parse((HERE / filename).read_text(encoding="utf-8"))
            modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.add(node.module.split(".")[0])
            self.assertTrue(not modules.intersection({"torch", "transformers", "unsloth", "trl", "peft"}),
                            f"{filename}: model/training import exceeds source-only scope")


if __name__ == "__main__":
    unittest.main()
