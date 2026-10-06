# SPDX-License-Identifier: Apache-2.0
"""Pure train-only message conversion from freshly captured v4 bytes.

Only ``capture_train_only`` may create the input used by a future runner. Its
snapshot is a trusted-process convenience, not a capability or authorization.
Do not deserialize snapshots or accept saved reports instead of fresh capture.
This module does not read paths, rerun gates, import a model runtime, or train.
The conformance report is retained opaque provenance, never a launch permit.
"""
from __future__ import annotations

from collections import Counter
import hashlib

import curriculum_admission as admission
import json_contract as contract


def training_messages(
    snapshot: admission.TrainOnlySnapshot,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Return immutable role/content triples from the exact captured train buffer.

    Complete pair validation and both real gates belong to fresh capture. They
    are not repeated here: the pair validator itself reads parent schema paths.
    The frozen manifest hash and its train commitment bind the retained bytes;
    the checks below defend the conversion shape without any path I/O. Nothing
    returned here authenticates a curriculum or permits a GPU/model invocation.
    """
    require = admission._require
    require(type(snapshot) is admission.TrainOnlySnapshot,
            "training input must be a trusted-process TrainOnlySnapshot")
    manifest_raw = snapshot.manifest_bytes
    train_raw = snapshot.train_bytes
    report_raw = snapshot.conformance_report_bytes
    for raw, maximum, label in ((manifest_raw, contract.MAX_BYTES, "manifest"),
                                (train_raw, admission.MAX_FILE_BYTES, "train"),
                                (report_raw, admission.MAX_FILE_BYTES, "report")):
        require(type(raw) is bytes and 0 < len(raw) <= maximum,
                f"captured {label} must be bounded exact bytes")
    require(hashlib.sha256(manifest_raw).hexdigest() == admission.MANIFEST_SHA256,
            "captured manifest differs from the frozen commitment")
    manifest = admission._strict(manifest_raw)
    require(manifest_raw == contract.canonical_json(manifest).encode("utf-8") + b"\n",
            "captured manifest must retain its canonical LF bytes")
    admission._keys(manifest, admission.MANIFEST_KEYS, "captured manifest")
    for key, expected in (("schema", admission.SCHEMA),
                          ("candidate_id", contract.PROFILE),
                          ("contract_source_revision", admission.CONTRACT_REVISION),
                          ("contract_source_sha256", admission.CONTRACT_SOURCE_SHA256),
                          ("evidence_class", "SIMULATED"),
                          ("signature", "UNAVAILABLE"),
                          ("runtime_binding", "UNAVAILABLE")):
        require(manifest[key] == expected, f"captured manifest {key} differs")
    require(type(manifest["seed"]) is int and manifest["seed"] == admission.SEED,
            "captured manifest seed differs")
    for key in ("training_eligible", "publication_eligible", "execution_authority"):
        require(manifest[key] is False, "conversion cannot grant authority")
    admission._keys(manifest["splits"], {"train", "dev", "test"}, "captured splits")
    train = manifest["splits"]["train"]
    admission._keys(train, {"path", "sha256", "rows", "kind_counts"}, "captured train")
    require(train["path"] == "train.jsonl" and type(train["rows"]) is int
            and train["rows"] == admission.MAX_ROWS, "captured train identity differs")
    admission._sha(train["sha256"], "captured train digest")
    require(hashlib.sha256(train_raw).hexdigest() == train["sha256"],
            "captured train bytes differ from the manifest")
    require(train_raw.endswith(b"\n") and b"\r" not in train_raw,
            "captured train bytes must be LF JSONL")
    lines = train_raw.split(b"\n")[:-1]
    require(len(lines) == admission.MAX_ROWS, "captured train row count differs")
    rows: list[tuple[tuple[str, str], ...]] = []
    identities: set[str] = set()
    kinds: Counter[str] = Counter()
    for line in lines:
        row = admission._strict(line)
        admission._keys(row, admission.ROW_KEYS, "captured row")
        require(line == contract.canonical_json(row).encode("utf-8"),
                "captured row must retain canonical JSON")
        require(row["split"] == "train", "nontraining split cannot be converted")
        require(isinstance(row["family"], str) and row["family"] in admission.TRAIN_FAMILIES,
                "unknown or held-out family cannot be converted")
        require(isinstance(row["kind"], str) and row["kind"] in contract.CLAIMS,
                "unknown response kind cannot be converted")
        identity = row["id"]
        require(isinstance(identity, str) and identity and identity not in identities,
                "captured row identity must be nonempty and unique")
        identities.add(identity)
        messages = row["messages"]
        require(isinstance(messages, list) and len(messages) == 3,
                "captured rows require exactly three messages")
        converted: list[tuple[str, str]] = []
        for message, role in zip(messages, ("system", "user", "assistant")):
            admission._keys(message, {"role", "content"}, "captured message")
            require(message["role"] == role and isinstance(message["content"], str)
                    and bool(message["content"]), "captured message role/content differs")
            converted.append((role, message["content"]))
        require(converted[0][1] == contract.SYSTEM_PROMPT,
                "captured system prompt differs")
        rows.append(tuple(converted))
        kinds[row["kind"]] += 1
    expected_counts = {kind: 120 for kind in contract.CLAIMS}
    require(type(train["kind_counts"]) is dict
            and set(train["kind_counts"]) == set(expected_counts)
            and all(type(value) is int for value in train["kind_counts"].values())
            and train["kind_counts"] == expected_counts and dict(kinds) == expected_counts,
            "captured response-kind coverage differs")
    return tuple(rows)
