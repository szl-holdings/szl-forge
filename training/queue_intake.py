#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline queue intake and exact-response smoke characterization, never a launcher.

Extends Forge's existing launch contract. It cannot load/train a model, allocate
compute, sign an approval, change a Hub artifact, or turn a smoke score into GO.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from typing import Any

MAX_BYTES = 2_000_000
SHA = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SECRET = re.compile(r"hf_[A-Za-z0-9]{20,}")


class IntakeError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise IntakeError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON member")
        result[key] = value
    return result


def decode(raw: bytes) -> Any:
    require(0 < len(raw) <= MAX_BYTES, "empty or oversized input")
    try:
        text = raw.decode("utf-8")
        require(not SECRET.search(text), "credential-like input refused; redact at source")
        result = json.loads(text, object_pairs_hook=unique,
                            parse_constant=lambda _: require(False, "nonfinite JSON"))
        require(not SECRET.search(json.dumps(result, ensure_ascii=False)),
                "credential-like input refused; redact at source")
        return result
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise IntakeError("invalid or credential-bearing JSON") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "missing canonical module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_index(rows: Any) -> dict:
    require(type(rows) is list, "metadata snapshot must be an array")
    index = {}
    for row in rows:
        require(type(row) is dict and row.get("kind") in {"models", "datasets"},
                "invalid metadata namespace")
        rid = row.get("id")
        require(type(rid) is str and REPO.fullmatch(rid) is not None, "invalid repository identity")
        key = (row["kind"], rid.casefold())
        require(key not in index, "duplicate namespace/repository observation")
        index[key] = row
    return index


def available(row: dict | None) -> bool:
    return (type(row) is dict and type(row.get("status")) is int
            and row["status"] == 200 and type(row.get("sha")) is str
            and SHA.fullmatch(row["sha"]) is not None)


def intake(queue: Any, observations: Any, *, config_dir: Path) -> dict:
    """Call the canonical non-executing contract for each proposed wave.

    Snapshot revisions are proposals, never automatically approved pins. No
    missing corpus, budget, admission, or holdout is invented by this function.
    """
    require(type(queue) is dict and queue.get("schema") == "szl.training-queue-import/v1",
            "unsupported queue schema")
    entries = queue.get("entries")
    total = queue.get("declared_total")
    require(type(entries) is list and 0 < len(entries) <= 1000, "empty or oversized queue")
    require(type(total) is int and total >= 0, "invalid declared total")
    idx = snapshot_index(observations)
    contract = load_module("szl_queue_launch_contract", Path(__file__).with_name("launch_contract.py"))
    identities = set()
    rows = []
    for entry in entries:
        require(type(entry) is dict, "queue entry must be an object")
        name, base, dataset = (entry.get(k) for k in ("name", "base", "dataset"))
        require(type(name) is str and "/" not in name and REPO.fullmatch("SZLHOLDINGS/" + name),
                "invalid target name")
        target = "SZLHOLDINGS/" + name
        require(target.casefold() not in identities, "duplicate target identity")
        identities.add(target.casefold())
        require(all(type(x) is str and REPO.fullmatch(x) for x in (base, dataset)),
                "invalid base or dataset identity")
        model = idx.get(("models", target.casefold()))
        upstream = idx.get(("models", base.casefold()))
        corpus = idx.get(("datasets", dataset.casefold()))
        blockers = ["OPERATOR_ADMISSION_REQUIRED", "GPU_LEASE_AND_BUDGET_REQUIRED",
                    "HYPERPARAMETER_CONTRACT_NOT_ESTABLISHED"]
        if total != len(entries):
            blockers.append("QUEUE_POPULATION_MISMATCH")
        for role, obj in (("TARGET", model), ("BASE", upstream), ("DATASET", corpus)):
            if not available(obj):
                blockers.append(role + "_OBSERVATION_UNAVAILABLE")
        model = model or {}
        corpus = corpus or {}
        pipeline = model.get("pipeline_tag")
        if pipeline != "text-generation":
            blockers.append("TARGET_NOT_QUALIFIED_FOR_CAUSAL_LM_SFT")
        card = model.get("cardData") or {}
        require(type(card) is dict, "invalid target card metadata")
        lineage = card.get("base_model")
        lineage = [lineage] if isinstance(lineage, str) else lineage
        if type(lineage) is not list or not lineage or not all(type(x) is str for x in lineage):
            blockers.append("TARGET_BASE_LINEAGE_UNAVAILABLE")
        elif base.casefold() not in {x.casefold() for x in lineage}:
            blockers.append("SUPPLIED_BASE_DIFFERS_FROM_DECLARED_LINEAGE")
        adapter = (model.get("adapter_config") or {}).get("data") or {}
        if adapter.get("base_model_name_or_path") not in (None, base):
            blockers.append("ADAPTER_BASE_MAPPING_REQUIRES_REVIEW")
        files = model.get("files", [])
        require(type(files) is list and all(type(x) is str for x in files), "invalid artifact list")
        if any(x.endswith(".npz") for x in files):
            blockers.append("MIXED_SOFTWARE_WEIGHT_DESTINATION_REQUIRES_REVIEW")
        proposal = {"schema": contract.CONFIG_SCHEMA, "kind": "sft", "method": "lora16",
                    "student": {"id": base, "revision": upstream.get("sha") if available(upstream) else None},
                    "dataset": {"id": dataset, "revision": corpus.get("sha") if available(corpus) else None,
                                "split": "train"},
                    "target_repo": target, "local_only": True,
                    "admission": None, "train_jsonl": None, "holdout_jsonl": None}
        result = contract.evaluate(proposal, config_dir=config_dir, expected_kind="sft")
        blockers.extend(result["codes"])
        rows.append({"target": target, "observed_target_revision": model.get("sha"),
                     "pipeline_tag": pipeline, "supplied_base": base, "declared_lineage": lineage,
                     "dataset": dataset, "proposed_wave": proposal,
                     "launch_contract": {k: result[k] for k in ("schema", "ok", "verdict", "codes",
                                                                "accepted_examples", "training_started", "hub_write")},
                     "blockers": sorted(set(blockers)), "training_authorized": False,
                     "disposition": "HOLD_NOT_LAUNCHABLE"})
    return {"schema": "szl.training-queue-intake/v1", "declared_total": total,
            "actual_entries": len(rows), "population_matches": total == len(rows),
            "entries": rows, "training_authorized": False, "hub_write": False,
            "scope": "SUPPLIED_QUEUE_AND_RECORDED_METADATA_NOT_WEIGHT_OR_RIGHTS_VERIFICATION"}


def serving_metadata_projection(catalog: Any) -> dict:
    """Compare advertised settings, NOT weights or authenticated identity.

    vLLM's observed catalog generates created timestamps and permission IDs per
    response. Preserve all other fields; do not erase policy/config changes.
    """
    require(type(catalog) is dict and catalog.get("object") == "list", "invalid serving catalog")
    data = catalog.get("data")
    require(type(data) is list and data, "empty serving catalog")
    chosen, seen = [], set()
    for model in data:
        require(type(model) is dict and type(model.get("id")) is str and model["id"], "missing served identity")
        require(model["id"] not in seen, "duplicate served identity")
        seen.add(model["id"])
        require(type(model.get("created")) is int and model["created"] >= 0, "invalid served timestamp")
        permissions = model.get("permission")
        require(type(permissions) is list, "missing serving permission metadata")
        projected = []
        for permission in permissions:
            require(type(permission) is dict and type(permission.get("id")) is str,
                    "invalid serving permission identifier")
            require(type(permission.get("created")) is int and permission["created"] >= 0,
                    "invalid serving permission timestamp")
            projected.append({k: v for k, v in permission.items() if k not in {"id", "created"}})
        record = {k: v for k, v in model.items() if k not in {"created", "permission"}}
        record["permission"] = sorted(projected, key=digest)
        chosen.append(record)
    result = {k: v for k, v in catalog.items() if k != "data"}
    result["data"] = sorted(chosen, key=lambda row: row["id"])
    return result


def normalize(value: str) -> str:
    return " ".join(value.casefold().split()).rstrip(".! ")


def characterize(fixtures: Any, run: Any, *, fixture_sha256: str) -> dict:
    """A new exact-response smoke protocol. No semantic benchmark or promotion."""
    require(type(fixture_sha256) is str and SHA256.fullmatch(fixture_sha256), "fixture hash required")
    require(type(fixtures) is dict and fixtures.get("schema") == "szl.exact-response-fixtures/v2",
            "wrong fixture schema")
    cases = fixtures.get("cases")
    require(type(cases) is list and 0 < len(cases) <= 1000, "empty or excessive fixtures")
    require(type(run) is dict and run.get("schema") == "szl.raw-smoke-responses/v2", "wrong response schema")
    require(run.get("fixture_sha256") == fixture_sha256, "response fixture binding mismatch")
    require(run.get("evidence_class") in {"SIMULATED", "LIVE_SMOKE"}, "missing evidence class")
    require(type(run.get("model_binding")) is dict, "missing model-binding record")
    records = run.get("responses")
    require(type(records) is list, "missing response list")
    by_id = {}
    for record in records:
        require(type(record) is dict and type(record.get("id")) is str, "invalid response identity")
        require(record["id"] not in by_id, "duplicate response identity")
        by_id[record["id"]] = record
    expected_ids = [f.get("id") if type(f) is dict else None for f in cases]
    require(all(type(x) is str and x for x in expected_ids) and len(set(expected_ids)) == len(cases),
            "invalid or duplicate fixture identity")
    require(set(by_id) == set(expected_ids), "response population does not match fixture identities")
    rows = []
    for fixture in cases:
        require(fixture.get("kind") in {"answerable", "unanswerable_under_fixture"}, "unsupported fixture kind")
        accepted = fixture.get("accepted")
        require(type(accepted) is list and accepted and all(type(s) is str and s.strip() for s in accepted),
                "explicit nonempty acceptance set required")
        require(type(fixture.get("question")) is str and fixture["question"], "missing question")
        record = by_id[fixture["id"]]
        response = record.get("response")
        require(response is None or (type(response) is str and len(response.encode()) <= 131072),
                "response is not bounded text")
        complete = record.get("status") == "OK" and record.get("finish_reason") in {"stop", "eos_token"}
        if not complete or not response or not response.strip():
            verdict = "UNAVAILABLE"
        elif normalize(response) in {normalize(x) for x in accepted}:
            verdict = "ACCEPTED_EXACT_RESPONSE"
        elif normalize(response) in {"abstain", "i don't know", "i do not know"}:
            verdict = "ABSTAINED_ON_ANSWERABLE" if fixture["kind"] == "answerable" else "REVIEW_REQUIRED"
        elif fixture["kind"] == "answerable" and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", normalize(response)):
            verdict = "INCORRECT_EXACT_ANSWER"
        else:
            verdict = "REVIEW_REQUIRED"
        rows.append({"id": fixture["id"], "kind": fixture["kind"], "verdict": verdict,
                     "response": response, "response_sha256": hashlib.sha256(response.encode()).hexdigest()
                     if response is not None else None, "finish_reason": record.get("finish_reason"),
                     "status": record.get("status")})
    return {"schema": "szl.exact-response-characterization/v2", "fixture_sha256": fixture_sha256,
            "evidence_class": run["evidence_class"], "model_binding": run["model_binding"],
            "expected_count": len(cases), "observed_count": len(records),
            "complete_outputs": sum(r["verdict"] != "UNAVAILABLE" for r in rows),
            "outcomes": dict(Counter(r["verdict"] for r in rows)), "rows": rows,
            "training_authorized": False, "promotion_authorized": False, "signature_verified": False,
            "scope": "EXACT_RESPONSE_SMOKE_ONLY_NOT_SEMANTIC_ACCURACY_OR_HISTORICAL_COMPARISON"}


def write_new(path: Path, report: dict) -> None:
    body = {**report, "receipt_sha256": digest(report), "digest_scope": "OBJECT_WITHOUT_RECEIPT_SHA256_AND_DIGEST_SCOPE"}
    raw = (json.dumps(body, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("intake")
    plan.add_argument("--queue", type=Path, required=True)
    plan.add_argument("--snapshot", type=Path, required=True)
    score = sub.add_parser("characterize")
    score.add_argument("--fixtures", type=Path, required=True)
    score.add_argument("--fixture-sha256", required=True)
    score.add_argument("--responses", type=Path, required=True)
    for command in (plan, score):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "intake":
            result = intake(decode(args.queue.read_bytes()), decode(args.snapshot.read_bytes()), config_dir=args.queue.parent)
        else:
            raw = args.fixtures.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == args.fixture_sha256, "fixture bytes changed")
            result = characterize(decode(raw), decode(args.responses.read_bytes()), fixture_sha256=args.fixture_sha256)
        write_new(args.out, result)
        print(json.dumps({"status": "RECORDED_NOT_AUTHORIZED", "schema": result["schema"], "output": str(args.out)}))
        return 0
    except (IntakeError, OSError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "REJECT", "error_class": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
