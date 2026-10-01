#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local tokenizer observations and offline paired comparison; never promotion.

Run capture in a fresh, separately prepared interpreter for each build/thread
setting. No package installation, Hub fetch, subprocess, model load or training.
The native library must be trusted separately: this is not a native-code sandbox.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
from typing import Any

SCHEMA = "szl.tokenizer-capture/v1"
PROFILE = "PYTHON_RICH_SOURCE_CONFIG_V1"
AUTHORITY = dict.fromkeys(("training", "publication", "promotion", "serving"), False)
FIELDS = ("ids", "tokens", "offsets", "attention_mask", "special_tokens_mask",
          "type_ids", "word_ids", "sequence_ids", "overflowing", "decode_all", "decode_skip")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SOURCE = re.compile(r"[0-9a-f]{40}\Z")
CASE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}\Z")
LIMIT = 8 * 1024 * 1024


class Invalid(ValueError):
    """Fixed diagnostic only; do not put local paths or corpus text here."""


class Unavailable(Invalid):
    """The selected interpreter cannot execute the declared profile."""


def require(ok: bool, code: str) -> None:
    if not ok:
        raise Invalid(code)


def integer(v: Any, low: int = 0, high: int = 10**18) -> bool:
    return type(v) is int and low <= v <= high


def hash_ok(v: Any) -> bool:
    return type(v) is str and SHA.fullmatch(v) is not None


def packed(v: Any) -> bytes:
    return json.dumps(v, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("ascii")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def strict(raw: bytes) -> Any:
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT, "JSON_SIZE")

    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, "DUPLICATE_KEY")
            result[k] = v
        return result

    def finite(v):
        f = float(v)
        require(math.isfinite(f), "NONFINITE")
        return f

    def reject(_):
        raise Invalid("NONFINITE")

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                            parse_float=finite, parse_constant=reject)
        stack = [(result, 0)]
        while stack:
            value, depth = stack.pop()
            require(depth <= 128, "JSON_DEPTH")
            if type(value) is dict:
                stack.extend((v, depth + 1) for v in value.values())
            elif type(value) is list:
                stack.extend((v, depth + 1) for v in value)
        return result
    except Invalid:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise Invalid("INVALID_JSON") from None


def read_bound(path: Path, expected: str, limit: int = LIMIT) -> bytes:
    require(hash_ok(expected), "EXPECTED_DIGEST")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and sha(raw) == expected, "INPUT_BYTES_DIFFER")
    return raw


def file_hash(path: Path) -> str:
    """Bound module hashing without retaining a large native library in memory."""
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            require(size <= 256 * 1024 * 1024, "MODULE_SIZE")
            h.update(block)
    require(size > 0, "EMPTY_MODULE")
    return h.hexdigest()


def corpus(raw: bytes) -> list[dict[str, Any]]:
    doc = strict(raw)
    require(type(doc) is dict and set(doc) == {"schema", "cases"}
            and doc["schema"] == "szl.tokenizer-corpus/v1", "CORPUS_SCHEMA")
    cases = doc["cases"]
    require(type(cases) is list and 1 <= len(cases) <= 256, "CASE_BOUND")
    seen = set()
    for row in cases:
        require(type(row) is dict and set(row) == {"id", "text", "pair"}, "CASE_FIELDS")
        name = row["id"]
        require(type(name) is str and CASE_ID.fullmatch(name) is not None
                and name not in seen, "CASE_ID")
        seen.add(name)
        for name in ("text", "pair"):
            text = row[name]
            if name == "pair" and text is None:
                continue
            require(type(text) is str, "TEXT_TYPE")
            try:
                require(len(text.encode("utf-8")) <= 32768, "TEXT_SIZE")
            except UnicodeError:
                raise Invalid("TEXT_ENCODING") from None
    return cases


def runtime_identity() -> dict[str, Any]:
    try:
        module = importlib.import_module("tokenizers")
        native = importlib.import_module("tokenizers.tokenizers")
        version = module.__version__
        require(type(version) is str and re.fullmatch(r"[A-Za-z0-9.+_-]{1,80}", version)
                is not None, "VERSION_SHAPE")
        return {"version": version, "native_sha256": file_hash(Path(native.__file__)),
                "binding_sha256": file_hash(Path(module.__file__)),
                "python": platform.python_version(), "system": platform.system(),
                "machine": platform.machine(), "host_digest": sha(platform.node().encode()),
                "cpu_count": os.cpu_count()}
    except (ImportError, AttributeError, OSError):
        raise Unavailable("NATIVE_BINDING_UNAVAILABLE") from None


def encoding_snapshot(engine, enc, depth: int = 0, budget: list[int] | None = None) -> dict[str, Any]:
    """Compare rich Python semantics, not IDs alone; absent fields never equal None."""
    if budget is None:
        budget = [262144, 256]
    require(depth <= 4 and budget[1] > 0, "OVERFLOW_BOUND")
    budget[1] -= 1
    try:
        data = {k: getattr(enc, k) for k in FIELDS[:8]}
        ids = data["ids"]
        require(type(ids) is list and len(ids) <= budget[0], "TOKEN_BOUND")
        budget[0] -= len(ids)
        require(all(integer(x, 0, 2**32 - 1) for x in ids), "TOKEN_ID")
        n = len(ids)
        require(all(type(data[k]) is list and len(data[k]) == n for k in FIELDS[1:8]), "ENCODING_LENGTH")
        require(all(type(x) is str for x in data["tokens"]), "TOKEN_TEXT")
        for k in ("attention_mask", "special_tokens_mask"):
            require(all(integer(x, 0, 1) for x in data[k]), "MASK_VALUE")
        require(all(integer(x, 0, 2**32 - 1) for x in data["type_ids"]), "TYPE_ID")
        for k in ("word_ids", "sequence_ids"):
            require(all(x is None or integer(x, 0, 2**32 - 1) for x in data[k]), "ALIGNMENT_ID")
        require(all(type(x) in (tuple, list) and len(x) == 2 and integer(x[0])
                    and integer(x[1]) and x[0] <= x[1] for x in data["offsets"]), "OFFSET")
        overflow = enc.overflowing
        require(type(overflow) is list and len(overflow) <= budget[1], "OVERFLOW_BOUND")
        data["overflowing"] = [encoding_snapshot(engine, x, depth + 1, budget) for x in overflow]
        data["decode_all"] = engine.decode(ids, skip_special_tokens=False)
        data["decode_skip"] = engine.decode(ids, skip_special_tokens=True)
        require(type(data["decode_all"]) is str and type(data["decode_skip"]) is str, "DECODE_TYPE")
        require(len(packed(data)) <= LIMIT, "ENCODING_BYTES")
        return data
    except (AttributeError, NotImplementedError):
        raise Unavailable("RICH_ENCODING_API_UNSUPPORTED") from None


def fingerprint(engine, enc) -> dict[str, Any]:
    data = encoding_snapshot(engine, enc)
    return {"token_count": len(data["ids"]), "fields": {k: sha(packed(data[k])) for k in FIELDS}}


def exercise(engine, cases: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    """Fixed corpus repeated-cache workload, not cache-cold or Rust-only throughput."""
    require(integer(repeats, 3, 30), "REPEAT_BOUND")
    rows, first_times, batch_times, decode_times = [], [], [], []
    singles, expected_batches = {}, {}
    for add in (False, True):
        singles[add] = []
        for case in cases:
            t = time.perf_counter_ns()
            enc = engine.encode(case["text"], pair=case["pair"], add_special_tokens=add)
            first_times.append(max(1, time.perf_counter_ns() - t))
            singles[add].append(enc)
            rows.append({"id": case["id"], "input_sha256": sha(packed(case)), "add_special_tokens": add,
                         "call": "single", **fingerprint(engine, enc)})
        inputs = [c["text"] if c["pair"] is None else (c["text"], c["pair"]) for c in cases]
        batch = engine.encode_batch(inputs, add_special_tokens=add)
        require(type(batch) is list and len(batch) == len(cases), "BATCH_COVERAGE")
        expected_batches[add] = [fingerprint(engine, enc) for enc in batch]
        rows.extend({"id": c["id"], "input_sha256": sha(packed(c)), "add_special_tokens": add,
                     "call": "batch", **fp} for c, fp in zip(cases, expected_batches[add]))
    # Single/batch shapes can legitimately differ under dynamic padding. Each
    # call mode is compared to its own oracle, never stripped to hide padding.
    for _ in range(repeats):
        batch_ns = decode_ns = 0
        for add in (False, True):
            t = time.perf_counter_ns()
            actual = engine.encode_batch(inputs, add_special_tokens=add)
            batch_ns += max(1, time.perf_counter_ns() - t)
            require(type(actual) is list and len(actual) == len(cases), "BATCH_COVERAGE")
            require([fingerprint(engine, enc) for enc in actual] == expected_batches[add], "WARM_OUTPUT_DRIFT")
            t = time.perf_counter_ns()
            decoded = [engine.decode(enc.ids, skip_special_tokens=skip)
                       for enc in singles[add] for skip in (False, True)]
            decode_ns += max(1, time.perf_counter_ns() - t)
            previous = [r for r in rows if r["call"] == "single" and r["add_special_tokens"] is add]
            expected = [r["fields"][k] for r in previous for k in ("decode_all", "decode_skip")]
            require([sha(packed(x)) for x in decoded] == expected, "DECODE_DRIFT")
        batch_times.append(batch_ns)
        decode_times.append(decode_ns)
    return {"rows": rows, "timing": {"first_sweep_ns": first_times,
             "warm_batch_ns": batch_times, "warm_decode_ns": decode_times}}


def peak_rss() -> int | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if sys.platform == "darwin" else 1024
        return int(value * scale) if value > 0 and sys.platform in ("linux", "darwin") else None
    except (ImportError, OSError):
        return None


def capture(args) -> dict[str, Any]:
    require(type(args.source) is str and SOURCE.fullmatch(args.source) is not None, "SOURCE_DECLARATION")
    require(args.threads in (1, 2, 4, 8) and integer(args.repeats, 3, 30), "RUN_BOUND")
    require(not any(x == "tokenizers" or x.startswith("tokenizers.") for x in sys.modules), "FRESH_PROCESS_REQUIRED")
    tokenizer_raw = read_bound(args.tokenizer, args.tokenizer_sha256)
    corpus_raw = read_bound(args.corpus, args.corpus_sha256, 4 * 1024 * 1024)
    cases = corpus(corpus_raw)
    require(type(strict(tokenizer_raw)) is dict, "TOKENIZER_OBJECT")
    require(hash_ok(args.native_sha256), "EXPECTED_NATIVE_DIGEST")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    identity = runtime_identity()
    require(identity["version"] == args.version and identity["native_sha256"] == args.native_sha256,
            "INSTALLED_BUILD_DIFFERS")
    try:
        cls = importlib.import_module("tokenizers").Tokenizer
        t = time.perf_counter_ns()
        engine = cls.from_str(tokenizer_raw.decode("utf-8"))
        load_ns = max(1, time.perf_counter_ns() - t)
        measured = exercise(engine, cases, args.repeats)
    except Invalid:
        raise
    except (AttributeError, NotImplementedError):
        raise Unavailable("PYTHON_API_UNSUPPORTED") from None
    except Exception:
        raise Unavailable("TOKENIZER_EXECUTION_FAILED") from None
    result = {"schema": SCHEMA, "profile": PROFILE, "state": "RECORDED_NOT_QUALIFIED",
              "authority": AUTHORITY.copy(), "observed_at": datetime.now(timezone.utc).isoformat(),
              "source_revision_declared": args.source, "observer_sha256": file_hash(Path(__file__)),
              "tokenizer_sha256": sha(tokenizer_raw), "corpus_sha256": sha(corpus_raw),
              "case_count": len(cases), "threads_requested": args.threads, "repeats": args.repeats,
              "runtime": identity, "load_ns": load_ns, "peak_process_rss_bytes": peak_rss(),
              "workload": "REPEATED_FIXED_CORPUS_PYTHON_CALLS", **measured}
    result["content_sha256"] = sha(packed(result))
    return result


def validate(doc: Any) -> None:
    keys = {"schema", "profile", "state", "authority", "observed_at", "source_revision_declared",
            "observer_sha256", "tokenizer_sha256", "corpus_sha256", "case_count", "threads_requested",
            "repeats", "runtime", "load_ns", "peak_process_rss_bytes", "workload", "rows", "timing", "content_sha256"}
    require(type(doc) is dict and set(doc) == keys and doc["schema"] == SCHEMA
            and doc["profile"] == PROFILE and doc["state"] == "RECORDED_NOT_QUALIFIED", "CAPTURE_SCHEMA")
    require(type(doc["authority"]) is dict and set(doc["authority"]) == set(AUTHORITY)
            and all(v is False for v in doc["authority"].values()), "AUTHORITY_AMPLIFICATION")
    require(hash_ok(doc["content_sha256"]) and doc["content_sha256"] ==
            sha(packed({k: v for k, v in doc.items() if k != "content_sha256"})), "CAPTURE_DIGEST")
    for k in ("observer_sha256", "tokenizer_sha256", "corpus_sha256"):
        require(hash_ok(doc[k]), "CAPTURE_IDENTITY")
    require(type(doc["source_revision_declared"]) is str and SOURCE.fullmatch(doc["source_revision_declared"]) is not None,
            "CAPTURE_SOURCE")
    require(type(doc["observed_at"]) is str, "CAPTURE_TIME")
    stamp = datetime.fromisoformat(doc["observed_at"])
    require(stamp.tzinfo is not None and stamp <= datetime.now(timezone.utc), "CAPTURE_TIME")
    n, repeats = doc["case_count"], doc["repeats"]
    require(integer(n, 1, 256) and integer(repeats, 3, 30) and type(doc["threads_requested"]) is int
            and doc["threads_requested"] in (1, 2, 4, 8), "CAPTURE_BOUNDS")
    require(doc["workload"] == "REPEATED_FIXED_CORPUS_PYTHON_CALLS" and integer(doc["load_ns"], 1), "MEASUREMENT_SCOPE")
    require(doc["peak_process_rss_bytes"] is None or integer(doc["peak_process_rss_bytes"], 1), "RSS_VALUE")
    runtime = doc["runtime"]
    require(type(runtime) is dict and set(runtime) == {"version", "native_sha256", "binding_sha256", "python",
            "system", "machine", "host_digest", "cpu_count"}, "RUNTIME_SHAPE")
    for key in ("native_sha256", "binding_sha256", "host_digest"):
        require(hash_ok(runtime[key]), "RUNTIME_DIGEST")
    for key in ("version", "python", "system", "machine"):
        require(type(runtime[key]) is str and re.fullmatch(r"[A-Za-z0-9.+_ -]{1,80}", runtime[key]) is not None, "RUNTIME_VALUE")
    require(runtime["cpu_count"] is None or integer(runtime["cpu_count"], 1, 65536), "CPU_COUNT")
    rows, found, inputs = doc["rows"], set(), {}
    require(type(rows) is list and len(rows) == 4 * n, "ROW_COVERAGE")
    for row in rows:
        require(type(row) is dict and set(row) == {"id", "input_sha256", "add_special_tokens", "call", "token_count", "fields"}, "ROW_SHAPE")
        require(type(row["id"]) is str and CASE_ID.fullmatch(row["id"]) is not None
                and type(row["add_special_tokens"]) is bool and type(row["call"]) is str
                and row["call"] in ("single", "batch"), "ROW_KEY")
        key = (row["id"], row["add_special_tokens"], row["call"])
        require(key not in found and hash_ok(row["input_sha256"]), "ROW_DUPLICATE_OR_DIGEST")
        found.add(key)
        require(inputs.setdefault(row["id"], row["input_sha256"]) == row["input_sha256"], "ROW_INPUT_CHANGED")
        require(integer(row["token_count"], 0, 262144) and type(row["fields"]) is dict
                and set(row["fields"]) == set(FIELDS) and all(hash_ok(v) for v in row["fields"].values()), "FIELD_COVERAGE")
    require(len(inputs) == n and found == {(i, a, c) for i in inputs for a in (False, True) for c in ("single", "batch")}, "ROW_MATRIX")
    timing = doc["timing"]
    require(type(timing) is dict and set(timing) == {"first_sweep_ns", "warm_batch_ns", "warm_decode_ns"}, "TIMING_SHAPE")
    for key, length in (("first_sweep_ns", 2 * n), ("warm_batch_ns", repeats), ("warm_decode_ns", repeats)):
        require(type(timing[key]) is list and len(timing[key]) == length
                and all(integer(v, 1, 10**15) for v in timing[key]), "TIMING_VALUES")


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    validate(baseline)
    validate(candidate)
    for key in ("profile", "observer_sha256", "source_revision_declared", "tokenizer_sha256", "corpus_sha256", "case_count"):
        require(baseline[key] == candidate[key], "COMPARISON_INPUTS_DIFFER")
    def index(doc):
        return {(r["id"], r["add_special_tokens"], r["call"]): r for r in doc["rows"]}
    left, right = index(baseline), index(candidate)
    require(set(left) == set(right), "COMPARISON_CASES_DIFFER")
    mismatches = []
    for key in sorted(left):
        a, b = left[key], right[key]
        require(a["input_sha256"] == b["input_sha256"], "COMPARISON_CASE_INPUT_DIFFERS")
        fields = [k for k in FIELDS if a["fields"][k] != b["fields"][k]]
        if a["token_count"] != b["token_count"]:
            fields.append("token_count")
        if fields:
            mismatches.append({"id": key[0], "add_special_tokens": key[1], "call": key[2], "fields": fields})
    same_build = all(baseline["runtime"][k] == candidate["runtime"][k]
                     for k in ("version", "native_sha256", "binding_sha256"))
    comparable = (not mismatches and all(baseline["runtime"][k] == candidate["runtime"][k]
                   for k in ("python", "system", "machine", "host_digest", "cpu_count"))
                   and baseline["threads_requested"] == candidate["threads_requested"]
                   and baseline["repeats"] == candidate["repeats"])
    ratios = {k: (statistics.median(baseline["timing"][k]) / statistics.median(candidate["timing"][k])
                  if comparable else None) for k in ("warm_batch_ns", "warm_decode_ns")}
    return {"schema": "szl.tokenizer-comparison/v1", "state": "FAIL_PARITY" if mismatches else
            "SAME_BUILD_CONTROL_PARITY" if same_build else "PARITY_OBSERVED_FOR_DECLARED_PROFILE",
            "authority": AUTHORITY.copy(), "baseline_capture": baseline["content_sha256"],
            "candidate_capture": candidate["content_sha256"], "mismatches": mismatches,
            "matching_observed_environment": comparable, "baseline_over_candidate_median_ratio": ratios,
            "candidate_benefit_established": False, "independently_attested": False,
            "scope": "SUPPLIED_LOCAL_OBSERVATIONS_NOT_GENERAL_COMPATIBILITY_OR_SPEEDUP_PROOF"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("identity")
    cap = actions.add_parser("capture")
    for name in ("tokenizer", "corpus"):
        cap.add_argument("--" + name, type=Path, required=True)
        cap.add_argument("--" + name + "-sha256", required=True)
    cap.add_argument("--source", required=True)
    cap.add_argument("--version", required=True)
    cap.add_argument("--native-sha256", required=True)
    cap.add_argument("--threads", type=int, choices=(1, 2, 4, 8), default=1)
    cap.add_argument("--repeats", type=int, default=5)
    cmp = actions.add_parser("compare")
    for name in ("baseline", "candidate"):
        cmp.add_argument("--" + name, type=Path, required=True)
        cmp.add_argument("--" + name + "-sha256", required=True)
    for p in (cap, cmp):
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "identity":
            print(packed(runtime_identity()).decode())
            return 0
        # Reserve before library work. Failure retains an honest diagnostic file.
        # On abrupt termination an empty/incomplete file is not valid evidence.
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            try:
                if args.action == "capture":
                    result = capture(args)
                    validate(result)
                else:
                    result = compare(strict(read_bound(args.baseline, args.baseline_sha256)),
                                     strict(read_bound(args.candidate, args.candidate_sha256)))
                code = 2 if result["state"] == "FAIL_PARITY" else 0
            except Invalid as exc:
                result = {"state": "UNAVAILABLE" if isinstance(exc, Unavailable) else "REJECTED",
                          "reason_code": str(exc), "authority": AUTHORITY.copy()}
                code = 2 if isinstance(exc, Unavailable) else 1
            except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
                result, code = {"state": "REJECTED", "reason_code": "LOCAL_INPUT_FAILURE", "authority": AUTHORITY.copy()}, 1
            stream.write(packed(result) + b"\n")
        print(packed({"state": result["state"], "authority": AUTHORITY}).decode())
        return code
    except (OSError, Invalid):
        print('{"state":"UNAVAILABLE","reason_code":"OUTPUT_OR_BINDING_UNAVAILABLE"}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
