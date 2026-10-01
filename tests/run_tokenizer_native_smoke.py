#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic native control in fresh processes, NOT a v1 or model benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "eval/tokenizer_qualification.py"
SPEC = importlib.util.spec_from_file_location("tokenizer_observer", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def run(arguments: list[str], expected: int = 0) -> None:
    completed = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), *arguments],
        capture_output=True, timeout=60, check=False)
    if completed.returncode != expected:
        # Child stderr may include native-library errors: never echo it.
        raise RuntimeError("NATIVE_CONTROL_EXIT_DIFFERED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    M.require(M.SOURCE.fullmatch(args.source) is not None, "SOURCE")
    # Import only inside this explicit synthetic test action.
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    identity = M.runtime_identity()
    M.require(identity["version"] == "0.23.2", "NATIVE_CONTROL_VERSION")
    vocab = {s: i for i, s in enumerate(["[UNK]", "[PAD]", "[BOS]", "[EOS]",
        "hello", "world", "café", "雪", "paired", "print", "x", "=", "1", "(", ")", "<", ">", "|"])}
    engine = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    engine.normalizer = normalizers.NFC()
    engine.pre_tokenizer = pre_tokenizers.Whitespace()
    engine.post_processor = processors.TemplateProcessing(single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1", special_tokens=[("[BOS]", 2), ("[EOS]", 3)])
    texts = [("empty", "", None), ("ascii", "hello world", None),
             ("unicode", "cafe\u0301 雪 🛰️", None), ("pair", "hello world", "paired"),
             ("code", 'print(x)\r\n\tx = 1', None), ("special", "[BOS] hello [EOS]", None),
             ("long", "hello world " * 120, None)]
    data = {"schema": "szl.tokenizer-corpus/v1", "cases": [dict(id=i, text=t, pair=p) for i, t, p in texts]}
    args.output.mkdir(parents=True, exist_ok=False)
    controls = []
    with tempfile.TemporaryDirectory(prefix="szl-tokenizer-smoke-") as temp:
        work = Path(temp)
        corp = work / "corpus.json"; corp.write_bytes(M.packed(data))
        for config in ("plain", "fixed-pad-truncate", "batch-pad"):
            if config == "fixed-pad-truncate":
                engine.enable_truncation(max_length=32, stride=4)
                engine.enable_padding(length=32, pad_id=1, pad_token="[PAD]", direction="left")
            elif config == "batch-pad":
                engine.no_truncation()
                engine.enable_padding(pad_id=1, pad_token="[PAD]", direction="right")
            tok = work / (config + ".json"); tok.write_text(engine.to_str(), encoding="utf-8")
            for threads in (1, 2, 4, 8):
                files = []
                for role in ("baseline", "candidate"):
                    out = args.output / f"{config}-{threads}-{role}.json"
                    run(["capture", "--tokenizer", str(tok), "--tokenizer-sha256", M.sha(tok.read_bytes()),
                         "--corpus", str(corp), "--corpus-sha256", M.sha(corp.read_bytes()),
                         "--source", args.source, "--version", identity["version"],
                         "--native-sha256", identity["native_sha256"], "--threads", str(threads),
                         "--repeats", "3", "--output", str(out)])
                    files.append(out)
                result = args.output / f"{config}-{threads}-comparison.json"
                run(["compare", "--baseline", str(files[0]), "--baseline-sha256", M.sha(files[0].read_bytes()),
                     "--candidate", str(files[1]), "--candidate-sha256", M.sha(files[1].read_bytes()), "--output", str(result)])
                report = M.strict(result.read_bytes())
                M.require(report["state"] == "SAME_BUILD_CONTROL_PARITY", "CONTROL_PARITY")
                controls.append({"config": config, "threads_requested": threads, "state": report["state"],
                                 "report_sha256": M.sha(result.read_bytes())})
        # Negative control is stored only in temporary scratch. It is deliberately
        # mutated observation DATA, never passed off as a real native observation.
        source = files[1]
        negative = M.strict(source.read_bytes())
        negative["rows"][0]["fields"]["ids"] = "0" * 64
        negative["content_sha256"] = M.sha(M.packed({k: v for k, v in negative.items() if k != "content_sha256"}))
        bad = work / "synthetic-negative.json"; bad.write_bytes(M.packed(negative))
        fail = work / "negative-result.json"
        run(["compare", "--baseline", str(files[0]), "--baseline-sha256", M.sha(files[0].read_bytes()),
             "--candidate", str(bad), "--candidate-sha256", M.sha(bad.read_bytes()), "--output", str(fail)], expected=2)
        M.require(M.strict(fail.read_bytes())["state"] == "FAIL_PARITY", "NEGATIVE_NOT_REJECTED")
    summary = {"schema": "szl.tokenizer-native-control/v1", "state": "PASS_SYNTHETIC_NATIVE_CONTROL",
        "source_revision_declared": args.source, "observer_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "runtime": identity, "controls": controls, "negative_control_rejected": True,
        "rc_executed": False, "representative_model_benchmark": False, "authority": M.AUTHORITY.copy()}
    (args.output / "native-control.json").write_bytes(M.packed(summary) + b"\n")
    print(M.packed({"state": summary["state"], "cells": len(controls), "rc_executed": False}).decode())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("NATIVE_CONTROL_FAILED; no production or RC qualification", file=sys.stderr)
        raise SystemExit(1) from None
