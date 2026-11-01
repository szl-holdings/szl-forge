#!/usr/bin/env python3
"""Bounded synthetic CPU research, with pinned source and separate dev/test data.

Never a production qualifier or policy decision-maker. No network or GPU use.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import numpy as np

SOURCE_REVISION = "8ba6893099aecfe1e7cced1b50075ccdd169e744"
PINS = {"receipt_agent.py": "bda481e78238e001100356dbb40ef2a8dd53b3490942f2a810d6d0aac5408609",
        "tiny_khipu.py": "4a1929574477ea356897b8eb1aec6ca655fe03b964061eea34332c5dadfb5cbe"}

def load_source(directory, filename):
    path = directory / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[filename]:
        raise ValueError("Source hash mismatch: " + filename)
    spec = importlib.util.spec_from_file_location("nano_research_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def row_hashes(x, y):
    return {hashlib.sha256(np.asarray(row, dtype="<f8").tobytes() +
                          np.asarray([label], dtype="<i8").tobytes()).hexdigest()
            for row, label in zip(x, y)}

def split_evidence(splits):
    hashes = {name: row_hashes(x, y) for name, (x, y) in splits.items()}
    for name, (x, y) in splits.items():
        if len(x) != len(y) or len(x) == 0 or len(hashes[name]) != len(x):
            raise ValueError("Empty, misaligned or duplicated split")
    for a, b in (("train","dev"), ("train","test"), ("dev","test")):
        if hashes[a] & hashes[b]:
            raise ValueError("Cross-split row overlap")
    return {name: {"n": len(splits[name][0]),
                    "row_set_sha256": hashlib.sha256("\n".join(sorted(value)).encode()).hexdigest()}
            for name, value in hashes.items()}

def metrics(pred, gold):
    pred, gold = np.asarray(pred), np.asarray(gold)
    if pred.shape != gold.shape or pred.ndim != 1 or pred.size == 0:
        raise ValueError("Named nonempty evaluation required")
    if not np.isin(pred, range(4)).all() or not np.isin(gold, range(4)).all():
        raise ValueError("Unknown class")
    counts = np.zeros((4,4), dtype=int)
    for y, p in zip(gold, pred):
        counts[int(y), int(p)] += 1
    n, k = int(len(gold)), int(np.sum(pred == gold))
    p, z = k/n, 1.959963984540054
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"correct": k, "n": n, "agreement": p,
            "wilson_95": [float(center-half), float(center+half)],
            "confusion_gold_by_prediction": counts.tolist(),
            "unsafe_allow_count": int(np.sum((pred == 0) & (gold != 0)))}

def tensor_digest(weights):
    h=hashlib.sha256()
    for name, value in sorted(weights.items()):
        arr=np.asarray(value,dtype="<f8")
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite weights")
        h.update(name.encode()+b"\0"+json.dumps(arr.shape).encode()+b"\0"+arr.tobytes())
    return h.hexdigest()

def receipt_run(module, seed, output):
    splits = {"train": module.synth_features(600, seed),
              "dev": module.synth_features(200, seed+100_000),
              "test": module.synth_features(400, seed+200_000)}
    evidence = split_evidence(splits)
    x, y=splits["train"]
    xd, yd=splits["dev"]
    w=module.init_weights(seed)
    initial={k:v.copy() for k,v in w.items()}
    best, best_agree, selected, used = None, -1, 0, 0
    for step in range(600):
        loss=module._step(w, x, y, 0.08)
        used=step+1
        if not np.isfinite(loss):
            raise ValueError("Nonfinite loss")
        if step % 15 == 0 or step == 599:
            score=module.agree(w,xd,yd)
            if score > best_agree:
                best, best_agree, selected={k:v.copy() for k,v in w.items()},score,used
            if score >= 0.90:
                break
    if best is None:
        raise ValueError("No development checkpoint")
    # The test labels are used only here, after checkpoint selection.
    xt, yt=splits["test"]
    weights_path=output / ("receipt-agent-seed-"+str(seed)+".npz")
    np.savez(weights_path, **best)
    readback=np.load(weights_path, allow_pickle=False)
    readback_weights={k:readback[k] for k in readback.files}
    readback.close()
    expected=tensor_digest(best)
    if tensor_digest(readback_weights) != expected:
        raise ValueError("Weight archive readback mismatch")
    policy_ok=all(module.decide(features,best)["decision"] == module.LABELS[module.rule_check(features)] for features in xt)
    if not policy_ok:
        raise ValueError("Advisory model changed authoritative rule decision")
    return {"seed":seed,"steps_executed":used,"selected_step":selected,
            "split_evidence":evidence,"development_agreement":best_agree,
            "untouched_test":metrics(module.predict(best,xt),yt),
            "untrained_initialization_test":metrics(module.predict(initial,xt),yt),
            "tensor_sha256":expected,"weight_file":weights_path.name,
            "weight_file_sha256":hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "archive_readback":True,"deterministic_policy_retained":policy_ok,
            "promotion_authorized":False}

def tiny_counterfactual(module, seed):
    w, historical_style=module.train(seed=seed, steps=280, n=80)
    matched={"query":"resolve F4 handle", "handles":[{"id":"h.match","note":"F4 node"}]}
    missing={"query":matched["query"],"handles":[{"id":"h.unrelated","note":"unrelated theorem"}]}
    a,b=module.forward(w,matched),module.forward(w,missing)
    exact_equal=np.array_equal(a["p"],b["p"])
    correct=int(a["decision"] == module.NAVIGATE)+int(b["decision"] == module.ABSTAIN)
    return {"seed":seed,"steps":280,"historical_style_synthetic_metrics":historical_style,
            "counterfactual":{"same_query":True,"offered_handles_changed":True,
                              "probabilities_identical":exact_equal,
                              "matched_decision":int(a["decision"]),
                              "missing_decision":int(b["decision"]),
                              "expected_decisions":[1,0],"correct":correct,"n":2},
            "tensor_sha256":tensor_digest(w),
            "finding":"HANDLE_CONTEXT_NOT_IN_DECISION_HEAD",
            "promotion_authorized":False}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--synthetic-cpu",action="store_true",required=True)
    args=parser.parse_args(argv)
    receipt=load_source(args.source_directory,"receipt_agent.py")
    tiny=load_source(args.source_directory,"tiny_khipu.py")
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report={"schema":"szl.nano-synthetic-research/v1",
            "timestamp":datetime.now(timezone.utc).isoformat(),
            "source_repository":"szl-holdings/szl-khipu","source_revision":SOURCE_REVISION,
            "source_sha256":PINS,"runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python":platform.python_version(),"numpy":np.__version__,
            "seeds":[11,23,37],"label":"MEASURED_SYNTHETIC_LOCAL_ONLY",
            "data_source":"Programmatically generated synthetic fixtures, no estate corpus rows",
            "training_authorized_by_this_receipt":False,"promotion_authorized":False,
            "independent_review":False,
            "receipt_agent":[receipt_run(receipt,s,args.output) for s in (11,23,37)],
            "tiny_khipu":[tiny_counterfactual(tiny,s) for s in (11,23,37)],
            "limitations":["Synthetic-generator generalization only; same generator family in all splits",
                           "New experiment, not reconstruction of historical unsigned training receipts",
                           "Counterfactual is a public diagnostic, not a sealed promotion benchmark",
                           "No policy authority is delegated to either learned model"]}
    report["elapsed_seconds"]=time.monotonic()-started
    (args.output/"experiment.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"elapsed_seconds":report["elapsed_seconds"],
                      "receipt_test":[r["untouched_test"]["agreement"] for r in report["receipt_agent"]],
                      "tiny_counterfactual":[r["counterfactual"]["correct"] for r in report["tiny_khipu"]],
                      "promotion_authorized":False}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

