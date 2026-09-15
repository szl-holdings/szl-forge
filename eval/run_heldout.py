#!/usr/bin/env python3
"""Run the held-out generation eval (Gate 2).

FAIL-CLOSED by design: the eval set must exist, prompt execution must complete,
and remote Hugging Face models can be pinned to an exact immutable revision.
An evaluation receipt records the model reference and revision that were used.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    print("::error::run_heldout: pyyaml is required (pip install pyyaml)", file=sys.stderr)
    raise SystemExit(1)


def die(msg: str) -> "SystemExit":
    print(f"::error::run_heldout: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_config(path: str) -> dict:
    cp = pathlib.Path(path)
    if not cp.is_file():
        die(f"held-out eval set missing: {cp} (fail-closed)")
    try:
        cfg = yaml.safe_load(cp.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        die(f"eval config not valid YAML: {exc}")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("prompts"), list) or not cfg["prompts"]:
        die("eval config has no non-empty 'prompts' list")
    for prompt in cfg["prompts"]:
        if not (
            isinstance(prompt, dict)
            and prompt.get("id")
            and prompt.get("prompt")
            and prompt.get("expected")
        ):
            die(f"malformed prompt entry (need id/prompt/expected): {prompt!r}")
    return cfg


def matches(output: str, prompt: dict) -> bool:
    mode = prompt.get("matching", "contains")
    out = output or ""
    for expected in prompt["expected"]:
        if mode == "exact" and out.strip() == str(expected).strip():
            return True
        if mode == "contains" and str(expected).lower() in out.lower():
            return True
    return False


def _stage_merged_model(model_ref, revision):
    # Stage a published merged full model at an exact revision.
    # Repos may ship model.safetensors (merged) alongside a preserved adapter.
    # transformers auto-detects adapter_config.json and rebuilds base+adapter
    # from base_model_name_or_path instead of loading the published merged
    # weights; that path depends on a mutable third-party base and can mask a
    # defective merge. Staging the merged files gates the published bytes.
    import os
    import shutil
    import tempfile

    if not ("/" in model_ref) or os.path.isdir(model_ref):
        return None
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        return None
    try:
        hf_hub_download(repo_id=model_ref, filename="model.safetensors", revision=revision)
    except EntryNotFoundError:
        return None
    staging = tempfile.mkdtemp(prefix="szl_gate_merged_")
    for filename in (
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        try:
            fetched = hf_hub_download(repo_id=model_ref, filename=filename, revision=revision)
        except EntryNotFoundError:
            continue
        real = os.path.realpath(fetched)
        dst = os.path.join(staging, filename)
        try:
            os.link(real, dst)
        except OSError:
            shutil.copyfile(real, dst)
    return staging


def run_transformers_cpu(model_ref: str, prompts: list, revision: str | None = None) -> list[str]:
    """Generate with a local/Hub model; exact Hub revisions are forwarded."""
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    staging = _stage_merged_model(model_ref, revision)
    if staging is not None:
        tokenizer = AutoTokenizer.from_pretrained(staging)
        model = AutoModelForCausalLM.from_pretrained(staging)
    else:
        kwargs = {"revision": revision} if revision else {}
        tokenizer = AutoTokenizer.from_pretrained(model_ref, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
    outputs: list[str] = []
    for prompt in prompts:
        ids = tokenizer(prompt["prompt"], return_tensors="pt")
        generated = model.generate(**ids, max_new_tokens=64, do_sample=False)
        outputs.append(
            tokenizer.decode(
                generated[0][ids["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
        )
    return outputs


def emit(out_path: str, doc: dict) -> None:
    output = pathlib.Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Local model dir or HF id.")
    parser.add_argument(
        "--revision",
        default=None,
        help="Exact Hub revision for remote --model. Required by callers that claim exact published bytes.",
    )
    parser.add_argument("--config", default="eval/heldout_generate.yaml")
    parser.add_argument("--results", default=None, help="Optional pre-computed outputs JSON.")
    parser.add_argument("--out", default="release/heldout-eval.json")
    parser.add_argument(
        "--assert-min-pass-rate",
        type=float,
        default=None,
        help="If set, heldout_passed=pass_rate>=this; else heldout_passed stays false.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompts = cfg["prompts"]

    outputs: dict[str, str] = {}
    mode: str
    if args.results:
        result_path = pathlib.Path(args.results)
        if not result_path.is_file():
            die(f"--results file not found: {result_path}")
        data = json.loads(result_path.read_text(encoding="utf-8"))
        rows = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            die("--results must be a list of {id, output} or {'results': [...]}")
        outputs = {
            str(row["id"]): str(row.get("output", ""))
            for row in rows
            if isinstance(row, dict) and "id" in row
        }
        missing = [prompt["id"] for prompt in prompts if prompt["id"] not in outputs]
        if missing:
            die(f"results file missing outputs for prompt ids: {missing}")
        mode = "results-file"
    else:
        try:
            generated = run_transformers_cpu(args.model, prompts, args.revision)
            outputs = {prompt["id"]: output for prompt, output in zip(prompts, generated)}
            mode = "transformers-cpu"
        except Exception as exc:
            die(
                "cannot execute held-out prompts (no --results; transformers run failed: "
                f"{exc}). Fail-closed: no eval, no publication."
            )

    results: list[dict] = []
    n_passed = 0
    for prompt in prompts:
        passed = matches(outputs[prompt["id"]], prompt)
        n_passed += int(passed)
        results.append(
            {
                "id": prompt["id"],
                "passed": bool(passed),
                "output_head": outputs[prompt["id"]][:160],
            }
        )
    n_total = len(prompts)
    pass_rate = n_passed / n_total if n_total else 0.0
    heldout_passed = bool(
        args.assert_min_pass_rate is not None and pass_rate >= args.assert_min_pass_rate
    )

    doc = {
        "model": args.model,
        "model_revision": args.revision,
        "config": args.config,
        "mode": mode,
        "n_total": n_total,
        "n_passed": n_passed,
        "pass_rate": round(pass_rate, 6),
        "heldout_passed": heldout_passed,
        "refusal_no_regression": False,
        "results": results,
    }
    emit(args.out, doc)
    print(f"held-out pass rate: {pass_rate:.4f} ({n_passed}/{n_total}), mode={mode}")
    print(f"heldout_passed={heldout_passed} (asserted threshold: {args.assert_min_pass_rate})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
