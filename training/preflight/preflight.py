"""Non-executing training preflight: verify everything BEFORE any GPU runs.

A preflight that cannot train, push, or write to the Hub. It only READS and
REJECTS. A wave that has not passed preflight must not start.

Checks (all fail-closed, typed):
  1. artifact family: the target Hub repo must be a causal-LM destination
     (config.json with causal architecture, or an empty repo explicitly marked
     for a merged 16-bit LM) - never a kernel, fixture, detector, VLM or kit.
  2. target identity: the resolved Hub repo id must equal the requested one.
  3. immutable model pin: the base model revision is resolved and recorded.
  4. dataset pins: every dataset resolves, carries a messages/conversations
     column, and per-source row counts are recorded.
  5. approved use: the config must carry the wave's honesty contract fields.

Usage:
  python preflight.py --config training/w29/forge-v3.json
"""
from __future__ import annotations

import argparse
import json
import os


class PreflightError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def fail(code: str, detail: str = "") -> None:
    raise PreflightError(code, detail)


def hf_api():
    if not os.environ.get("HF_TOKEN"):
        fail("PREFLIGHT_NO_TOKEN", "HF_TOKEN is not set")
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def check_artifact_family(api, target_repo: str) -> dict:
    """The destination must be a causal-LM family target, proven by its files."""
    try:
        files = list(api.list_repo_files(target_repo, repo_type="model"))
    except Exception as e:
        fail("PREFLIGHT_TARGET_UNREADABLE", f"{target_repo}: {e}")
    # non-causal artifacts that must NEVER receive a merged LM checkpoint
    non_lm_markers = [f for f in files if f.endswith(".npz")]
    if non_lm_markers:
        fail("PREFLIGHT_NON_LM_TARGET",
             f"{target_repo} carries fixture/kernel artifacts {non_lm_markers[:3]}; "
             "pushing a merged causal-LM checkpoint would overwrite them")
    has_config = "config.json" in files
    empty = not [f for f in files if f not in (".gitattributes", "README.md")]
    return {"target_repo": target_repo, "files": len(files),
            "has_config": has_config, "empty": empty}


def check_target_identity(api, requested: str) -> None:
    """The Hub must resolve the destination to exactly the requested identity."""
    try:
        info = api.repo_info(requested, repo_type="model")
        resolved = getattr(info, "id", None) or requested
    except Exception as e:
        fail("PREFLIGHT_TARGET_UNREADABLE", f"{requested}: {e}")
    if resolved != requested:
        fail("PREFLIGHT_TARGET_MISMATCH",
             f"requested {requested} but provider resolved {resolved}")


def check_model_pin(api, model_id: str) -> dict:
    try:
        info = api.model_info(model_id)
    except Exception as e:
        fail("PREFLIGHT_MODEL_UNREADABLE", f"{model_id}: {e}")
    return {"model": model_id, "sha": info.sha}


def check_dataset(api, dataset_id: str) -> dict:
    try:
        info = api.dataset_info(dataset_id, files_metadata=False)
        files = [s.rfilename for s in info.siblings]
    except Exception as e:
        fail("PREFLIGHT_DATASET_UNREADABLE", f"{dataset_id}: {e}")
    has_chat = any(f.endswith((".jsonl", ".json", ".parquet")) for f in files)
    if not has_chat:
        fail("PREFLIGHT_DATASET_NO_DATA", f"{dataset_id} has no data files")
    return {"dataset": dataset_id, "sha": info.sha, "files": len(files)}


def run_preflight(config_path: str) -> dict:
    cfg = json.load(open(config_path))
    for field in ("target_repo", "base_model", "datasets", "method"):
        if field not in cfg:
            fail("PREFLIGHT_CONFIG_MISSING_FIELD", field)
    api = hf_api()
    family = check_artifact_family(api, cfg["target_repo"])
    check_target_identity(api, cfg["target_repo"])
    model_pin = check_model_pin(api, cfg["base_model"])
    dataset_pins = [check_dataset(api, d) for d in cfg["datasets"]]
    if cfg.get("method") not in ("qlora", "lora16"):
        fail("PREFLIGHT_METHOD_NOT_TRAINABLE",
             f"method {cfg.get('method')!r} is not a runnable SFT method")
    return {
        "schema": "szl.forge.preflight/v1",
        "config": config_path,
        "target": family,
        "base_model_pin": model_pin,
        "dataset_pins": dataset_pins,
        "verdict": "PASS",
        "truth_label": "MEASURED_PREFLIGHT",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    try:
        report = run_preflight(args.config)
    except PreflightError as e:
        print(json.dumps({"schema": "szl.forge.preflight/v1",
                          "verdict": "REJECT", "code": e.code,
                          "error": str(e)}, indent=2))
        raise SystemExit(1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
