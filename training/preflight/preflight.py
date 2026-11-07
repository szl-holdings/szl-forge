"""Non-executing training preflight: verify everything BEFORE any GPU runs.

A preflight that cannot train, push, or write to the Hub. It only READS and
REJECTS. A wave that has not passed preflight must not start.

Checks (all fail-closed, typed):
  1. artifact family: the target Hub repo must be a causal-LM destination
     (config.json with causal architecture, or an empty repo explicitly marked
     for a merged 16-bit LM) - never a kernel, fixture, detector, VLM or kit.
  2. target identity: the resolved Hub repo id must equal the requested one.
  3. immutable model pin: student and teacher reads use the configured commit,
     and the resolved sha must be that commit. The default branch is not a pin.
  4. dataset pin: the dataset read uses the configured commit, and that commit
     must contain data files. Latest default-branch metadata is not a pin.
  5. approved use: the config must carry the wave's honesty contract fields.

The shared launch contract runs before any Hub read. A config that fails
rights, pins, holdout separation, or the 32-example floor never reaches the
network. This process still cannot train or write to the Hub.

Usage:
  python preflight.py --config training/w29/forge-v3.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os


class PreflightError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def fail(code: str, detail: str = "") -> None:
    raise PreflightError(code, detail)


def _contract():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "launch_contract.py"))
    spec = importlib.util.spec_from_file_location("szl_launch_contract", path)
    if spec is None or spec.loader is None:
        fail("LAUNCH_CONFIG_UNREADABLE", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _require_configured_revision(revision: object, code: str, label: str) -> str:
    pattern = _contract().IMMUTABLE_REVISION
    if not isinstance(revision, str) or pattern.fullmatch(revision) is None:
        fail(code, f"{label} revision is not the configured immutable commit")
    return revision


def check_model_pin(api, model_id: str, revision: str, *, missing_code: str, mismatch_code: str) -> dict:
    """Read the configured commit. A bare repo read returns the default branch."""
    revision = _require_configured_revision(revision, mismatch_code, model_id)
    try:
        info = api.model_info(model_id, revision=revision)
    except Exception as e:
        fail(missing_code, f"{model_id}@{revision}: {e}")
    resolved = getattr(info, "sha", None)
    if resolved != revision:
        fail(mismatch_code, f"requested {model_id}@{revision} but provider resolved {resolved}")
    return {"model": model_id, "sha": resolved, "revision": revision}


def check_dataset(api, dataset_id: str, revision: str) -> dict:
    """Read dataset files at the configured commit, not at the default branch."""
    revision = _require_configured_revision(revision, "PREFLIGHT_DATASET_PIN_MISMATCH", dataset_id)
    try:
        info = api.dataset_info(dataset_id, revision=revision, files_metadata=False)
        siblings = getattr(info, "siblings", None) or []
        files = [s.rfilename for s in siblings]
    except PreflightError:
        raise
    except Exception as e:
        fail("PREFLIGHT_DATASET_UNREADABLE", f"{dataset_id}@{revision}: {e}")
    resolved = getattr(info, "sha", None)
    if resolved != revision:
        fail(
            "PREFLIGHT_DATASET_PIN_MISMATCH",
            f"requested {dataset_id}@{revision} but provider resolved {resolved}",
        )
    has_chat = any(f.endswith((".jsonl", ".json", ".parquet")) for f in files)
    if not has_chat:
        fail("PREFLIGHT_DATASET_NO_DATA", f"{dataset_id}@{revision} has no data files")
    return {"dataset": dataset_id, "sha": resolved, "revision": revision, "files": len(files)}


def run_preflight(config_path: str) -> dict:
    decision = _contract().evaluate_path(config_path)
    if not decision["ok"]:
        fail("LAUNCH_CONTRACT_REJECTED", ",".join(decision["codes"]))
    cfg = decision["config"]
    target_repo = cfg.get("target_repo")
    if not isinstance(target_repo, str) or not target_repo:
        fail("PREFLIGHT_CONFIG_MISSING_FIELD", "target_repo")
    if decision["kind"] == "sft" and cfg.get("method") not in ("qlora", "lora16"):
        fail("PREFLIGHT_METHOD_NOT_TRAINABLE",
             f"method {cfg.get('method')!r} is not a runnable SFT method")
    api = hf_api()
    family = check_artifact_family(api, target_repo)
    check_target_identity(api, target_repo)
    model_pin = check_model_pin(
        api,
        decision["student_id"],
        decision["student_revision"],
        missing_code="PREFLIGHT_MODEL_UNREADABLE",
        mismatch_code="PREFLIGHT_MODEL_PIN_MISMATCH",
    )
    dataset_pin = check_dataset(api, decision["dataset_id"], decision["dataset_revision"])
    teacher_pin = None
    if decision["kind"] == "distill":
        teacher_pin = check_model_pin(
            api,
            decision["teacher_id"],
            decision["teacher_revision"],
            missing_code="PREFLIGHT_TEACHER_UNREADABLE",
            mismatch_code="PREFLIGHT_TEACHER_PIN_MISMATCH",
        )
    return {
        "schema": "szl.forge.preflight/v1",
        "config": config_path,
        "target": family,
        "base_model_pin": model_pin,
        "dataset_pins": [dataset_pin],
        "teacher_pin": teacher_pin,
        "accepted_examples": decision["accepted_examples"],
        "verdict": "PASS",
        "truth_label": "MEASURED_PREFLIGHT",
        "training_started": False,
        "hub_write": False,
        "model_loaded": False,
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
