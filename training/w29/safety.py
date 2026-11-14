# SPDX-License-Identifier: Apache-2.0
"""Fail-closed run policy for Wave 29.

``run_w29.sh`` asks this module before it starts ``train_wave.py``. A config
may stay in the catalog as documentation. It is trained only when its Hub
target is on the causal-LM allowlist and its method is ``qlora``.

Protected targets are live non-causal-LM artifacts. Pushing a merged 16-bit
causal-LM checkpoint into them would overwrite the artifact. A ``qlora`` tag
on a protected or otherwise non-allowlisted repo refuses the whole wave
before any trainer process starts. Skipping only when ``method != qlora`` is
not enough: that is how the numpy fixtures stayed runnable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Live Hub artifacts that are not causal-LM SFT targets. Reasons are the
# trainer they actually need, or the artifact a merged checkpoint would replace.
PROTECTED_REPOS = {
    "SZLHOLDINGS/YARQA-ATTN": (
        "attention kernel (yarqa_attn.npz and the kernels pointer), not a causal-LM checkpoint"
    ),
    "SZLHOLDINGS/chakana": "embedding kit; needs a contrastive trainer, not SFT",
    "SZLHOLDINGS/tinku": "reranker kit; needs a contrastive trainer, not SFT",
    "SZLHOLDINGS/MiniEmbed-Nano": "numpy embedding fixture (mini_embed.npz)",
    "SZLHOLDINGS/Moons-Nano": "numpy MLP fixture (moons.npz), not a causal LM",
    "SZLHOLDINGS/ReceiptAgent-Nano": "numpy MLP fixture (receipt_agent.npz), not a causal LM",
    "SZLHOLDINGS/TinyKhipu-Nano": "numpy fixture (tiny_khipu.npz), not a causal LM",
    "SZLHOLDINGS/waman": "RF-DETR detector; needs a detection trainer",
    "SZLHOLDINGS/KILLINCHU-EYE": "RF-DETR detector alias of waman, not a second model",
    "SZLHOLDINGS/qantu": "VLM; needs a vision-language trainer",
    "SZLHOLDINGS/oac-clinical-transport-health-v1": "logistic health model, not a causal LM",
    "SZLHOLDINGS/oac-ops-health-v2": "logistic health model, not a causal LM",
    "SZLHOLDINGS/oac-system-health-v1": "logistic health model, not a causal LM",
    "SZLHOLDINGS/szl-triage-linear-development-v1": "linear text classifier, not a causal LM",
    "SZLHOLDINGS/szl-triage-retrain": "linear text classifier, not a causal LM",
}

# Causal-LM candidate that must not be plain QLoRA. The public repo is a
# report (README + training-eval-report.json) with no weights. Two DPO rounds
# failed the recorded 3/6 confabulation floor; the next round is rejection
# sampling. An SFT push would overwrite that report.
WITHHELD_SFT_REPOS = {
    "SZLHOLDINGS/khipu-abstain-r3-candidate": (
        "report-only abstention candidate; rejection sampling then SFT, not plain QLoRA "
        "(two DPO rounds failed the 3/6 confabulation floor)"
    ),
}

# Candidate repo ids this wave may SFT. Not the live WILLAY checkpoint and
# not the numpy nano fixtures. Hub file listings for these three ids were not
# publicly readable from this checkout; they are allowlisted because they are
# the remaining causal-LM candidate ids, and every other target is refused.
RUNNABLE_REPOS = {
    "SZLHOLDINGS/willay-r1-candidate": "willay-r1",
    "SZLHOLDINGS/willay-r2-candidate": "willay-r2",
    "SZLHOLDINGS/forge-v3": "forge-v3",
}


def classify(config: object) -> tuple[str, str]:
    """Return ``(RUN|SKIP|REFUSE, reason)`` for one wave config."""
    if not isinstance(config, dict):
        return "REFUSE", "config is not a JSON object"
    target = config.get("target_repo")
    method = config.get("method")
    if not isinstance(target, str) or not target:
        return "REFUSE", "missing target_repo"
    if not isinstance(method, str) or not method:
        return "REFUSE", "missing method"
    if target in PROTECTED_REPOS:
        reason = PROTECTED_REPOS[target]
        if method == "qlora":
            return "REFUSE", f"{target} is a protected non-causal-LM repo ({reason}) but method is qlora"
        return "SKIP", reason
    if target in WITHHELD_SFT_REPOS:
        reason = WITHHELD_SFT_REPOS[target]
        if method == "qlora":
            return "REFUSE", f"{target} is withheld from plain SFT ({reason}) but method is qlora"
        return "SKIP", reason
    if method != "qlora":
        return "SKIP", f"method={method}; not on the causal-LM allowlist"
    if target not in RUNNABLE_REPOS:
        return "REFUSE", f"{target} is method=qlora but not on the causal-LM allowlist"
    expected_name = RUNNABLE_REPOS[target]
    name = config.get("name")
    if name != expected_name:
        return "REFUSE", f"{target} allowlist name is {expected_name}, config name is {name!r}"
    for field in ("base_model", "epochs", "seqlen"):
        if config.get(field) in (None, ""):
            return "REFUSE", f"runnable config {expected_name} is missing {field}"
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets or not all(isinstance(item, str) and item for item in datasets):
        return "REFUSE", f"runnable config {expected_name} has no dataset list"
    return "RUN", "causal-LM SFT candidate on the allowlist"


def load_configs(directory: Path) -> list[tuple[Path, object]]:
    paths = sorted(directory.glob("*.json"))
    loaded: list[tuple[Path, object]] = []
    for path in paths:
        try:
            loaded.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except json.JSONDecodeError as exc:
            loaded.append((path, exc))
    return loaded


def preflight(directory: Path) -> int:
    """Refuse the wave when any config would train a non-allowlisted repo."""
    rows = load_configs(directory)
    if not rows:
        print(f"REFUSE {directory}: no wave configs", file=sys.stderr)
        return 1
    failed = False
    seen_targets: dict[str, str] = {}
    for path, payload in rows:
        if isinstance(payload, json.JSONDecodeError):
            print(f"REFUSE {path.name}: invalid JSON ({payload})", file=sys.stderr)
            failed = True
            continue
        action, reason = classify(payload)
        if action == "REFUSE":
            print(f"REFUSE {path.name}: {reason}", file=sys.stderr)
            failed = True
            continue
        if isinstance(payload, dict):
            target = payload.get("target_repo")
            if isinstance(target, str):
                previous = seen_targets.get(target)
                if previous is not None:
                    print(
                        f"REFUSE {path.name}: target_repo {target} is also used by {previous}",
                        file=sys.stderr,
                    )
                    failed = True
                else:
                    seen_targets[target] = path.name
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "preflight":
        return preflight(Path(argv[2]))
    if len(argv) == 3 and argv[1] == "decide":
        path = Path(argv[2])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"REFUSE\tinvalid JSON ({exc})")
            return 0
        action, reason = classify(payload)
        print(f"{action}\t{reason}")
        return 0
    print("usage: safety.py preflight DIR | safety.py decide CONFIG", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
