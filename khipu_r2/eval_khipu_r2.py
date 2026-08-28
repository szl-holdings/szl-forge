#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""KHIPU-R2 eval stamp. Live Hub abstain is MEASURED 3/6 (not a pass).

Hub eval_measured.json: grounding 5/5, plan 11/11, abstain 3/6.
This-SKU evals are not-this-run. This-kit jobs UNKNOWN.
publication_eligible false. Signed SZL-Khipu-1.5B abstain stays MEASURED 2/6
on that card only. Does not overwrite signed 1.5B. Lab stays signed Khipu GGUF.

CHAWPI extra lock: Hub receipt ddf6c50 publication_eligible false is the
public claim. stale profile key dropped. Launcher still no --run-job.
r=32 α=64 this SKU. No Hub PUT. Do not merge #64.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KHIPU = ROOT / "khipu"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HUB = "SZLHOLDINGS/KHIPU-R2"
FORBIDDEN_HUB = "SZLHOLDINGS/SZL-Khipu-1.5B"
SIGNED_RECEIPT = KHIPU / "eval_receipt.signed.json"
HUB_JOB_ID = "6a91bf11984507d9db4ea104"
HUB_JOB_STATUS = "COMPLETED"
HUB_ADAPTER_STATUS = "AVAILABLE"
HUB_ADAPTER_SIZE = "147.8MB"
HUB_ABSTAIN_CORRECT = 3
HUB_ABSTAIN_TOTAL = 6
HUB_ABSTAIN_LABEL = "MEASURED"
HUB_GROUNDING_CORRECT = 5
HUB_GROUNDING_TOTAL = 5
HUB_PLAN_VALID = 11
HUB_PLAN_TOTAL = 11
HUB_RECEIPT_COMMIT = "ddf6c50d8baa9f818b9f478086e7b5919eb773cf"
CHAWPI = "hub-receipt-ddf6c50-publication-eligible-false"
SIGNED_ABSTAIN_CORRECT = 2
SIGNED_ABSTAIN_TOTAL = 6


def signed_abstain() -> tuple[int, int]:
    payload = json.loads(SIGNED_RECEIPT.read_text(encoding="utf-8"))["payload"]
    correct = int(payload["abstainCorrect"])
    total = int(payload["abstainTotal"])
    if (correct, total) != (SIGNED_ABSTAIN_CORRECT, SIGNED_ABSTAIN_TOTAL):
        raise SystemExit(
            f"[khipu-r2-eval] signed abstain is {correct}/{total}, "
            f"expected {SIGNED_ABSTAIN_CORRECT}/{SIGNED_ABSTAIN_TOTAL}"
        )
    return correct, total


def main() -> int:
    signed_correct, signed_total = signed_abstain()
    report = {
        "kind": "szl-khipu-r2-eval-report",
        "artifact": HUB,
        "sku": "KHIPU-R2",
        "separate_sku": True,
        "does_not_overwrite": FORBIDDEN_HUB,
        "base_model": BASE_MODEL,
        "lab": "signed Khipu GGUF",
        "inference_lab_pin": False,
        "hub_job_id": HUB_JOB_ID,
        "hub_job_status": HUB_JOB_STATUS,
        "hub_adapter": HUB_ADAPTER_STATUS,
        "hub_adapter_size": HUB_ADAPTER_SIZE,
        "hub_abstain": f"{HUB_ABSTAIN_CORRECT}/{HUB_ABSTAIN_TOTAL}",
        "hub_abstain_correct": HUB_ABSTAIN_CORRECT,
        "hub_abstain_total": HUB_ABSTAIN_TOTAL,
        "hub_abstain_label": HUB_ABSTAIN_LABEL,
        "hub_abstain_pass": False,
        "hub_grounding": f"{HUB_GROUNDING_CORRECT}/{HUB_GROUNDING_TOTAL}",
        "hub_grounding_correct": HUB_GROUNDING_CORRECT,
        "hub_grounding_total": HUB_GROUNDING_TOTAL,
        "hub_plan": f"{HUB_PLAN_VALID}/{HUB_PLAN_TOTAL}",
        "hub_plan_valid": HUB_PLAN_VALID,
        "hub_plan_total": HUB_PLAN_TOTAL,
        "hub_receipt_commit": HUB_RECEIPT_COMMIT,
        "chawpi": CHAWPI,
        "signed_original_repo": FORBIDDEN_HUB,
        "signed_original_abstain": f"{signed_correct}/{signed_total}",
        "signed_original_abstain_correct": signed_correct,
        "signed_original_abstain_total": signed_total,
        "signed_original_label": "MEASURED",
        "jobs": "UNKNOWN",
        "jobs_scope": "this-kit",
        "evals": "not-this-run",
        "evals_scope": "this-sku",
        "this_sku_k_over_n": "not-this-run",
        "publication_eligible": False,
        "autonomy_eligible": False,
        "hub_put": False,
        "claim_boundary": (
            f"Live Hub KHIPU-R2 eval_measured.json is {HUB_ABSTAIN_LABEL} "
            f"abstain {HUB_ABSTAIN_CORRECT}/{HUB_ABSTAIN_TOTAL} (not a pass), "
            f"grounding {HUB_GROUNDING_CORRECT}/{HUB_GROUNDING_TOTAL}, "
            f"plan {HUB_PLAN_VALID}/{HUB_PLAN_TOTAL}. "
            f"Signed SZL-Khipu-1.5B abstain stays MEASURED "
            f"{signed_correct}/{signed_total} on that card only. "
            "This-SKU evals not-this-run. This-kit jobs UNKNOWN. "
            "Does not overwrite signed SZL-Khipu-1.5B. publication_eligible false. "
            "CHAWPI extra lock: Hub receipt ddf6c50 publication_eligible false "
            "is the public claim. stale profile key dropped. "
            "Launcher still no --run-job. r=32 α=64 this SKU. "
            "Do not merge #64. Lab stays signed Khipu GGUF."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = HERE / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[khipu-r2-eval] hub_abstain={HUB_ABSTAIN_LABEL} "
        f"{HUB_ABSTAIN_CORRECT}/{HUB_ABSTAIN_TOTAL} (not a pass) "
        f"grounding={HUB_GROUNDING_CORRECT}/{HUB_GROUNDING_TOTAL} "
        f"plan={HUB_PLAN_VALID}/{HUB_PLAN_TOTAL}"
    )
    print(
        f"[khipu-r2-eval] signed original abstain MEASURED "
        f"{signed_correct}/{signed_total} (signed 1.5B card only)"
    )
    print(
        "[khipu-r2-eval] this-sku evals=not-this-run "
        "this-kit jobs=UNKNOWN publication_eligible=false"
    )
    print(
        f"[khipu-r2-eval] hub_job={HUB_JOB_ID} {HUB_JOB_STATUS} "
        f"adapter={HUB_ADAPTER_STATUS} ({HUB_ADAPTER_SIZE})"
    )
    print(
        f"[khipu-r2-eval] CHAWPI hub_receipt={HUB_RECEIPT_COMMIT} "
        "publication_eligible=false is the public claim"
    )
    print(f"[khipu-r2-eval] base_model={BASE_MODEL}")
    print(f"[khipu-r2-eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
