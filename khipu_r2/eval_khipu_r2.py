#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""KHIPU-R2 eval stamp. Live Hub abstain is MEASURED 3/6 (not a pass).

This-SKU evals are not-this-run. This-kit jobs UNKNOWN. publication_eligible
false. Does not overwrite signed SZL-Khipu-1.5B. Lab stays signed Khipu GGUF.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HUB = "SZLHOLDINGS/KHIPU-R2"
FORBIDDEN_HUB = "SZLHOLDINGS/SZL-Khipu-1.5B"
HUB_JOB_ID = "6a91bf11984507d9db4ea104"
HUB_JOB_STATUS = "COMPLETED"
HUB_ADAPTER_STATUS = "AVAILABLE"
HUB_ADAPTER_SIZE = "147.8MB"
HUB_ABSTAIN_CORRECT = 3
HUB_ABSTAIN_TOTAL = 6
HUB_ABSTAIN_LABEL = "MEASURED"


def main() -> int:
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
        "jobs": "UNKNOWN",
        "jobs_scope": "this-kit",
        "evals": "not-this-run",
        "evals_scope": "this-sku",
        "this_sku_k_over_n": "not-this-run",
        "publication_eligible": False,
        "autonomy_eligible": False,
        "hub_put": False,
        "claim_boundary": (
            f"Live Hub KHIPU-R2 abstain is {HUB_ABSTAIN_LABEL} "
            f"{HUB_ABSTAIN_CORRECT}/{HUB_ABSTAIN_TOTAL} (not a pass). "
            "This-SKU evals not-this-run. This-kit jobs UNKNOWN. "
            "Does not overwrite signed SZL-Khipu-1.5B. publication_eligible false. "
            "Lab stays signed Khipu GGUF."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = HERE / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[khipu-r2-eval] hub_abstain={HUB_ABSTAIN_LABEL} "
        f"{HUB_ABSTAIN_CORRECT}/{HUB_ABSTAIN_TOTAL} (not a pass)"
    )
    print(
        "[khipu-r2-eval] this-sku evals=not-this-run "
        "this-kit jobs=UNKNOWN publication_eligible=false"
    )
    print(
        f"[khipu-r2-eval] hub_job={HUB_JOB_ID} {HUB_JOB_STATUS} "
        f"adapter={HUB_ADAPTER_STATUS} ({HUB_ADAPTER_SIZE})"
    )
    print(f"[khipu-r2-eval] base_model={BASE_MODEL}")
    print(f"[khipu-r2-eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
