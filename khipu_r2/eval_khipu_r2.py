#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""KHIPU-R2 eval. Honest about signed 2/6 abstain. publication_eligible false.

This checkout does not re-run held-out generate. Signed SZL-Khipu-1.5B
abstain remains MEASURED 2/6 (blocker). Do not invent 6/6. Jobs UNKNOWN.
Separate SKU — not an overwrite of signed 1.5B.
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


def signed_abstain() -> tuple[int, int, int, int]:
    payload = json.loads(SIGNED_RECEIPT.read_text(encoding="utf-8"))["payload"]
    return (
        int(payload["abstainCorrect"]),
        int(payload["abstainTotal"]),
        int(payload["groundingCorrect"]),
        int(payload["groundingTotal"]),
    )


def main() -> int:
    abstain_correct, abstain_total, grounding_correct, grounding_total = (
        signed_abstain()
    )
    if (abstain_correct, abstain_total) != (2, 6):
        raise SystemExit(
            f"[khipu-r2-eval] signed abstain is {abstain_correct}/{abstain_total}, "
            "expected 2/6"
        )
    report = {
        "kind": "szl-khipu-r2-eval-report",
        "artifact": HUB,
        "sku": "KHIPU-R2",
        "separate_sku": True,
        "does_not_overwrite": FORBIDDEN_HUB,
        "base_model": BASE_MODEL,
        "card_status": "ROADMAP",
        "jobs": "UNKNOWN",
        "evals": "not-this-run",
        "this_sku_k_over_n": "not-this-run",
        "signed_original_repo": FORBIDDEN_HUB,
        "signed_original_abstain": f"{abstain_correct}/{abstain_total}",
        "signed_original_abstain_correct": abstain_correct,
        "signed_original_abstain_total": abstain_total,
        "signed_original_grounding": f"{grounding_correct}/{grounding_total}",
        "signed_original_label": "MEASURED",
        "publication_eligible": False,
        "autonomy_eligible": False,
        "hub_put": False,
        "claim_boundary": (
            "Signed SZL-Khipu-1.5B held-out abstain is MEASURED 2/6 (blocker). "
            "KHIPU-R2 is a separate ROADMAP SKU. This checkout did not generate "
            "held-out rows and does not invent a passing abstain score. "
            "publication_eligible false."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = HERE / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[khipu-r2-eval] signed original abstain MEASURED "
        f"{abstain_correct}/{abstain_total} (blocker)"
    )
    print(
        "[khipu-r2-eval] this-sku evals=not-this-run "
        "publication_eligible=false jobs=UNKNOWN"
    )
    print(f"[khipu-r2-eval] base_model={BASE_MODEL}")
    print(f"[khipu-r2-eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
