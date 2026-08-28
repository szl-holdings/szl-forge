#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 Lutar, Stephen P. - SZL Holdings
"""A11OY-MINI eval — inherit none-this-run. No fabricated k/n.

Without a local .gguf this is honest ROADMAP. A file on disk is not an eval.
Parent evals stay none-this-run (live SZLHOLDINGS/chaski). Not 5/5. Not MEASURED.
Quality ROADMAP. Bytes MEASURED only if a local GGUF hash is written.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert_a11oy_mini_gguf as convert  # noqa: E402


def local_gguf() -> Path | None:
    for name in (convert.Q4_NAME, convert.F16_NAME):
        path = HERE / name
        if path.is_file():
            return path
    found = sorted(HERE.glob("*.gguf"))
    return found[0] if found else None


def report_payload() -> dict:
    gguf = local_gguf()
    bytes_info = convert.gguf_bytes_label(gguf)
    return {
        "kind": "szl-a11oy-mini-eval-report",
        "artifact": convert.SKU,
        "parent": convert.PARENT,
        "forbidden_parent": convert.FORBIDDEN_PARENT,
        "base_model": convert.CANONICAL_BASE,
        "seed": convert.SEED,
        "doctrine": convert.DOCTRINE,
        "evals": "none-this-run",
        "parent_evals": "none-this-run",
        "quality": "ROADMAP",
        "label": "ROADMAP",
        "publication_eligible": False,
        "autonomy_eligible": False,
        "gguf": "LOCAL" if gguf else "UNAVAILABLE",
        "gguf_exists": bool(gguf),
        "bytes": bytes_info,
        "bytes_measured": bytes_info["label"] == "MEASURED",
        "khipu_lab_pin": False,
        "inference_lab_pin": False,
        "tok_s_claim": False,
        "hub_put": False,
        "third_llm": False,
        "new_train": False,
        "base_model_relation_quantized": False,
        "claim_boundary": (
            "Evals none-this-run. Inherited from live SZLHOLDINGS/chaski. "
            "No JSON/refusal gate ran. Do not claim 5/5 or 6/6. "
            "A local GGUF hash is bytes MEASURED, not an eval. "
            "Quality stays ROADMAP until a gate runs. Not 5050. "
            "Lab stays Khipu. No tok/s. No Hub PUT."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    report = report_payload()
    path = HERE / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "[a11oy-mini-eval] evals=none-this-run quality=ROADMAP "
        f"gguf={report['gguf']} parent={convert.PARENT}"
    )
    print(
        "[a11oy-mini-eval] publication_eligible=false "
        f"bytes_measured={report['bytes_measured']} lab=Khipu"
    )
    print(f"[a11oy-mini-eval] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
