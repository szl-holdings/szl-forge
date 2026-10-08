"""Prepare, but never publish, the bounded ReceiptAgent v3 Hub-card correction.

The public Hub README and the Forge authoring card have different layouts. This
one-shot preparer withdraws two reviewed loader spans in the exact observed Hub
README and preserves every other byte. It has no credential or Hub write path.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CARD = "receiptagent-v3/card/README.md"
SOURCE_CARD_SHA256 = "d15037c8a084b9ea2e0b9f0c71324632f6e97cc177f93120e8485dbb8d8931ea"
HUB_REPO = "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3"
HUB_PARENT = "d674c37cae7127021ba36820ea7a6774cec06255"
HUB_README_SHA256 = "79fa68b40bcd8cdfff3c4c0a50a7f4c452bf62b080fa95080baffb02a9430555"
CANDIDATE_README_SHA256 = "21f74eebf9a1033054bfd6bc9306c3322ade2a134a3b88125a895e2f6006dd9c"
CANDIDATE_KIND = "LOADER_EXAMPLE_WITHDRAWAL_ONLY"
ADAPTER_SHA256 = "90b82ade0b2c84eca92e90cdeb56137d61d7da4b4863ff4d29fa75e9dc037870"
MAX_REMOTE_BYTES = 1_000_000


class CardSyncError(ValueError):
    """The candidate cannot be prepared without weakening its evidence bound."""


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def bind_source(root: Path, revision: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise CardSyncError("source revision must be an exact 40-character SHA")
    local = (root / SOURCE_CARD).read_bytes()
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{revision}:{SOURCE_CARD}"],
        capture_output=True,
        check=False,
    )
    if result.returncode or digest(result.stdout) != SOURCE_CARD_SHA256:
        raise CardSyncError("reviewed immutable Forge card bytes changed")
    # Git's LF blob may be checked out as CRLF on Windows. Only that EOL
    # conversion is tolerated; the immutable blob, not the checkout, is bound.
    if b"\r" in local.replace(b"\r\n", b"") or local.replace(b"\r\n", b"\n") != result.stdout:
        raise CardSyncError("Forge card differs from the declared immutable source")
    return digest(result.stdout)


def _get(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "szl-forge-card-review/1"})
    with urlopen(request, timeout=20) as response:
        body = response.read(MAX_REMOTE_BYTES + 1)
    if len(body) > MAX_REMOTE_BYTES:
        raise CardSyncError("Hub response exceeds review bound")
    return body


def read_hub_parent() -> tuple[bytes, dict, dict]:
    info = json.loads(_get(f"https://huggingface.co/api/models/{HUB_REPO}"))
    if (
        info.get("id") != HUB_REPO
        or info.get("sha") != HUB_PARENT
        or info.get("private") is not False
        or info.get("gated") is not False
    ):
        raise CardSyncError("public Hub model identity or parent drifted")
    base = f"https://huggingface.co/{HUB_REPO}/raw/{HUB_PARENT}/"
    readme = _get(base + "README.md")
    if digest(readme) != HUB_README_SHA256:
        raise CardSyncError("reviewed Hub README bytes changed")
    historical = json.loads(_get(base + "eval_dev_named_n.json"))
    additive = json.loads(_get(base + "eval_dev_named_n.bound.json"))
    return readme, historical, additive


def validate_dev_records(historical: dict, additive: dict) -> None:
    shared = {"artifact": HUB_REPO, "split": "DEV", "n": 12,
              "opened_test_jsonl": False, "publication_eligible": False}
    for label, record, correct, recovery in (
        ("historical", historical, 11, "3/4"),
        ("additive", additive, 12, "4/4"),
    ):
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in shared.items()):
            raise CardSyncError(f"{label} DEV identity or release boundary drifted")
        if (record.get("correct"), record.get("draft"), record.get("recovery"),
                record.get("refuse")) != (correct, "4/4", recovery, "4/4"):
            raise CardSyncError(f"{label} DEV counts drifted")
    if (
        additive.get("promotion_effect") != "NONE"
        or additive.get("base_model") != "UNRECORDED"
        or additive.get("artifact_sha256", {}).get("adapter_model.safetensors") != ADAPTER_SHA256
        or "not superseded" not in additive.get("relationship", "")
    ):
        raise CardSyncError("additive adapter binding or non-promotion boundary drifted")


OLD_CLAIM = (
    "| Development named-N check | DEV n=12 MEASURED **11/12** (4/4 draft · 3/4 recovery · 4/4 refuse); "
    "`test.jsonl` never opened. The szl-forge card's \"12/12\" row contradicts the committed receipt and is corrected here. |"
)
NEW_CLAIM = (
    "| Historical development check (2026-08-29) | DEV n=12 MEASURED **11/12** "
    "(4/4 draft · 3/4 recovery · 4/4 refuse); one recovery output did not parse; `test.jsonl` never opened. |\n"
    "| Additive adapter-bound development check (2026-09-29) | DEV n=12 MEASURED **12/12** "
    "(4/4 draft · 4/4 recovery · 4/4 refuse); separate owner-run record, base model `UNRECORDED`, "
    "`promotion_effect=NONE`, not a release gate. |"
)
OLD_EXPLANATION = (
    "The earlier card's 12/12 statement contradicted this committed receipt.\n"
    "The failure remains part of the evidence. This is a small owner-run development\n"
    "check, not a held-out test, 1024-case evaluation, public leaderboard result,\n"
    "or third-party benchmark. The receipt does not establish exact current public\n"
    "adapter/merged-byte binding or transfer its result to the later merge."
)
NEW_EXPLANATION = (
    "The historical 11/12 failure remains part of the evidence. A separate additive\n"
    "2026-09-29 owner-run DEV check reports 12/12 on adapter SHA-256\n"
    f"`{ADAPTER_SHA256}`. It explicitly does not supersede the older record,\n"
    "records `base_model=UNRECORDED` and `promotion_effect=NONE`, and is not a\n"
    "held-out test, independent benchmark, release gate, or evaluation of the\n"
    "derived merged checkpoint. Neither check establishes complete reproducible\n"
    "inference lineage or deployment readiness."
)
OLD_EVIDENCE = (
    "| Evidence | Train loss MEASURED on the 180-row committed split. "
    "DEV named-N 11/12 MEASURED (owner-run, not held-out). |"
)
NEW_EVIDENCE = (
    "| Evidence | Train loss MEASURED on the 180-row committed split. "
    "Separate owner-run DEV n=12 checks: historical 11/12 and additive adapter-bound 12/12; "
    "neither held-out nor a merged-checkpoint qualification. |"
)
OLD_EVALS = "  evals: DEV_NAMED_N_11_OF_12_MEASURED"
NEW_EVALS = (
    "  evals: HISTORICAL_DEV_11_OF_12_AND_ADDITIVE_ADAPTER_BOUND_DEV_12_OF_12_UNQUALIFIED"
)
OLD_SOURCE_NOTE = (
    "with the receipt-bound 11/12 retained where the source card still says 12/12)."
)
NEW_SOURCE_NOTE = (
    "with the historical 11/12 receipt retained; current Forge source distinguishes "
    "it from the separate additive adapter-bound 12/12 DEV record)."
)

# The preceding evidence strings describe the already published correction.
# This new one-shot candidate preserves those results and release boundaries.
OLD_LOAD_SECTION = (
    "## Load\n\n"
    "Untested in this card pass. Adapter path (the declared artifact):\n\n"
    "```python\n"
    "from peft import PeftModel\n"
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
    'base_id = "Qwen/Qwen3.5-0.8B"\n'
    "tok = AutoTokenizer.from_pretrained(base_id)\n"
    "base = AutoModelForCausalLM.from_pretrained(base_id)\n"
    'model = PeftModel.from_pretrained(base, "SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3")\n'
    "```\n\n"
    "The root also contains `config.json` and the merged `model.safetensors`; callers must verify "
    "which path their library selects rather than relying on loader precedence.\n\n"
)
NEW_LOAD_SECTION = (
    "## Load\n\n"
    "Executable loading guidance is withdrawn pending exact-lineage and loader qualification.\n\n"
    "Use only for isolated, reviewed research with explicitly pinned model, base and runtime identities. "
    "The root contains both adapter and merged artifacts, so select the load path explicitly. "
    "Before publishing an executable load recipe, establish the matching immutable base revision and "
    "qualify the exact Transformers/PEFT versions and architecture class. The adapter configuration "
    "references a conditional-generation base class while the merged configuration names "
    "`Qwen3_5ForCausalLM`; do not hide this distinction behind a generic loader example.\n\n"
    "Loader compatibility is UNKNOWN. No weights were loaded and no inference was run in this card review.\n\n"
)
OLD_LOADER_PHRASE = "(`model.safetensors` + `config.json`, loadable with `transformers`)"
NEW_LOADER_PHRASE = "(`model.safetensors` + `config.json`; loader compatibility UNKNOWN)"


def candidate(readme: bytes) -> bytes:
    if b"\r\n" in readme:
        raise CardSyncError("Hub line endings changed from reviewed LF")
    updated = readme
    for old, new in ((OLD_LOAD_SECTION, NEW_LOAD_SECTION),
                     (OLD_LOADER_PHRASE, NEW_LOADER_PHRASE)):
        before, after = old.encode("utf-8"), new.encode("utf-8")
        if updated.count(before) != 1 or updated.count(after):
            raise CardSyncError("Hub card anchor is absent, duplicated, or already edited")
        updated = updated.replace(before, after, 1)
    return updated


def prepare(root: Path, source_revision: str) -> tuple[bytes, str, dict]:
    source_hash = bind_source(root, source_revision)
    readme, historical, additive = read_hub_parent()
    validate_dev_records(historical, additive)
    changed = candidate(readme)
    if digest(changed) != CANDIDATE_README_SHA256:
        raise CardSyncError("reviewed two-span loader-withdrawal Hub candidate bytes changed")
    diff = "".join(difflib.unified_diff(
        readme.decode("utf-8").splitlines(keepends=True),
        changed.decode("utf-8").splitlines(keepends=True),
        fromfile=f"{HUB_REPO}@{HUB_PARENT}/README.md",
        tofile="review-candidate/README.md",
    ))
    receipt = {
        "schema": "szl.receiptagent-v3-card-review/v1",
        "source_repository": "szl-holdings/szl-forge",
        "source_revision": source_revision,
        "source_card_path": SOURCE_CARD,
        "source_card_sha256": source_hash,
        "target_repository": HUB_REPO,
        "observed_hub_parent": HUB_PARENT,
        "observed_readme_sha256": digest(readme),
        "candidate_readme_sha256": digest(changed),
        "candidate_kind": CANDIDATE_KIND,
        "changed_paths": ["README.md"],
        "disposition": "REVIEW_ONLY_NO_HUB_WRITE",
        "release_status": "UNQUALIFIED",
    }
    return changed, diff, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    changed, diff, receipt = prepare(ROOT, args.source_revision)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "README.md").write_bytes(changed)
        (args.output_dir / "review.diff").write_text(diff, encoding="utf-8", newline="\n")
        (args.output_dir / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
