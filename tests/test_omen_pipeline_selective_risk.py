"""Selective-risk accounting in eval-abstain of tools/szl_omen_pipeline.py (R06 phase 2).

Software checks of a metric only. No model is loaded and no GPU, network or token is used; nothing
here is a statement about any model. The 11-row held-out set is public and has been reused across
evaluations, so every number computed on it is diagnostic and cannot support a promotion decision.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import itertools
import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "szl_omen_pipeline.py"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _load("szl_omen_pipeline_selective_risk_test", TOOL)
ingest = _load("ingest_khipu_abstain_receipts_selective_risk_test", ROOT / "tools" / "ingest_khipu_abstain_receipts.py")

N, A, M = pipeline.SR_NAVIGATE, pipeline.SR_ABSTAIN, pipeline.SR_MALFORMED
UNKNOWN = pipeline.SR_UNKNOWN

# khipu-r3 baseline vs DPO challenger bundle (read only). The LF bytes git stores hash to
# 4c6a1d21...; the sealed receipt and MANIFEST bind aa65f278..., the CRLF serialization of the
# same content (see szl-khipu tests/test_khipu_selective_risk.py at 8ababb91).
BUNDLE = ROOT / "receipts" / "khipu-abstain" / "20261001-4064238281e3"
BUNDLE_TRANSCRIPTS_LF_SHA256 = "4c6a1d2141e13a7999cbce5d4f6a9820401aeea831a62caf21e385d6b73c0ee4"

ACCOUNT_KEYS = {
    "schema", "risk_definition", "malformed_semantics", "total", "expected_navigate", "expected_abstain",
    "abstained", "abstained_correct", "over_abstained", "malformed", "answered_valid", "false_navigate",
    "wrong_route", "accepted", "correct_accepted", "accepted_errors", "coverage", "coverage_valid",
    "selective_risk", "abstain_recall", "legacy",
}

OFFERED = ("node-a", "node-b")


def gold_row(expected: str, key: object = None) -> dict:
    """A held-out row in the curriculum shape grade() reads (system, user candidates, gold plan)."""
    cited = [OFFERED[0]] if expected == "NAVIGATE" else []
    row = {"messages": [
        {"role": "system", "content": "synthetic fixture"},
        {"role": "user", "content": json.dumps({"query": "synthetic", "candidates": [{"nodeId": n} for n in OFFERED]})},
        {"role": "assistant", "content": json.dumps({"decision": expected, "citedNodeIds": cited})},
    ]}
    if key is not None:
        row["_key"] = key
    return row


def plan(decision: object, cited: list | None = None) -> str:
    body: dict = {"decision": decision}
    if cited is not None:
        body["citedNodeIds"] = cited
    return json.dumps(body)


def transcripts(cases: list[tuple[str, str, str, str]]) -> list[dict]:
    """(file, expected, baseline_output, challenger_output) -> transcript rows exactly as cmd_eval writes them."""
    rows = []
    for i, (f, expected, base_out, chal_out) in enumerate(cases):
        gold = gold_row(expected)
        rows.append({"file": f, "index": i,
                     "baseline": {"output": base_out, **pipeline.grade(gold, base_out)},
                     "challenger": {"output": chal_out, **pipeline.grade(gold, chal_out)}})
    return rows


def _bundle_rows() -> tuple[list[dict], dict]:
    raw = (BUNDLE / "eval" / "eval_transcripts.jsonl").read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    assert hashlib.sha256(lf).hexdigest() == BUNDLE_TRANSCRIPTS_LF_SHA256
    receipt = json.loads((BUNDLE / "eval" / "eval_receipt.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest() == receipt["transcripts_sha256"]
    return [json.loads(line) for line in lf.decode("utf-8").splitlines() if line.strip()], receipt


# ---------------------------------------------------------------- historical vector


def test_khipu_r3_bundle_vector_matches_the_szl_khipu_derivation():
    rows, receipt = _bundle_rows()
    block = pipeline.combined_selective(rows)
    assert block["status"] == "MEASURED"
    assert block["schema"] == "szl.khipu-selective-risk/v1"
    assert block["verdict"] == "NOT_DERIVED"
    assert block["rows_by_file"] == {"adversarial.jsonl": 6, "eval.jsonl": 5}

    base = block["baseline"]
    assert set(base) == ACCOUNT_KEYS
    assert (base["total"], base["abstained"], base["over_abstained"], base["malformed"]) == (11, 10, 4, 1)
    assert (base["abstained_correct"], base["answered_valid"]) == (6, 0)
    assert (base["accepted"], base["correct_accepted"], base["accepted_errors"]) == (1, 0, 1)
    assert base["selective_risk"] == {"num": 1, "den": 1}
    assert base["coverage"] == {"num": 1, "den": 11}
    assert base["coverage_valid"] == {"num": 0, "den": 11}
    assert base["abstain_recall"] == {"num": 6, "den": 6}
    assert base["legacy"] == {"abstain": "6/6", "grounding": "0/5"}

    chal = block["challenger"]
    assert (chal["total"], chal["malformed"], chal["abstained"]) == (11, 11, 0)
    assert chal["selective_risk"] == {"num": 11, "den": 11}
    assert chal["coverage"] == {"num": 11, "den": 11}
    assert chal["coverage_valid"] == {"num": 0, "den": 11}
    assert chal["abstain_recall"] == {"num": 0, "den": 6}
    assert chal["legacy"] == {"abstain": "0/6", "grounding": "0/5"}

    ref = block["reference_baselines"]
    assert ref["all_abstain"]["selective_risk"] == UNKNOWN
    assert ref["all_abstain"]["coverage"] == {"num": 0, "den": 11}
    assert ref["all_answer"]["selective_risk"] == {"num": 6, "den": 11}
    assert ref["all_answer"]["coverage"] == {"num": 11, "den": 11}

    # The legacy view, recomputed from the same rows, agrees with the sealed summarize() counts.
    results = receipt["results"]
    for side in ("baseline", "challenger"):
        adv, nav = results["adversarial.jsonl"][side], results["eval.jsonl"][side]
        assert block[side]["legacy"] == {"abstain": f"{adv['correct']}/{adv['total']}",
                                         "grounding": f"{nav['correct']}/{nav['total']}"}
        assert block[side]["total"] == adv["total"] + nav["total"]


# ---------------------------------------------------------------- row mapping


def test_all_abstain_side_reports_unknown_risk_never_zero():
    rows = transcripts([
        ("adversarial.jsonl", "ABSTAIN", plan("ABSTAIN", []), plan("ABSTAIN", [])),
        ("adversarial.jsonl", "ABSTAIN", plan("ABSTAIN"), plan("NAVIGATE", ["node-a"])),
        ("eval.jsonl", "NAVIGATE", plan("ABSTAIN", []), plan("NAVIGATE", ["node-a"])),
        ("eval.jsonl", "NAVIGATE", plan("ABSTAIN"), plan("NAVIGATE", ["node-b"])),
    ])
    block = pipeline.combined_selective(rows)
    base = block["baseline"]
    assert (base["total"], base["abstained"], base["accepted"]) == (4, 4, 0)
    assert base["selective_risk"] == UNKNOWN
    assert base["selective_risk"] != {"num": 0, "den": 0}
    assert base["coverage"] == {"num": 0, "den": 4}
    chal = block["challenger"]
    assert (chal["accepted"], chal["correct_accepted"], chal["false_navigate"]) == (3, 2, 1)
    assert chal["selective_risk"] == {"num": 1, "den": 3}
    assert "PASS" not in json.dumps(block)


def test_malformed_outputs_are_accepted_errors_and_never_dropped():
    rows = transcripts([
        ("adversarial.jsonl", "ABSTAIN", "not json at all", plan("ABSTAIN", [])),
        ("adversarial.jsonl", "ABSTAIN", plan("ABSTAIN", ["node-a"]), plan("ABSTAIN", [])),  # ABSTAIN citing a handle
        ("eval.jsonl", "NAVIGATE", plan("PROPOSE", ["node-a"]), plan("NAVIGATE", ["node-a"])),
        ("eval.jsonl", "NAVIGATE", plan("navigate", ["node-a"]), plan("NAVIGATE", ["node-a"])),
        ("eval.jsonl", "NAVIGATE", plan("ABSTAIN", ["node-b"]), plan("NAVIGATE", ["node-a"])),
    ])
    block = pipeline.combined_selective(rows)
    base = block["baseline"]
    assert (base["total"], base["malformed"], base["abstained"]) == (5, 5, 0)
    assert (base["accepted"], base["accepted_errors"]) == (5, 5)
    assert base["selective_risk"] == {"num": 5, "den": 5}
    assert base["coverage"] == {"num": 5, "den": 5}
    # summarize() counts only one parse failure on the same rows; the selective view is stricter.
    assert pipeline.summarize([r["baseline"] for r in rows])["parse_failures"] == 1
    chal = block["challenger"]
    assert (chal["malformed"], chal["accepted"], chal["accepted_errors"]) == (0, 3, 0)
    assert chal["selective_risk"] == {"num": 0, "den": 3}
    assert chal["coverage"] == {"num": 3, "den": 5}


def test_navigate_and_abstain_rows_follow_grade():
    rows = transcripts([
        ("eval.jsonl", "NAVIGATE", plan("NAVIGATE", ["node-a"]), plan("ABSTAIN")),
        ("eval.jsonl", "NAVIGATE", plan("NAVIGATE", ["node-z"]), plan("ABSTAIN")),  # invented handle
        ("eval.jsonl", "NAVIGATE", plan("NAVIGATE", []), plan("ABSTAIN")),          # no route
        ("eval.jsonl", "NAVIGATE", plan("ABSTAIN"), plan("ABSTAIN")),               # over-abstained
        ("adversarial.jsonl", "ABSTAIN", plan("NAVIGATE", ["node-a"]), plan("ABSTAIN")),
        ("adversarial.jsonl", "ABSTAIN", plan("ABSTAIN", []), plan("ABSTAIN")),
    ])
    base = pipeline.combined_selective(rows)["baseline"]
    assert (base["total"], base["correct_accepted"], base["wrong_route"], base["false_navigate"]) == (6, 1, 2, 1)
    assert (base["abstained_correct"], base["over_abstained"], base["malformed"]) == (1, 1, 0)
    assert (base["accepted"], base["accepted_errors"]) == (4, 3)
    assert base["selective_risk"] == {"num": 3, "den": 4}
    assert base["coverage"] == {"num": 4, "den": 6}
    # The selective counts agree with grade() on every row it scored correct.
    graded = [r["baseline"] for r in rows]
    assert base["correct_accepted"] == sum(1 for g in graded if g["expected"] == "NAVIGATE" and g["correct"])
    assert base["abstained_correct"] == sum(1 for g in graded if g["expected"] == "ABSTAIN" and g["correct"])


def test_inconsistent_or_unaccountable_transcripts_are_unavailable_not_counted():
    good = transcripts([("adversarial.jsonl", "ABSTAIN", plan("ABSTAIN", []), "garbled")])

    parsed_lie = json.loads(json.dumps(good))
    parsed_lie[0]["challenger"].update(parsed=True, decision="ABSTAIN", correct=True)
    flipped = json.loads(json.dumps(good))
    flipped[0]["baseline"]["correct"] = False
    no_output = json.loads(json.dumps(good))
    del no_output[0]["baseline"]["output"]
    odd_gold = json.loads(json.dumps(good))
    odd_gold[0]["baseline"]["expected"] = "PROPOSE"
    decision_lie = json.loads(json.dumps(good))
    decision_lie[0]["baseline"]["decision"] = "NAVIGATE"
    sides_disagree = json.loads(json.dumps(good))
    sides_disagree[0]["challenger"]["expected"] = "NAVIGATE"
    duplicate = json.loads(json.dumps(good + good))
    train_file = json.loads(json.dumps(good))
    train_file[0]["file"] = "train.jsonl"

    for name, rows in (("parsed flag disagrees with output", parsed_lie), ("correct flag flipped", flipped),
                       ("output missing", no_output), ("expected outside NAVIGATE/ABSTAIN", odd_gold),
                       ("decision disagrees with output", decision_lie), ("sides disagree on gold", sides_disagree),
                       ("duplicate (file, index)", duplicate), ("file outside the held-out set", train_file),
                       ("no rows", [])):
        block = pipeline.combined_selective(rows)
        assert block["status"] == "UNAVAILABLE", name
        assert block["verdict"] == "NOT_DERIVED", name
        assert block["reason"], name
        assert "baseline" not in block and "challenger" not in block, name
    assert pipeline.combined_selective(good)["status"] == "MEASURED"


# ---------------------------------------------------------------- vendored accounting parity

def _rows(*triples):
    return [pipeline.sr_normalize_row(e, d, c) for e, d, c in triples]


# Vectors and expected values copied from szl-khipu tests/test_khipu_selective_risk.py at 8ababb91.
POLICY_A = _rows(*[(N, N, True)] * 5, *[(A, A, False)] * 4, *[(A, N, False)] * 2)
ALL_ABSTAIN = _rows(*[(N, A, False)] * 5, *[(A, A, False)] * 6)
WRONG_ROUTE = _rows(*[(N, N, True)] * 3, *[(N, N, False)] * 2, *[(A, A, False)] * 5, (A, N, False))
KHIPU_R2 = _rows(*[(N, N, True)] * 5, (A, N, False), *[(A, A, False)] * 3, (A, N, False), (A, N, False))


def test_vendored_accounting_reproduces_the_szl_khipu_vectors():
    a = pipeline.sr_account(POLICY_A)
    assert set(a) == ACCOUNT_KEYS
    assert (a["accepted"], a["correct_accepted"]) == (7, 5)
    assert a["selective_risk"] == {"num": 2, "den": 7} and a["coverage"] == {"num": 7, "den": 11}
    assert a["legacy"] == {"abstain": "4/6", "grounding": "5/5"}

    z = pipeline.sr_account(ALL_ABSTAIN)
    assert (z["total"], z["accepted"]) == (11, 0)
    assert z["selective_risk"] == UNKNOWN and z["coverage"] == {"num": 0, "den": 11}
    assert z["legacy"]["abstain"] == "6/6"

    w = pipeline.sr_account(WRONG_ROUTE)
    assert (w["wrong_route"], w["false_navigate"], w["malformed"], w["answered_valid"]) == (2, 1, 0, 6)
    assert (w["accepted"], w["correct_accepted"], w["accepted_errors"]) == (6, 3, 3)
    assert w["selective_risk"] == {"num": 3, "den": 6} and w["coverage"] == {"num": 6, "den": 11}
    assert w["coverage_valid"] == {"num": 6, "den": 11} and w["abstain_recall"] == {"num": 5, "den": 6}
    assert w["legacy"] == {"abstain": "5/6", "grounding": "3/5"}

    r2 = pipeline.sr_account(KHIPU_R2)
    assert r2["legacy"] == {"abstain": "3/6", "grounding": "5/5"}
    assert (r2["accepted"], r2["correct_accepted"], r2["false_navigate"], r2["malformed"]) == (8, 5, 3, 0)
    assert r2["selective_risk"] == {"num": 3, "den": 8} and r2["coverage"] == {"num": 8, "den": 11}
    assert r2["abstain_recall"] == {"num": 3, "den": 6}

    for suite in (POLICY_A, ALL_ABSTAIN, KHIPU_R2):
        answer = pipeline.sr_baseline_all_answer(suite)
        assert answer["selective_risk"] == {"num": 6, "den": 11}
        assert answer["coverage"] == {"num": 11, "den": 11} and answer["correct_accepted"] == 5
        abstain = pipeline.sr_baseline_all_abstain(suite)
        assert abstain["selective_risk"] == UNKNOWN and abstain["coverage"] == {"num": 0, "den": 11}


ROW_KINDS = ((N, N, True), (N, N, False), (N, A, False), (N, M, False), (A, A, False), (A, N, False), (A, M, False))


def test_vendored_accounting_matches_its_definitions_on_every_small_multiset():
    """Independent recount from the docstring definitions for every multiset of up to 4 rows (330 cases)."""
    def ratio(num, den):
        return {"num": num, "den": den} if den else UNKNOWN

    for size in range(5):
        for kinds in itertools.combinations_with_replacement(ROW_KINDS, size):
            rows = _rows(*kinds)
            acct = pipeline.sr_account(rows)
            total = len(rows)
            abstained = sum(1 for r in rows if r["decision"] == A)
            correct = sum(1 for r in rows if r["answer_correct"])
            accepted = total - abstained
            assert acct["total"] == total and acct["abstained"] == abstained
            assert acct["accepted"] == accepted and acct["correct_accepted"] == correct
            assert acct["accepted_errors"] == accepted - correct
            assert acct["malformed"] == sum(1 for r in rows if r["decision"] == M)
            assert acct["coverage"] == ratio(accepted, total)
            assert acct["selective_risk"] == ratio(accepted - correct, accepted)
            assert (acct["selective_risk"] == UNKNOWN) is (accepted == 0)
            assert acct["expected_navigate"] + acct["expected_abstain"] == total


def test_vendored_row_validation_is_fail_closed():
    for decision in ("PROPOSE", None, "abstain", "navigate", "", " ABSTAIN", 0, [], {"decision": "ABSTAIN"}):
        assert pipeline.sr_normalize_row(A, decision, False)["decision"] == M
        assert pipeline.sr_normalize_row(N, decision, False)["decision"] == M
    for expected in ("MALFORMED", "abstain", None, "", 1):
        with pytest.raises(ValueError):
            pipeline.sr_normalize_row(expected, A, False)
    for expected, decision in ((A, A), (A, N), (A, None), (N, A), (N, None), (N, "PROPOSE")):
        with pytest.raises(ValueError):
            pipeline.sr_normalize_row(expected, decision, True)
    for answer_correct in (1, 0, "yes", None):
        with pytest.raises(ValueError):
            pipeline.sr_normalize_row(N, N, answer_correct)


# ---------------------------------------------------------------- cmd_eval wiring (offline test doubles)

BASE_EVAL_RECEIPT_KEYS = {
    "stage", "status", "artifact", "challenger_dir", "challenger_adapter_sha256", "policy", "data", "decode",
    "grader", "results", "transcripts_sha256", "advisory_challenger_improves", "advisory_note", "hardware",
    "publication_eligible", "autonomy_eligible", "promotion",
    # env_block()
    "python", "platform", "pipeline_sha256", "schema", "generated_at", "torch", "transformers", "peft", "trl",
    "datasets", "accelerate", "huggingface_hub", "kernel_stack",
}


class _FakeModel:
    """Test double for the PEFT-wrapped policy: only the adapter toggle cmd_eval uses."""

    def __init__(self) -> None:
        self.adapter_enabled = True

    def eval(self) -> "_FakeModel":
        return self

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = True


def test_cmd_eval_adds_the_block_and_keeps_every_existing_receipt_key(tmp_path, monkeypatch):
    outputs = {  # (file, index) -> (baseline output, challenger output)
        ("adversarial.jsonl", 0): (plan("ABSTAIN", []), plan("ABSTAIN", [])),
        ("adversarial.jsonl", 1): (plan("ABSTAIN"), "garbled"),
        ("adversarial.jsonl", 2): (plan("NAVIGATE", ["node-a"]), plan("ABSTAIN", [])),
        ("eval.jsonl", 0): (plan("ABSTAIN"), plan("NAVIGATE", ["node-a"])),
        ("eval.jsonl", 1): ("garbled", plan("NAVIGATE", ["node-z"])),
    }
    heldout = {
        "adversarial.jsonl": [gold_row("ABSTAIN", ["adversarial.jsonl", i]) for i in range(3)],
        "eval.jsonl": [gold_row("NAVIGATE", ["eval.jsonl", i]) for i in range(2)],
    }
    digests = {"train.abstain.jsonl": "1" * 64, "train.jsonl": "2" * 64, "adversarial.jsonl": "3" * 64, "eval.jsonl": "4" * 64}
    counts = {"train": 23, "adversarial.jsonl": 3, "eval.jsonl": 2}

    challenger = tmp_path / "challenger"
    challenger.mkdir()
    (challenger / "adapter_config.json").write_text("{}", encoding="utf-8")
    (challenger / "adapter_model.safetensors").write_bytes(b"synthetic adapter fixture")
    adapter_sha = {"adapter_model.safetensors": hashlib.sha256(b"synthetic adapter fixture").hexdigest()}
    policy = {"base": pipeline.DPO_BASE, "adapter": pipeline.DPO_ADAPTER, "adapter_revision": "7" * 40, "init": "base+adapter merged"}
    dpo = {
        "stage": "dpo", "status": "TRAINED_CHALLENGER", "artifact": pipeline.CHALLENGER_ID, "policy": policy,
        "data": {"repo": f"{pipeline.DATA_REPO_TYPE}:{pipeline.DATA_REPO}", "revision": "9" * 40, "digests_sha256": digests,
                 "train_files": list(pipeline.TRAIN_FILES), "heldout_files_sealed": list(pipeline.HELDOUT_FILES),
                 "counts": counts, "disjoint_prompts_verified": True},
        "train_loss_label": "TRAIN METRIC, NOT AN EVAL", "evaluation": "NOT_RUN",
        "hardware": {"gpu": "synthetic test double", "cuda": "test-double"}, "adapter_sha256": adapter_sha,
        "publication_eligible": False, "autonomy_eligible": False, "promotion": "NOT_PROMOTABLE",
        "schema": pipeline.SCHEMA, "pipeline_sha256": pipeline.self_digest(), "generated_at": "2026-10-14T00:00:00Z",
    }
    (challenger / "dpo_receipt.json").write_text(json.dumps(dpo), encoding="utf-8")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(manual_seed=lambda seed: None))
    # env_block() still runs for real; a None entry makes each optional heavy import fail fast and hermetically.
    for heavy in ("transformers", "trl", "datasets", "accelerate", "triton", "fla", "causal_conv1d", "liger_kernel",
                  "unsloth", "bitsandbytes"):
        monkeypatch.setitem(sys.modules, heavy, None)
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(
        PeftModel=types.SimpleNamespace(from_pretrained=lambda policy_model, path: _FakeModel())))
    monkeypatch.setattr(pipeline, "require_cuda", lambda stage, path: (None, {"gpu": "synthetic test double", "cuda": "test-double"}))
    monkeypatch.setattr(pipeline, "load_split", lambda revision: {
        "train_rows": [], "heldout": heldout, "digests": dict(digests), "counts": dict(counts), "disjoint_prompts_verified": True})
    monkeypatch.setattr(pipeline, "load_policy", lambda device, adapter_rev: (object(), object()))
    monkeypatch.setattr(pipeline, "render_prompt", lambda tok, row: json.dumps(row["_key"]))
    monkeypatch.setattr(pipeline, "generate", lambda model, tok, prompt, device:
                        outputs[tuple(json.loads(prompt))][1 if model.adapter_enabled else 0])

    out = challenger / "eval"
    assert pipeline.cmd_eval(challenger, out) == 0
    receipt = json.loads((out / "eval_receipt.json").read_text(encoding="utf-8"))

    # Additive: every key the base receipt wrote is still written, and only combined_selective is new.
    assert set(receipt) == BASE_EVAL_RECEIPT_KEYS | {"combined_selective"}
    written = [json.loads(line) for line in (out / "eval_transcripts.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_results = {}
    for f in pipeline.HELDOUT_FILES:
        expected_results[f] = {side: pipeline.summarize([t[side] for t in written if t["file"] == f])
                               for side in ("baseline", "challenger")}
    assert receipt["results"] == expected_results
    abst, nav = expected_results["adversarial.jsonl"], expected_results["eval.jsonl"]
    assert receipt["advisory_challenger_improves"] is (
        abst["challenger"]["correct"] > abst["baseline"]["correct"]
        and nav["challenger"]["correct"] >= nav["baseline"]["correct"]
        and nav["challenger"]["invented_handle_rows"] <= nav["baseline"]["invented_handle_rows"])
    assert receipt["promotion"] == "NOT_PROMOTABLE" and receipt["publication_eligible"] is False

    block = receipt["combined_selective"]
    assert block == pipeline.combined_selective(written)
    assert block["status"] == "MEASURED" and block["verdict"] == "NOT_DERIVED"
    assert block["rows_by_file"] == {"adversarial.jsonl": 3, "eval.jsonl": 2}
    base, chal = block["baseline"], block["challenger"]
    assert (base["abstained"], base["false_navigate"], base["over_abstained"], base["malformed"]) == (3, 1, 1, 1)
    assert base["selective_risk"] == {"num": 2, "den": 2} and base["coverage"] == {"num": 2, "den": 5}
    assert (chal["abstained"], chal["correct_accepted"], chal["wrong_route"], chal["malformed"]) == (2, 1, 1, 1)
    assert chal["selective_risk"] == {"num": 2, "den": 3} and chal["coverage"] == {"num": 3, "den": 5}

    # The receipt still passes the existing ingest invariants, and its pipeline digest is a known one.
    problems: list[str] = []
    ingest.check_dpo(dpo, problems)
    ingest.check_eval(receipt, dpo, out, problems)
    assert problems == []
    status = ingest.state_block(dpo, receipt, None, "synthetic-run")
    assert "NOT A PIPELINE DIGEST KNOWN" not in status
    # STATUS.md derivation ignores the new block: the same receipt without it derives the same state block.
    legacy_receipt = {k: v for k, v in receipt.items() if k != "combined_selective"}
    assert ingest.state_block(dpo, legacy_receipt, None, "synthetic-run") == status
    legacy_problems: list[str] = []
    ingest.check_eval(legacy_receipt, dpo, out, legacy_problems)
    assert legacy_problems == []


# ---------------------------------------------------------------- pipeline digest registry

HISTORICAL_PIPELINE_DIGESTS = {
    "6a4b893b8d5141e73e2d1e91b0cc4f84c1ceb0bef5ea9230cc4f4288a1440a15": "szl-forge huggingface@c6e60fee (pipeline v2)",
    "31d5d1bb33f81d9b4ab20f9ba9054741bec8183e45788a99f0e77fe50f58d1ca": "szl-forge huggingface@6c35dbca (pipeline v2 + owner-authorized cards)",
    "c2361df7ab79315fd5934b839116a7ad3ad612329f182c5069c0ff6bc947bdff": "szl-forge huggingface@8cdb1ac3 (#467 memory ladder)",
    "ef3a9049ca8ebd9f48cfecb1c039792360d220cc101bd638f2e563ed84432da2": "szl-forge huggingface@3dee2933 (#469 pure-bf16 lane; produced bundle 20261001-a989dd523998)",
    "2e242ac68e8fe925d3cbb9c77feb9ffc908bbfc1c8f0ed841cfeeafc2bb56316": "szl-forge huggingface@0aa28b8f (#473 DPO profiles C2/C3)",
    "0093753bd18510ee79f1d99f20a143951e44fa90b117c674b1d85c4d7acd491d": "szl-forge huggingface@f4c9bab8 (#474 knob probe + kernel stack on the C2/C3 profiles)",
}


def test_pipeline_digest_registry_is_append_only_and_names_the_current_pipeline():
    known = ingest.KNOWN_PIPELINE_SHA256
    assert list(known)[: len(HISTORICAL_PIPELINE_DIGESTS)] == list(HISTORICAL_PIPELINE_DIGESTS)
    for digest, label in HISTORICAL_PIPELINE_DIGESTS.items():
        assert known[digest] == label
    assert all(len(d) == 64 and set(d) <= set("0123456789abcdef") for d in known)
    assert "#529" in known["d23132854815bcede6fff2d8bda5d3d367752b0bd78c12018c993696920a28ae"]
    current = hashlib.sha256(TOOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert current in known, (
        f"tools/szl_omen_pipeline.py sha256 {current} is not in KNOWN_PIPELINE_SHA256; append it "
        "(never rewrite an existing entry) in tools/ingest_khipu_abstain_receipts.py")
