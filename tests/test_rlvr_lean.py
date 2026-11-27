"""Lean RLVR reward: static hacks fail closed; the kernel path skips without lean."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from training.rlvr.lean import (
    ALLOWED_AXIOMS,
    AXIOM_COUNT,
    BASELINE_AXIOMS,
    PINNED_COMMIT,
    TRACKED_SORRIES,
    VERIFIER,
    lean_argv,
    lean_reward,
    score_lean,
)
from training.rlvr.receipt import sha256_hex

CLEAN = "theorem t : True := trivial\n"
ROOT = Path(__file__).resolve().parents[1]


def _ok_probe(_checkout):
    return {"ok": True, "head": PINNED_COMMIT}


def _pass_runner(source, checkout, timeout):
    assert "trivial" in source
    assert timeout <= 120
    return {"ok": True, "axioms": ["propext"], "detail": "ok"}


def test_clean_candidate_scores_one_when_kernel_accepts():
    outcome = score_lean(CLEAN, checkout=Path("."), probe=_ok_probe, runner=_pass_runner)
    assert outcome.reward == 1.0
    receipt = outcome.receipt
    assert receipt["verifier"] == VERIFIER
    assert receipt["version"]
    assert receipt["verdict"] == "pass"
    assert receipt["input_sha256"] == sha256_hex(CLEAN)
    assert receipt["baseline"]["axioms"] == 14
    assert receipt["baseline"]["tracked_sorries"] == 163
    assert receipt["baseline"]["commit"] == PINNED_COMMIT
    assert len(BASELINE_AXIOMS) == AXIOM_COUNT == 14
    assert TRACKED_SORRIES == 163
    assert "sorryAx" not in ALLOWED_AXIOMS


@pytest.mark.parametrize(
    "source,fragment",
    [
        ("theorem t : True := by sorry\n", "sorry"),
        ("-- not a proof\ntheorem t : True := by admit\n", "admit"),
        ("axiom evilAx : False\ntheorem t : False := evilAx\n", "axiom"),
        ("theorem t : 2 ^ 10 = 1024 := by native_decide\n", "native_decide"),
        ('theorem t : True := trivial\n#eval IO.println "hi"\n', "#eval"),
        ("theorem t : True := trivial\ndef spawned := IO.Process.spawn { }\n", "IO.Process"),
        ("theorem t : True := trivial\nrun_cmd echo hi\n", "run_cmd"),
    ],
)
def test_reward_hacks_score_zero_without_running_lean(source, fragment):
    called = {}

    def runner(*_args):
        called["ran"] = True
        return {"ok": True, "axioms": []}

    outcome = score_lean(source, checkout=Path("."), probe=_ok_probe, runner=runner)
    assert outcome.reward == 0.0
    assert outcome.receipt["verdict"] == "fail"
    assert fragment in outcome.receipt["reason"]
    assert called == {}


def test_comment_and_string_sorry_are_not_tactics():
    source = '-- sorry in a comment\ntheorem t : String := "sorry and admit"\n'

    def runner(*_args):
        return {"ok": True, "axioms": ["propext"], "detail": ""}

    outcome = score_lean(source, checkout=Path("."), probe=_ok_probe, runner=runner)
    assert outcome.reward == 1.0


def test_block_comment_sorry_is_ignored():
    source = "/- sorry -/\ntheorem t : True := trivial\n"
    outcome = score_lean(source, checkout=Path("."), probe=_ok_probe, runner=_pass_runner)
    assert outcome.reward == 1.0


def test_kernel_sorry_ax_is_rejected():
    def runner(*_args):
        return {"ok": True, "axioms": ["propext", "sorryAx"], "detail": "depends on axioms: [sorryAx]"}

    outcome = score_lean(CLEAN, checkout=Path("."), probe=_ok_probe, runner=runner)
    assert outcome.reward == 0.0
    assert "sorryAx" in outcome.receipt["reason"]


def test_new_kernel_axiom_is_rejected():
    def runner(*_args):
        return {"ok": True, "axioms": ["propext", "evilAx"], "detail": ""}

    outcome = score_lean(CLEAN, checkout=Path("."), probe=_ok_probe, runner=runner)
    assert outcome.reward == 0.0
    assert "evilAx" in outcome.receipt["reason"]


def test_pinned_baseline_axiom_is_allowed():
    def runner(*_args):
        return {"ok": True, "axioms": ["propext", "pinsker", "Classical.choice"], "detail": ""}

    outcome = score_lean(CLEAN, checkout=Path("."), probe=_ok_probe, runner=runner)
    assert outcome.reward == 1.0
    assert "pinsker" in outcome.receipt["baseline_axioms_used"]


def test_wrong_commit_and_timeout_score_zero():
    def bad_probe(_checkout):
        return {"ok": False, "reason": "checkout HEAD abc is not pinned " + PINNED_COMMIT}

    called = {}

    def runner(*_args):
        called["ran"] = True
        return {"ok": True, "axioms": []}

    mismatch = score_lean(CLEAN, checkout=Path("."), probe=bad_probe, runner=runner)
    assert mismatch.reward == 0.0
    assert PINNED_COMMIT in mismatch.receipt["reason"]
    assert called == {}

    def timeout_runner(*_args):
        return {"ok": False, "timed_out": True, "axioms": []}

    timed = score_lean(CLEAN, checkout=Path("."), probe=_ok_probe, runner=timeout_runner)
    assert timed.reward == 0.0
    assert timed.receipt["reason"] == "lean timed out"


def test_candidate_is_not_a_shell_command():
    source = "theorem t : True := trivial\n; rm -rf /\n"
    argv = lean_argv("lake", Path("/tmp/candidate.lean"))
    assert argv == ["lake", "env", "lean", "/tmp/candidate.lean"]
    assert all("rm" not in part for part in argv)
    assert source not in " ".join(argv)


def test_missing_toolchain_reports_absence(monkeypatch):
    monkeypatch.setattr("training.rlvr.lean.shutil.which", lambda _name: None)
    outcome = score_lean(CLEAN, checkout=Path("/tmp"), probe=_ok_probe)
    assert outcome.reward == 0.0
    assert outcome.receipt["reason"] == "lean/lake absent"
    assert outcome.receipt["verdict"] == "fail"


def test_cpu_path_skips_when_lean_or_lake_is_absent():
    if shutil.which("lean") is None or shutil.which("lake") is None:
        pytest.skip("lean/lake absent")
    checkout = ROOT.joinpath("..", "lutar-lean")
    if not checkout.is_dir():
        pytest.skip("lean/lake present but the pinned lutar-lean checkout is absent")
    outcome = score_lean(CLEAN, checkout=checkout, timeout=5)
    assert outcome.reward in (0.0, 1.0)
    assert outcome.receipt["verifier"] == VERIFIER


def test_trl_batch_returns_floats_and_receipts():
    batch = lean_reward(
        [CLEAN, "theorem t : True := by sorry\n"],
        checkout=Path("."),
        probe=_ok_probe,
        runner=_pass_runner,
    )
    assert list(batch) == [1.0, 0.0]
    assert [item["verdict"] for item in batch.receipts] == ["pass", "fail"]
    assert batch.receipts[0]["input_sha256"] == hashlib.sha256(CLEAN.encode()).hexdigest()


def test_chat_completion_shape_is_accepted():
    messages = [{"role": "assistant", "content": CLEAN}]
    outcome = score_lean(messages, checkout=Path("."), probe=_ok_probe, runner=_pass_runner)
    assert outcome.reward == 1.0
