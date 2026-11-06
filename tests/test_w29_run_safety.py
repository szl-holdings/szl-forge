# SPDX-License-Identifier: Apache-2.0
"""Wave 29 must not SFT or push into live non-causal-LM Hub repos.

These tests do not train and do not contact the Hub. They lock the runner
policy: protected targets are skipped, a qlora tag on one of them refuses the
wave before the trainer starts, and only the three allowlisted candidate repos
are runnable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W29 = ROOT / "training" / "w29"

# Named by the overwrite finding. A later edit that drops one of these from
# the protected set fails here even if the prose still mentions it.
REQUIRED_PROTECTED = {
    "SZLHOLDINGS/YARQA-ATTN",
    "SZLHOLDINGS/chakana",
    "SZLHOLDINGS/tinku",
    "SZLHOLDINGS/MiniEmbed-Nano",
    "SZLHOLDINGS/Moons-Nano",
    "SZLHOLDINGS/ReceiptAgent-Nano",
    "SZLHOLDINGS/TinyKhipu-Nano",
    "SZLHOLDINGS/waman",
    "SZLHOLDINGS/KILLINCHU-EYE",
    "SZLHOLDINGS/qantu",
    "SZLHOLDINGS/oac-clinical-transport-health-v1",
    "SZLHOLDINGS/oac-ops-health-v2",
    "SZLHOLDINGS/oac-system-health-v1",
    "SZLHOLDINGS/szl-triage-linear-development-v1",
    "SZLHOLDINGS/szl-triage-retrain",
}


def _safety():
    spec = importlib.util.spec_from_file_location("w29_safety", W29 / "safety.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configs():
    rows = []
    for path in sorted(W29.glob("*.json")):
        rows.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return rows


def test_catalog_covers_every_protected_repo_and_only_allowlisted_runs():
    safety = _safety()
    assert REQUIRED_PROTECTED <= set(safety.PROTECTED_REPOS)
    configs = _configs()
    assert len(configs) == 19
    actions = {}
    targets = set()
    for _path, config in configs:
        action, _reason = safety.classify(config)
        assert action != "REFUSE"
        actions[config["name"]] = (action, config["target_repo"], config["method"])
        targets.add(config["target_repo"])
    assert targets == set(safety.PROTECTED_REPOS) | set(safety.WITHHELD_SFT_REPOS) | set(safety.RUNNABLE_REPOS)
    runnable = {name for name, (action, _target, _method) in actions.items() if action == "RUN"}
    assert runnable == {"willay-r1", "willay-r2", "forge-v3"}
    for name in ("receiptagent-nano", "tinykhipu-nano", "moons-nano", "miniembed-nano", "yarqa-attn"):
        action, target, method = actions[name]
        assert action == "SKIP"
        assert method != "qlora"
        assert target in safety.PROTECTED_REPOS
    action, target, method = actions["khipu-abstain-r3"]
    assert action == "SKIP"
    assert method != "qlora"
    assert target in safety.WITHHELD_SFT_REPOS
    assert target not in safety.PROTECTED_REPOS


def test_qlora_on_a_protected_repo_is_refused():
    safety = _safety()
    for target in REQUIRED_PROTECTED:
        action, reason = safety.classify(
            {"name": "probe", "target_repo": target, "method": "qlora", "datasets": ["x"]}
        )
        assert action == "REFUSE", target
        assert target in reason
    action, _reason = safety.classify(
        {
            "name": "khipu-abstain-r3",
            "target_repo": "SZLHOLDINGS/khipu-abstain-r3-candidate",
            "method": "qlora",
            "datasets": ["x"],
        }
    )
    assert action == "REFUSE"


def test_readme_matches_the_skip_and_run_sets():
    readme = (W29 / "README.md").read_text(encoding="utf-8")
    for name in (
        "willay-r1",
        "willay-r2",
        "forge-v3",
        "receiptagent-nano",
        "tinykhipu-nano",
        "moons-nano",
        "miniembed-nano",
        "yarqa-attn",
        "chakana",
        "tinku",
        "qantu",
        "waman",
        "killinchu-eye",
        "khipu-abstain-r3",
        "triage-linear-v1",
        "triage-retrain",
    ):
        assert name in readme
    assert "six causal-LM" not in readme
    assert "numpy" in readme
    assert "UNKNOWN" in readme
    run, skipped = readme.split("## Skipped — live non-causal-LM artifacts", 1)
    runnable_section = run.split("## Run", 1)[1]
    assert "receiptagent-nano" not in runnable_section
    assert "tinykhipu-nano" not in runnable_section
    assert "moons-nano" not in runnable_section
    assert "receiptagent-nano" in skipped
    assert "moons-nano" in skipped
    assert "tinykhipu-nano" in skipped


def _run(config_dir: Path, plan: bool, trainer: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["W29_CONFIG_DIR"] = str(config_dir)
    env["W29_PLAN"] = "1" if plan else "0"
    if trainer is not None:
        env["W29_TRAINER"] = str(trainer)
    return subprocess.run(
        ["bash", str(W29 / "run_w29.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_plan_skips_numpy_fixtures_and_does_not_start_a_trainer(tmp_path: Path):
    trainer = tmp_path / "trainer.py"
    trainer.write_text("import sys\nsys.stderr.write('TRAINER_STARTED\\n')\nraise SystemExit(0)\n", encoding="utf-8")
    result = _run(W29, plan=True, trainer=trainer)
    assert result.returncode == 0, result.stderr
    assert "TRAINER_STARTED" not in result.stderr
    assert "=== w29:" not in result.stdout
    for name in ("receiptagent-nano", "tinykhipu-nano", "moons-nano", "yarqa-attn", "qantu", "khipu-abstain-r3"):
        assert f"SKIP {name} " in result.stdout
    for name in ("willay-r1", "willay-r2", "forge-v3"):
        assert f"PLAN {name} " in result.stdout
    assert result.stdout.count("PLAN ") == 3


def test_runner_refuses_qlora_protected_target_before_the_trainer(tmp_path: Path):
    source = json.loads((W29 / "moons-nano.json").read_text(encoding="utf-8"))
    source["method"] = "qlora"
    (tmp_path / "moons-nano.json").write_text(json.dumps(source), encoding="utf-8")
    marker = tmp_path / "started"
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        textwrap.dedent(
            f"""\
            from pathlib import Path
            Path({str(marker)!r}).write_text("started", encoding="utf-8")
            raise SystemExit(0)
            """
        ),
        encoding="utf-8",
    )
    result = _run(tmp_path, plan=False, trainer=trainer)
    assert result.returncode != 0
    assert "REFUSE" in result.stderr
    assert "Moons-Nano" in result.stderr
    assert not marker.exists()
    assert "TRAINER" not in result.stdout


def test_runner_trains_only_the_allowlisted_repo(tmp_path: Path):
    allow = json.loads((W29 / "willay-r1.json").read_text(encoding="utf-8"))
    protected = json.loads((W29 / "yarqa-attn.json").read_text(encoding="utf-8"))
    (tmp_path / "willay-r1.json").write_text(json.dumps(allow), encoding="utf-8")
    (tmp_path / "yarqa-attn.json").write_text(json.dumps(protected), encoding="utf-8")
    calls = tmp_path / "calls.txt"
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        textwrap.dedent(
            f"""\
            import sys
            from pathlib import Path
            Path({str(calls)!r}).write_text("\\n".join(sys.argv), encoding="utf-8")
            raise SystemExit(0)
            """
        ),
        encoding="utf-8",
    )
    result = _run(tmp_path, plan=False, trainer=trainer)
    assert result.returncode == 0, result.stderr
    assert "SKIP yarqa-attn " in result.stdout
    assert "=== w29: willay-r1 ===" in result.stdout
    recorded = calls.read_text(encoding="utf-8")
    assert "--repo-id" in recorded
    assert "SZLHOLDINGS/willay-r1-candidate" in recorded
    assert "SZLHOLDINGS/YARQA-ATTN" not in recorded
    assert "--method" in recorded and "qlora" in recorded
