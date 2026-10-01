"""Dependency-light ownership guards; numerical execution belongs to the CPU lane."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TESTS = ("test_merge_lora_hardened.py", "test_merge_lora_numerics.py")


def workflow():
    # BaseLoader retains YAML's literal on key and prevents implicit bool coercion.
    return yaml.load((ROOT / ".github/workflows/model-candidates.yml").read_text(),
                     Loader=yaml.BaseLoader)


def test_tensor_suites_have_one_cpu_owner_not_silent_skips():
    for name in TESTS:
        path = ROOT / "candidate-tests" / name
        assert path.is_file(), f"required numerical test missing: {name}"
        assert not (ROOT / "tests" / name).exists(), "duplicate minimal-lane collection"
        text = path.read_text()
        assert "import torch" in text
        assert "importorskip" not in text and "pytest.mark.skip" not in text


def test_merger_source_and_regressions_trigger_both_admission_events():
    data = workflow()
    required = {"scripts/merge_lora_hardened.py", "candidate-tests/**",
                "tests/test_lora_ci_ownership.py", ".github/workflows/model-candidates.yml"}
    for event in ("pull_request", "push"):
        assert required <= set(data["on"][event]["paths"])
    assert data["on"]["push"]["branches"] == ["main"]


def test_cpu_matrix_runs_real_tests_and_keeps_failures_blocking():
    data = workflow()
    assert data["permissions"] == {"contents": "read"}
    job = data["jobs"]["cpu-contracts"]
    assert {(row["os"], row["python"]) for row in job["strategy"]["matrix"]["include"]} == {
        ("ubuntu-latest", "3.11"), ("windows-latest", "3.12")}
    assert job.get("continue-on-error", "false") == "false"
    numerical = [s for s in job["steps"] if "pytest -q candidate-tests" in s.get("run", "")]
    assert len(numerical) == 1
    assert numerical[0]["run"] == "python -m pytest -q candidate-tests --junitxml=reports/model-candidate-tests.xml"
    assert "if" not in numerical[0] and numerical[0].get("continue-on-error", "false") == "false"
    install = next(s["run"] for s in job["steps"] if s.get("name") == "Install isolated CPU test dependencies")
    assert "--index-url https://download.pytorch.org/whl/cpu" in install
    assert "candidate-tests/requirements.txt" in install
    receipts = [s for s in job["steps"] if s.get("with", {}).get("path") == "reports/model-candidate-tests.xml"]
    assert len(receipts) == 1 and receipts[0]["if"] == "always()"
    assert receipts[0]["with"]["if-no-files-found"] == "error"
