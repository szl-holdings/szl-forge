import json
import subprocess
import pytest
from szl_model_lab.cli import main, parser
from szl_model_lab.training import verify_source

def test_plan_is_inert(capsys):
    assert main(["plan"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["training_started"] and not result["hf_publication"]

def test_train_requires_acknowledgement():
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--track", "router", "--data", "x", "--output", "y", "--source-revision", "a"*40])

def test_evaluation_requires_acknowledgement():
    with pytest.raises(SystemExit):
        parser().parse_args(["evaluate-test", "--artifact", "x", "--data", "y", "--output", "z"])

def test_floating_revision_rejected():
    with pytest.raises(ValueError, match="full_source_revision"):
        verify_source("main")

def test_source_identity_and_dirtiness(tmp_path, monkeypatch):
    import szl_model_lab.training as module
    package = tmp_path / "model-lab" / "src" / "szl_model_lab"
    package.mkdir(parents=True)
    p = package / "training.py"; p.write_text("# source fixture\n")
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()
    git("init"); git("config", "user.name", "Contract Test"); git("config", "user.email", "fixture@example.invalid")
    git("add", "."); git("commit", "-m", "test fixture")
    sha = git("rev-parse", "HEAD")
    monkeypatch.setattr(module, "__file__", str(p))
    result = verify_source(sha)
    assert result["verification"] == "GIT_CLEAN" and result["upstream_admission_verified"] is False
    with pytest.raises(ValueError, match="mismatch"):
        verify_source("0"*40)
    p.write_text("# changed\n")
    with pytest.raises(ValueError, match="dirty"):
        verify_source(sha)
