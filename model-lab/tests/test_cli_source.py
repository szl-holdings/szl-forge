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


@pytest.mark.parametrize("flags", [
    ("assume-unchanged",), ("skip-worktree",),
    ("assume-unchanged", "skip-worktree"),
])
@pytest.mark.parametrize("action", ["unchanged", "edit", "delete"])
@pytest.mark.parametrize("outside_package", [False, True])
def test_source_rejects_hidden_index_flags(tmp_path, monkeypatch, flags, action, outside_package):
    """Hidden index state cannot establish cleanliness, even without an edit."""
    import szl_model_lab.training as module
    package = tmp_path / "model-lab" / "src" / "szl_model_lab"
    package.mkdir(parents=True)
    source = package / "training.py"
    source.write_bytes(b"# source fixture\n")
    flagged = tmp_path / "policy flag check.txt" if outside_package else source
    if outside_package:
        flagged.write_bytes(b"# policy fixture\n")

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    git("init")
    git("config", "user.name", "Contract Test")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "commit.gpgsign", "false")  # Unsigned synthetic fixture only.
    git("config", "core.autocrlf", "false")
    git("add", ".")
    git("commit", "-m", "test fixture")
    sha = git("rev-parse", "HEAD")
    relative = flagged.relative_to(tmp_path).as_posix()
    for flag in flags:
        git("update-index", "--" + flag, "--", relative)
    if action == "edit":
        flagged.write_bytes(b"# hidden change\n")
    elif action == "delete":
        flagged.unlink()
    assert git("status", "--porcelain", "--untracked-files=all") == ""
    index = tmp_path / ".git" / "index"
    index_before = index.read_bytes()
    flags_before = git("ls-files", "-v", "-z")
    bytes_before = flagged.read_bytes() if flagged.exists() else None
    monkeypatch.setattr(module, "__file__", str(source))
    with pytest.raises(ValueError, match="hidden_index_flags"):
        verify_source(sha)
    assert index.read_bytes() == index_before
    assert git("ls-files", "-v", "-z") == flags_before
    assert (flagged.read_bytes() if flagged.exists() else None) == bytes_before


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_source_clean_git_line_endings_remain_supported(tmp_path, monkeypatch, line_ending):
    import szl_model_lab.training as module
    package = tmp_path / "model-lab" / "src" / "szl_model_lab"
    package.mkdir(parents=True)
    source = package / "training.py"
    source.write_bytes(b"# source fixture" + line_ending)

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    git("init")
    git("config", "user.name", "Contract Test")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "commit.gpgsign", "false")  # Unsigned synthetic fixture only.
    git("config", "core.autocrlf", "true")
    git("add", ".")
    git("commit", "-m", "test fixture")
    sha = git("rev-parse", "HEAD")
    monkeypatch.setattr(module, "__file__", str(source))
    assert verify_source(sha)["verification"] == "GIT_CLEAN"
