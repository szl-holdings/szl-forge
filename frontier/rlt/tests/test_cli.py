import json
import subprocess
import sys

from frontier.rlt.runtime import canonical, digest


def test_cli_writes_measured_receipt_and_never_overwrites(tmp_path):
    dest = tmp_path / "run"
    result = subprocess.run([sys.executable, "-m", "frontier.rlt", "--output", str(dest)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads((dest / "receipt.json").read_text())
    signature = report.pop("receipt_sha256")
    assert digest(canonical(report)) == signature
    assert report["max_abs_restore_error"] == 0.0
    assert report["trained"] is False and report["production_qualified"] is False
    assert digest((dest / "checkpoint.safetensors").read_bytes()) == report["checkpoint_sha256"]
    assert "<script" not in (dest / "index.html").read_text()
    retry = subprocess.run([sys.executable, "-m", "frontier.rlt", "--output", str(dest)],
                           capture_output=True, text=True, timeout=30)
    assert retry.returncode != 0
