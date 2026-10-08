from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.run_governed_live_verifier import (
    VerifierEntrypointError,
    bind_release_manifest,
    invoke,
    load_main,
    main,
)


ACTUAL_VERIFIER = Path(
    "spaces/szl-model-inference-lab/verify_governed_live.py"
)


def test_actual_governed_verifier_loads_without_executing_network() -> None:
    entrypoint = load_main(ACTUAL_VERIFIER)
    assert callable(entrypoint)


def test_late_helper_is_defined_before_main_is_invoked(tmp_path: Path) -> None:
    script = tmp_path / "late_helper.py"
    script.write_text(
        """from __future__ import annotations


def main(argv=None):
    assert argv == [\"--probe\", \"ok\"]
    return helper_defined_after_guard()


if __name__ == \"__main__\":
    raise SystemExit(main())


def helper_defined_after_guard():
    return 0
""",
        encoding="utf-8",
    )
    assert invoke(script, ["--probe", "ok"]) == 0


def test_missing_main_fails_closed(tmp_path: Path) -> None:
    script = tmp_path / "missing_main.py"
    script.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(VerifierEntrypointError, match="callable main"):
        load_main(script)


def test_non_integer_exit_fails_closed(tmp_path: Path) -> None:
    script = tmp_path / "invalid_exit.py"
    script.write_text(
        "def main(argv=None):\n    return True\n",
        encoding="utf-8",
    )
    with pytest.raises(VerifierEntrypointError, match="return an integer"):
        invoke(script, [])


def test_cli_binds_checkout_manifest_digest_to_governed_verifier(tmp_path: Path) -> None:
    script = tmp_path / "verify_governed_live.py"
    script.write_text(
        "def main(argv=None):\n"
        "    return 0 if '--expected-release-manifest-sha256' in argv else 7\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "release.json"
    manifest.write_bytes(b'{"schema":"fixture"}\r\n')
    expected = hashlib.sha256(b'{"schema":"fixture"}\n').hexdigest()
    assert bind_release_manifest(script, ["--probe", "ok"]) == [
        "--probe", "ok", "--expected-release-manifest-sha256", expected,
    ]
    assert main(["--script", str(script), "--probe", "ok"]) == 0


def test_cli_rejects_manifest_digest_override_or_absence(tmp_path: Path) -> None:
    script = tmp_path / "verify_governed_live.py"
    script.write_text("def main(argv=None):\n    return 0\n", encoding="utf-8")
    with pytest.raises(VerifierEntrypointError, match="unavailable"):
        bind_release_manifest(script, [])
    (tmp_path / "release.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(VerifierEntrypointError, match="override"):
        bind_release_manifest(script, ["--expected-release-manifest-sha256", "0" * 64])
