from __future__ import annotations

from pathlib import Path

import pytest

from tools.run_governed_live_verifier import (
    VerifierEntrypointError,
    invoke,
    load_main,
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
