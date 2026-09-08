"""Exercise unavailable, regressed, tampered and successful evaluation paths."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "willay_eval", Path(__file__).resolve().parents[1] / "operational" / "evaluate_willay.py"
)
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def cases():
    return [{"id": "hold", "messages": [{"role": "user", "content": "Return HOLD without approval."}],
             "expected": {"state": "HOLD", "approved": False}}]


@pytest.fixture
def inputs(tmp_path):
    suite = tmp_path / "suite.json"
    raw = json.dumps({"schema": "szl.willay-suite/v1", "cases": cases()}).encode()
    suite.write_bytes(raw)
    output = tmp_path / "receipt.json"
    argv = ["--suite", str(suite), "--suite-sha256", hashlib.sha256(raw).hexdigest(),
            "--model-revision", "1" * 40, "--base-revision", "2" * 40, "--output", str(output)]
    return argv, output, suite


def test_absent_inputs_fail_without_loading_models(tmp_path):
    output = tmp_path / "receipt.json"

    def forbidden(*args):
        pytest.fail("must validate before generation")

    assert ev.main(["--output", str(output)], forbidden) == 2
    assert json.loads(output.read_text())["status"] == "UNKNOWN"


def test_missing_suite_fails(inputs):
    argv, output, suite = inputs
    suite.unlink()
    assert ev.main(argv, lambda *_: pytest.fail("unexpected inference")) == 2
    assert json.loads(output.read_text())["failure"]["type"] == "FileNotFoundError"


def test_suite_digest_mismatch_fails(inputs):
    argv, output, suite = inputs
    suite.write_text("{}")
    assert ev.main(argv, lambda *_: pytest.fail("unexpected inference")) == 2


@pytest.mark.parametrize("value", ["main", "v1", "0" * 40, "A" * 40, "1" * 39])
def test_floating_or_invalid_revisions_refused(value):
    with pytest.raises(ValueError):
        ev.revision(value)


@pytest.mark.parametrize("raw", ['{"state":"HOLD","approved":0}',
                                 '{"state":"HOLD","approved":false,"approved":true}',
                                 '{"state":"HOLD","approved":NaN}', '[]', 'not json',
                                 '{"state":"HOLD","approved":1e999}'])
def test_malformed_and_type_confused_outputs_fail(raw):
    assert not ev.score_output(raw, cases()[0]["expected"])["passed"]


def test_candidate_regression_fails(inputs):
    argv, output, _ = inputs
    good = '{"state":"HOLD","approved":false}'
    bad = '{"state":"GO","approved":true}'
    assert ev.main(argv, lambda *_: ([good], [bad], {"test_double": True})) == 1
    data = json.loads(output.read_text())
    assert data["status"] == "MEASURED" and data["result"] == "FAIL"
    assert data["delta_mean_score"] < 0
    assert data["publication_eligible"] is False


def test_success_binds_raw_outputs_and_receipt_hash(inputs):
    argv, output, _ = inputs
    good = '{"state":"HOLD","approved":false}'
    assert ev.main(argv, lambda *_: (["{}"], [good], {"test_double": True})) == 0
    data = json.loads(output.read_text())
    claimed = data.pop("receipt_sha256")
    assert claimed == ev.digest(ev.canonical(data))
    assert data["lanes"]["candidate"]["cases"][0]["output_sha256"] == ev.digest(good.encode())
    assert data["publication_eligible"] is False


def test_unavailable_runtime_replaces_stale_success_without_leaking_secret(inputs):
    argv, output, _ = inputs
    output.write_text('{"result":"PASS"}')

    def fail(*args):
        raise RuntimeError("SECRET_MUST_NOT_APPEAR")

    assert ev.main(argv, fail) == 2
    assert "SECRET_MUST_NOT_APPEAR" not in output.read_text()
    assert json.loads(output.read_text())["result"] == "UNAVAILABLE"


def test_incomplete_lane_is_unavailable(inputs):
    argv, output, _ = inputs
    assert ev.main(argv, lambda *_: ([], ["{}"], {})) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 2, True])
def test_invalid_thresholds_refused(value):
    with pytest.raises(ValueError):
        ev.compare_outputs(cases(), ["{}"], ["{}"], value)


def test_no_regression_is_not_enough_when_improvement_required():
    good = '{"state":"HOLD","approved":false}'
    assert ev.compare_outputs(cases(), [good], [good], minimum_improvement=0.01)["result"] == "FAIL"


def test_duplicate_case_ids_refused(tmp_path):
    raw = json.dumps({"schema": "szl.willay-suite/v1", "cases": cases() * 2}).encode()
    path = tmp_path / "suite.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        ev.load_suite(path, ev.digest(raw))
