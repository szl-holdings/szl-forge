"""Stage-1 review, lens (a): adversarial attack on the v2 kernel CLI and artifact validation.

SYNTHETIC; DEV data only.  Builds a temporary DEV artifact (tests' freeze_dev helper, TEST-ONLY
source-state override) in a temp directory, then runs the kernel CLI in a fresh subprocess for
every probe and records exit code, stdout/stderr prefixes and whether a Python traceback leaked.
Field names such as 'patient' appear only as KEYS with dummy numeric values.

Echo bound (fixer round, verified issue A5-scope): every refusal's stderr must be at most
ECHO_LIMIT bytes, whatever the attacker puts in keys or values of the input, the model or the
(non hash-bound) receipt; probes with ids starting "echo_" target this.

Usage (from ROOT):  PYTHONUTF8=1 py -3.12 -B logs/v2/review/kernel_attack.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from v2.research import registry  # noqa: E402
from v2.research.gates import VALID_FEATURES  # noqa: E402
from v2.tests import _support as S  # noqa: E402

KERNEL = registry.KERNEL_PATH
PY = [sys.executable, "-B"]
ECHO_LIMIT = 1100  # bytes of stderr for any refusal (kernel caps a refusal message at 1024 chars)


def run_cli(model: Path, receipt: Path, data: bytes, tmp: Path, name: str) -> dict:
    inp = tmp / f"in_{name}.json"
    inp.write_bytes(data)
    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.run(
        PY + [str(KERNEL), "--model", str(model), "--receipt", str(receipt), "--input", str(inp)],
        capture_output=True, env=env, timeout=120,
    )
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    return {
        "code": proc.returncode,
        "traceback": "Traceback" in err or "Traceback" in out,
        "raw_escape_in_stderr": "\x1b" in err,
        "stdout": out[:160].replace("\n", " "),
        "stderr": err[:220].replace("\n", " "),
        "stderr_len": len(err),
        "neg_zero_in_stdout": "-0.0" in out,
    }


def feats(**changes) -> dict:
    f = dict(VALID_FEATURES)
    f.update(changes)
    return f


def wrap(f) -> bytes:
    return json.dumps({"features": f}).encode("utf-8")


def text(s: str) -> bytes:
    return s.encode("utf-8")


VALID_TXT = json.dumps({"features": VALID_FEATURES})
INNER = json.dumps(VALID_FEATURES)[1:-1]


def input_probes() -> list[tuple[str, bytes, str]]:
    """(id, raw bytes, expectation 'refuse' or 'accept')."""
    p: list[tuple[str, bytes, str]] = []
    add = lambda i, b, e="refuse": p.append((i, b, e))  # noqa: E731
    add("control_valid", text(VALID_TXT), "accept")
    add("control_json_bools_for_binary", wrap(feats(listener_running=True, tls_enabled=False)), "accept")
    # CONTRACT 2.6 classes
    f = feats(); del f["tls_enabled"]; add("c26_missing_field", wrap(f))
    add("c26_extra_field", wrap(feats(uptime_seconds=5)))
    add("c26_nan", text(VALID_TXT.replace('"queue_utilization": 0.12', '"queue_utilization": NaN')))
    add("c26_inf", text(VALID_TXT.replace('"seconds_since_last_success": 14.0', '"seconds_since_last_success": Infinity')))
    add("c26_neg_inf", text(VALID_TXT.replace('"consecutive_failures": 0', '"consecutive_failures": -Infinity')))
    add("c26_out_of_range", wrap(feats(queue_utilization=1.5)))
    add("c26_string_for_number", wrap(feats(queue_utilization="0.5")))
    for name in ("patient", "hl7", "fhir", "specimen", "order", "result", "mrn"):
        add(f"c26_prohibited_{name}", wrap(feats(**{name: 0})))
    # duplicate keys
    add("dup_key_inner", text('{"features": {' + INNER + ', "listener_running": 1}}'))
    add("dup_key_inner_conflicting", text('{"features": {' + INNER + ', "queue_utilization": 0.99}}'))
    add("dup_key_wrapper", text('{"features": {' + INNER + '}, "features": {' + INNER + '}}'))
    # numeric edge cases
    add("num_1e308", wrap(feats(seconds_since_last_success=1e308)))
    add("num_neg_1e308", wrap(feats(seconds_since_last_success=-1e308)))
    add("num_1e400_literal", text(VALID_TXT.replace('"seconds_since_last_success": 14.0', '"seconds_since_last_success": 1e400')))
    add("num_neg_zero_continuous", text(VALID_TXT.replace('"queue_utilization": 0.12', '"queue_utilization": -0.0')), "accept")
    add("num_neg_zero_binary", text(VALID_TXT.replace('"tls_enabled": 1', '"tls_enabled": -0.0')))
    add("num_float_binary", text(VALID_TXT.replace('"tls_enabled": 1', '"tls_enabled": 1.0')))
    add("num_subnormal_negative", wrap(feats(queue_utilization=-5e-324)))
    add("num_huge_int_digits", text(VALID_TXT.replace('"consecutive_failures": 0', '"consecutive_failures": 1' + "0" * 5000)))
    add("num_int_10e400", text(VALID_TXT.replace('"consecutive_failures": 0', '"consecutive_failures": 1' + "0" * 400)))
    add("num_as_string", wrap(feats(consecutive_failures="3")))
    add("num_as_string_nan", wrap(feats(queue_utilization="NaN")))
    add("num_null", wrap(feats(queue_utilization=None)))
    # unicode look-alike and whitespace keys
    add("key_fullwidth_patient_extra", wrap(feats(**{"ｐａｔｉｅｎｔ": 0})))
    add("key_fullwidth_PATIENT_extra", wrap(feats(**{"ＰＡＴＩＥＮＴ": 0})))
    add("key_zero_width_space_in_patient", wrap(feats(**{"pat​ient": 0})))
    add("key_zero_width_joiner_in_patient_id", wrap(feats(**{"patient‍_id": 0})))
    add("key_soft_hyphen_in_mrn", wrap(feats(**{"m­rn": 0})))
    add("key_bom_in_specimen", wrap(feats(**{"﻿specimen": 0})))
    add("key_cyrillic_homoglyph_patient", wrap(feats(**{"раtient": 0})))
    f = feats(); v = f.pop("listener_running"); f["listener​_running"] = v
    add("key_zero_width_in_feature_name", wrap(f))
    f = feats(); v = f.pop("tls_enabled"); f["tls_enabled "] = v
    add("key_trailing_space_feature_name", wrap(f))
    f = feats(); v = f.pop("tls_enabled"); f["ｔls_enabled"] = v
    add("key_fullwidth_feature_name", wrap(f))
    add("key_trailing_space_patient", wrap(feats(**{"patient ": 0})))
    add("key_nbsp_patient", wrap(feats(**{"patient id": 0})))
    add("key_ansi_escape", wrap(feats(**{"\x1b[31mred": 0})))
    add("key_lone_surrogate", text('{"features": {' + INNER + ', "\\ud800": 0}}'))
    add("key_long_1k", wrap(feats(**{"k" * 1000: 0})))
    # structure
    add("deep_nesting_list_20000", text('{"features": {' + INNER + ', "x": ' + "[" * 20000 + "]" * 20000 + "}}"))
    add("deep_nesting_dict_200", text('{"features": {' + INNER + ', "x": ' + '{"a": ' * 200 + "0" + "}" * 200 + "}}"))
    pad = 1_000_000 - len(VALID_TXT)
    add("size_exactly_limit", text(VALID_TXT + " " * pad), "accept")
    add("size_limit_plus_1", text(VALID_TXT + " " * (pad + 1)))
    add("size_huge_5MB", text(VALID_TXT + " " * 5_000_000))
    add("bom_utf8", b"\xef\xbb\xbf" + text(VALID_TXT))
    add("utf16", VALID_TXT.encode("utf-16"))
    add("invalid_utf8", text(VALID_TXT)[:-2] + b"\xff}")
    add("empty", b"")
    add("json_null", b"null")
    add("json_list", text("[" + VALID_TXT + "]"))
    add("json_number", b"0")
    add("json_string", b'"features"')
    add("trailing_garbage", text(VALID_TXT + " x"))
    add("two_documents", text(VALID_TXT + VALID_TXT))
    add("js_comment", text("/*c*/" + VALID_TXT))
    add("single_quotes", text(VALID_TXT.replace('"', "'")))
    add("unwrapped", text(json.dumps(VALID_FEATURES)))
    add("nested_prohibited", wrap(feats(meta={"mrn": 0})))
    # echo bound: long keys, many keys, long keys along a deep path
    add("echo_input_extra_100000_char_key", wrap(feats(**{"i" * 100_000: 0})))
    add("echo_input_60000_extra_short_keys", wrap(feats(**{f"x{i:05d}": 0 for i in range(60_000)})))
    nested: object = 0
    for _ in range(40):
        nested = {"n" * 1000: nested}
    add("echo_input_long_keys_deep_path", wrap(feats(**{"n" * 1000: nested})))
    nested = "text"
    for _ in range(30):
        nested = {"t" * 1000: nested}
    add("echo_input_long_keys_path_to_text", wrap(feats(**{"t" * 1000: nested})))
    return p


def artifact_attacks(model_path: Path, receipt_path: Path, tmp: Path) -> list[tuple[str, Path, Path, str]]:
    """(id, model, receipt, expectation) with receipts regenerated consistently."""
    base_model = json.loads(model_path.read_text("utf-8"))
    base_receipt = json.loads(receipt_path.read_text("utf-8"))
    out = []

    def emit(name, model_bytes: bytes, receipt_change=None, expect="refuse", receipt_bytes=None):
        d = tmp / f"art_{name}"
        d.mkdir()
        m = d / "model.json"
        m.write_bytes(model_bytes)
        r = d / "artifact_receipt.json"
        if receipt_bytes is None:
            rec = json.loads(json.dumps(base_receipt))
            rec["model_sha256"] = hashlib.sha256(model_bytes).hexdigest()
            if receipt_change:
                receipt_change(rec)
            receipt_bytes = (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode("utf-8")
        r.write_bytes(receipt_bytes)
        out.append((name, m, r, expect))

    def mod(change) -> bytes:
        a = json.loads(json.dumps(base_model))
        change(a)
        return (json.dumps(a, indent=2, sort_keys=True) + "\n").encode("utf-8")

    emit("control_consistent", model_path.read_bytes(), expect="accept")
    emit("stop_reason_list", mod(lambda a: a["training"].__setitem__("stop_reason", ["CONVERGED"])))
    emit("stop_reason_dict", mod(lambda a: a["training"].__setitem__("stop_reason", {"x": 1})))
    emit("calibrator_kind_list", mod(lambda a: a["calibrator"].__setitem__("kind", ["platt"])))
    emit("conformal_method_list", mod(lambda a: a["conformal"].__setitem__("method", ["m"])))
    emit("weight_1e308", mod(lambda a: a["weights"].__setitem__("tls_enabled", 1e308)))
    emit("intercept_string", mod(lambda a: a.__setitem__("intercept", "1.0")))
    emit("threshold_off_grid", mod(lambda a: a.__setitem__("decision_threshold", 0.155)))
    emit("alpha_0_5_not_registered", mod(lambda a: a["conformal"].__setitem__("alpha", 0.5)))
    emit("l2_0_5_not_on_grid", mod(lambda a: a["training"].__setitem__("l2", 0.5)))
    txt = model_path.read_text("utf-8")
    emit("model_duplicate_key", txt.replace('"intercept":', '"intercept": 0.0,\n  "intercept":', 1).encode("utf-8"))
    emit("model_nan", txt.replace('"intercept": ', '"intercept": NaN, "x_": ', 1).encode("utf-8"))
    emit("model_bom", b"\xef\xbb\xbf" + model_path.read_bytes())
    emit("receipt_bom", model_path.read_bytes(),
         receipt_bytes=b"\xef\xbb\xbf" + receipt_path.read_bytes())
    emit("receipt_prereg_sha_altered", model_path.read_bytes(),
         receipt_change=lambda r: r.__setitem__("preregistration_sha256", "0" * 64))
    # The standalone kernel cannot run git: these two are accepted BY DESIGN at the kernel and
    # are caught by the research-side origin layer freeze.verify_origin (expectation "origin";
    # evidence: v2/tests/test_review_hardening.py::OriginVerificationTest).
    emit("receipt_manifest_sha_altered", model_path.read_bytes(),
         receipt_change=lambda r: r.__setitem__("data_manifest_sha256", "0" * 64), expect="origin")
    emit("receipt_commit_altered", model_path.read_bytes(),
         receipt_change=lambda r: r["source"].__setitem__("commit", "f" * 40), expect="origin")
    emit("receipt_stop_reason_list", model_path.read_bytes(),
         receipt_change=lambda r: r["selection"].__setitem__("stop_reason", ["x"]))
    emit("receipt_kernel_sha_altered", model_path.read_bytes(),
         receipt_change=lambda r: r.__setitem__("kernel_sha256", "0" * 64))
    emit("receipt_metrics_key_added", model_path.read_bytes(),
         receipt_change=lambda r: r.__setitem__("metrics", {"balanced_accuracy": 0.99}))
    # echo bound (verified issue: _exact_keys and the calibrator kind echoed in full)
    emit("echo_receipt_extra_100000_char_key", model_path.read_bytes(),
         receipt_change=lambda r: r.__setitem__("r" * 100_000, 0))
    emit("echo_receipt_20000_extra_short_keys", model_path.read_bytes(),
         receipt_change=lambda r: r.update({f"r{i:05d}": 0 for i in range(20_000)}))
    emit("echo_model_extra_100000_char_key", mod(lambda a: a.__setitem__("m" * 100_000, 0)))
    emit("echo_model_20000_extra_short_keys", mod(lambda a: a.update({f"m{i:05d}": 0 for i in range(20_000)})))
    emit("echo_weights_extra_100000_char_key", mod(lambda a: a["weights"].__setitem__("w" * 100_000, 0.0)))
    emit("echo_calibrator_kind_100000_char_value", mod(lambda a: a["calibrator"].__setitem__("kind", "k" * 100_000)))
    emit("echo_calibrator_kind_big_list", mod(lambda a: a["calibrator"].__setitem__("kind", ["k" * 100] * 1000)))
    emit("echo_calibrator_params_extra_100000_char_key",
         mod(lambda a: a["calibrator"]["params"].__setitem__("p" * 100_000, 0)))
    emit("receipt_duplicate_key", model_path.read_bytes(),
         receipt_bytes=receipt_path.read_text("utf-8").replace(
             '"attests":', '"attests": "x",\n  "attests":', 1).encode("utf-8"))
    return out


def main() -> int:
    print("# v2 kernel adversarial review (lens a) -- SYNTHETIC, DEV data only")
    print(f"python: {sys.version.split()[0]}  kernel: {KERNEL.relative_to(ROOT).as_posix()}")
    print(f"kernel_sha256: {hashlib.sha256(KERNEL.read_bytes()).hexdigest()}")
    print(f"echo bound: stderr_len <= {ECHO_LIMIT} bytes for every refusal")
    with tempfile.TemporaryDirectory(prefix="oac_v2_review_") as tmp_s:
        tmp = Path(tmp_s)
        info = S.freeze_dev(tmp / "art")
        model, receipt = Path(info["model_path"]), Path(info["receipt_path"])
        print(f"DEV artifact model_sha256: {info['model_sha256']}")
        bad = []
        print("\n## input probes (CLI subprocess)")
        print("id | expect | exit | traceback | outcome | stderr_len | stderr/stdout prefix")
        for name, data, expect in input_probes():
            r = run_cli(model, receipt, data, tmp, name)
            ok = (r["code"] == 0) if expect == "accept" else (r["code"] == 2)
            ok = ok and not r["traceback"] and not r["raw_escape_in_stderr"]
            ok = ok and r["stderr_len"] <= ECHO_LIMIT
            if not ok:
                bad.append(name)
            shown = r["stdout"] if r["code"] == 0 else r["stderr"]
            extra = " neg_zero_in_output" if r["neg_zero_in_stdout"] else ""
            print(f"{name} | {expect} | {r['code']} | {r['traceback']} | {'OK' if ok else 'FINDING'}{extra} | "
                  f"stderr_len={r['stderr_len']} | {shown[:200]}")
        print("\n## artifact/receipt attacks (consistent receipts; CLI subprocess with the valid input)")
        print("id | expect | exit | traceback | outcome | stderr_len | stderr prefix")
        for name, m, rc, expect in artifact_attacks(model, receipt, tmp):
            r = run_cli(m, rc, text(VALID_TXT), tmp, "art_" + name)
            ok = (r["code"] == 0) if expect in ("accept", "origin") else (r["code"] == 2)
            ok = ok and not r["traceback"] and r["stderr_len"] <= ECHO_LIMIT
            if not ok:
                bad.append("artifact:" + name)
            shown = r["stdout"] if r["code"] == 0 else r["stderr"]
            print(f"{name} | {expect} | {r['code']} | {r['traceback']} | {'OK' if ok else 'FINDING'} | "
                  f"stderr_len={r['stderr_len']} | {shown[:200]}")
        print(f"\nSUMMARY: {len(bad)} finding(s): {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
