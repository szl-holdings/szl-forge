"""Stage-1 review regressions (logs/v2/review/STAGE1_REVIEW.md): one test per fixed defect.

SYNTHETIC; DEV data only.  Field names such as 'patient' are used only as KEYS with dummy
numeric values.  Nothing under v2/data/sealed/ is opened and no registered split is read.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from v2.research import freeze, metrics, registry
from v2.research.gates import VALID_FEATURES
from v2.tests import _support as S

K = S.kernel()
VALID_TXT = json.dumps({"features": VALID_FEATURES})
INNER = json.dumps(VALID_FEATURES)[1:-1]


def _write_pair(directory: Path, artifact: dict, receipt: dict) -> tuple[Path, Path]:
    """Write a model and a receipt whose model_sha256 is regenerated consistently."""
    directory.mkdir(parents=True, exist_ok=True)
    model_bytes = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    receipt = dict(receipt, model_sha256=hashlib.sha256(model_bytes).hexdigest())
    model, rec = directory / "model.json", directory / "artifact_receipt.json"
    model.write_bytes(model_bytes)
    rec.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model, rec


class _Frozen(unittest.TestCase):
    def setUp(self):
        self._tmp = S.TempDir()
        self.tmp = self._tmp.__enter__()
        self.info = S.freeze_dev(self.tmp / "art")
        self.model = Path(self.info["model_path"])
        self.receipt_path = Path(self.info["receipt_path"])
        self.artifact = json.loads(self.model.read_text(encoding="utf-8"))
        self.receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def cli(self, model: Path, receipt: Path, data: bytes) -> tuple[int, str, str]:
        path = self.tmp / "input.json"
        path.write_bytes(data)
        err, out = io.StringIO(), io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            code = K.main(["--model", str(model), "--receipt", str(receipt), "--input", str(path)])
        return code, out.getvalue(), err.getvalue()


class ArtifactHardeningTest(_Frozen):
    # A1: unhashable provenance used to raise TypeError (stack trace, exit 1)
    def test_unhashable_stop_reason_is_refused_cleanly(self):
        for bad in (["CONVERGED"], {"x": 1}, 7, None):
            with self.subTest(stop_reason=bad):
                art = json.loads(json.dumps(self.artifact))
                art["training"]["stop_reason"] = bad
                model, rec = _write_pair(self.tmp / f"sr_{type(bad).__name__}", art, self.receipt)
                with self.assertRaises(K.ModelArtifactError) as ctx:
                    K.OpsHealthKernel(model, rec)
                self.assertIn("training stop provenance invalid", str(ctx.exception))
                code, out, err = self.cli(model, rec, VALID_TXT.encode("utf-8"))
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertFalse(json.loads(err)["ok"])

    def test_cli_subprocess_never_prints_a_traceback(self):
        art = json.loads(json.dumps(self.artifact))
        art["training"]["stop_reason"] = ["CONVERGED"]
        model, rec = _write_pair(self.tmp / "sub", art, self.receipt)
        inp = self.tmp / "valid.json"
        inp.write_text(VALID_TXT, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-B", str(registry.KERNEL_PATH), "--model", str(model), "--receipt",
             str(rec), "--input", str(inp)],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(json.loads(proc.stderr)["ok"])

    def test_unexpected_internal_error_is_a_refusal_without_traceback(self):
        with mock.patch.object(K, "OpsHealthKernel", side_effect=RuntimeError("boom")):
            code, out, err = self.cli(self.model, self.receipt_path, VALID_TXT.encode("utf-8"))
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(err), {"ok": False, "error": "internal error (RuntimeError); refusing"})

    # A2: registered constants are pinned
    def test_alpha_must_be_registered(self):
        for alpha in (0.05, 0.2, 0.5, 0.1000000001):
            art = json.loads(json.dumps(self.artifact))
            art["conformal"]["alpha"] = alpha
            with self.subTest(alpha=alpha), self.assertRaises(K.ModelArtifactError) as ctx:
                K.validate_artifact(art)
            self.assertIn("conformal.alpha must be the registered 0.10", str(ctx.exception))
        self.assertEqual(K.validate_artifact(self.artifact)["conformal"]["alpha"], 0.1)

    def test_l2_must_be_on_registered_grid(self):
        for l2 in (0.5, 0.02, 1e-5, 0.0011):
            art = json.loads(json.dumps(self.artifact))
            art["training"]["l2"] = l2
            with self.subTest(l2=l2), self.assertRaises(K.ModelArtifactError) as ctx:
                K.validate_artifact(art)
            self.assertIn("registered grid", str(ctx.exception))
        for l2 in (0, 0.0, 1e-4, 1e-3, 1e-2):
            art = json.loads(json.dumps(self.artifact))
            art["training"]["l2"] = l2
            with self.subTest(l2=l2):
                K.validate_artifact(art)

    # A3: preregistration digest pinned in the kernel
    def test_receipt_preregistration_digest_is_pinned(self):
        self.assertEqual(K.PREREGISTRATION_SHA256, registry.PREREG_SHA256)
        self.assertEqual(K.PREREGISTRATION_SHA256, S.sha256_file(registry.PREREG_PATH))
        model, rec = _write_pair(self.tmp / "prereg", self.artifact,
                                 dict(self.receipt, preregistration_sha256="0" * 64))
        with self.assertRaises(K.ModelArtifactError) as ctx:
            K.OpsHealthKernel(model, rec)
        self.assertIn("not the registered protocol", str(ctx.exception))


class InputHardeningTest(_Frozen):
    # A4: Unicode look-alike keys are refused AS PROHIBITED (not only by the schema check)
    def test_unicode_folded_prohibited_keys(self):
        scorer = K.ArtifactScorer(self.artifact)
        keys = {
            "fullwidth_lower": "ｐａｔｉｅｎｔ",
            "fullwidth_upper": "ＰＡＴＩＥＮＴ",
            "zero_width_space": "pat​ient",
            "zero_width_joiner": "sp‍ecimen",
            "word_joiner": "⁠hl7",
            "soft_hyphen": "m­rn",
            "fullwidth_token": "ｆｈｉｒ_port",
            "nbsp_separator": "order id",
        }
        for label, key in keys.items():
            with self.subTest(key=label), self.assertRaises(K.ModelInputError) as ctx:
                scorer.advise({"features": {**VALID_FEATURES, key: 0}})
            self.assertIn("prohibited non-operational field", str(ctx.exception))
        # cross-script homoglyphs do not fold; the exact eight-field schema still refuses them
        with self.assertRaises(K.ModelInputError) as ctx:
            scorer.advise({"features": {**VALID_FEATURES, "раtient": 0}})
        self.assertIn("feature schema mismatch", str(ctx.exception))
        for name in K.FEATURE_NAMES:
            self.assertIsNone(K.prohibited_match(name))

    def test_look_alike_and_padded_feature_names_are_refused(self):
        scorer = K.ArtifactScorer(self.artifact)
        for bad in ("tls_enabled ", " tls_enabled", "tls​_enabled", "ｔls_enabled",
                    "TLS_ENABLED"):
            features = dict(VALID_FEATURES)
            features[bad] = features.pop("tls_enabled")
            with self.subTest(name=ascii(bad)), self.assertRaises(K.ModelInputError):
                scorer.advise({"features": features})

    # duplicate keys refused at every level (v1 accepted them)
    def test_duplicate_keys_refused_everywhere(self):
        cases = {
            "inner_same": '{"features": {' + INNER + ', "listener_running": 1}}',
            "inner_conflicting": '{"features": {' + INNER + ', "queue_utilization": 0.99}}',
            "wrapper": '{"features": {' + INNER + '}, "features": {' + INNER + '}}',
            "nested": '{"features": {' + INNER + ', "x": {"a": 0, "a": 1}}}',
        }
        for name, text in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(K.ModelInputError) as ctx:
                    K.loads_strict(text, K.ModelInputError, allow_nonfinite=True)
                self.assertIn("duplicate JSON key", str(ctx.exception))
                code, out, err = self.cli(self.model, self.receipt_path, text.encode("utf-8"))
                self.assertEqual((code, out), (2, ""))

    def test_numeric_edge_cases(self):
        refused = {
            "1e308": VALID_TXT.replace('"seconds_since_last_success": 14.0', '"seconds_since_last_success": 1e308'),
            "-1e308": VALID_TXT.replace('"seconds_since_last_success": 14.0', '"seconds_since_last_success": -1e308'),
            "1e400": VALID_TXT.replace('"seconds_since_last_success": 14.0', '"seconds_since_last_success": 1e400'),
            "string_number": VALID_TXT.replace('"consecutive_failures": 0', '"consecutive_failures": "0"'),
            "neg_zero_binary": VALID_TXT.replace('"tls_enabled": 1', '"tls_enabled": -0.0'),
            "float_binary": VALID_TXT.replace('"tls_enabled": 1', '"tls_enabled": 1.0'),
            "negative_subnormal": VALID_TXT.replace('"queue_utilization": 0.12', '"queue_utilization": -5e-324'),
        }
        for name, text in refused.items():
            with self.subTest(case=name):
                self.assertNotEqual(text, VALID_TXT)
                code, out, err = self.cli(self.model, self.receipt_path, text.encode("utf-8"))
                self.assertEqual((code, out), (2, ""))
                self.assertFalse(json.loads(err)["ok"])

    # A6: -0.0 is the value 0; the advisory bytes equal those of +0.0 and carry no negative zero
    def test_negative_zero_is_canonical(self):
        zero = VALID_TXT.replace('"queue_utilization": 0.12', '"queue_utilization": 0.0')
        neg = VALID_TXT.replace('"queue_utilization": 0.12', '"queue_utilization": -0.0')
        c1, out_zero, _ = self.cli(self.model, self.receipt_path, zero.encode("utf-8"))
        c2, out_neg, _ = self.cli(self.model, self.receipt_path, neg.encode("utf-8"))
        self.assertEqual((c1, c2), (0, 0))
        self.assertEqual(out_zero, out_neg)
        off = {**VALID_FEATURES, "tls_enabled": 0, "listener_running": False}
        code, out, _ = self.cli(self.model, self.receipt_path, json.dumps({"features": off}).encode("utf-8"))
        self.assertEqual(code, 0)
        self.assertNotIn("-0.0", out)

    def test_bom_deep_nesting_and_size_are_refused(self):
        cases = {
            "bom": b"\xef\xbb\xbf" + VALID_TXT.encode("utf-8"),
            "deep_list": ('{"features": {' + INNER + ', "x": ' + "[" * 20000 + "]" * 20000 + "}}").encode("utf-8"),
            "deep_dict": ('{"features": {' + INNER + ', "x": ' + '{"a": ' * 200 + "0" + "}" * 200 + "}}").encode("utf-8"),
            "over_limit": (VALID_TXT + " " * (K.MAX_INPUT_BYTES - len(VALID_TXT) + 1)).encode("utf-8"),
        }
        for name, data in cases.items():
            with self.subTest(case=name):
                code, out, err = self.cli(self.model, self.receipt_path, data)
                self.assertEqual((code, out), (2, ""))
                self.assertFalse(json.loads(err)["ok"])
        at_limit = (VALID_TXT + " " * (K.MAX_INPUT_BYTES - len(VALID_TXT))).encode("utf-8")
        self.assertEqual(self.cli(self.model, self.receipt_path, at_limit)[0], 0)

    # A5: long keys are not echoed in full
    def test_long_keys_are_truncated_in_messages(self):
        scorer = K.ArtifactScorer(self.artifact)
        for key in ("k" * 10_000, "patient_" + "x" * 10_000):
            with self.subTest(prefix=key[:8]), self.assertRaises(K.ModelInputError) as ctx:
                scorer.advise({"features": {**VALID_FEATURES, key: 0}})
            self.assertLess(len(str(ctx.exception)), 300)
            self.assertIn("chars)", str(ctx.exception))
        with self.assertRaises(K.ModelInputError) as ctx:
            K.loads_strict('{"' + "d" * 5000 + '": 0, "' + "d" * 5000 + '": 1}', K.ModelInputError)
        self.assertLess(len(str(ctx.exception)), 300)
        # short keys are still shown verbatim (probe fragments rely on it)
        with self.assertRaises(K.ModelInputError) as ctx:
            scorer.advise({"features": {**VALID_FEATURES, "uptime_seconds": 1}})
        self.assertIn("unexpected=['uptime_seconds']", str(ctx.exception))


class EchoBoundTest(_Frozen):
    """A5 scope (fixer round): no refusal echoes an unbounded key, key list, path or value.

    Every case goes through the CLI and asserts exit 2, no traceback and a short stderr.
    """

    def assert_bounded_refusal(self, model: Path, rec: Path, data: bytes, fragment: str,
                               limit: int = 300) -> str:
        code, out, err = self.cli(model, rec, data)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertLessEqual(len(err.encode("utf-8")), limit, err[:200])
        self.assertFalse(json.loads(err)["ok"])
        self.assertIn(fragment, err)
        return err

    def test_receipt_extra_long_key(self):
        rec = dict(self.receipt, **{"r" * 100_000: 0})
        model, rec_path = _write_pair(self.tmp / "rk", self.artifact, rec)
        err = self.assert_bounded_refusal(model, rec_path, VALID_TXT.encode("utf-8"),
                                          "artifact receipt fields mismatch")
        self.assertIn("r" * K.MAX_SHOWN_KEY + "...(+99936 chars)", err)

    def test_model_extra_long_key_with_consistent_receipt(self):
        art = dict(self.artifact, **{"m" * 100_000: 0})
        model, rec = _write_pair(self.tmp / "mk", art, self.receipt)
        self.assert_bounded_refusal(model, rec, VALID_TXT.encode("utf-8"),
                                    "model artifact fields mismatch")

    def test_nested_artifact_extra_long_keys(self):
        cases = {
            "weights": lambda a: a["weights"].__setitem__("w" * 100_000, 0.0),
            "calibrator.params": lambda a: a["calibrator"]["params"].__setitem__("p" * 100_000, 0),
        }
        for label, change in cases.items():
            with self.subTest(label=label):
                art = json.loads(json.dumps(self.artifact))
                change(art)
                model, rec = _write_pair(self.tmp / f"nk_{label}", art, self.receipt)
                self.assert_bounded_refusal(model, rec, VALID_TXT.encode("utf-8"),
                                            f"{label} fields mismatch")

    def test_calibrator_kind_value_is_not_echoed(self):
        for name, kind in (("long_text", "k" * 100_000), ("big_list", ["k" * 100] * 1000),
                           ("short_text", "sigmoid")):
            with self.subTest(kind=name):
                art = json.loads(json.dumps(self.artifact))
                art["calibrator"]["kind"] = kind
                model, rec = _write_pair(self.tmp / f"ck_{name}", art, self.receipt)
                with self.assertRaises(K.ModelArtifactError) as ctx:
                    K.OpsHealthKernel(model, rec)
                self.assertLess(len(str(ctx.exception)), 120)
                err = self.assert_bounded_refusal(model, rec, VALID_TXT.encode("utf-8"),
                                                  "unsupported calibrator kind (not shown)")
                self.assertNotIn("kk", err)
                self.assertNotIn("sigmoid", err)
        with self.assertRaises(K.ModelArtifactError) as ctx:
            K.apply_calibrator({"kind": "k" * 100_000, "params": {}}, 0.5)
        self.assertEqual(str(ctx.exception), "unknown calibrator kind")

    def test_many_extra_keys_are_counted_not_listed(self):
        rec = dict(self.receipt, **{f"r{i:05d}": 0 for i in range(20_000)})
        model, rec_path = _write_pair(self.tmp / "many_r", self.artifact, rec)
        self.assert_bounded_refusal(model, rec_path, VALID_TXT.encode("utf-8"),
                                    f"'...(+{20_000 - K.MAX_SHOWN_KEYS} more)'")
        art = dict(self.artifact, **{f"m{i:05d}": 0 for i in range(20_000)})
        model, rec_path = _write_pair(self.tmp / "many_m", art, self.receipt)
        self.assert_bounded_refusal(model, rec_path, VALID_TXT.encode("utf-8"),
                                    f"'...(+{20_000 - K.MAX_SHOWN_KEYS} more)'")
        data = json.dumps({"features": {**VALID_FEATURES,
                                        **{f"x{i:05d}": 0 for i in range(60_000)}}}).encode("utf-8")
        self.assertLessEqual(len(data), K.MAX_INPUT_BYTES)
        self.assert_bounded_refusal(self.model, self.receipt_path, data,
                                    f"'...(+{60_000 - K.MAX_SHOWN_KEYS} more)'")

    def test_deep_path_of_long_keys_is_bounded(self):
        for name, leaf, depth, fragment in (("depth", 0, 40, "input nesting exceeds"),
                                            ("text", "text", 30, "text values are not accepted")):
            with self.subTest(case=name):
                nested = leaf
                for _ in range(depth):
                    nested = {"n" * 1000: nested}
                data = json.dumps({"features": {**VALID_FEATURES, "n" * 1000: nested}}).encode("utf-8")
                err = self.assert_bounded_refusal(self.model, self.receipt_path, data, fragment,
                                                  limit=K.MAX_SHOWN_PATH + 200)
                self.assertIn("chars)", err)

    def test_refusal_line_is_byte_bounded_ascii_json(self):
        for message in ("short refusal", "", "a" * 5000, "\U0001F600" * 5000, "\x1b" * 5000,
                        '"\\' * 5000, "é" * 959 + "x" * 200):
            with self.subTest(prefix=message[:4], length=len(message)):
                line = K._refusal_line(message)
                self.assertLessEqual(len(line.encode("utf-8")), K.MAX_SHOWN_ERROR)
                line.encode("ascii")  # JSON-escaped: never raw non-ASCII or control bytes
                parsed = json.loads(line)
                self.assertEqual(set(parsed), {"error", "ok"})
                self.assertIs(parsed["ok"], False)
                if len(message) > K.MAX_SHOWN_ERROR:
                    self.assertIn("chars)", parsed["error"])
        # messages that fit are shown unchanged
        for message in ("short refusal", "", "b" * (K.MAX_SHOWN_ERROR - 64)):
            self.assertEqual(json.loads(K._refusal_line(message))["error"], message)

    def test_cli_subprocess_receipt_extra_long_key_stderr_bytes(self):
        rec = dict(self.receipt, **{"r" * 100_000: 0})
        model, rec_path = _write_pair(self.tmp / "sub_rk", self.artifact, rec)
        inp = self.tmp / "valid.json"
        inp.write_text(VALID_TXT, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-B", str(registry.KERNEL_PATH), "--model", str(model), "--receipt",
             str(rec_path), "--input", str(inp)],
            capture_output=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, b"")
        self.assertNotIn(b"Traceback", proc.stderr)
        self.assertLessEqual(len(proc.stderr), 300)
        self.assertFalse(json.loads(proc.stderr.decode("ascii"))["ok"])


class InvalidBaselineReportTest(unittest.TestCase):
    # B1: an INVALID_BASELINE report must not allow sensitivity/specificity to be solved for
    def test_invalid_report_withholds_accuracy(self):
        y = [1] * 40 + [0] * 60
        d = [1] * 25 + [0] * 15 + [1] * 10 + [0] * 50
        invalid = metrics.baseline_validity(y, d, receipt_verified=True, probes_refused=3, probes_total=4)
        report = metrics.binary_report(y, d, invalid)
        self.assertEqual(report["per_class_rates"], metrics.INVALID_BASELINE)
        self.assertNotIn("accuracy", report)
        self.assertNotIn("confusion", report)
        self.assertEqual(set(report), {"rows", "balanced_accuracy", "prevalence", "baseline_validity",
                                       "per_class_rates"})
        valid = metrics.baseline_validity(y, d, receipt_verified=True, probes_refused=4, probes_total=4)
        report = metrics.binary_report(y, d, valid)
        self.assertEqual(report["accuracy"], 75 / 100)
        self.assertEqual(report["confusion"], {"tp": 25, "fp": 10, "tn": 50, "fn": 15})


class OriginVerificationTest(unittest.TestCase):
    """A3: freeze.verify_origin catches an altered source commit (hermetic temp git repo)."""

    def _git(self, repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=oac-test", "-c", "user.email=oac-test@localhost",
             "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
            check=True, capture_output=True, text=True,
        ).stdout

    def test_origin_is_verified_against_committed_blobs(self):
        with S.TempDir() as tmp:
            repo = tmp / "repo"
            for _key, rel in freeze.ORIGIN_BLOBS:
                target = repo / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(registry.ROOT / rel, target)
            self._git(repo, "init", "-q")
            self._git(repo, "add", *[rel for _k, rel in freeze.ORIGIN_BLOBS])
            self._git(repo, "commit", "-q", "-m", "fixture")
            head = self._git(repo, "rev-parse", "HEAD").strip()
            model_bytes = freeze.canonical_model_bytes(S.candidate())
            receipt = freeze.build_receipt(model_bytes, commit=head, tree_clean=True, selection=S.SELECTION)
            result = freeze.verify_origin(receipt, root=repo)
            self.assertEqual(result["commit"], head)
            self.assertEqual(set(result["verified"]), {k for k, _r in freeze.ORIGIN_BLOBS})
            # an altered (well-formed) source commit is caught
            altered = json.loads(json.dumps(receipt))
            altered["source"]["commit"] = "f" * 40
            with self.assertRaises(freeze.OriginMismatch):
                freeze.verify_origin(altered, root=repo)
            # a commit whose generator differs from the receipt's digest is caught
            gen = repo / "v2" / "research" / "generator.py"
            gen.write_bytes(gen.read_bytes() + b"# drift\n")
            self._git(repo, "commit", "-q", "-am", "drift")
            drifted = json.loads(json.dumps(receipt))
            drifted["source"]["commit"] = self._git(repo, "rev-parse", "HEAD").strip()
            with self.assertRaises(freeze.OriginMismatch) as ctx:
                freeze.verify_origin(drifted, root=repo)
            self.assertIn("generator_sha256", str(ctx.exception))
            # a tampered manifest digest is caught at the original commit
            tampered = json.loads(json.dumps(receipt))
            tampered["data_manifest_sha256"] = "0" * 64
            with self.assertRaises(freeze.OriginMismatch) as ctx:
                freeze.verify_origin(tampered, root=repo)
            self.assertIn("data_manifest_sha256", str(ctx.exception))

    def test_dev_receipt_with_fake_commit_fails_origin(self):
        with S.TempDir() as tmp:
            info = S.freeze_dev(tmp / "art")
            receipt = json.loads(Path(info["receipt_path"]).read_text(encoding="utf-8"))
            with self.assertRaises(freeze.OriginMismatch):
                freeze.verify_origin(receipt)


class ForbiddenWordTest(unittest.TestCase):
    """CONTRACT section 6: the banned word appears in v2 only as v1's authority-map key
    'clinical_decision' (kernel + tests asserting it).  Known exception pending amendment
    request AR-1: the upstream v1 source path recorded in PREREGISTRATION.receipt.json."""

    def test_banned_word_only_in_authority_key(self):
        needle = "clin" + "ical"
        allowed_token = needle + "_decision"
        offenders = []
        for path in sorted(registry.V2.rglob("*")):
            rel = path.relative_to(registry.ROOT).as_posix()
            if not path.is_file() or rel.startswith("v2/data/") or path.suffix not in (".py", ".md", ".json", ".txt"):
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                low = line.lower()
                if needle not in low:
                    continue
                if low.count(needle) == low.count(allowed_token) and (
                    rel == "v2/ops-health/ops_health.py" or rel.startswith("v2/tests/")
                ):
                    continue
                if rel == "v2/PREREGISTRATION.receipt.json" and '"v1_generator_source"' in line:
                    continue  # AR-1 (v2/AMENDMENT_REQUESTS.md)
                offenders.append(f"{rel}:{lineno}")
        self.assertEqual(offenders, [])
        kernel_lines = [l for l in registry.KERNEL_PATH.read_text(encoding="utf-8").splitlines()
                        if needle in l.lower()]
        self.assertEqual([l.strip() for l in kernel_lines], [f'"{allowed_token}": False,'])


if __name__ == "__main__":
    unittest.main()
