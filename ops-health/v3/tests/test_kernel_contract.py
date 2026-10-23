"""Kernel v3 contract tests (prereg s.6.7, frozen at registration). SYNTHETIC; DEV data only.

Required by s.6.7: the 76 probes (v2's 66 input probes and TT1-TT10); duplicate-key refusal in
inputs and in the artifact; the all-false authority map on ALERT, NO_ALERT and ABSTAIN; the
standard-library-only import check; receipt mismatch. Field names such as 'patient' are used
only as KEYS with dummy numeric values. Every artifact is a DEV candidate frozen into a
temporary directory with the TEST-ONLY source-state override.
"""

from __future__ import annotations

import ast
import copy
import io
import json
import math
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from v3.research import freeze, gates, registry
from v3.tests import _support as S

K = S.kernel()
VALID = dict(gates.VALID_FEATURES)
VALID_TXT = json.dumps({"features": VALID})
INNER = json.dumps(VALID)[1:-1]
TT_FRAGMENTS = {
    "TT1": "differs from the registered representation R",
    "TT2": "scale must be positive",
    "TT3": "scale must be positive",
    "TT4": "non-finite JSON constant NaN",
    "TT5": "non-finite JSON constant Infinity",
    "TT6": "differs from the registered representation R",
    "TT7": "transform_table.tls_enabled differs from the registered representation R",
    "TT8": "transform_table fields mismatch: missing=['seconds_since_last_success']",
    "TT9": "transform_table fields mismatch: missing=[], unexpected=['uptime_seconds']",
    "TT10": "duplicate JSON key 'seconds_since_last_success'",
}


class _Frozen(unittest.TestCase):
    def setUp(self):
        self._tmp = S.TempDir()
        self.tmp = self._tmp.__enter__()
        self.info = S.freeze_dev(self.tmp / "art")
        self.model = Path(self.info["model_path"])
        self.receipt_path = Path(self.info["receipt_path"])
        self.model_bytes = self.model.read_bytes()
        self.artifact = json.loads(self.model_bytes)
        self.receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def pair(self, name: str, model_bytes: bytes, receipt: dict | None = None, consistent: bool = True):
        receipt = self.receipt if receipt is None else receipt
        if consistent:
            return S.write_consistent_pair(self.tmp / name, model_bytes, receipt)
        return freeze.write_pair(self.tmp / name, model_bytes, receipt)

    def artifact_pair(self, name: str, artifact: dict, receipt: dict | None = None):
        data = (json.dumps(artifact, sort_keys=True, indent=2, allow_nan=True) + "\n").encode("utf-8")
        return self.pair(name, data, receipt)

    def assertLoadRefused(self, model, rec, fragment: str):
        with self.assertRaises(K.ModelArtifactError) as ctx:
            K.OpsHealthKernel(model, rec)
        self.assertIn(fragment, str(ctx.exception))

    def cli(self, model: Path, receipt: Path, data: bytes) -> tuple[int, str, str]:
        path = self.tmp / "input.json"
        path.write_bytes(data)
        err, out = io.StringIO(), io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            code = K.main(["--model", str(model), "--receipt", str(receipt), "--input", str(path)])
        return code, out.getvalue(), err.getvalue()


class FailClosedProbeTest(_Frozen):
    def test_probe_set_is_76(self):
        self.assertEqual(len(gates.INPUT_PROBES), 66)
        self.assertEqual(gates.TT_PROBE_IDS, tuple(f"TT{i}" for i in range(1, 11)))
        self.assertEqual(gates.N_PROBES, 76)
        from v2.research import gates as v2_gates

        self.assertIs(gates.INPUT_PROBES, v2_gates.FAILCLOSED_PROBES)

    def test_every_input_probe_is_refused_with_its_message(self):
        kernel = K.OpsHealthKernel(self.model, self.receipt_path)
        for probe_id, payload, fragment in gates.INPUT_PROBES:
            with self.subTest(probe=probe_id):
                with self.assertRaises(K.ModelInputError) as ctx:
                    kernel.advise(copy.deepcopy(payload))
                self.assertIn(fragment, str(ctx.exception))

    def test_every_transform_table_probe_is_refused_at_load(self):
        probes = gates.transform_table_probes(self.artifact)
        self.assertEqual([p[0] for p in probes], list(gates.TT_PROBE_IDS))
        for probe_id, model_bytes in probes:
            with self.subTest(probe=probe_id):
                model, rec = self.pair(probe_id, model_bytes)
                self.assertLoadRefused(model, rec, TT_FRAGMENTS[probe_id])
                with self.assertRaises(K.ModelArtifactError):
                    K.load_artifact_bytes(model_bytes)

    def test_failclosed_gate_counts_all_76(self):
        kernel = K.OpsHealthKernel(self.model, self.receipt_path)
        report = gates.run_failclosed(kernel.advise, K, kernel.artifact, self.receipt)
        self.assertEqual((report["refused"], report["total"], report["escaped"]), (76, 76, []))
        self.assertEqual(report["fraction"], 1.0)
        self.assertEqual(report["transform_table_probes"]["refused"], 10)

    def test_valid_input_is_accepted(self):
        out = K.OpsHealthKernel(self.model, self.receipt_path).advise({"features": dict(VALID)})
        self.assertIn(out["advisory"], ("ALERT", "NO_ALERT", "ABSTAIN"))
        self.assertEqual(set(out), {
            "schema", "advisory", "abstain_reason", "prediction_set", "operator_attention_score",
            "threshold_decision", "decision_threshold", "threshold_index", "score_semantics",
            "purpose", "authority", "normalized_feature_contributions", "coverage_statement"})
        self.assertEqual(out["schema"], "szl-oac/ops-health-advisory/v3")
        self.assertEqual(list(out["normalized_feature_contributions"]), list(K.FEATURE_NAMES))

    def test_exactly_the_eight_v2_fields(self):
        v2 = S.v2_kernel()
        self.assertEqual(K.FEATURE_SPECS, v2.FEATURE_SPECS)
        self.assertEqual(len(K.FEATURE_NAMES), 8)
        scorer = K.ArtifactScorer(self.artifact)
        for name in K.FEATURE_NAMES:
            payload = {"features": {k: v for k, v in VALID.items() if k != name}}
            with self.subTest(missing=name), self.assertRaises(K.ModelInputError) as ctx:
                scorer.advise(payload)
            self.assertIn(f"missing=['{name}']", str(ctx.exception))

    def test_prohibited_screen_runs_before_the_schema_check(self):
        scorer = K.ArtifactScorer(self.artifact)
        for replaced, name in (("queue_utilization", "patient_id"), ("tls_enabled", "HL7"),
                               ("consecutive_failures", "Order-ID"), ("listener_running", "result value"),
                               ("configuration_valid", "xMRNx"), ("ledger_integrity_ok", "FhirPort")):
            features = {k: v for k, v in VALID.items() if k != replaced}
            features[name] = 0
            with self.subTest(name=name), self.assertRaises(K.ModelInputError) as ctx:
                scorer.advise({"features": features})
            self.assertIn("prohibited non-operational field", str(ctx.exception))
            self.assertNotIn("schema mismatch", str(ctx.exception))

    def test_match_kinds_normalized_token_substring(self):
        cases = {"patient": "exact", "PATIENT": "exact", " Patient-ID ": "exact", "order id": "exact",
                 "source_hl7_port": "token", "last-order-count": "token", "queue name": "token",
                 "PatientCount": "substring", "xhl7x": "substring", "specimens": "substring"}
        for key, kind in cases.items():
            with self.subTest(key=key):
                self.assertEqual(K.prohibited_match(key), kind)
        for key in K.FEATURE_NAMES + ("sortorder", "resultant", "uptime_seconds"):
            self.assertIsNone(K.prohibited_match(key))

    def test_look_alike_keys_are_refused_as_prohibited(self):
        scorer = K.ArtifactScorer(self.artifact)
        keys = ("ｐａｔｉｅｎｔ", "pat​ient", "sp‍ecimen", "⁠hl7",
                "m­rn", "ｆｈｉｒ_port", "order id")
        for key in keys:
            with self.subTest(key=ascii(key)), self.assertRaises(K.ModelInputError) as ctx:
                scorer.advise({"features": {**VALID, key: 0}})
            self.assertIn("prohibited non-operational field", str(ctx.exception))
        with self.assertRaises(K.ModelInputError) as ctx:  # cross-script: schema check refuses
            scorer.advise({"features": {**VALID, "раtient": 0}})
        self.assertIn("feature schema mismatch", str(ctx.exception))
        for bad in ("tls_enabled ", " tls_enabled", "tls​_enabled", "ｔls_enabled", "TLS_ENABLED"):
            features = dict(VALID)
            features[bad] = features.pop("tls_enabled")
            with self.subTest(name=ascii(bad)), self.assertRaises(K.ModelInputError):
                scorer.advise({"features": features})

    def test_screen_matches_v2_kernel_on_every_probe_key(self):
        v2 = S.v2_kernel()
        names = set()
        for _pid, payload, _frag in gates.INPUT_PROBES:
            if isinstance(payload, dict) and isinstance(payload.get("features"), dict):
                names |= set(map(str, payload["features"]))
        for name in sorted(names):
            self.assertEqual(K.prohibited_match(name), v2.prohibited_match(name), name)


class StrictJsonTest(_Frozen):
    def test_duplicate_keys_in_inputs_are_refused(self):
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
                code, out, _err = self.cli(self.model, self.receipt_path, text.encode("utf-8"))
                self.assertEqual((code, out), (2, ""))

    def test_duplicate_keys_in_the_artifact_are_refused(self):
        text = self.model_bytes.decode("utf-8")
        cases = {
            "top_level": text.replace('{\n  "authority"', '{\n  "intercept": 0.0,\n  "authority"', 1),
            "weights": text.replace('"weights": {', '"weights": {"tls_enabled": 0.0, ', 1),
            "transform_table": text.replace('"transform_table": {',
                                            '"transform_table": {"tls_enabled": {"offset": 0.0, "scale": 1.0}, ', 1),
            "conformal": text.replace('"conformal": {', '"conformal": {"alpha": 0.1, ', 1),
        }
        for name, variant in cases.items():
            with self.subTest(case=name):
                self.assertNotEqual(variant, text)
                model, rec = self.pair(f"dup_{name}", variant.encode("utf-8"))
                self.assertLoadRefused(model, rec, "duplicate JSON key")

    def test_duplicate_keys_in_the_receipt_are_refused(self):
        text = freeze.canonical_json_bytes(self.receipt).decode("utf-8")
        variant = text.replace('{\n  "amendments_sha256"', '{\n  "attests": "x",\n  "amendments_sha256"', 1)
        self.assertNotEqual(variant, text)
        rec = self.tmp / "dup_receipt.json"
        rec.write_text(variant, encoding="utf-8")
        self.assertLoadRefused(self.model, rec, "duplicate JSON key")

    def test_non_finite_constants_in_the_artifact_are_refused(self):
        for name, token in (("nan", "NaN"), ("inf", "Infinity"), ("ninf", "-Infinity")):
            art = copy.deepcopy(self.artifact)
            art["weights"]["tls_enabled"] = float(token.replace("Infinity", "inf"))
            model, rec = self.artifact_pair(f"nf_{name}", art)
            with self.subTest(token=token):
                self.assertLoadRefused(model, rec, "non-finite JSON constant")


class AuthorityMapTest(_Frozen):
    def crafted(self, name: str, q0: float, q1: float):
        art = copy.deepcopy(self.artifact)
        art["conformal"]["q_hat"] = {"0": q0, "1": q1}
        model, rec = self.artifact_pair(name, art)
        return K.OpsHealthKernel(model, rec)

    def test_authority_map_is_v2s_constant(self):
        v2 = S.v2_kernel()
        self.assertEqual(K.AUTHORITY_MAP, v2.AUTHORITY_MAP)
        self.assertEqual(list(K.AUTHORITY_MAP), list(v2.AUTHORITY_MAP))
        self.assertEqual(len(K.AUTHORITY_MAP), 5)
        self.assertTrue(all(v is False for v in K.AUTHORITY_MAP.values()))

    def test_all_false_on_alert_no_alert_and_abstain(self):
        v2_map = S.v2_kernel().AUTHORITY_MAP
        # q_hat (0.5, 0.5): p >= 0.5 -> {1} ALERT, p < 0.5 -> {0} NO_ALERT;
        # (1.0, 1.0): always {0, 1} ABSTAIN BOTH; (0.0, 0.0): EMPTY unless p is exactly 0 or 1.
        low = dict(VALID)
        high = dict(VALID, listener_running=0, ledger_integrity_ok=0, configuration_valid=0,
                    consecutive_failures=20, queue_utilization=1.0)
        seen = {}
        for name, (q0, q1), features in (
            ("alert", (0.5, 0.5), high), ("no_alert", (0.5, 0.5), low),
            ("abstain_both", (1.0, 1.0), low), ("abstain_empty", (0.0, 0.0), low),
        ):
            kernel = self.crafted(name, q0, q1)
            out = kernel.advise({"features": features})
            seen[name] = (out["advisory"], out["abstain_reason"])
            with self.subTest(case=name):
                self.assertEqual(out["authority"], v2_map)
                self.assertTrue(all(v is False for v in out["authority"].values()))
        self.assertEqual(seen, {"alert": ("ALERT", None), "no_alert": ("NO_ALERT", None),
                                "abstain_both": ("ABSTAIN", "BOTH"), "abstain_empty": ("ABSTAIN", "EMPTY")})

    def test_returned_map_is_a_copy(self):
        kernel = K.OpsHealthKernel(self.model, self.receipt_path)
        out = kernel.advise({"features": dict(VALID)})
        out["authority"]["device_control"] = True
        self.assertFalse(kernel.advise({"features": dict(VALID)})["authority"]["device_control"])
        self.assertFalse(K.AUTHORITY_MAP["device_control"])


class ImportTest(unittest.TestCase):
    @staticmethod
    def imports(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    names.add("." * node.level + (node.module or ""))
                else:
                    names.add(node.module.split(".")[0])
        return names

    def test_kernel_imports_the_standard_library_only(self):
        names = self.imports(registry.KERNEL_PATH)
        self.assertTrue(names)
        self.assertEqual(sorted(n for n in names if n not in sys.stdlib_module_names), [])

    def test_research_code_imports_the_standard_library_and_the_research_tree_only(self):
        allowed = set(sys.stdlib_module_names) | {"v2", "v3", "."}
        for path in sorted((registry.V3 / "research").glob("*.py")) + sorted((registry.V3 / "tools").glob("*.py")):
            if path.name in ("v3_ots_stamp.py",):  # drafting stamp tool: uses the public client
                continue
            with self.subTest(file=path.name):
                names = self.imports(path)
                self.assertEqual(sorted(n for n in names if n not in allowed), [], path.name)


class ReceiptTest(_Frozen):
    def test_valid_pair_loads_and_binds_everything(self):
        kernel = K.OpsHealthKernel(self.model, self.receipt_path)
        r = kernel.receipt
        self.assertEqual(r["model_sha256"], S.sha256_file(self.model))
        self.assertEqual(r["kernel_sha256"], S.sha256_file(registry.KERNEL_PATH))
        self.assertEqual(r["kernel_sha256"], K.kernel_file_sha256())
        self.assertEqual(r["preregistration_sha256"], registry.PREREG_SHA256)
        self.assertEqual(r["reregistrations_sha256"], [registry.REREG_R1_SHA256])
        self.assertEqual(r["source"], {"repository": "https://github.com/szl-holdings/szl-forge",
                                       "commit": S.FAKE_COMMIT, "tag": "oac-v3-fit", "tree_clean": True})
        self.assertEqual(r["attests"], "integrity and origin only; not quality")

    def test_model_byte_flip_is_refused(self):
        data = bytearray(self.model_bytes)
        data[10] ^= 1
        model, rec = self.pair("flip", bytes(data), consistent=False)
        self.assertLoadRefused(model, rec, "model artifact hash does not match receipt")

    def test_kernel_self_hash_mismatch_is_refused(self):
        module, path = S.load_kernel_copy(self.tmp, mutate=lambda t: t + "\n# drift\n")
        with self.assertRaises(module.ModelArtifactError) as ctx:
            module.OpsHealthKernel(self.model, self.receipt_path)
        self.assertIn("kernel sha256 does not match receipt", str(ctx.exception))
        rebound = S.rebind_receipt(self.receipt_path, path, self.tmp / "rebound.json")
        module.OpsHealthKernel(self.model, rebound)

    def test_generator_digest_mismatch_is_refused(self):
        receipt = dict(self.receipt, generator_sha256="0" * 64)
        model, rec = self.pair("gen", self.model_bytes, receipt, consistent=False)
        self.assertLoadRefused(model, rec, "generator sha256 differs between model and receipt")

    def test_receipt_field_refusals(self):
        def changed(**fields):
            r = json.loads(json.dumps(self.receipt))
            for key, value in fields.items():
                if "." in key:
                    outer, inner = key.split(".")
                    r[outer][inner] = value
                else:
                    r[key] = value
            return r

        cases = {
            "schema v2": (changed(schema="szl-oac/ops-health-artifact-receipt/v2"), "receipt schema mismatch"),
            "metrics added": (dict(self.receipt, metrics={"ba": 0.99}), "artifact receipt fields mismatch"),
            "prereg": (changed(preregistration_sha256="0" * 64), "not the registered protocol"),
            "tree dirty": (changed(**{"source.tree_clean": False}), "not clean"),
            "repository": (changed(**{"source.repository": "https://example.invalid/x"}), "public szl-forge"),
            "short commit": (changed(**{"source.commit": "abc1234"}), "40-hex"),
            "tag": (changed(**{"source.tag": "oac-v3-frozen"}), "registered step tag"),
            "segment": (changed(**{"fit.segment": 2}), "does not match its step tag"),
            "rereg form": (changed(reregistrations_sha256=["XYZ"]), "reregistrations_sha256[0]"),
            "rereg type": (changed(reregistrations_sha256="0" * 64), "list of digests"),
            "amendments": (changed(amendments_sha256="nothex"), "amendments_sha256"),
            "trace": (changed(**{"fit.trace_sha256": None}), "trace_sha256"),
            "attests": (changed(attests="quality"), "attestation text"),
        }
        for label, (receipt, fragment) in cases.items():
            with self.subTest(case=label):
                model, rec = self.pair(f"r_{label.replace(' ', '_')}", self.model_bytes, receipt, consistent=False)
                self.assertLoadRefused(model, rec, fragment)

    def test_kernel_prereg_pin_is_the_registered_digest(self):
        self.assertEqual(K.PREREGISTRATION_SHA256, registry.PREREG_SHA256)
        self.assertEqual(K.PREREGISTRATION_SHA256, S.sha256_file(registry.PREREG_PATH))
        self.assertEqual(registry.REREG_R1_SHA256, S.sha256_file(registry.REREG_R1_PATH))


class ArtifactPinTest(_Frozen):
    def refused(self, change, fragment):
        art = copy.deepcopy(self.artifact)
        change(art)
        with self.assertRaises(K.ModelArtifactError) as ctx:
            K.validate_artifact(art)
        self.assertIn(fragment, str(ctx.exception))

    def test_registered_constants_are_pinned(self):
        for l2 in (0.0, 1e-4, 1e-2, 0.0011):
            self.refused(lambda a, v=l2: a["training"].__setitem__("l2", v), "registered 1e-3")
        for alpha in (0.05, 0.2, 0.1000000001):
            self.refused(lambda a, v=alpha: a["conformal"].__setitem__("alpha", v), "registered 0.10")
        for index in (0, 100, 50.0, True, "50"):
            self.refused(lambda a, v=index: a.__setitem__("threshold_index", v), "threshold_index")
        self.refused(lambda a: a["calibrator"].__setitem__("kind", "isotonic"), "unsupported calibrator kind")
        self.refused(lambda a: a["generator"].__setitem__("version", "ops-health-generator/2.0.0"),
                     "generator version mismatch")
        self.refused(lambda a: a["generator"].__setitem__("namespace", "oac-ops-health-v2"),
                     "generator namespace mismatch")
        self.refused(lambda a: a.__setitem__("decision_threshold", 0.2), "model artifact fields mismatch")

    def test_transform_table_rule(self):
        T = K.canonical_transform_table()
        self.assertEqual(K.validate_transform_table(T),
                         {n: (float(o), float(s)) for n, (o, s) in K.TRANSFORM_TABLE.items()})
        ints = {n: {"offset": int(e["offset"]), "scale": int(e["scale"])} for n, e in T.items()}
        K.validate_transform_table(ints)  # numerically equal entries are the registered table
        bad = {
            "seconds 86400": ("seconds_since_last_success", "scale", 86400.0, "differs from the registered"),
            "zero": ("seconds_since_last_success", "scale", 0.0, "must be positive"),
            "negative": ("seconds_since_last_success", "scale", -900.0, "must be positive"),
            "nan": ("seconds_since_last_success", "scale", float("nan"), "not finite"),
            "inf": ("seconds_since_last_success", "scale", float("inf"), "not finite"),
            "offset": ("seconds_since_last_success", "offset", 1.0, "differs from the registered"),
            "tls": ("tls_enabled", "scale", 2.0, "differs from the registered"),
            "bool": ("queue_utilization", "scale", True, "must be numeric"),
            "text": ("queue_utilization", "scale", "1", "must be numeric"),
        }
        for label, (name, key, value, fragment) in bad.items():
            table = json.loads(json.dumps(T))
            table[name][key] = value
            with self.subTest(case=label), self.assertRaises(K.ModelArtifactError) as ctx:
                K.validate_transform_table(table)
            self.assertIn(fragment, str(ctx.exception))
        table = json.loads(json.dumps(T))
        table["seconds_since_last_success"]["unit"] = "s"
        with self.assertRaises(K.ModelArtifactError):
            K.validate_transform_table(table)

    def test_normalization_uses_representation_r(self):
        scorer = K.ArtifactScorer(self.artifact)
        vector = scorer.normalize(dict(VALID, seconds_since_last_success=1800.0, consecutive_failures=5,
                                       queue_utilization=0.5))
        self.assertEqual(vector[5], 2.0)
        self.assertEqual(vector[4], 0.25)
        self.assertEqual(vector[3], 0.5)
        with self.assertRaises(K.ModelInputError):
            scorer.normalize(dict(VALID, seconds_since_last_success=86400.5))
        self.assertEqual(scorer.normalize(dict(VALID, seconds_since_last_success=86400.0))[5], 96.0)


class CliTest(_Frozen):
    def test_cli_refuses_with_exit_code_2_and_accepts_valid_input(self):
        cases = {
            "nan_constant": VALID_TXT.replace('"queue_utilization": 0.12', '"queue_utilization": NaN'),
            "duplicate_key": '{"features": {"listener_running": 1, "listener_running": 1}}',
            "prohibited": json.dumps({"features": {**VALID, "specimen_id": 0}}),
            "not_json": "{",
            "bom": "﻿" + VALID_TXT,
        }
        for name, text in cases.items():
            with self.subTest(case=name):
                code, out, err = self.cli(self.model, self.receipt_path, text.encode("utf-8"))
                self.assertEqual((code, out), (2, ""))
                self.assertFalse(json.loads(err)["ok"])
        code, out, _err = self.cli(self.model, self.receipt_path, VALID_TXT.encode("utf-8"))
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertTrue(math.isfinite(result["operator_attention_score"]))
        self.assertTrue(all(v is False for v in result["authority"].values()))

    def test_cli_subprocess_never_prints_a_traceback(self):
        art = copy.deepcopy(self.artifact)
        art["training"]["stop_reason"] = ["CONVERGED"]
        model, rec = self.artifact_pair("sub", art)
        inp = self.tmp / "valid.json"
        inp.write_text(VALID_TXT, encoding="utf-8")
        proc = subprocess.run([sys.executable, "-I", "-B", str(registry.KERNEL_PATH), "--model", str(model),
                               "--receipt", str(rec), "--input", str(inp)],
                              capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(json.loads(proc.stderr)["ok"])

    def test_refusals_are_bounded(self):
        scorer = K.ArtifactScorer(self.artifact)
        for key in ("k" * 10_000, "patient_" + "x" * 10_000):
            with self.subTest(prefix=key[:8]), self.assertRaises(K.ModelInputError) as ctx:
                scorer.advise({"features": {**VALID, key: 0}})
            self.assertLess(len(str(ctx.exception)), 300)
        rec = dict(self.receipt, **{"r" * 100_000: 0})
        model, rec_path = self.pair("long_key", self.model_bytes, rec, consistent=False)
        code, out, err = self.cli(model, rec_path, VALID_TXT.encode("utf-8"))
        self.assertEqual((code, out), (2, ""))
        self.assertLessEqual(len(err.encode("utf-8")), 300)


class ForbiddenWordTest(unittest.TestCase):
    """The banned word appears in new v3 text only as the frozen v1 authority-map key, in the
    kernel. v3/evidence/ holds registered evidence published unchanged (s.14.2) and is not new text."""

    def test_banned_word_only_in_the_kernel_authority_key(self):
        needle = "clin" + "ical"
        allowed = needle + "_decision"
        offenders = []
        for path in sorted(registry.V3.rglob("*")):
            if not path.is_file() or path.suffix not in (".py", ".md", ".json", ".txt", ".yml"):
                continue
            rel = path.relative_to(registry.ROOT).as_posix()
            if rel.startswith("v3/evidence/"):
                continue  # registered evidence published byte-identically before registration (s.14.2)
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                low = line.lower()
                if needle not in low:
                    continue
                if rel == "v3/ops-health/ops_health.py" and line.strip() == f'"{allowed}": False,':
                    continue
                offenders.append(f"{rel}:{lineno}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
