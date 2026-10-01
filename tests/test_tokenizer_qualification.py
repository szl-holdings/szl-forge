# SPDX-License-Identifier: Apache-2.0
"""Offline controls use deterministic engine doubles, not an upstream benchmark."""
import contextlib
import copy
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("forge_tokenizer_qualification",
    Path(__file__).resolve().parents[1] / "eval/tokenizer_qualification.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def cases():
    return [{"id": "empty", "text": "", "pair": None},
            {"id": "unicode", "text": "e\u0301 雪 🛰️\n", "pair": "paired"},
            {"id": "code", "text": 'print("<|end|>")\t', "pair": None}]


class Engine:
    def encode(self, text, pair=None, add_special_tokens=False):
        text = text + ("|" + pair if pair is not None else "")
        ids = [ord(c) for c in text]
        if add_special_tokens:
            ids = [1] + ids + [2]
        n = len(ids)
        return SimpleNamespace(ids=ids, tokens=[str(i) for i in ids], offsets=[(0, 0)] * n,
            attention_mask=[1] * n, special_tokens_mask=[int(i in (1, 2)) for i in ids],
            type_ids=[0] * n, word_ids=list(range(n)), sequence_ids=[0] * n, overflowing=[])

    def encode_batch(self, values, add_special_tokens=False):
        return [self.encode(*v, add_special_tokens=add_special_tokens) if isinstance(v, tuple)
                else self.encode(v, add_special_tokens=add_special_tokens) for v in values]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids if not skip_special_tokens or i not in (1, 2))


def runtime():
    return {"version": "0.23.2", "native_sha256": "a" * 64, "binding_sha256": "b" * 64,
            "python": "3.13.5", "system": "Linux", "machine": "x86_64", "host_digest": "c" * 64, "cpu_count": 8}


def seal(doc):
    doc["content_sha256"] = M.sha(M.packed({k: v for k, v in doc.items() if k != "content_sha256"}))
    return doc


def observation():
    data = {"schema": M.SCHEMA, "profile": M.PROFILE, "state": "RECORDED_NOT_QUALIFIED",
        "authority": M.AUTHORITY.copy(), "observed_at": datetime.now(timezone.utc).isoformat(),
        "source_revision_declared": "a" * 40, "observer_sha256": "d" * 64,
        "tokenizer_sha256": "e" * 64, "corpus_sha256": "f" * 64, "case_count": len(cases()),
        "threads_requested": 1, "repeats": 3, "runtime": runtime(), "load_ns": 10,
        "peak_process_rss_bytes": None, "workload": "REPEATED_FIXED_CORPUS_PYTHON_CALLS",
        **M.exercise(Engine(), cases(), 3)}
    data["timing"] = {"first_sweep_ns": [10] * 6, "warm_batch_ns": [100, 200, 300], "warm_decode_ns": [40, 50, 60]}
    return seal(data)


class InputTests(unittest.TestCase):
    def test_valid_corpus_with_empty_unicode_pair(self):
        doc = {"schema": "szl.tokenizer-corpus/v1", "cases": cases()}
        self.assertEqual(M.corpus(M.packed(doc)), cases())

    def test_duplicate_json_keys(self):
        with self.assertRaises(M.Invalid): M.strict(b'{"a":1,"a":2}')

    def test_nonfinite_constants_and_exponents(self):
        for value in (b'NaN', b'Infinity', b'1e999'):
            with self.subTest(value=value), self.assertRaises(M.Invalid): M.strict(value)

    def test_invalid_utf8_and_empty_json(self):
        for value in (b'\xff', b''):
            with self.assertRaises(M.Invalid): M.strict(value)

    def test_json_size_bound(self):
        with self.assertRaises(M.Invalid): M.strict(b' ' * (M.LIMIT + 1))

    def test_json_depth_error_is_sanitized(self):
        with self.assertRaises(M.Invalid): M.strict(b'[' * 2000 + b']' * 2000)

    def test_corpus_unknown_fields_rejected(self):
        doc = {"schema": "szl.tokenizer-corpus/v1", "cases": cases(), "token": "never log"}
        with self.assertRaises(M.Invalid): M.corpus(M.packed(doc))

    def test_duplicate_case_id(self):
        c = cases(); c[1]["id"] = c[0]["id"]
        with self.assertRaises(M.Invalid): M.corpus(M.packed({"schema": "szl.tokenizer-corpus/v1", "cases": c}))

    def test_no_empty_corpus(self):
        with self.assertRaises(M.Invalid): M.corpus(M.packed({"schema": "szl.tokenizer-corpus/v1", "cases": []}))

    def test_case_limit(self):
        c = [{"id": str(i), "text": "", "pair": None} for i in range(257)]
        with self.assertRaises(M.Invalid): M.corpus(M.packed({"schema": "szl.tokenizer-corpus/v1", "cases": c}))

    def test_case_types_and_surrogates(self):
        for key, value in (("text", []), ("pair", 2), ("text", "\ud800"), ("id", "private/path"), ("id", True)):
            c = cases(); c[0][key] = value
            with self.subTest(key=key), self.assertRaises(M.Invalid): M.corpus(M.packed({"schema": "szl.tokenizer-corpus/v1", "cases": c}))

    def test_text_byte_limit_not_character_limit(self):
        c = cases(); c[0]["text"] = "雪" * 12000
        with self.assertRaises(M.Invalid): M.corpus(M.packed({"schema": "szl.tokenizer-corpus/v1", "cases": c}))

    def test_read_requires_matching_exact_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "x"; p.write_bytes(b'{}')
            self.assertEqual(M.read_bound(p, M.sha(b'{}')), b'{}')
            with self.assertRaises(M.Invalid): M.read_bound(p, "0" * 64)


class SemanticsTests(unittest.TestCase):
    def test_all_fields_and_call_modes_recorded(self):
        out = M.exercise(Engine(), cases(), 3)
        self.assertEqual(len(out["rows"]), 12)
        self.assertTrue(all(set(r["fields"]) == set(M.FIELDS) for r in out["rows"]))
        self.assertEqual(len(out["timing"]["warm_batch_ns"]), 3)

    def test_fingerprints_do_not_emit_text_or_ids(self):
        out = M.exercise(Engine(), cases(), 3)
        self.assertNotIn('print(', json.dumps(out))
        self.assertTrue(all(M.hash_ok(r["fields"]["ids"]) for r in out["rows"]))

    def test_missing_offsets_is_unsupported_not_none(self):
        enc = Engine().encode("x"); del enc.offsets
        with self.assertRaises(M.Unavailable): M.fingerprint(Engine(), enc)

    def test_malformed_lengths_masks_offsets(self):
        for key, value in (("ids", [True]), ("offsets", [(2, 1)]), ("attention_mask", [2]),
                           ("word_ids", [False]), ("tokens", []), ("type_ids", [-1])):
            enc = Engine().encode("x"); setattr(enc, key, value)
            with self.subTest(key=key), self.assertRaises(M.Invalid): M.fingerprint(Engine(), enc)

    def test_decode_must_return_text(self):
        e = Engine(); e.decode = lambda *a, **k: None
        with self.assertRaises(M.Invalid): M.fingerprint(e, e.encode("x"))

    def test_overflow_hash_includes_nested_ids(self):
        e = Engine(); enc = e.encode("x"); enc.overflowing = [e.encode("z")]
        first = M.fingerprint(e, enc)
        enc.overflowing[0].ids[0] += 1
        self.assertNotEqual(first["fields"]["overflowing"], M.fingerprint(e, enc)["fields"]["overflowing"])

    def test_overflow_recursion_bound(self):
        e = Engine(); enc = e.encode("x"); enc.overflowing = [enc]
        with self.assertRaises(M.Invalid): M.fingerprint(e, enc)

    def test_missing_batch_rows_fails(self):
        e = Engine(); e.encode_batch = lambda *a, **k: []
        with self.assertRaises(M.Invalid): M.exercise(e, cases(), 3)

    def test_warm_drift_cannot_improve_benchmark(self):
        e = Engine(); original = e.encode_batch; count = 0
        def changing(*a, **k):
            nonlocal count
            count += 1
            rows = original(*a, **k)
            if count > 2: rows[1].ids[0] += 1
            return rows
        e.encode_batch = changing
        with self.assertRaisesRegex(M.Invalid, "WARM_OUTPUT_DRIFT"): M.exercise(e, cases(), 3)

    def test_dynamic_padding_difference_not_falsely_rejected(self):
        e = Engine(); original = e.encode_batch
        def batch(*a, **k):
            rows = original(*a, **k)
            for enc in rows: enc.type_ids = [7] * len(enc.ids)
            return rows
        e.encode_batch = batch
        out = M.exercise(e, cases(), 3)
        self.assertNotEqual(out["rows"][1]["fields"]["type_ids"], out["rows"][4]["fields"]["type_ids"])

    def test_repeat_bounds(self):
        for value in (0, 2, 31, True):
            with self.subTest(value=value), self.assertRaises(M.Invalid): M.exercise(Engine(), cases(), value)


class ReportTests(unittest.TestCase):
    def test_positive_same_build_control_not_benefit(self):
        a = observation(); b = copy.deepcopy(a)
        result = M.compare(a, b)
        self.assertEqual(result["state"], "SAME_BUILD_CONTROL_PARITY")
        self.assertFalse(result["candidate_benefit_established"])
        self.assertTrue(all(v is False for v in result["authority"].values()))

    def test_different_native_build_parity_scope(self):
        a = observation(); b = copy.deepcopy(a); b["runtime"]["native_sha256"] = "9" * 64
        self.assertEqual(M.compare(a, seal(b))["state"], "PARITY_OBSERVED_FOR_DECLARED_PROFILE")

    def test_fast_candidate_with_wrong_id_has_no_speed_ratio(self):
        a = observation(); b = copy.deepcopy(a); b["rows"][0]["fields"]["ids"] = "0" * 64
        b["timing"]["warm_batch_ns"] = [1, 1, 1]
        out = M.compare(a, seal(b))
        self.assertEqual(out["state"], "FAIL_PARITY")
        self.assertIsNone(out["baseline_over_candidate_median_ratio"]["warm_batch_ns"])

    def test_every_field_can_independently_fail_parity(self):
        for key in M.FIELDS:
            a = observation(); b = copy.deepcopy(a); b["rows"][1]["fields"][key] = "0" * 64
            with self.subTest(field=key): self.assertEqual(M.compare(a, seal(b))["state"], "FAIL_PARITY")

    def test_reordered_rows_are_identity_matched(self):
        a = observation(); b = copy.deepcopy(a); b["rows"].reverse()
        self.assertEqual(M.compare(a, seal(b))["state"], "SAME_BUILD_CONTROL_PARITY")

    def test_different_threads_compare_semantics_without_ratio(self):
        a = observation(); b = copy.deepcopy(a); b["threads_requested"] = 8
        out = M.compare(a, seal(b))
        self.assertFalse(out["matching_observed_environment"])
        self.assertIsNone(out["baseline_over_candidate_median_ratio"]["warm_batch_ns"])

    def test_host_drift_blocks_ratio_not_parity(self):
        a = observation(); b = copy.deepcopy(a); b["runtime"]["host_digest"] = "1" * 64
        self.assertFalse(M.compare(a, seal(b))["matching_observed_environment"])

    def test_comparison_input_bindings(self):
        for key in ("observer_sha256", "tokenizer_sha256", "corpus_sha256", "source_revision_declared"):
            a = observation(); b = copy.deepcopy(a); b[key] = "1" * len(b[key])
            with self.subTest(key=key), self.assertRaises(M.Invalid): M.compare(a, seal(b))

    def test_mismatching_case_identity(self):
        a = observation(); b = copy.deepcopy(a)
        for row in b["rows"]:
            if row["id"] == "empty": row["id"] = "different"
        with self.assertRaises(M.Invalid): M.compare(a, seal(b))

    def test_tampering_requires_new_digest(self):
        a = observation(); a["load_ns"] += 1
        with self.assertRaisesRegex(M.Invalid, "CAPTURE_DIGEST"): M.validate(a)

    def test_rehashed_authority_escalation_still_rejected(self):
        for value in (True, 0, None):
            a = observation(); a["authority"]["promotion"] = value
            with self.subTest(value=value), self.assertRaises(M.Invalid): M.validate(seal(a))

    def test_unknown_report_field_not_preserved(self):
        a = observation(); a["raw_prompt"] = "private"
        with self.assertRaises(M.Invalid): M.validate(seal(a))

    def test_boolean_counts_and_memory_rejected(self):
        for key in ("case_count", "threads_requested", "repeats", "load_ns", "peak_process_rss_bytes"):
            a = observation(); a[key] = True
            with self.subTest(key=key), self.assertRaises(M.Invalid): M.validate(seal(a))

    def test_zero_negative_nan_timing_rejected(self):
        for value in (0, -1, True, 1.5):
            a = observation(); a["timing"]["warm_batch_ns"][0] = value
            with self.subTest(value=value), self.assertRaises(M.Invalid): M.validate(seal(a))

    def test_incomplete_matrix_is_not_success(self):
        a = observation(); a["rows"][-1] = copy.deepcopy(a["rows"][0])
        with self.assertRaises(M.Invalid): M.validate(seal(a))

    def test_same_case_cannot_change_input_between_calls(self):
        a = observation(); a["rows"][0]["input_sha256"] = "0" * 64
        with self.assertRaises(M.Invalid): M.validate(seal(a))

    def test_missing_rich_field_rejected(self):
        a = observation(); del a["rows"][0]["fields"]["offsets"]
        with self.assertRaises(M.Invalid): M.validate(seal(a))

    def test_missing_timing_repeat_rejected(self):
        a = observation(); a["timing"]["warm_batch_ns"].pop()
        with self.assertRaises(M.Invalid): M.validate(seal(a))

    def test_unavailable_memory_not_zero(self):
        a = observation(); M.validate(a)
        a["peak_process_rss_bytes"] = 0
        with self.assertRaises(M.Invalid): M.validate(seal(a))

    def test_time_zone_and_future_validation(self):
        for stamp in ("2026-01-01T00:00:00", "2099-01-01T00:00:00+00:00"):
            a = observation(); a["observed_at"] = stamp
            with self.assertRaises(M.Invalid): M.validate(seal(a))


class CaptureAndCliTests(unittest.TestCase):
    def setUp(self):
        # Other repository tests may import a tokenizer. These controlled capture
        # fixtures model a fresh process; native smoke uses actual subprocesses.
        modules = patch.dict(M.sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        for key in list(M.sys.modules):
            if key == "tokenizers" or key.startswith("tokenizers."):
                del M.sys.modules[key]

    def prepare(self, root):
        tok = root / "tokenizer.json"; tok.write_bytes(b'{}')
        c = root / "corpus.json"; c.write_bytes(M.packed({"schema": "szl.tokenizer-corpus/v1", "cases": cases()}))
        return SimpleNamespace(source="a" * 40, threads=1, repeats=3, tokenizer=tok, tokenizer_sha256=M.sha(tok.read_bytes()),
            corpus=c, corpus_sha256=M.sha(c.read_bytes()), version="0.23.2", native_sha256="a" * 64)

    def test_actual_capture_call_path_with_engine_double(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.prepare(Path(temp))
            binding = SimpleNamespace(Tokenizer=SimpleNamespace(from_str=lambda _: Engine()))
            with patch.object(M, "runtime_identity", return_value=runtime()), patch.object(M.importlib, "import_module", return_value=binding):
                result = M.capture(args)
            M.validate(result)
            self.assertEqual(result["state"], "RECORDED_NOT_QUALIFIED")
            self.assertEqual(result["case_count"], 3)

    def test_hash_mismatch_stops_before_native_load(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.prepare(Path(temp)); args.corpus_sha256 = "0" * 64
            with patch.object(M, "runtime_identity") as identity, self.assertRaises(M.Invalid): M.capture(args)
            identity.assert_not_called()

    def test_native_identity_mismatch_stops_before_tokenizer_load(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.prepare(Path(temp)); args.native_sha256 = "0" * 64
            with patch.object(M, "runtime_identity", return_value=runtime()), patch.object(M.importlib, "import_module") as load:
                with self.assertRaises(M.Invalid): M.capture(args)
                load.assert_not_called()

    def test_preloaded_binding_cannot_relabel_thread_count(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.prepare(Path(temp))
            with patch.dict(M.sys.modules, {"tokenizers": object()}), self.assertRaises(M.Invalid): M.capture(args)

    def test_capture_threads_set_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.prepare(Path(temp)); args.threads = 8
            def load():
                self.assertEqual(M.os.environ["RAYON_NUM_THREADS"], "8")
                self.assertEqual(M.os.environ["TOKENIZERS_PARALLELISM"], "true")
                raise M.Unavailable("NATIVE_BINDING_UNAVAILABLE")
            with patch.object(M, "runtime_identity", side_effect=load), self.assertRaises(M.Unavailable): M.capture(args)

    def test_missing_library_is_explicit(self):
        with patch.object(M.importlib, "import_module", side_effect=ImportError("private-path")):
            with self.assertRaisesRegex(M.Unavailable, "NATIVE_BINDING_UNAVAILABLE"): M.runtime_identity()

    def cli(self, root, bad=False, existing=False):
        a = observation(); b = copy.deepcopy(a)
        if bad: b["rows"][0]["fields"]["ids"] = "0" * 64; seal(b)
        paths = []
        for name, obj in (("base", a), ("candidate", b)):
            p = root / (name + ".json"); p.write_bytes(M.packed(obj)); paths.append(p)
        out = root / "result.json"
        if existing: out.write_text("original")
        argv = ["compare", "--baseline", str(paths[0]), "--candidate", str(paths[1]), "--output", str(out),
                "--baseline-sha256", M.sha(paths[0].read_bytes()), "--candidate-sha256", M.sha(paths[1].read_bytes())]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()): code = M.main(argv)
        return code, out.read_text()

    def test_cli_positive(self):
        with tempfile.TemporaryDirectory() as temp:
            code, text = self.cli(Path(temp))
            self.assertEqual(code, 0); self.assertEqual(json.loads(text)["state"], "SAME_BUILD_CONTROL_PARITY")

    def test_cli_negative_preserves_failure_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            code, text = self.cli(Path(temp), bad=True)
            self.assertEqual(code, 2); self.assertEqual(json.loads(text)["state"], "FAIL_PARITY")

    def test_cli_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(self.cli(Path(temp), existing=True), (1, "original"))

    def test_cli_error_redacts_source_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); p = root / "in"; p.write_text('private-corpus-no-echo')
            out = root / "out"
            argv = ["compare", "--baseline", str(p), "--candidate", str(p), "--output", str(out),
                    "--baseline-sha256", M.sha(p.read_bytes()), "--candidate-sha256", M.sha(p.read_bytes())]
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(M.main(argv), 1)
            self.assertNotIn('private-corpus', out.read_text() + stdout.getvalue())

    def test_cli_unsupported_capture_retains_nonpassing_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            argv = ["capture", "--tokenizer", "t", "--tokenizer-sha256", "0" * 64,
                    "--corpus", "c", "--corpus-sha256", "0" * 64, "--source", "a" * 40,
                    "--version", "1.0.0rc0", "--native-sha256", "0" * 64, "--output", str(out)]
            with patch.object(M, "capture", side_effect=M.Unavailable("PYTHON_API_UNSUPPORTED")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(M.main(argv), 2)
            self.assertEqual(json.loads(out.read_text())["state"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
