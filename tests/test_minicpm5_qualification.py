"""Offline contracts, not model-quality results. No torch or network required."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("minicpm_under_test", ROOT / "inference/minicpm5_qualification.py")
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def inventory():
    return [{"name": "config.json", "size": 20, "sha256": None},
            {"name": "tokenizer_config.json", "size": 10, "sha256": None},
            {"name": "model.safetensors", "size": 100, "sha256": "a" * 64}]


def raw_report():
    return {"cases": [], "plan": q.plan(), "modelLoaded": False}


class SuiteTests(unittest.TestCase):
    def test_suite_is_stable_public_and_bounded(self):
        cases = q.suite()
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({c['id'] for c in cases}), 12)
        self.assertEqual(q.digest(cases), q.digest(q.suite()))
        self.assertEqual(q.plan()['suiteClass'], 'PUBLIC_SYNTHETIC_CONTRACT_PROBES')

    def test_all_six_categories_are_present_twice(self):
        for category in ['lookup', 'missing', 'cross_tenant', 'revoked', 'injection', 'unicode']:
            self.assertEqual(sum(c['category'] == category for c in q.suite()), 2)

    def test_expected_output_is_not_sent_to_model(self):
        for case in q.suite():
            self.assertNotIn('expected', case['input'])
            self.assertNotIn('decision', case['input'])

    def test_revoked_missing_and_cross_tenant_require_abstention(self):
        for case in q.suite():
            if case['category'] in {'revoked', 'missing', 'cross_tenant'}:
                self.assertEqual(case['expected'], {'decision': 'ABSTAIN', 'evidence_id': None, 'value': None})

    def test_fixture_mutation_cannot_alter_next_run(self):
        q.suite()[0]['input']['documents'].clear()
        self.assertTrue(q.suite()[0]['input']['documents'])

    def test_unmeasured_fields_do_not_become_truth(self):
        plan = q.plan()
        self.assertEqual(plan['productionDisposition'], 'HOLD')
        for key in ['trainingAuthorized', 'toolExecutionAuthorized', 'publicationEligible', 'automaticDownload']:
            self.assertFalse(plan[key])
        self.assertEqual(plan['modelRevision'], q.REVISION)
        self.assertRegex(q.REVISION, r'^[a-f0-9]{40}$')


class OutputTests(unittest.TestCase):
    def test_exact_answers_pass(self):
        for case in q.suite():
            self.assertTrue(q.grade(json.dumps(case['expected']), case['expected'])['passed'])

    def test_duplicate_keys_rejected(self):
        expected = q.suite()[1]['expected']
        self.assertFalse(q.grade('{"decision":"LOOKUP","decision":"ABSTAIN","evidence_id":null,"value":null}', expected)['passed'])

    def test_nonfinite_and_overflow_numbers_rejected(self):
        for text in ['NaN', 'Infinity', '-Infinity', '1e999', '{"x":1e999}']:
            with self.subTest(text=text), self.assertRaises(q.QualificationError):
                q.strict_json(text)

    def test_markdown_and_reasoning_not_repaired_into_pass(self):
        case = q.suite()[0]
        for text in ['```json\n' + json.dumps(case['expected']) + '\n```', '<think>reason</think>' + json.dumps(case['expected'])]:
            self.assertFalse(q.grade(text, case['expected'])['passed'])

    def test_valid_json_cannot_pass_wrong_grounding(self):
        case = q.suite()[0]
        answer = dict(case['expected'], value='invented')
        self.assertFalse(q.grade(json.dumps(answer), case['expected'])['passed'])

    def test_extra_fields_and_wrong_root_rejected(self):
        expected = q.suite()[1]['expected']
        for value in [[], None, 1, 'ABSTAIN', dict(expected, execute=True)]:
            self.assertFalse(q.grade(json.dumps(value), expected)['passed'])

    def test_truncated_correct_output_is_failure(self):
        case = q.suite()[0]
        self.assertFalse(q.grade(json.dumps(case['expected']), case['expected'], truncated=True)['passed'])

    def test_outputs_are_hashed_not_persisted(self):
        raw = json.dumps(q.suite()[0]['expected'])
        result = q.grade(raw, q.suite()[0]['expected'])
        self.assertEqual(result['outputSha256'], hashlib.sha256(raw.encode()).hexdigest())
        self.assertNotIn(raw, json.dumps(result))

    def test_output_size_bound(self):
        with self.assertRaises(q.QualificationError):
            q.strict_json('x' * 16385)


class ArtifactTests(unittest.TestCase):
    def test_only_known_config_and_safetensors_are_downloadable(self):
        for name in ['model.safetensors', 'model-00001-of-00002.safetensors', 'config.json']:
            self.assertTrue(q.allowed_file(name))
        for name in ['model.bin', 'pytorch_model.bin', 'model.py', '../model.safetensors', 'foo/model.safetensors', 'README.md']:
            self.assertFalse(q.allowed_file(name))

    def test_complete_inventory_selected_deterministically(self):
        rows = inventory()
        self.assertEqual(q.select_files(rows), q.select_files(list(reversed(rows))))

    def test_weight_digest_required(self):
        rows = inventory()
        rows[-1]['sha256'] = None
        with self.assertRaises(q.QualificationError):
            q.select_files(rows)

    def test_boolean_or_unknown_size_rejected(self):
        for size in [True, None, -1]:
            rows = inventory()
            rows[-1]['size'] = size
            with self.assertRaises(q.QualificationError):
                q.select_files(rows)

    def test_missing_components_rejected(self):
        for i in range(3):
            rows = inventory()
            rows.pop(i)
            with self.assertRaises(q.QualificationError):
                q.select_files(rows)

    def test_duplicate_inventory_rejected(self):
        rows = inventory()
        rows.append(rows[-1])
        with self.assertRaises(q.QualificationError):
            q.select_files(rows)

    def test_total_bytes_budget_enforced(self):
        rows = inventory()
        rows[-1]['size'] = q.MAX_BYTES + 1
        with self.assertRaises(q.QualificationError):
            q.select_files(rows)

    def test_model_bytes_and_architecture_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = {'config.json': b'{"model_type":"llama"}', 'tokenizer_config.json': b'{}', 'model.safetensors': b'fixture-not-weights'}
            rows = []
            for name, value in data.items():
                (root / name).write_bytes(value)
                rows.append({'name': name, 'size': len(value), 'sha256': hashlib.sha256(value).hexdigest() if name.endswith('.safetensors') else None})
            self.assertEqual(len(q.verify_files(root, rows)), 3)
            (root / 'model.safetensors').write_bytes(b'FIXTURE-not-weights')
            with self.assertRaises(q.QualificationError):
                q.verify_files(root, rows)

    def test_remote_code_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = {'config.json': b'{"model_type":"llama","auto_map":{"AutoModel":"evil.Model"}}', 'tokenizer_config.json': b'{}', 'model.safetensors': b'fixture'}
            rows = []
            for name, value in data.items():
                (root / name).write_bytes(value)
                rows.append({'name': name, 'size': len(value), 'sha256': hashlib.sha256(value).hexdigest()})
            with self.assertRaises(q.QualificationError):
                q.verify_files(root, rows)


class EvidenceTests(unittest.TestCase):
    def test_empty_run_cannot_be_smoke_pass(self):
        report = q.finalize(raw_report())
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertEqual(report['evidenceClass'], 'INCOMPLETE_NO_GENERATION')
        self.assertIsNone(report['p95GenerationMs'])

    def test_failed_quality_is_not_success(self):
        report = raw_report()
        report['cases'] = [{'passed': False, 'generationMs': 1.0} for _ in q.suite()]
        self.assertEqual(q.finalize(report)['status'], 'SMOKE_FAIL')

    def test_complete_pass_still_cannot_promote(self):
        report = raw_report()
        report['cases'] = [{'passed': True, 'generationMs': float(i+1)} for i in range(12)]
        final = q.finalize(report)
        self.assertEqual(final['status'], 'SMOKE_PASS')
        self.assertEqual(final['productionDisposition'], 'HOLD')
        for key in ['runtimeQualified', 'publicationEligible', 'trainingAuthorized', 'toolExecuted', 'sealed']:
            self.assertFalse(final[key])
        for key in ['joules', 'costUsd', 'ttftMs']:
            self.assertIsNone(final[key])

    def test_record_is_self_consistent_and_not_signed(self):
        report = q.finalize(raw_report())
        sha = report.pop('recordSha256')
        self.assertEqual(sha, q.digest(report))
        self.assertFalse(report['sealed'])

    def test_latency_quantile_definition(self):
        self.assertEqual(q.quantile([1, 2, 3, 4], .5), 2)
        self.assertEqual(q.quantile([1, 2, 3, 4], .95), 4)
        self.assertIsNone(q.quantile([], .95))
        for sample in [[float('nan')], [-1], [float('inf')]]:
            with self.assertRaises(q.QualificationError):
                q.quantile(sample, .5)

    def test_atomic_failure_replaces_old_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            q.atomic_json(path, {'status': 'SMOKE_PASS'})
            q.atomic_json(path, {'status': 'INCOMPLETE'})
            self.assertEqual(json.loads(path.read_text())['status'], 'INCOMPLETE')
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_cli_plan_has_no_heavy_dependency_or_inference(self):
        result = subprocess.run([sys.executable, str(ROOT/'inference/minicpm5_qualification.py')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['modelId'], q.MODEL)

    def test_live_run_requires_explicit_source_identity(self):
        result = subprocess.run([sys.executable, str(ROOT/'inference/minicpm5_qualification.py'), '--run'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
