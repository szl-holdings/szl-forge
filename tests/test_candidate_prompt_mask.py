# SPDX-License-Identifier: Apache-2.0
"""Source-extracted masking controls; no model, GPU, corpus or candidate import."""
import ast
import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    'willay-candidate-20261001',
    'brain-navigator-r2-candidate-20261001',
    'szl-triage-qwen3-5-0-8b-lora-candidate-20261001',
    'szl-triage-qwen3-5-0-8b-lora-study5-candidate-20261001',
)
PATHS = [ROOT / 'tools/estate_payload/kit/train_candidate.py'] + [
    ROOT / 'frontier' / name / 'train_candidate.py' for name in CANDIDATES]


def load_functions(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {'response_only_labels', 'build_examples'}]
    if {n.name for n in nodes} != {'response_only_labels', 'build_examples'}:
        raise AssertionError('Masking implementation missing from ' + path.name)
    env = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


class PromptMaskTests(unittest.TestCase):
    def setUp(self):
        self.modules = [load_functions(p) for p in PATHS]

    def each(self):
        return zip(PATHS, self.modules)

    def test_all_five_owners_have_identical_mask_contract(self):
        dumped = []
        for path in PATHS:
            functions = [n for n in ast.parse(path.read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name in {'response_only_labels', 'build_examples'}]
            dumped.append(ast.dump(ast.Module(body=functions, type_ignores=[])))
        self.assertEqual(len(set(dumped)), 1)

    def test_exact_prefix_preserves_tokens_and_assistant_labels(self):
        for path, env in self.each():
            with self.subTest(path=path):
                self.assertEqual(env['response_only_labels']([7, 8, 9, 10], [7, 8], 8),
                                 ([7, 8, 9, 10], [-100, -100, 9, 10], 0))

    def test_equal_length_wrong_prefix_is_rejected(self):
        for path, env in self.each():
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'MASK_PROMPT_PREFIX_MISMATCH'):
                env['response_only_labels']([7, 8, 9, 10], [70, 80], 8)

    def test_shared_partial_prefix_still_rejected(self):
        for path, env in self.each():
            with self.subTest(path=path), self.assertRaises(ValueError):
                env['response_only_labels']([7, 8, 9, 10], [7, 80], 8)

    def test_prompt_longer_than_full_is_rejected(self):
        for path, env in self.each():
            with self.subTest(path=path), self.assertRaises(ValueError):
                env['response_only_labels']([7, 8], [7, 8, 9], 8)

    def test_no_assistant_suffix_is_not_silently_dropped(self):
        for path, env in self.each():
            with self.subTest(path=path), self.assertRaises(ValueError):
                env['response_only_labels']([7, 8], [7, 8], 8)

    def test_truncation_with_one_target_preserves_boundary(self):
        for path, env in self.each():
            with self.subTest(path=path):
                self.assertEqual(env['response_only_labels']([7, 8, 9, 10], [7, 8], 3),
                                 ([7, 8, 9], [-100, -100, 9], 1))

    def test_truncation_without_targets_is_rejected(self):
        for path, env in self.each():
            for limit in (1, 2):
                with self.subTest(path=path, limit=limit), self.assertRaisesRegex(ValueError, 'MASK_NO_ASSISTANT'):
                    env['response_only_labels']([7, 8, 9, 10], [7, 8], limit)

    def test_nonpositive_boolean_and_noninteger_limits_rejected(self):
        for path, env in self.each():
            for limit in (0, -1, True, False, 2.5, '4', None):
                with self.subTest(path=path, limit=limit), self.assertRaisesRegex(ValueError, 'MASK_MAX_LENGTH'):
                    env['response_only_labels']([7, 8], [7], limit)

    def test_token_ids_must_be_nonempty_nonnegative_integer_lists(self):
        invalid = [[], (), None, '1', [True], [-1], [1.1], [None], [[1]]]
        for path, env in self.each():
            for value in invalid:
                for is_prompt in (False, True):
                    full, prompt = ([7, 8], value) if is_prompt else (value, [7])
                    with self.subTest(path=path, value=value, prompt=is_prompt), self.assertRaisesRegex(ValueError, 'MASK_TOKEN_IDS'):
                        env['response_only_labels'](full, prompt, 8)

    def test_zero_is_a_valid_token_id(self):
        for path, env in self.each():
            with self.subTest(path=path):
                self.assertEqual(env['response_only_labels']([0, 1, 0], [0, 1], 3)[1], [-100, -100, 0])

    def test_inputs_never_modified(self):
        for path, env in self.each():
            full, prompt = [1, 2, 3, 4], [1, 2]
            before = copy.deepcopy((full, prompt))
            env['response_only_labels'](full, prompt, 3)
            with self.subTest(path=path): self.assertEqual((full, prompt), before)

    def test_seeded_prefix_and_truncation_properties(self):
        rng = random.Random(1704)
        cases = []
        for _ in range(100):
            n, extra = rng.randint(1, 20), rng.randint(1, 20)
            full = [rng.randint(0, 10000) for _ in range(n + extra)]
            cases.append((full, full[:n], rng.randint(n + 1, n + extra + 8)))
        for path, env in self.each():
            for full, prompt, limit in cases:
                clipped, labels, truncated = env['response_only_labels'](full, prompt, limit)
                with self.subTest(path=path):
                    self.assertEqual(clipped, full[:limit])
                    self.assertEqual(labels[:len(prompt)], [-100] * len(prompt))
                    self.assertEqual(labels[len(prompt):], clipped[len(prompt):])
                    self.assertEqual(truncated, int(len(full) > limit))
                    self.assertEqual(len(labels), len(clipped))

    def test_actual_build_examples_uses_verified_mask_and_integer_dtype(self):
        class TorchDouble:
            long = 'int64'
            @staticmethod
            def tensor(value, dtype=None): return {'values': value, 'dtype': dtype}
            @staticmethod
            def ones(n, dtype=None): return {'values': [1] * n, 'dtype': dtype}
        for path, env in self.each():
            env['lib'] = SimpleNamespace(text_only_encode=lambda tok, messages, add_generation_prompt:
                                        ('', [1, 2] if add_generation_prompt else [1, 2, 3, 4]))
            with patch.dict(sys.modules, {'torch': TorchDouble}):
                values, count = env['build_examples']([{'messages': [0, 1]}], object(), 3)
            with self.subTest(path=path):
                self.assertEqual(count, 1)
                self.assertEqual(values[0]['labels'], {'values': [-100, -100, 3], 'dtype': 'int64'})
                self.assertEqual(values[0]['input_ids']['values'], [1, 2, 3])
                self.assertEqual(values[0]['attention_mask']['values'], [1, 1, 1])

    def test_invalid_build_stops_before_tensor_creation(self):
        for path, env in self.each():
            env['lib'] = SimpleNamespace(text_only_encode=lambda tok, messages, add_generation_prompt:
                                        ('', [8, 9] if add_generation_prompt else [1, 2, 3, 4]))
            torch = SimpleNamespace(tensor=lambda *a, **k:self.fail('tensor construction on invalid mask'), long='int64')
            with patch.dict(sys.modules, {'torch': torch}), self.subTest(path=path), self.assertRaises(ValueError):
                env['build_examples']([{'messages': [0, 1]}], object(), 4)

    def test_error_message_has_no_token_values(self):
        for path, env in self.each():
            with self.subTest(path=path):
                try: env['response_only_labels']([876543, 99999], [765432], 10)
                except ValueError as e: self.assertEqual(str(e), 'MASK_PROMPT_PREFIX_MISMATCH')
                else: self.fail('incorrect prefix was accepted')


if __name__ == '__main__':
    unittest.main()
