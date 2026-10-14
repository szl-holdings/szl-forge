# SPDX-License-Identifier: Apache-2.0
"""SIMULATED character-tokenizer fixtures, never real tokenizer qualification.

Only independently constructed synthetic snapshots are used. No actual train,
dev or held-out file, model runtime, tokenizer download or provider is accessed.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import io
from pathlib import Path
import sys
import traceback
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum_admission as admission
import json_contract as contract
import training_tokens as tokens
from test_curriculum_admission import fixture_manifest, fixture_row, serialize

END = 0x110000
PAD = END + 1
ENDING = "<END>"
FRAGMENTS = ("<system>\n", "\n<user>\n", "\n<assistant>\n")


def binding(**changes):
    return replace(tokens.TokenBinding(
        tokens.BASE_ID, tokens.BASE_REVISION, *("a" * 64 for _ in range(6)),
        8192, 8192, 8192, 360 * 8192, PAD + 1, PAD, FRAGMENTS,
        ((END, ENDING), (PAD, "<PAD>")), (END,)), **changes)


def encode(messages):
    """Transparent single-character test double; no Hugging Face code."""
    rendered = ""
    spans = []
    for fragment, (_, content) in zip(FRAGMENTS, messages):
        rendered += fragment
        spans.append((len(rendered), len(rendered) + len(content)))
        rendered += content
    end_start = len(rendered)
    ids = tuple(map(ord, rendered)) + (END, PAD, PAD)
    labels = (tokens.IGNORE_INDEX,) * spans[-1][0] + tuple(map(ord, messages[-1][1]))
    offsets = tuple((i, i + 1) for i in range(end_start)) + ((0, 0),) * 3
    return tokens.EncodedRow(rendered + ENDING, ids, (1,) * (end_start + 1) + (0, 0),
                             offsets, labels + (END, -100, -100), tuple(spans),
                             (end_start, end_start + len(ENDING)))


def decode(ids):
    return "".join(ENDING if token == END else "<PAD>" if token == PAD else chr(token)
                   for token in ids)


def change_at(values, index, value):
    return values[:index] + (value,) + values[index + 1:]


class TrainingTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [fixture_row(family, kind, index)
                for family in sorted(admission.TRAIN_FAMILIES)
                for kind in contract.CLAIMS for index in range(30)]
        cls.raw = serialize(rows)
        manifest = fixture_manifest(cls.raw)
        cls.manifest_raw = (contract.canonical_json(manifest) + "\n").encode("utf-8")
        cls.snapshot = admission.TrainOnlySnapshot(cls.manifest_raw, cls.raw,
                                                   b'{"evidence_class":"SIMULATED"}\n')
        cls.messages = tuple(tuple((m["role"], m["content"]) for m in row["messages"])
                             for row in rows)
        cls.first = encode(cls.messages[0])

    def check(self, *, policy=None, encoder=encode, decoder=decode, snapshot=None):
        with patch.object(admission, "MANIFEST_SHA256", hashlib.sha256(self.manifest_raw).hexdigest()):
            return tokens.check_training_tokens(self.snapshot if snapshot is None else snapshot,
                                                binding() if policy is None else policy,
                                                encoder, decoder)

    def reject_row(self, row):
        with self.assertRaises(admission.AdmissionError):
            self.check(encoder=lambda messages: row)

    def test_all_360_rows_ordered_redacted_immutable_without_io(self):
        seen = []

        def recording(messages):
            seen.append(messages)
            return encode(messages)

        with patch("builtins.open", side_effect=AssertionError("unexpected file read")), \
             patch.object(io, "open", side_effect=AssertionError("unexpected file read")), \
             patch.object(admission, "run_gates", side_effect=AssertionError("unexpected gate run")):
            report = self.check(encoder=recording)
        self.assertEqual(tuple(seen), self.messages)
        self.assertEqual(tuple(row.index for row in report.rows), tuple(range(360)))
        self.assertEqual(report.total_tokens, sum(row.sequence_tokens for row in report.rows))
        self.assertEqual(report.ordered_tensors_sha256,
                         tokens._json_digest(tuple((row.index, row.tensors_sha256) for row in report.rows)))
        self.assertNotIn(self.messages[0][-1][1], repr(report))
        self.assertNotIn("input_ids", asdict(report))
        self.assertEqual(report.evidence_class, "REPORTED")
        for key in ("assets_authenticated", "training_eligible", "publication_eligible", "execution_authority"):
            self.assertIs(getattr(report, key), False)
        with self.assertRaises(FrozenInstanceError):
            report.rows = ()
        with self.assertRaises(FrozenInstanceError):
            report.rows[0].sequence_tokens = 0

    def test_base_asset_digest_and_capacity_bindings_fail_closed(self):
        cases = (("base_id", "other/base"), ("base_revision", "0" * 40),
                 ("template_sha256", "a" * 64 + "\n"), ("tokenizer_sha256", "A" * 64),
                 ("dependency_sha256", "unknown"), ("masking_sha256", False),
                 ("eos_sha256", ""), ("thinking_sha256", "0" * 63),
                 ("tokenizer_limit", True), ("model_limit", 0),
                 ("admitted_limit", tokens.MAX_SEQUENCE + 1), ("total_token_limit", 0),
                 ("vocab_size", 2**31), ("pad_token_id", -1),
                 ("render_fragments", list(FRAGMENTS)), ("render_fragments", ("x" * 1025, "", "")),
                 ("special_tokens", ((END, ENDING), (END, ENDING))),
                 ("special_tokens", ((True, ENDING),)), ("termination_ids", (False,)),
                 ("termination_ids", (10,)), ("termination_ids", ()))
        for field, value in cases:
            with self.subTest(field=field), self.assertRaises(admission.AdmissionError):
                self.check(policy=binding(**{field: value}))

    def test_minimum_capacity_not_just_one_declared_limit_is_enforced(self):
        for field in ("tokenizer_limit", "model_limit", "admitted_limit"):
            with self.subTest(field=field), self.assertRaises(admission.AdmissionError):
                self.check(policy=binding(**{field: 2, "total_token_limit": 720}))

    def test_aggregate_cap_includes_padding_and_stops_sequentially(self):
        calls = []

        def recording(messages):
            calls.append(messages)
            return encode(messages)

        with self.assertRaisesRegex(admission.AdmissionError, "aggregate"):
            self.check(policy=binding(total_token_limit=len(self.first.input_ids)), encoder=recording)
        self.assertEqual(len(calls), 2)

    def test_tensor_types_booleans_ranges_lengths_and_nested_packing_rejected(self):
        row = self.first
        cases = (("input_ids", list(row.input_ids)), ("input_ids", ((1, 2),) * len(row.input_ids)),
                 ("input_ids", change_at(row.input_ids, 0, True)),
                 ("input_ids", change_at(row.input_ids, 0, -1)),
                 ("input_ids", change_at(row.input_ids, 0, PAD + 1)),
                 ("attention_mask", row.attention_mask[:-1]),
                 ("attention_mask", change_at(row.attention_mask, 0, True)),
                 ("labels", change_at(row.labels, 0, -101)),
                 ("labels", change_at(row.labels, 0, False)),
                 ("offsets", list(row.offsets)))
        for field, value in cases:
            with self.subTest(field=field):
                self.reject_row(replace(row, **{field: value}))

    def test_rendered_message_rewrite_reorder_or_extra_packed_text_rejected(self):
        row = self.first
        for changed in (replace(row, rendered="X" + row.rendered[1:]),
                        replace(row, rendered=row.rendered + "another example"),
                        replace(row, rendered=row.rendered[:-1]),
                        replace(row, message_spans=tuple(reversed(row.message_spans))),
                        replace(row, message_spans=row.message_spans[:2]),
                        replace(row, termination_span=(row.termination_span[0] + 1, len(row.rendered)))):
            self.reject_row(changed)

    def test_unreliable_offsets_gap_overlap_bool_and_unmapped_zero_rejected(self):
        row = self.first
        for offset in ((0, 0), (1, 2), (-1, 0), (0, len(row.rendered) + 1),
                       (False, 1), [0, 1], (0, 1, 2)):
            self.reject_row(replace(row, offsets=change_at(row.offsets, 0, offset)))
        self.reject_row(replace(row, offsets=change_at(row.offsets, 1, (0, 1))))

    def test_first_json_final_json_and_eos_targets_cannot_be_masked(self):
        row = self.first
        start, end = row.message_spans[-1]
        for position in (start, end - 1, end):
            self.reject_row(replace(row, labels=change_at(row.labels, position, -100)))

    def test_prompt_and_padding_must_be_masked(self):
        row = self.first
        for position in (0, row.message_spans[-1][0] - 1, len(row.input_ids) - 1):
            self.reject_row(replace(row, labels=change_at(row.labels, position, row.input_ids[position])))

    def test_wrong_collator_label_or_pre_shifted_labels_rejected(self):
        row = self.first
        position = row.message_spans[-1][0]
        self.reject_row(replace(row, labels=change_at(row.labels, position, ord("!"))))
        self.reject_row(replace(row, labels=row.labels[1:] + (-100,)))

    def test_prompt_assistant_boundary_straddling_is_rejected(self):
        row = self.first
        position = row.message_spans[-1][0] - 1
        self.reject_row(replace(row, offsets=change_at(row.offsets, position, (position, position + 2))))

    def test_padding_must_be_right_only_and_have_bound_id_and_zero_offset(self):
        row = self.first
        self.reject_row(replace(row, attention_mask=change_at(row.attention_mask, 0, 0)))
        self.reject_row(replace(row, input_ids=change_at(row.input_ids, -1 + len(row.input_ids), ord("x"))))
        self.reject_row(replace(row, offsets=change_at(row.offsets, len(row.offsets) - 1, (1, 2))))

    def test_truncation_missing_or_substituted_termination_rejected(self):
        row = self.first
        cutoff = row.termination_span[0]
        self.reject_row(replace(row, input_ids=row.input_ids[:cutoff],
                                attention_mask=row.attention_mask[:cutoff],
                                offsets=row.offsets[:cutoff], labels=row.labels[:cutoff]))
        self.reject_row(replace(row, input_ids=change_at(row.input_ids, cutoff, ord("x")),
                                labels=change_at(row.labels, cutoff, ord("x"))))

    def test_explicit_special_offset_is_accepted_as_well_as_bound_zero_offset(self):
        def explicit(messages):
            row = encode(messages)
            index = len(row.input_ids) - 3
            return replace(row, offsets=change_at(row.offsets, index, row.termination_span))

        self.assertEqual(len(self.check(encoder=explicit).rows), 360)

    def test_full_decode_target_decode_and_eos_decode_each_fail_closed(self):
        for which in (1, 2, 3):
            calls = []

            def wrong(ids):
                calls.append(ids)
                return "incorrect" if len(calls) == which else decode(ids)

            with self.subTest(which=which), self.assertRaises(admission.AdmissionError):
                self.check(decoder=wrong)

    def test_adapter_exception_text_is_redacted(self):
        def failing(*args):
            raise RuntimeError("SIMULATED-private-prompt-marker")

        for kwargs in ({"encoder": failing}, {"decoder": failing}):
            try:
                self.check(**kwargs)
            except admission.AdmissionError as exc:
                self.assertNotIn("SIMULATED-private-prompt-marker", "".join(traceback.format_exception(exc)))
                self.assertIn("details redacted", str(exc))
            else:
                self.fail("adapter failure was not rejected")

    def test_earlier_conversion_still_rejects_non_snapshot_and_changed_train_bytes(self):
        for value in ({"state": "VALID"}, replace(self.snapshot, train_bytes=self.raw + b" ")):
            with self.assertRaises(admission.AdmissionError):
                self.check(snapshot=value)

    def test_unicode_and_repeated_target_use_exact_character_spans(self):
        target = contract.canonical_json({"note": "caf\u00e9 e\u0301 \U0001f642"})
        messages = (("system", "SIMULATED unicode \u03bb"),
                    ("user", "Quoted target twice: " + target + " then " + target),
                    ("assistant", target))
        row = encode(messages)
        policy = binding()
        limit, special = tokens._binding(policy)
        ledger = tokens._row(0, messages, row, policy, special, limit, decode)
        self.assertEqual(ledger.assistant_tokens, len(target))
        self.assertEqual(ledger.target_sha256, hashlib.sha256(target.encode("utf-8")).hexdigest())
        # A byte-offset adapter cannot silently masquerade as a character adapter.
        byte_offsets = tuple((len(row.rendered[:start].encode("utf-8")),
                              len(row.rendered[:end].encode("utf-8")))
                             if (start, end) != (0, 0) else (0, 0)
                             for start, end in row.offsets)
        with self.assertRaises(admission.AdmissionError):
            tokens._row(0, messages, replace(row, offsets=byte_offsets),
                        policy, special, limit, decode)
        # Repointing supervision at the quoted copy must also fail.
        quoted_start = row.rendered.index(target)
        wrong_spans = row.message_spans[:2] + ((quoted_start, quoted_start + len(target)),)
        with self.assertRaises(admission.AdmissionError):
            tokens._row(0, messages, replace(row, message_spans=wrong_spans),
                        policy, special, limit, decode)

    def test_padding_id_may_equal_eos_without_masking_active_eos(self):
        def shared_pad(messages):
            row = encode(messages)
            return replace(row, input_ids=row.input_ids[:-2] + (END, END))

        report = self.check(policy=binding(pad_token_id=END), encoder=shared_pad)
        self.assertTrue(all(row.termination_tokens == 1 for row in report.rows))
        row = shared_pad(self.messages[0])
        eos_index = len(row.input_ids) - 3
        with self.assertRaises(admission.AdmissionError):
            self.check(policy=binding(pad_token_id=END), encoder=lambda messages: replace(
                row, labels=change_at(row.labels, eos_index, -100)))

    def test_multi_token_termination_is_exact_and_ordered(self):
        second = PAD + 1
        policy = binding(vocab_size=second + 1,
                         special_tokens=((END, ENDING), (PAD, "<PAD>"), (second, "<STOP>")),
                         termination_ids=(END, second))

        def multi_end(messages):
            row = encode(messages)
            return replace(row, rendered=row.rendered + "<STOP>",
                           input_ids=row.input_ids[:-2] + (second, PAD, PAD),
                           attention_mask=row.attention_mask[:-2] + (1, 0, 0),
                           offsets=row.offsets[:-2] + ((0, 0),) * 3,
                           labels=row.labels[:-2] + (second, -100, -100),
                           termination_span=(row.termination_span[0], len(row.rendered) + 6))

        def multi_decode(ids):
            return "".join("<STOP>" if value == second else decode((value,)) for value in ids)

        report = self.check(policy=policy, encoder=multi_end, decoder=multi_decode)
        self.assertTrue(all(row.termination_tokens == 2 for row in report.rows))
        row = multi_end(self.messages[0])
        for ending in ((second, END), (END, END)):
            with self.subTest(ending=ending), self.assertRaises(admission.AdmissionError):
                self.check(policy=policy, decoder=multi_decode, encoder=lambda messages: replace(
                    row, input_ids=row.input_ids[:-4] + ending + (PAD, PAD),
                    labels=row.labels[:-4] + ending + (-100, -100)))

    def test_ledger_is_deterministic_and_sensitive_to_binding_and_tensors(self):
        first = self.check()
        self.assertEqual(first, self.check())
        changed_binding = self.check(policy=binding(masking_sha256="b" * 64))
        self.assertNotEqual(first.binding_sha256, changed_binding.binding_sha256)
        self.assertEqual(first.ordered_tensors_sha256, changed_binding.ordered_tensors_sha256)

        def unpadded(messages):
            row = encode(messages)
            return replace(row, input_ids=row.input_ids[:-2], attention_mask=row.attention_mask[:-2],
                           offsets=row.offsets[:-2], labels=row.labels[:-2])

        changed_tensors = self.check(encoder=unpadded)
        self.assertNotEqual(first.ordered_tensors_sha256, changed_tensors.ordered_tensors_sha256)
        self.assertEqual(first.total_tokens - changed_tensors.total_tokens, 360 * 2)
        self.assertEqual(tuple(row.target_sha256 for row in first.rows),
                         tuple(row.target_sha256 for row in changed_tensors.rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
