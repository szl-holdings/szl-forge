"""SIMULATED contracts for an independent standard-library algebra oracle."""

from copy import deepcopy
from dataclasses import replace
import json
import math
import unittest

from toy_lora_oracle import (
    AdamWPolicy, ContractError, OracleSession, SAMPLE_ASSISTANT_MASK,
    SAMPLE_INPUTS, SAMPLE_LABELS, ToyProfile, adamw_step, adapter_shapes,
    artificial_adapters, assert_gradient_close, assert_parity, finite_difference,
    make_dropout_replay, resident, streamed_recompute, zero_optimizer_state,
)


class AlgebraOracleTests(unittest.TestCase):
    def setUp(self):
        self.profile = ToyProfile()
        self.adapters = artificial_adapters(self.profile)

    def run_fixture(self, function, profile=None, replay=None):
        return function(profile or self.profile, self.adapters, SAMPLE_INPUTS,
                        SAMPLE_LABELS, SAMPLE_ASSISTANT_MASK, replay)

    def test_resident_and_streamed_match_all_thirty_nonzero_adapter_gradients(self):
        reference = self.run_fixture(resident)
        streamed = self.run_fixture(streamed_recompute)
        assert_parity(reference, streamed)
        gradients = [value for matrix in reference.gradients.values() for row in matrix for value in row]
        self.assertEqual(len(gradients), 30)
        self.assertTrue(all(abs(value) > 1e-12 for value in gradients))
        self.assertTrue(math.isfinite(reference.loss))

    def test_every_adapter_gradient_matches_independent_central_differences(self):
        reference = self.run_fixture(resident)
        numerical = finite_difference(self.profile, self.adapters, SAMPLE_INPUTS,
                                      SAMPLE_LABELS, SAMPLE_ASSISTANT_MASK)
        assert_gradient_close(reference.gradients, numerical, absolute=1e-7, relative=1e-5)

    def test_stream_event_schedule_has_one_owner_and_reverse_backward_order(self):
        result = self.run_fixture(streamed_recompute)
        active = None
        backward_order = []
        for operation, index in result.events:
            if operation == "load":
                self.assertIsNone(active)
                active = index
            elif operation == "evict":
                self.assertEqual(active, index)
                active = None
            else:
                self.assertEqual(active, index)
                if operation == "backward":
                    backward_order.append(index)
        self.assertIsNone(active)
        self.assertEqual(backward_order, [1, 0])

    def test_loss_uses_only_assistant_positions_and_correct_denominator(self):
        masked = self.run_fixture(resident)
        individual = [resident(self.profile, self.adapters, (SAMPLE_INPUTS[index],),
                               (SAMPLE_LABELS[index],), (True,)).loss for index in (1, 2)]
        self.assertAlmostEqual(masked.loss, sum(individual) / 2.0, places=14)
        changed = resident(self.profile, self.adapters, SAMPLE_INPUTS, (4, 2, 4),
                           SAMPLE_ASSISTANT_MASK)
        self.assertEqual(masked.loss, changed.loss)

    def test_empty_supervision_after_truncation_fails_closed(self):
        for function in (resident, streamed_recompute):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ContractError):
                    function(self.profile, self.adapters, SAMPLE_INPUTS, SAMPLE_LABELS,
                             (False, False, False))

    def test_invalid_labels_masks_shapes_and_nonfinite_inputs_are_rejected(self):
        cases = [
            (SAMPLE_INPUTS, (0, 5, 4), SAMPLE_ASSISTANT_MASK),
            (SAMPLE_INPUTS, (0, -100, 4), SAMPLE_ASSISTANT_MASK),
            (SAMPLE_INPUTS, (0, True, 4), SAMPLE_ASSISTANT_MASK),
            (SAMPLE_INPUTS, SAMPLE_LABELS, (False, 1, True)),
            (SAMPLE_INPUTS, SAMPLE_LABELS, (True,)),
            ((), (), ()),
            (((float("nan"), 0.0, 1.0),), (1,), (True,)),
            (((1.0, 2.0),), (1,), (True,)),
        ]
        for inputs, labels, mask in cases:
            with self.subTest(inputs=inputs, labels=labels, mask=mask):
                with self.assertRaises(ContractError):
                    resident(self.profile, self.adapters, inputs, labels, mask)

    def test_fixed_nonzero_dropout_replay_matches_recompute_and_finite_difference(self):
        profile = replace(self.profile, dropout_probability=0.25)
        replay = make_dropout_replay(profile, len(SAMPLE_INPUTS), seed=17)
        self.assertEqual(replay, make_dropout_replay(profile, len(SAMPLE_INPUTS), seed=17))
        self.assertTrue(any(value == 0.0 for matrix in replay.values() for row in matrix for value in row))
        reference = self.run_fixture(resident, profile, replay)
        streamed = self.run_fixture(streamed_recompute, profile, replay)
        assert_parity(reference, streamed)
        numerical = finite_difference(profile, self.adapters, SAMPLE_INPUTS,
                                      SAMPLE_LABELS, SAMPLE_ASSISTANT_MASK, replay)
        assert_gradient_close(reference.gradients, numerical, absolute=1e-7, relative=1e-5)

    def test_missing_or_mismatched_dropout_replay_is_rejected(self):
        profile = replace(self.profile, dropout_probability=0.25)
        with self.assertRaises(ContractError):
            self.run_fixture(resident, profile)
        replay = make_dropout_replay(profile, len(SAMPLE_INPUTS), seed=17)
        mismatch = make_dropout_replay(profile, len(SAMPLE_INPUTS), seed=19)
        self.assertNotEqual(replay, mismatch)
        with self.assertRaises(ContractError):
            streamed_recompute(profile, self.adapters, SAMPLE_INPUTS, SAMPLE_LABELS,
                               SAMPLE_ASSISTANT_MASK, replay, mismatch)

    def test_injected_wrong_missing_zero_and_nonfinite_gradients_fail_parity(self):
        reference = self.run_fixture(resident)
        for mutation in ("wrong", "missing", "zero", "nan"):
            observed = deepcopy(self.run_fixture(streamed_recompute))
            if mutation == "wrong":
                observed.gradients["layer.0.A"][0][0] += 0.001
            elif mutation == "missing":
                del observed.gradients["layer.1.B"]
            elif mutation == "zero":
                observed.gradients["layer.0.A"] = [[0.0] * 3 for _ in range(2)]
            else:
                observed.gradients["layer.1.A"][0][0] = float("nan")
            with self.subTest(mutation=mutation):
                with self.assertRaises(ContractError):
                    assert_parity(reference, observed)

    def test_injected_loss_input_gradient_and_authority_changes_fail_parity(self):
        reference = self.run_fixture(resident)
        bad_input = tuple((value + 0.01 if index == 0 else value)
                          for index, value in enumerate(reference.input_gradients[0]))
        cases = [replace(reference, loss=reference.loss + 0.01),
                 replace(reference, input_gradients=(bad_input,) + reference.input_gradients[1:]),
                 replace(reference, training_eligible=True),
                 replace(reference, publication_eligible=0),
                 replace(reference, execution_authority=True)]
        for observed in cases:
            with self.subTest(observed=observed):
                with self.assertRaises(ContractError):
                    assert_parity(reference, observed)

    def test_adapters_and_frozen_base_are_unchanged_by_derivatives(self):
        before, base = deepcopy(self.adapters), deepcopy(self.profile.identity)
        self.run_fixture(resident)
        self.run_fixture(streamed_recompute)
        self.assertEqual(self.adapters, before)
        self.assertEqual(self.profile.identity, base)

    def test_first_adamw_update_matches_independent_closed_form(self):
        policy = AdamWPolicy()
        gradients = self.run_fixture(resident).gradients
        values, state = adamw_step(self.profile, self.adapters, gradients,
                                  zero_optimizer_state(self.profile), policy)
        self.assertEqual(state["step"], 1)
        for key, (rows, columns) in adapter_shapes(self.profile).items():
            for row in range(rows):
                for column in range(columns):
                    old, g = self.adapters[key][row][column], gradients[key][row][column]
                    expected = old * (1.0 - policy.lr * policy.weight_decay)
                    expected -= policy.lr * g / (abs(g) + policy.epsilon)
                    self.assertAlmostEqual(values[key][row][column], expected, places=14)


class CheckpointOracleTests(unittest.TestCase):
    def setUp(self):
        self.profile = ToyProfile(dropout_probability=0.25)
        self.policy = AdamWPolicy()

    def advance(self, session, mode="resident"):
        return session.advance(SAMPLE_INPUTS, SAMPLE_LABELS, SAMPLE_ASSISTANT_MASK, mode)

    def test_two_step_resume_matches_uninterrupted_and_streamed_updates(self):
        uninterrupted = OracleSession(self.profile, seed=17, policy=self.policy)
        first = self.advance(uninterrupted)
        checkpoint = uninterrupted.checkpoint()
        reference = self.advance(uninterrupted)
        resumed = OracleSession.from_checkpoint(self.profile, self.policy,
                                                json.loads(json.dumps(checkpoint)))
        observed = self.advance(resumed, "streamed")
        assert_parity(reference, observed)
        self.assertEqual(uninterrupted.checkpoint(), resumed.checkpoint())
        self.assertEqual(resumed.optimizer["step"], 2)
        self.assertEqual(resumed.dataset_cursor, {"batches_seen": 2, "samples_seen": 6})
        self.assertNotEqual(first.gradients, reference.gradients)

    def test_checkpoint_and_restore_deep_copy_adapter_optimizer_and_rng_state(self):
        session = OracleSession(self.profile, policy=self.policy)
        self.advance(session)
        saved = session.checkpoint()
        expected = deepcopy(saved)
        session.adapters["layer.0.A"][0][0] = 99.0
        session.optimizer["moments"]["layer.0.A"]["m"][0][0] = 99.0
        session.rng.random()
        self.assertEqual(saved, expected)
        resumed = OracleSession.from_checkpoint(self.profile, self.policy, saved)
        saved["adapters"]["layer.1.B"][0][0] = 77.0
        saved["optimizer_state"]["moments"]["layer.1.B"]["v"][0][0] = 77.0
        saved["rng_state"]["state"][0] = 0
        self.assertEqual(resumed.checkpoint(), expected)

    def test_wrong_identity_rank_expert_coverage_and_optimizer_policy_are_rejected(self):
        checkpoint = OracleSession(self.profile, policy=self.policy).checkpoint()
        for field in ("base", "rank", "alpha_type", "expert_coverage", "optimizer_policy"):
            altered = deepcopy(checkpoint)
            if field == "base":
                altered["identity"]["base"][0][0][0] += 0.01
            elif field == "rank":
                altered["identity"]["rank"] = 3
            elif field == "alpha_type":
                altered["identity"]["alpha"] = 4
            elif field == "expert_coverage":
                altered["identity"]["expert_coverage"] = "ALL_EXPERTS"
            else:
                altered["optimizer_policy"]["epsilon"] = 0.001
            with self.subTest(field=field):
                with self.assertRaises(ContractError):
                    OracleSession.from_checkpoint(self.profile, self.policy, altered)

    def test_missing_unknown_keys_bad_shapes_and_nonfinite_state_are_rejected(self):
        checkpoint = OracleSession(self.profile, policy=self.policy).checkpoint()
        for mutation in ("adapter_missing", "adapter_unknown", "optimizer_missing",
                         "top_unknown", "shape", "nan", "negative_variance", "rng", "cursor"):
            altered = deepcopy(checkpoint)
            if mutation == "adapter_missing":
                del altered["adapters"]["layer.0.A"]
            elif mutation == "adapter_unknown":
                altered["adapters"]["unknown"] = [[1.0]]
            elif mutation == "optimizer_missing":
                del altered["optimizer_state"]["moments"]["layer.1.B"]
            elif mutation == "top_unknown":
                altered["launch_gpu"] = True
            elif mutation == "shape":
                altered["adapters"]["layer.0.B"].pop()
            elif mutation == "nan":
                altered["optimizer_state"]["moments"]["layer.0.A"]["m"][0][0] = float("nan")
            elif mutation == "negative_variance":
                altered["optimizer_state"]["moments"]["layer.0.A"]["v"][0][0] = -1.0
            elif mutation == "rng":
                altered["rng_state"]["state"][-1] = 625
            else:
                altered["dataset_cursor"]["batches_seen"] = 1
            with self.subTest(mutation=mutation):
                with self.assertRaises(ContractError):
                    OracleSession.from_checkpoint(self.profile, self.policy, altered)

    def test_simulation_flags_remain_false_and_attempted_promotion_is_rejected(self):
        checkpoint = OracleSession(self.profile, policy=self.policy).checkpoint()
        self.assertEqual(checkpoint["evidence_class"], "SIMULATED")
        for flag in ("training_eligible", "publication_eligible", "execution_authority"):
            self.assertIs(checkpoint[flag], False)
            for value in (True, 0, None):
                altered = deepcopy(checkpoint)
                altered[flag] = value
                with self.subTest(flag=flag, value=value):
                    with self.assertRaises(ContractError):
                        OracleSession.from_checkpoint(self.profile, self.policy, altered)

    def test_failed_step_does_not_consume_rng_update_parameters_or_move_cursor(self):
        session = OracleSession(self.profile, policy=self.policy)
        before = session.checkpoint()
        with self.assertRaises(ContractError):
            session.advance(SAMPLE_INPUTS, SAMPLE_LABELS, (False, False, False))
        self.assertEqual(session.checkpoint(), before)

    def test_invalid_constructor_inputs_do_not_silently_default(self):
        for options in ({"adapters": {}}, {"profile": False}, {"policy": {}}, {"seed": True}):
            with self.subTest(options=options):
                with self.assertRaises(ContractError):
                    OracleSession(**options)

    def test_optimizer_state_changes_step_two_even_with_same_observed_step_one_loss(self):
        session = OracleSession(self.profile, policy=self.policy)
        self.advance(session)
        checkpoint = session.checkpoint()
        altered = deepcopy(checkpoint)
        altered["optimizer_state"]["moments"]["layer.0.A"]["m"][0][0] += 0.001
        original = OracleSession.from_checkpoint(self.profile, self.policy, checkpoint)
        corrupted = OracleSession.from_checkpoint(self.profile, self.policy, altered)
        original_result, corrupted_result = self.advance(original), self.advance(corrupted)
        self.assertEqual(original_result.loss, corrupted_result.loss)
        self.assertNotEqual(original.adapters, corrupted.adapters)


if __name__ == "__main__":
    unittest.main()
