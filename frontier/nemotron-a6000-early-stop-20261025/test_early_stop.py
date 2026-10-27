"""Synthetic state tests only: no model, corpus, files, clock, network, or GPU."""
from dataclasses import FrozenInstanceError, replace
from decimal import localcontext
import hashlib
import json
import unittest

import early_stop as es


def config(**changes):
    values = dict(candidate_id="SIMULATED-candidate", run_id="SIMULATED-run",
                  development_sha256="1" * 64, evaluation_interval=100,
                  direction="min", min_delta="0.01", patience=2)
    values.update(changes)
    return es.Config(**values)


def checkpoint(step):
    return es.Checkpoint("SIMULATED-step-" + str(step),
                         hashlib.sha256(("SIMULATED-" + str(step)).encode()).hexdigest())


def advance(state, metric):
    step = state.last_step + state.config.evaluation_interval
    return es.observe(state, config=state.config, step=step, metric=metric, checkpoint=checkpoint(step))


def reserialize(document):
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


class EarlyStopTests(unittest.TestCase):
    def test_worsening_trajectory_stops_and_preserves_first_best(self):
        state = es.new_state(config())
        for metric in ("3.0", "3.2", "3.5"):
            state = advance(state, metric)
        self.assertTrue(state.stopped)
        self.assertEqual((state.last_step, state.bad_evaluations, state.best_step), (300, 2, 100))
        self.assertEqual((state.best_metric, state.best_checkpoint), ("3", checkpoint(100)))
        self.assertEqual(state.stop_reason, "PATIENCE_EXHAUSTED")

    def test_improved_resume_matches_uninterrupted_exact_bytes(self):
        cfg = config(patience=3)
        losses = ("3", "2.9", "2.91", "2.7", "2.71", "2.8")
        uninterrupted = es.new_state(cfg)
        for loss in losses:
            uninterrupted = advance(uninterrupted, loss)
        resumed = es.new_state(cfg)
        for loss in losses[:3]:
            resumed = advance(resumed, loss)
        raw = es.canonical_bytes(resumed)
        resumed = es.restore(raw, config=cfg, expected_state_sha256=es.state_sha256(resumed))
        for loss in losses[3:]:
            resumed = advance(resumed, loss)
        self.assertEqual(es.canonical_bytes(uninterrupted), es.canonical_bytes(resumed))
        self.assertEqual((resumed.best_step, resumed.bad_evaluations), (400, 2))

    def test_stopped_latch_survives_resume_and_blocks_new_metric(self):
        state = advance(advance(es.new_state(config(patience=1)), "1"), "1.1")
        restored = es.restore(es.canonical_bytes(state), config=state.config,
                              expected_state_sha256=es.state_sha256(state))
        self.assertTrue(restored.stopped)
        with self.assertRaisesRegex(es.StopStateError, "EVALUATION_AFTER_STOP"):
            advance(restored, "0.1")

    def test_min_delta_is_strict_and_ties_consume_patience(self):
        state = advance(es.new_state(config(patience=4, min_delta="0.1")), "1.0")
        state = advance(state, "0.9")
        self.assertEqual((state.bad_evaluations, state.best_metric), (1, "1"))
        state = advance(state, "1.00")
        self.assertEqual(state.bad_evaluations, 2)
        state = advance(state, "0.899")
        self.assertEqual((state.bad_evaluations, state.best_step, state.best_metric), (0, 400, "0.899"))

    def test_decimal_comparison_ignores_ambient_precision(self):
        with localcontext() as ambient:
            ambient.prec = 1
            ambient.Emax = 2
            ambient.Emin = -2
            state = advance(es.new_state(config(min_delta="0")), "0.3000000000000000000001")
            state = advance(state, "0.3000000000000000000000")
            large = advance(es.new_state(config(min_delta="0.1")), "9000000000000000000000000000000")
            large = advance(large, "8000000000000000000000000000000")
        self.assertEqual((state.best_step, state.bad_evaluations, state.best_metric), (200, 0, "0.3"))
        self.assertEqual(large.best_step, 200)

    def test_decimal_canonicalization(self):
        for raw, canonical in (("-0.000", "0"), ("2.500", "2.5"), ("-3.100", "-3.1")):
            with self.subTest(raw=raw):
                state = advance(es.new_state(config()), raw)
                self.assertEqual(state.best_metric, canonical)
        self.assertEqual(config(min_delta="0.0100").min_delta, "0.01")

    def test_malformed_or_nonfinite_metric_does_not_change_original_state(self):
        state = advance(es.new_state(config()), "1")
        before = es.canonical_bytes(state)
        for value in (True, False, 1, 1.1, None, "NaN", "Infinity", "-Infinity", "1e2",
                      "+1", "01", " .1", "1.", "1\n", "9" * 65, [], {}):
            with self.subTest(value=value):
                with self.assertRaises(es.StopStateError):
                    advance(state, value)
                self.assertEqual(es.canonical_bytes(state), before)

    def test_repeated_backward_skipped_and_boolean_steps_rejected(self):
        state = advance(es.new_state(config()), "1")
        for step in (100, 99, 0, -1, 201, 300, True, 200.0):
            with self.subTest(step=step):
                with self.assertRaises(es.StopStateError):
                    es.observe(state, config=state.config, step=step, metric="0.5", checkpoint=checkpoint(200))
        with self.assertRaisesRegex(es.StopStateError, "INVALID_STEP"):
            es.observe(es.new_state(config()), config=config(), step=0, metric="1", checkpoint=checkpoint(0))
        with self.assertRaisesRegex(es.StopStateError, "EVALUATION_STEP_MISMATCH"):
            es.observe(es.new_state(config()), config=config(), step=50, metric="1", checkpoint=checkpoint(50))

    def test_all_valid_config_drift_is_rejected_on_observe_and_restore(self):
        state = advance(es.new_state(config()), "1")
        changes = ({"candidate_id": "other"}, {"run_id": "other"}, {"development_sha256": "2" * 64},
                   {"evaluation_interval": 200}, {"min_delta": "0.02"}, {"patience": 3})
        for change in changes:
            with self.subTest(change=change):
                changed = replace(state.config, **change)
                with self.assertRaisesRegex(es.StopStateError, "CONFIG_BINDING_MISMATCH"):
                    es.observe(state, config=changed, step=200, metric="0.5", checkpoint=checkpoint(200))
                with self.assertRaisesRegex(es.StopStateError, "CONFIG_BINDING_MISMATCH"):
                    es.restore(es.canonical_bytes(state), config=changed, expected_state_sha256=es.state_sha256(state))

    def test_bad_config_and_checkpoint_types_rejected(self):
        for change in ({"direction": "max"}, {"direction": True}, {"evaluation_interval": True},
                       {"evaluation_interval": 0}, {"patience": True}, {"patience": 0},
                       {"min_delta": "-0.1"}, {"min_delta": float("nan")},
                       {"development_sha256": "A" * 64}, {"candidate_id": ""}, {"run_id": None}):
            with self.subTest(change=change):
                with self.assertRaises(es.StopStateError):
                    config(**change)
        with self.assertRaises(es.StopStateError):
            es.Checkpoint("checkpoint", "invalid")
        with self.assertRaises(es.StopStateError):
            es.observe(es.new_state(config()), config=config(), step=100, metric="1", checkpoint={})

    def test_empty_state_roundtrip_and_objects_are_frozen(self):
        state = es.new_state(config())
        self.assertEqual(state, es.restore(es.canonical_bytes(state), config=state.config,
                                         expected_state_sha256=es.state_sha256(state)))
        with self.assertRaises(FrozenInstanceError):
            state.stopped = True
        with self.assertRaises(FrozenInstanceError):
            state.config.patience = 999

    def test_tampering_and_old_state_rejected_by_external_latest_pin(self):
        old = advance(es.new_state(config()), "1")
        latest = advance(old, "1.1")
        with self.assertRaisesRegex(es.StopStateError, "STATE_HASH_MISMATCH"):
            es.restore(es.canonical_bytes(old), config=latest.config, expected_state_sha256=es.state_sha256(latest))
        with self.assertRaisesRegex(es.StopStateError, "STATE_HASH_MISMATCH"):
            es.restore(es.canonical_bytes(latest).replace(b'"1.1"', b'"0.1"'),
                       config=latest.config, expected_state_sha256=es.state_sha256(latest))

    def test_structurally_invalid_rehashed_state_fails_closed(self):
        stopped = advance(advance(es.new_state(config(patience=1)), "1"), "1.1")
        changes = ({"stopped": False}, {"stop_reason": None}, {"evaluations": True},
                   {"bad_evaluations": False}, {"last_step": 100}, {"best_step": 150},
                   {"best_metric": "1.00"}, {"last_metric": "0.5"}, {"best_checkpoint": None},
                   {"surprise": 1}, {"schema": "other"})
        for change in changes:
            with self.subTest(change=change):
                document = json.loads(es.canonical_bytes(stopped))
                document.update(change)
                raw = reserialize(document)
                with self.assertRaises(es.StopStateError):
                    es.restore(raw, config=stopped.config, expected_state_sha256=hashlib.sha256(raw).hexdigest())

    def test_noncanonical_duplicate_and_nonfinite_json_rejected(self):
        state = advance(es.new_state(config()), "1")
        raw = es.canonical_bytes(state)
        variants = (raw.rstrip(), json.dumps(json.loads(raw), indent=2).encode(),
                    raw.replace(b'"evaluations":1', b'"evaluations":1,"evaluations":1'),
                    raw.replace(b'"last_metric":"1"', b'"last_metric":NaN'),
                    raw.replace(b'"last_metric":"1"', b'"last_metric":1.0'),
                    b"\xff", b"[" * 2000 + b"]" * 2000)
        for changed in variants:
            with self.subTest(length=len(changed)):
                with self.assertRaises(es.StopStateError):
                    es.restore(changed, config=state.config, expected_state_sha256=hashlib.sha256(changed).hexdigest())

    def test_state_size_and_required_hash_limits(self):
        state = es.new_state(config())
        for raw in (b"", b" " * (es.MAX_STATE_BYTES + 1), "not-bytes"):
            with self.subTest(type=type(raw).__name__):
                with self.assertRaises(es.StopStateError):
                    es.restore(raw, config=state.config, expected_state_sha256="0" * 64)
        with self.assertRaises(es.StopStateError):
            es.restore(es.canonical_bytes(state), config=state.config, expected_state_sha256=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)

