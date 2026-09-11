# SPDX-License-Identifier: Apache-2.0
"""Fail-closed tests for the public sealed-run counter watcher."""
from __future__ import annotations

import json
import unittest

from frontier.evaluation.sealed_count import encode, genesis
from inference.sealed_count_watch import (
    SealedCountWatchError,
    observe,
)


class SealedCountWatchTests(unittest.TestCase):
    def test_missing_public_file_is_unavailable_not_zero(self):
        observation = observe(None, http_status=404, observed_at="2026-09-11T16:00:00Z")
        self.assertEqual(observation["observationStatus"], "UNAVAILABLE")
        self.assertEqual(observation["reasonCode"], "public_counter_absent")
        self.assertNotIn("total_sealed", observation)
        self.assertEqual(observation["productionDisposition"], "HOLD")
        self.assertEqual(observation["promotionEffect"], "NONE")

    def test_non_404_fetch_failure_is_unavailable_not_zero(self):
        observation = observe(None, http_status=0, observed_at="2026-09-11T16:00:00Z")
        self.assertEqual(observation["observationStatus"], "UNAVAILABLE")
        self.assertEqual(observation["reasonCode"], "public_counter_fetch_failed")
        self.assertNotIn("total_sealed", observation)

    def test_genesis_document_is_watch_with_honest_zero(self):
        raw = encode(genesis())
        observation = observe(raw, http_status=200, observed_at="2026-09-11T16:00:00Z")
        self.assertEqual(observation["observationStatus"], "WATCH")
        self.assertEqual(observation["total_sealed"], 0)
        self.assertEqual(observation["day_count"], 0)
        self.assertEqual(observation["productionDisposition"], "HOLD")

    def test_extra_key_fails_closed(self):
        payload = json.loads(encode(genesis()).decode("utf-8"))
        payload["provider"] = "glm"
        raw = json.dumps(payload).encode("utf-8")
        with self.assertRaises(SealedCountWatchError):
            observe(raw, http_status=200)

    def test_oversized_body_fails_closed(self):
        with self.assertRaises(SealedCountWatchError):
            observe(b"{" + (b"x" * (65 * 1024)), http_status=200)


if __name__ == "__main__":
    unittest.main()
