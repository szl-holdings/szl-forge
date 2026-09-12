"""Fixture-lane coverage for Ghlore retrieval. No upstream daemon."""
from __future__ import annotations

import unittest

from frontier.ghlore_retrieval_fixture import (
    DENIED_AUTHORITY,
    UPSTREAM_REVISION,
    RetrievalContractError,
    evaluate_fixture_lane,
    retrieve,
)


class GhloreRetrievalFixtureTests(unittest.TestCase):
    def test_exact_source_pin_is_preserved(self) -> None:
        result = evaluate_fixture_lane()
        self.assertEqual(result["sourceRevision"], UPSTREAM_REVISION)
        self.assertEqual(result["sourceRevision"], "87290a46c79e26ebb0d47575263b7ee34c1c1390")
        self.assertFalse(result["upstreamRuntimeBenchmark"])
        self.assertFalse(result["ghloreDaemonExecuted"])

    def test_named_baselines_retrieve_expected_docs_with_citations(self) -> None:
        result = evaluate_fixture_lane()
        self.assertEqual(result["baselineHits"], result["expectedQueries"])
        self.assertEqual(result["disposition"], "EVALUATION")
        self.assertEqual(result["lane"], "FIXTURE_RETRIEVAL_MEASURED")
        router = result["evidence"]["Q-ROUTER"]
        self.assertEqual(router["hits"][0]["doc_id"], "DOC-FRESH-ROUTER")
        self.assertIn("DOC-FRESH-ROUTER", router["hits"][0]["citation"])
        verify = result["evidence"]["Q-VERIFY"]
        self.assertEqual(verify["hits"][0]["doc_id"], "DOC-CITE-COSIGN")
        self.assertTrue(verify["hits"][0]["truncated"] or len(verify["hits"][0]["excerpt"]) <= 160)

    def test_authority_flags_stay_false(self) -> None:
        result = evaluate_fixture_lane()
        for key, value in DENIED_AUTHORITY.items():
            self.assertIs(result[key], value)
            self.assertIs(result[key], False)

    def test_unavailable_backend_is_empty_and_named(self) -> None:
        down = retrieve("where is the inference flagship bound", backend_available=False)
        self.assertEqual(down["backend"], "UNAVAILABLE")
        self.assertEqual(down["hits"], [])
        self.assertEqual(down["reason"], "unavailable_backend")
        held = evaluate_fixture_lane(backend_available=False)
        self.assertEqual(held["disposition"], "HOLD")
        self.assertIn("upstream_runtime_unavailable", held["reasons"])

    def test_empty_query_and_injection_do_not_admit_private_corpus(self) -> None:
        empty = retrieve("   ")
        self.assertTrue(empty["empty"])
        self.assertEqual(empty["hits"], [])
        result = evaluate_fixture_lane()
        self.assertNotIn("injection_overclaimed_private_access", result["reasons"])
        self.assertTrue(result["evidence"]["empty_query"]["empty"])

    def test_forbidden_effectors_fail_closed(self) -> None:
        for kwargs in (
            {"trust_network_enabled": True},
            {"private_corpus_admitted": True},
            {"daemon_started": True},
        ):
            with self.subTest(kwargs=kwargs):
                result = evaluate_fixture_lane(**kwargs)
                self.assertEqual(result["disposition"], "HOLD")
                self.assertTrue(result["reasons"])

    def test_malformed_flags_raise_contract_error(self) -> None:
        with self.assertRaises(RetrievalContractError):
            evaluate_fixture_lane(trust_network_enabled="false")  # type: ignore[arg-type]
        with self.assertRaises(RetrievalContractError):
            retrieve("q", backend_available="true")  # type: ignore[arg-type]
        with self.assertRaises(RetrievalContractError):
            retrieve("q", max_chars=0)


if __name__ == "__main__":
    unittest.main()
