from __future__ import annotations

import unittest

from frontier.vllm_029_contract import (
    ARTIFACTS,
    FRONTIER_SELECTION_MERGE,
    INDEPENDENT_BYTE_EVIDENCE,
    TRL_COMPAT_REVISION,
    UPSTREAM_METADATA_EVIDENCE,
    VLLM_REVISION,
    QualificationContractError,
    evaluate_observation,
)


class Vllm029ContractTests(unittest.TestCase):
    def _hold(self, **overrides: object) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "artifact_key": "cpu-linux-x86_64-glibc234",
            "observed_size_bytes": None,
            "observed_sha256": None,
            "independent_hash_verified": False,
            "dependency_closure_captured": False,
            "hardware_identity_state": "NOT_RUN",
            "model_runner_v2_state": "NOT_RUN",
            "model_runner_v1_fallback_state": "NOT_RUN",
            "correctness_state": "NOT_RUN",
            "failure_restart_state": "NOT_RUN",
            "matched_performance_state": "NOT_RUN",
            "rollback_target_bound": False,
        }
        kwargs.update(overrides)
        return evaluate_observation(**kwargs)  # type: ignore[arg-type]

    def test_exact_frontier_source_pins_are_fixed(self) -> None:
        self.assertEqual(VLLM_REVISION, "98dff2a81d747d1dba01a47f939f48c3526d4206")
        self.assertEqual(TRL_COMPAT_REVISION, "488a0d34be07c7cb2c130e8ba44cb9a4103717b2")
        self.assertEqual(FRONTIER_SELECTION_MERGE, "e4661fe971fc222c7b91c42851d3bb43f89f68f9")
        self.assertEqual(ARTIFACTS["cpu-linux-x86_64-glibc234"].asset_id, 552379258)
        self.assertEqual(ARTIFACTS["cuda129-linux-x86_64-glibc228"].asset_id, 552379252)

    def test_upstream_metadata_never_claims_independent_byte_verification(self) -> None:
        result = self._hold(
            observed_size_bytes=ARTIFACTS["cpu-linux-x86_64-glibc234"].size_bytes,
            observed_sha256=ARTIFACTS["cpu-linux-x86_64-glibc234"].sha256,
        )
        self.assertEqual(result["evidenceClass"], UPSTREAM_METADATA_EVIDENCE)
        self.assertEqual(result["disposition"], "HOLD")
        self.assertIn("artifact_bytes_not_independently_verified", result["blockers"])

    def test_independent_verification_rejects_digest_or_size_drift(self) -> None:
        artifact = ARTIFACTS["cpu-linux-x86_64-glibc234"]
        with self.assertRaises(QualificationContractError):
            self._hold(
                observed_size_bytes=artifact.size_bytes,
                observed_sha256="0" * 64,
                independent_hash_verified=True,
            )
        with self.assertRaises(QualificationContractError):
            self._hold(
                observed_size_bytes=artifact.size_bytes + 1,
                observed_sha256=artifact.sha256,
                independent_hash_verified=True,
            )

    def test_model_runner_v2_pass_requires_real_prerequisites(self) -> None:
        artifact = ARTIFACTS["cuda129-linux-x86_64-glibc228"]
        with self.assertRaises(QualificationContractError):
            self._hold(
                artifact_key="cuda129-linux-x86_64-glibc228",
                observed_size_bytes=artifact.size_bytes,
                observed_sha256=artifact.sha256,
                model_runner_v2_state="PASS",
            )

    def test_fully_observed_lane_is_still_evaluation_only(self) -> None:
        artifact = ARTIFACTS["cuda129-linux-x86_64-glibc228"]
        result = self._hold(
            artifact_key="cuda129-linux-x86_64-glibc228",
            observed_size_bytes=artifact.size_bytes,
            observed_sha256=artifact.sha256,
            independent_hash_verified=True,
            dependency_closure_captured=True,
            hardware_identity_state="PASS",
            model_runner_v2_state="PASS",
            model_runner_v1_fallback_state="PASS",
            correctness_state="PASS",
            failure_restart_state="PASS",
            matched_performance_state="PASS",
            rollback_target_bound=True,
        )
        self.assertEqual(result["evidenceClass"], INDEPENDENT_BYTE_EVIDENCE)
        self.assertEqual(result["disposition"], "EVALUATION")
        self.assertEqual(result["blockers"], [])
        self.assertFalse(result["productionTrainingAuthorized"])
        self.assertFalse(result["productionServingAuthorized"])
        self.assertFalse(result["hubPublicationAuthorized"])
        self.assertFalse(result["weightRehostingAuthorized"])
        self.assertFalse(result["automaticPromotionAuthorized"])
        self.assertFalse(result["productionDefaultChangeAuthorized"])

    def test_malformed_observation_types_fail_closed(self) -> None:
        with self.assertRaises(QualificationContractError):
            self._hold(independent_hash_verified="false")
        with self.assertRaises(QualificationContractError):
            self._hold(model_runner_v2_state="success")
        with self.assertRaises(QualificationContractError):
            self._hold(observed_sha256="ABC")
        with self.assertRaises(QualificationContractError):
            self._hold(artifact_key="cuda")


if __name__ == "__main__":
    unittest.main()
