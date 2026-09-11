import unittest

import hf_core_stack_guard as guard


class CoreStackGuardTests(unittest.TestCase):
    def releases(self):
        return [
            {"package": name, "version": spec["version"], "revision": spec["revision"]}
            for name, spec in guard.EXPECTED.items()
        ]

    def checks(self, value="PASS"):
        return {name: value for name in guard.REQUIRED_CHECKS}

    def test_plan_is_hold_and_denies_authority(self):
        plan = guard.plan()
        self.assertEqual(plan["disposition"], "HOLD")
        self.assertTrue(all(v is False for v in plan["authority"].values()))
        self.assertIn("UNAVAILABLE", plan["allowedResults"])

    def test_exact_release_set_can_be_evaluated_but_not_promoted(self):
        result = guard.validate_evidence({"releases": self.releases(), "checks": self.checks()})
        self.assertTrue(result["allChecksPass"])
        self.assertEqual(result["disposition"], "HOLD")
        self.assertFalse(result["authority"]["automaticPromotion"])

    def test_unavailable_hardware_is_explicit_and_not_pass(self):
        checks = self.checks()
        checks["accelerate_fsdp2_checkpoint_offload"] = "UNAVAILABLE"
        result = guard.validate_evidence({"releases": self.releases(), "checks": checks})
        self.assertFalse(result["allChecksPass"])
        self.assertEqual(result["checks"]["accelerate_fsdp2_checkpoint_offload"], "UNAVAILABLE")

    def test_moving_or_wrong_revision_fails_closed(self):
        releases = self.releases()
        releases[0] = dict(releases[0], revision="0" * 40)
        with self.assertRaises(guard.CoreStackError):
            guard.validate_evidence({"releases": releases, "checks": self.checks()})

    def test_missing_check_fails_closed(self):
        checks = self.checks()
        checks.pop("trl_removed_ppo_migration")
        with self.assertRaises(guard.CoreStackError):
            guard.validate_evidence({"releases": self.releases(), "checks": checks})

    def test_duplicate_release_fails_closed(self):
        releases = self.releases()
        releases.append(dict(releases[0]))
        with self.assertRaises(guard.CoreStackError):
            guard.validate_evidence({"releases": releases, "checks": self.checks()})


if __name__ == "__main__":
    unittest.main()
