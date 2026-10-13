"""Artificial byte fixtures, not model memory measurements."""
from dataclasses import replace
import unittest

from residency_plan import Budget, InputError, PHASES, Reservation, evaluate_plan, simulate_abort


def fixture():
    return (
        Reservation("original", "disk-store", "disk", 100, PHASES, True, True, role="original"),
        Reservation("derived", "disk-store", "disk", 60, PHASES, True, role="derived"),
        Reservation("adapter", "s1", "gpu", 8, PHASES, True, device="cuda:1", role="adapter"),
        Reservation("grad", "s1", "gpu", 8, ("backward", "optimizer"), device="cuda:1"),
        Reservation("moments", "s1", "gpu", 16, ("optimizer", "checkpoint"), True, device="cuda:1"),
        Reservation("layer", "s1", "gpu", 30, ("forward", "backward"), device="cuda:1"),
        Reservation("prefetch", "s1", "gpu", 30, ("forward",), device="cuda:1"),
        Reservation("activations", "s1", "host", 40, ("forward", "backward")),
        Reservation("load", "s1", "host", 50, ("construction",)),
        Reservation("scratch", "s1", "gpu", 20, ("backward",), device="cuda:1"),
        Reservation("checkpoint-write", "s1", "disk", 70, ("checkpoint",)),
    )


def plan(items=None, budgets=None, **kw):
    return evaluate_plan(
        fixture() if items is None else items,
        {t: Budget(1000, 0) for t in ("gpu", "host", "disk")} if budgets is None else budgets,
        base_id="fixture/artificial-dense", base_revision="1" * 40,
        target_device="cuda:1", **kw)


class ResidencyTests(unittest.TestCase):
    def test_explicit_phase_peaks(self):
        result = plan()
        self.assertEqual(result["peak_required_bytes"], {"gpu": 68, "host": 50, "disk": 130})
        self.assertTrue(result["fits_declared_budget"])
        self.assertEqual(result["decision"], "MODELED")

    def test_no_authority_even_if_fitting(self):
        result = plan()
        for flag in ("training_eligible", "publication_eligible", "execution_authority"):
            self.assertIs(result[flag], False)
        self.assertEqual(result["manifest_completeness"], "UNKNOWN")

    def test_existing_original_retained_without_double_charging_free_space(self):
        result = plan(budgets={"gpu": Budget(1000, 0), "host": Budget(1000, 0), "disk": Budget(130, 0)})
        self.assertTrue(result["fits_declared_budget"])
        self.assertEqual(result["phase_accounting"][-1]["gross_bytes"]["disk"], 230)

    def test_new_original_is_charged(self):
        items = (replace(fixture()[0], already_allocated=False),) + fixture()[1:]
        self.assertEqual(plan(items)["peak_required_bytes"]["disk"], 230)

    def test_checkpoint_peak_blocks(self):
        result = plan(budgets={"gpu": Budget(1000, 0), "host": Budget(1000, 0), "disk": Budget(129, 0)})
        self.assertEqual(result["decision"], "BLOCKED")
        self.assertEqual(result["budget_excesses"][0]["phase"], "checkpoint")

    def test_prefetch_is_in_live_peak(self):
        result = plan(budgets={"gpu": Budget(67, 0), "host": Budget(1000, 0), "disk": Budget(1000, 0)})
        self.assertEqual(result["budget_excesses"][0]["phase"], "forward")

    def test_backward_scratch_and_optimizer_moments_are_counted(self):
        rows = {r["phase"]: r for r in plan()["phase_accounting"]}
        self.assertEqual(rows["backward"]["gross_bytes"]["gpu"], 66)
        self.assertEqual(rows["optimizer"]["gross_bytes"]["gpu"], 32)

    def test_construction_host_peak_blocks(self):
        result = plan(budgets={"gpu": Budget(1000, 0), "host": Budget(49, 0), "disk": Budget(1000, 0)})
        self.assertEqual(result["budget_excesses"][0]["phase"], "construction")

    def test_reserves_count_in_every_phase(self):
        result = plan(budgets={"gpu": Budget(70, 3), "host": Budget(1000, 0), "disk": Budget(1000, 0)})
        self.assertEqual(result["peak_required_bytes"]["gpu"], 71)
        self.assertFalse(result["fits_declared_budget"])

    def test_unknown_allocation_never_becomes_zero(self):
        items = tuple(replace(r, size_bytes=None) if r.name == "scratch" else r for r in fixture())
        result = plan(items)
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertIsNone(result["fits_declared_budget"])
        self.assertIsNone(result["peak_required_bytes"]["gpu"])

    def test_unknown_existing_original_still_unknown(self):
        items = (replace(fixture()[0], size_bytes=None),) + fixture()[1:]
        self.assertEqual(plan(items)["decision"], "UNKNOWN")

    def test_unknown_budget_never_becomes_infinite(self):
        result = plan(budgets={"gpu": Budget(None, 0), "host": Budget(1000, 0), "disk": Budget(1000, 0)})
        self.assertEqual(result["decision"], "UNKNOWN")

    def test_known_excess_overrides_unknown(self):
        result = plan(budgets={"gpu": Budget(None, 0), "host": Budget(1, 0), "disk": Budget(1000, 0)})
        self.assertEqual(result["decision"], "BLOCKED")

    def test_invalid_byte_types_rejected(self):
        for value in (-1, True, 1.5, float("nan"), float("inf"), "10"):
            with self.subTest(value=value), self.assertRaises(InputError):
                Budget(value)
            with self.subTest(value=value), self.assertRaises(InputError):
                replace(fixture()[0], size_bytes=value)

    def test_unknown_reserve_never_becomes_zero(self):
        result = plan(budgets={"gpu": Budget(100), "host": Budget(1000, 0), "disk": Budget(1000, 0)})
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertIsNone(result["peak_required_bytes"]["gpu"])

    def test_unknown_fields_cannot_hide_known_lower_bound_excess(self):
        result = plan(budgets={"gpu": Budget(60), "host": Budget(1000, 0), "disk": Budget(1000, 0)})
        self.assertEqual(result["decision"], "BLOCKED")
        self.assertEqual(result["budget_excesses"][0]["known_lower_bound_bytes"], 68)
        self.assertEqual(result["budget_excesses"][0]["over_bytes_at_least"], 8)

    def test_wrong_device_rejected(self):
        items = tuple(replace(r, device="cuda:0") if r.tier == "gpu" else r for r in fixture())
        with self.assertRaises(InputError):
            plan(items)

    def test_default_or_non_gpu_device_rejected(self):
        with self.assertRaises(InputError):
            replace(fixture()[2], device="cuda")
        with self.assertRaises(InputError):
            replace(fixture()[7], device="cuda:1")

    def test_missing_expert_adapter_coverage_rejected(self):
        with self.assertRaises(InputError):
            plan(expected_adapter_targets=("attention", "packed-expert"),
                 actual_adapter_targets=("attention",))

    def test_duplicate_target_rejected(self):
        with self.assertRaises(InputError):
            plan(expected_adapter_targets=("attention", "attention"), actual_adapter_targets=("attention",))

    def test_missing_budget_and_empty_manifest_rejected(self):
        with self.assertRaises(InputError):
            plan(budgets={"gpu": Budget(100)})
        with self.assertRaises(InputError):
            plan(())

    def test_duplicate_reservation_rejected(self):
        with self.assertRaises(InputError):
            plan(fixture() + (fixture()[0],))

    def test_no_unknown_phase_or_implicit_lifetime(self):
        for phases in ((), ("forward", "forward"), ("decode",), ["forward"]):
            with self.subTest(phases=phases), self.assertRaises(InputError):
                replace(fixture()[0], phases=phases)

    def test_original_cannot_be_deleted_in_backward(self):
        with self.assertRaises(InputError):
            replace(fixture()[0], phases=("forward",))
        with self.assertRaises(InputError):
            replace(fixture()[0], persistent=False)

    def test_persistent_asset_cannot_vanish_after_construction(self):
        with self.assertRaises(InputError):
            Reservation("derived", "disk", "disk", 60, ("construction",), True)

    def test_persistent_lifetime_cannot_have_gaps(self):
        with self.assertRaises(InputError):
            Reservation("resident", "s1", "gpu", 60,
                        ("forward", "backward", "checkpoint"), True, device="cuda:1")

    def test_existing_disk_asset_must_be_accounted_in_every_phase(self):
        with self.assertRaises(InputError):
            Reservation("existing-derived", "disk", "disk", 60,
                        ("optimizer", "checkpoint"), True, True)

    def test_already_allocated_cannot_hide_gpu_use(self):
        with self.assertRaises(InputError):
            replace(fixture()[2], already_allocated=True)

    def test_abort_keeps_persistent_and_other_owner_state(self):
        other = Reservation("other-session", "s2", "host", 1, ("backward",))
        result = simulate_abort(fixture() + (other,), owner="s1", phase="backward")
        self.assertEqual(set(result["released_temporary_names"]), {"grad", "layer", "activations", "scratch"})
        self.assertIn("original", result["retained_names"])
        self.assertIn("adapter", result["retained_names"])
        self.assertIn("other-session", result["retained_names"])
        self.assertEqual(result["real_deletions"], 0)
        self.assertIs(result["execution_authority"], False)

    def test_digest_binds_budget_and_lifetime(self):
        first = plan()["request_sha256"]
        self.assertEqual(first, plan()["request_sha256"])
        changed = plan(budgets={"gpu": Budget(999, 0), "host": Budget(1000, 0), "disk": Budget(1000, 0)})
        self.assertNotEqual(first, changed["request_sha256"])
        self.assertNotEqual(first, plan((replace(fixture()[0], size_bytes=101),) + fixture()[1:])["request_sha256"])

    def test_huge_integer_bytes_are_exact(self):
        items = (Reservation("original", "disk", "disk", 2**100, PHASES, True, role="original"),)
        result = plan(items, budgets={"gpu": Budget(0, 0), "host": Budget(0, 0), "disk": Budget(2**100 - 1, 0)})
        self.assertEqual(result["budget_excesses"][0]["over_bytes_at_least"], 1)

    def test_abort_duplicate_names_rejected(self):
        with self.assertRaises(InputError):
            simulate_abort(fixture() + (fixture()[0],), owner="s1", phase="backward")


if __name__ == "__main__":
    unittest.main()
