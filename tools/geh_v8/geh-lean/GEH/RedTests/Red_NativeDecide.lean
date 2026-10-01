import GEH.Compliance.GehGuard
-- RED TEST: must FAIL the build. native_decide => Lean.ofReduceBool outside the trust base.
theorem geh_red_native : 2 ^ 10 = 1024 := by native_decide
#geh_guard geh_red_native
