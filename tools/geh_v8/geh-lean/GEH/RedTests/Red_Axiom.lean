import GEH.Compliance.GehGuard
-- RED TEST: must FAIL the build. A user axiom is a foreign axiom.
axiom gehForeign : ∀ (p : Prop), p
theorem geh_red_axiom : 1 = 2 := gehForeign _
#geh_guard geh_red_axiom
