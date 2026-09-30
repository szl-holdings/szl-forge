import GEH.Compliance.GehGuard
axiom gehForeign : ∀ (p : Prop), p
theorem geh_k_red4 : 1 = 2 := gehForeign _
#geh_guard geh_k_red4
