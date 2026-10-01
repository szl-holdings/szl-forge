import GEH.Compliance.GehGuard

theorem geh_k_green : ∀ (p q : Prop), p → (p → q) → q := by
  intro p q hp hpq
  exact hpq hp

#geh_guard geh_k_green
