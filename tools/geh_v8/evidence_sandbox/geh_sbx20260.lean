import GEH.Compliance.GehGuard

theorem geh_sbx20260 : forall (p q : Prop), p -> (p -> q) -> q := by
  intro p q hp hpq
  exact hpq hp

#geh_guard geh_sbx20260
