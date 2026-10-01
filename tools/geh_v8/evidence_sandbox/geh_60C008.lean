import GEH.Compliance.GehGuard

def gehPae (pt payload : String) : String :=
  "DSSEv1 " ++ toString pt.length ++ " " ++ pt ++ " " ++ toString payload.length ++ " " ++ payload

def gehLink (prev m mod : Nat) : Nat := (prev * 31 + m) % mod

theorem geh_60C008 : ∀ (h60C0 : True),  gehLink 23399 11328 7919 = 230 := by
  decide

#geh_guard geh_60C008
