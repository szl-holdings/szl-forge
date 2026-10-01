import GEH.Compliance.GehGuard

def gehPae (pt payload : String) : String :=
  "DSSEv1 " ++ toString pt.length ++ " " ++ pt ++ " " ++ toString payload.length ++ " " ++ payload

def gehLink (prev m mod : Nat) : Nat := (prev * 31 + m) % mod

theorem geh_60C003 : gehLink 7989 31300 65521 = 16875 := by
  rfl

#geh_guard geh_60C003
