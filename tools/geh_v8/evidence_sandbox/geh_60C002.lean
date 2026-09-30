import GEH.Compliance.GehGuard

def gehPae (pt payload : String) : String :=
  "DSSEv1 " ++ toString pt.length ++ " " ++ pt ++ " " ++ toString payload.length ++ " " ++ payload

def gehLink (prev m mod : Nat) : Nat := (prev * 31 + m) % mod

theorem geh_60C002 : ∀ (b60C c002 : Bool),  Bool.xor (Bool.xor b60C c002) c002 = b60C := by
  intro b60C c002
  cases b60C <;> cases c002 <;> rfl

#geh_guard geh_60C002
