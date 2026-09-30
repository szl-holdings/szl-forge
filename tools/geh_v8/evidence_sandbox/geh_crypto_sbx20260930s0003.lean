/- GEH crypto suite run=sbx20260930s0003 salt=20260930. AUTO-GENERATED; do not hand-edit. -/
import GEH.Compliance.GehGuard

def gehPae (pt payload : String) : String :=
  "DSSEv1 " ++ toString pt.length ++ " " ++ pt ++ " " ++ toString payload.length ++ " " ++ payload

def gehLink (prev m mod : Nat) : Nat := (prev * 31 + m) % mod

theorem geh_60C000 : ∀ (h60C0 : True),  gehPae "bxc" "x" = "DSSEv1 3 bxc 1 x" := by
  intro h60C0
  decide
#geh_guard geh_60C000

theorem geh_60C001 : gehPae "cb" "d" ≠ gehPae "c" "bd" := by
  decide
#geh_guard geh_60C001

theorem geh_60C002 : ∀ (b60C c002 : Bool),  Bool.xor (Bool.xor b60C c002) c002 = b60C := by
  intro b60C c002
  cases b60C <;> cases c002 <;> rfl
#geh_guard geh_60C002

theorem geh_60C003 : ∀ (h60C0 : True), gehLink 75038 27007 65521 = 59950 := by
  intro h60C0
  rfl
#geh_guard geh_60C003

