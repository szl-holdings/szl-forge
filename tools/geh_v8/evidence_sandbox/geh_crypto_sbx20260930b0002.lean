/- GEH crypto suite run=sbx20260930b0002 salt=20260930. AUTO-GENERATED; do not hand-edit. -/
import GEH.Compliance.GehGuard

def gehPae (pt payload : String) : String :=
  "DSSEv1 " ++ toString pt.length ++ " " ++ pt ++ " " ++ toString payload.length ++ " " ++ payload

def gehLink (prev m mod : Nat) : Nat := (prev * 31 + m) % mod

theorem geh_60C000 : gehPae "az" "ba" = "DSSEv1 2 az 2 ba" := by
  decide
#geh_guard geh_60C000

theorem geh_60C001 : gehPae "ya" "a" ≠ gehPae "y" "aa" := by
  decide
#geh_guard geh_60C001

theorem geh_60C002 : ∀ (b60C c002 : Bool),  Bool.xor (Bool.xor b60C c002) c002 = b60C := by
  intro b60C c002
  cases b60C <;> cases c002 <;> rfl
#geh_guard geh_60C002

theorem geh_60C003 : gehLink 7989 31300 65521 = 16875 := by
  rfl
#geh_guard geh_60C003

theorem geh_60C004 : [141, 310, 429, 482].eraseDups = [141, 310, 429, 482] := by
  decide
#geh_guard geh_60C004

theorem geh_60C005 : gehPae "c" "bdbx" = "DSSEv1 1 c 4 bdbx" := by
  decide
#geh_guard geh_60C005

theorem geh_60C006 : gehPae "bx" "y" ≠ gehPae "b" "xy" := by
  decide
#geh_guard geh_60C006

theorem geh_60C007 : ∀ (b60C c007 : Bool), Bool.xor (Bool.xor b60C c007) c007 = b60C := by
  intro b60C c007
  cases b60C <;> cases c007 <;> rfl
#geh_guard geh_60C007

theorem geh_60C008 : ∀ (h60C0 : True),  gehLink 23399 11328 7919 = 230 := by
  intro h60C0
  rfl
#geh_guard geh_60C008

theorem geh_60C009 : [48, 275, 388, 433].eraseDups = [48, 275, 388, 433] := by
  decide
#geh_guard geh_60C009

