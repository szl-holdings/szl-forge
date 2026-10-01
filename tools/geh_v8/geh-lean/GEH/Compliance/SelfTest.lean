import GEH.Compliance.GehGuard

/-! Kernel-grounded self-tests for the guard. Green cases must PASS. -/

theorem geh_selftest_clean (p q : Prop) (hp : p) (hpq : p → q) : q := hpq hp
#geh_guard geh_selftest_clean

def gehPae (pt payload : String) : String :=
  "DSSEv1 " ++ toString pt.length ++ " " ++ pt ++ " " ++ toString payload.length ++ " " ++ payload

theorem geh_selftest_pae : gehPae "ab" "c" = "DSSEv1 2 ab 1 c" := by decide
#geh_guard geh_selftest_pae

theorem geh_selftest_pae_rfl : gehPae "a" "xyz" = "DSSEv1 1 a 3 xyz" := by rfl
#geh_guard geh_selftest_pae_rfl

theorem geh_selftest_disamb : gehPae "ab" "c" ≠ gehPae "a" "bc" := by decide
#geh_guard geh_selftest_disamb

def gehLink (prev m mod : Nat) : Nat := (prev * 31 + m) % mod

theorem geh_selftest_link : gehLink 123 456 7919 = 4269 := by rfl
#geh_guard geh_selftest_link

theorem geh_selftest_xor (a b : Bool) : Bool.xor (Bool.xor a b) b = a := by
  cases a <;> cases b <;> rfl
#geh_guard geh_selftest_xor

theorem geh_selftest_nodup : [3, 5, 7].eraseDups = [3, 5, 7] := by decide
#geh_guard geh_selftest_nodup

theorem geh_selftest_classical (p : Prop) : p ∨ ¬p := Classical.em p
#geh_guard geh_selftest_classical
