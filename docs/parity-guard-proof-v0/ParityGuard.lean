/-
ParityGuard.lean - SZL estate parity guard, formal contract (Phase 1).

Per szl-forge docs/frontier-20261115/parity-guard-verification-plan.md:
1. Formalize the guard's contract as Lean theorems (no sorry).
2. Referee: lake build must accept; no human-vibes acceptance.
3. Bind: proof files + sha256s; receipts cite the proof hash.

Scope honesty (fail-closed doctrine):
- This file formalizes the guard CONTRACT (CAS no-overwrite, monotonic
  revision advance, fail-closed default, hard-deny non-liftability) as an
  abstract typed model. It does NOT verify the Python implementation;
  binding model to code is Phase 4 of the plan.
- No new axioms; #print axioms blocks below show sorry-free status.

Estate artifact.
-/

namespace SZL.ParityGuard

/-- Revision identifiers are natural numbers, issued in increasing order. -/
abbrev Rev := Nat

/-- Guard verdict. Fail-closed: every unverified case is a deny. -/
inductive Verdict where
  | allow | deny
deriving DecidableEq, Repr

/-
The parity guard decision. A write is allowed only when BOTH hold:
- CAS: observed parent revision = expected parent revision
  (the writer proved knowledge of current state - no blind overwrite);
- monotonicity: the new revision strictly advances past the parent.
Everything else - including unknown observed state - is a deny.
-/
def guard (observedParent expectedParent newRev : Rev) : Verdict :=
  if observedParent = expectedParent ∧ expectedParent < newRev then Verdict.allow else Verdict.deny

/-- CAS property: an allowed write cannot have overwritten an unexpected parent. -/
theorem no_blind_overwrite {o e n : Rev} (h : guard o e n = Verdict.allow) : o = e := by
  unfold guard at h
  split at h
  next hc => exact hc.1
  simp at h

/-- Monotonicity: an allowed write strictly advances the revision. -/
theorem monotonic_advance {o e n : Rev} (h : guard o e n = Verdict.allow) : e < n := by
  unfold guard at h
  split at h
  next hc => exact hc.2
  simp at h

/-- Fail-closed completeness: matching parent + strict advance is allowed. -/
theorem guard_complete {o e n : Rev} (h1 : o = e) (h2 : e < n) : guard o e n = Verdict.allow := by
  unfold guard
  subst h1
  simp [h2]

/-- Fail-closed on mismatch: any unexpected parent is denied, whatever the content. -/
theorem guard_fails_closed_on_mismatch {o e n : Rev} (h : Not (o = e)) :
    guard o e n = Verdict.deny := by
  unfold guard
  by_cases hc : o = e ∧ e < n
  · exact absurd hc.1 h
  · simp [hc]

/-
Effect of the guard on state: an allowed write advances the revision,
a denied write leaves state unchanged (none).
-/
def advance (parent expected newRev : Rev) : Option Rev :=
  if guard parent expected newRev = Verdict.allow then some newRev else none

/-- Advancing strictly increases the revision (parent = expected < new). -/
theorem advance_strictly_increases {p e n : Rev} (h : advance p e n = some n) : p < n := by
  unfold advance at h
  split at h
  next hc =>
    rw [no_blind_overwrite hc]
    exact monotonic_advance hc
  simp at h

/-
Policy overlay (Memory Covenant v0.4 tie-in): a hard deny is absolute -
no number of approvals can lift it; without it, approvals enable;
zero approvals is still a deny (deny-by-default).
-/
def policyVerdict (hardDeny : Bool) (approvals : Nat) : Verdict :=
  if hardDeny then Verdict.deny
  else if approvals > 0 then Verdict.allow else Verdict.deny

/-- Hard denies cannot be lifted by any number of approvals. -/
theorem hard_deny_absolute (approvals : Nat) : policyVerdict true approvals = Verdict.deny := by
  simp [policyVerdict]

/-- Deny-by-default: no hard deny and zero approvals still denies. -/
theorem deny_by_default : policyVerdict false 0 = Verdict.deny := by
  simp [policyVerdict]

/-- With no hard deny, a positive number of approvals enables. -/
theorem approval_enables (approvals : Nat) (h : approvals > 0) :
    policyVerdict false approvals = Verdict.allow := by
  simp [policyVerdict, h]

#print axioms no_blind_overwrite
#print axioms monotonic_advance
#print axioms guard_complete
#print axioms guard_fails_closed_on_mismatch
#print axioms advance_strictly_increases
#print axioms hard_deny_absolute
#print axioms deny_by_default
#print axioms approval_enables

end SZL.ParityGuard
