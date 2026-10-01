/-
  GEH Axiom Whitelist Guard v3 — the Tao boundary, build-failing.

  Compliance-grade proofs may depend ONLY on the human-owned trust base
  {propext, Classical.choice, Quot.sound}. Anything else — `sorryAx`,
  `Lean.ofReduceBool` (native_decide), user axioms — FAILS THE BUILD.
  The agent cannot widen this list; changing it is a code-review event.

  Usage:   #geh_guard myTheorem
  Emits:   GEH_GUARD {"tool":"lean.query_axioms","theorem":"myTheorem",
                      "axioms":[...],"allowed":true}
  Pinned:  leanprover/lean4:v4.18.0
-/
import Lean
open Lean Elab Command

namespace GEH

/-- Human-owned trust base. Policy lives here and nowhere else. -/
def allowedAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

def axiomsAllowed (ax : List Name) : Bool :=
  ax.all fun a => allowedAxioms.contains a

/-- Stable emission order (dedup + sort) so receipts are deterministic. -/
def normalize (ax : Array Name) : List Name :=
  (ax.toList.eraseDups.toArray.qsort fun a b => a.toString < b.toString).toList

def guardJson (thm : Name) (axs : List Name) (allowed : Bool) : Json :=
  Json.mkObj
    [ ("tool", "lean.query_axioms")
    , ("theorem", toString thm)
    , ("axioms", toJson (axs.map toString))
    , ("allowed", allowed)
    , ("trust_base", toJson (allowedAxioms.map toString)) ]

end GEH

syntax (name := gehGuard) "#geh_guard " ident : command

/-- `#geh_guard thm` — emit the audit line for the receipt, or FAIL THE BUILD
    when `thm` depends on any axiom outside the trust base. -/
@[command_elab gehGuard] def elabGehGuard : CommandElab := fun stx => do
  match stx with
  | `(#geh_guard $n:ident) => do
    let name ← liftCoreM <| realizeGlobalConstNoOverload n
    let axs ← collectAxioms name
    let norm := GEH.normalize axs
    let ok := GEH.axiomsAllowed norm
    let line := "GEH_GUARD " ++ (GEH.guardJson name norm ok).compress
    if ok then
      logInfo line
    else
      throwError "{line}\nGEH GUARD FAILED: {name} depends on axioms outside the trust base {norm} (allowed: {GEH.allowedAxioms})"
  | _ => throwUnsupportedSyntax
