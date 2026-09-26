# Access Lab: first useful pilot

This is an execution plan, not evidence of customers, revenue, successful user
testing, or a trained accessibility model. The initial product is a local review
workbench for one restricted form-label repair. Do not sell it as a whole-site
audit, certification service, or autonomous remediation agent.

## First customer and task

Proposed audience: a small web agency or nonprofit maintaining an ordinary
contact, registration, or donation form. Validate the problem with a consenting
owner before adding more model families or infrastructure.

1. Ask the owner which form task is difficult to maintain, what they do today,
   and whether a reviewed, reproducible repair would save useful work. Record
   the answer without interpreting interest as a paid commitment.
2. Use at most five owner-approved, non-sensitive static examples within the
   accepted grammar. Remove real submissions, secrets, identifiers, and tracking.
   Keep written processing and reuse permission separate from the source.
3. Run the deterministic baseline, retain original and repaired source hashes,
   and review the exact change. A held case stays held; do not weaken the gate
   to improve the acceptance rate.
4. Test the owner's real workflow in a separate controlled application. Include
   keyboard operation and a relevant screen-reader/browser combination with a
   qualified reviewer or consenting user. The workbench itself does not execute
   submitted HTML or perform these evaluations.
5. Return an explicit accepted/held/regressed result for each case and ask whether
   the owner would use or pay for a repeatable review service. Do not contact
   anyone, publish their source, or incur participant costs without authorization.

## Record the outcome

For each permitted case record: a non-identifying case ID, source/output hashes,
tool version and source commit, repair state, independent human decision,
browser/assistive-technology versions, attempted task, completion outcome, any
regression, and measured review time. Missing observations are `NOT_MEASURED`.
Do not record a pass merely because the HTTP request or automated rule passed.

Suggested initial exit criteria are five reviewed flows, zero observed regressions
in those flows, explicit evidence for every accepted change, and one owner who
confirms the workflow is useful. These are proposed milestones, not a statistical
generalization claim. Pause the pilot on an unexplained behavior change or
unexpected sensitive data exposure.

## When another model is justified

Keep the public development suite for regression testing. Before training, obtain
permitted examples of a genuinely useful capability the deterministic baseline
cannot provide. Separate families/templates across train, validation, and an
unseen test set; deduplicate before splitting. Freeze the test set independently
of tuning and record all tested candidates rather than only the winner.

Pin the existing parent model, license, dataset revision, prompt, training recipe,
and resulting artifact digest. Compare baseline, unmodified parent, and candidate
under the same execution and review gates. Record exact repairs, refusals, unsafe
changes, latency, and measured local resource use. A candidate must add validated
capability without unacceptable regressions; producing weights is not itself
success. Publish unsuccessful results honestly and do not relabel an unchanged
upstream model as an SZL-trained model.

Use already available local compute only. No cloud job, paid endpoint, new model
download, or shared-server reconfiguration is authorized by this plan. Local
hardware is not costless: account for electricity, storage, operator time, and
the opportunity cost of occupying the GPU.

## How this could become a business

Test a paid, scoped review-and-regression service before investing in a hosted
multi-tenant product. Possible deliverables are reviewed patches and repeatable
tests for owner-controlled forms. Actual pricing, demand, delivery cost, and
customer outcomes remain unvalidated. Reputation should follow permissioned,
reproducible results and useful open-source work, not claims of novelty,
compliance, profit, or impact that the evidence does not support.
