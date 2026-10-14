# Workstream C verification record (C-verify)

Date: 2026-09-29. Verifier: the C-verify workstream of the OAC System Health contract (CONTRACT.md section 0 and
section 3). Reading material only: nothing here was posted, sent, uploaded or pushed.

Every claim carries one truth label. MEASURED = a check ran in this session and its output is saved under
`logs/C/verify/` (file named in the Evidence column). REPORTED = stated by a named source. MODELED = our judgement
over measured facts. CONJECTURE = unverified guess. UNKNOWN = not established. This file contains no OAC metric; the
OAC numbers that `evidence_matrix.md` refers to are SYNTHETIC and are not generalized.

People appear in this file only as OpenAlex author ids, and papers only by DOI or arXiv id.

## 0. Verdict

- MODELED: **PASS after fixes.** Every problem the prior verifier recorded is fixed (section 1), the new problems found by
  C-verify are fixed (sections 3 to 8), and the final scan of the commit set is clean (section 9). The open items in
  section 11 do not block local use of the reading material. They must be settled before anything from
  Workstream C is published.
- MEASURED (`logs/C/verify/final_scan.json`): the commit set holds 0 contact addresses, 0 third-party e-mail addresses,
  0 credential-shaped strings, 0 clinical affiliation strings and 0 occurrences of the party named in the CONTRACT
  isolation section.
- REPORTED (C-verify session record, `progress/C-verify.md`): C-verify made no remote write. Its network calls were
  GET requests to OpenAlex, Crossref and arXiv; its only git write is the local commit that adds this file.

## 1. Starting point: the prior verifier's findings

- REPORTED (`logs/verify/C-literature/verdict.json`, `progress/verify-C-literature.md`): verdict FIX with 3 minor
  issues and 4 nits. The on-disk record names four of them. First, two matrix rows (conformal risk control,
  selective classification with guaranteed risk) omitted their theorems' assumptions. Second, the "Not guaranteed"
  and "Reason" cells carried no truth label. Third, several OpenAlex author ids in leaders.md looked
  mis-disambiguated (7 flagged by a topic screen). Fourth, one REFORMS paraphrase shared a 7-word run with the source.
- UNKNOWN: the wording of the remaining itemised issues. They were returned to the orchestrator and are not on disk.
  C-verify re-checked every deliverable from scratch instead (sections 2 to 7).

| Prior finding | Status | Label | Evidence |
|---|---|---|---|
| Missing assumptions in the conformal-risk-control and selective-risk rows | Fixed: guarantees now state theorem and assumptions from the arXiv source (rows 23 and 26) | MEASURED | `evidence_matrix_before_fix.md`, `eprint_fetch.json`, `matrix_label_check_final_stdout.txt` |
| Unlabelled matrix cells | Fixed: every guarantee and not-guaranteed cell carries [R-abs], [R-src], [M] or [C] | MEASURED | `matrix_label_check_final_stdout.txt` (40 rows, 0 problems) |
| Mis-disambiguated author ids | Fixed: rule-based signals plus hand review of 26 ids; 11 ids have their institutions withheld | MEASURED (signals), CONJECTURE (classes) | `leaders_ids_check.json`, `leaders_id_review.json` |
| REFORMS paraphrase with a 7-word shared run | Fixed: items 3d, 5c, 6c reworded; longest shared run now 4 words per item, 5 against the whole source | MEASURED | `reforms_check.json`, `reforms_items_before_fix.json` |

## 2. Anchors (`literature/anchors.json`)

| Check | Result | Label | Evidence |
|---|---|---|---|
| Re-resolve all 51 anchors by id: OpenAlex singleton lookup, Crossref by DOI, one batched arXiv id_list call | 90 requests, 0 non-200 (2026-09-29T16:08:49Z) | MEASURED | `reresolve_anchors.json`, `reresolve_anchors_stdout.txt` |
| Compare id, title, year, venue, DOI and expected first-author surname with the stored values | 49 CONFIRMED, 2 CONFIRMED_WITH_DIFFS | MEASURED | same |
| The 2 diffs | Crossref year differs by one from the stored Crossref year (doi 10.1145/3143561: stored 2019, fresh 2018; doi 10.1007/s10618-024-01081-4: stored 2025, fresh 2024). Title, DOI, venue and surname agree | MEASURED | same |
| Cause of the 2 diffs | Crossref carries several dates for one work (print, online, issue); the two runs read different ones | MODELED | same |
| Action | anchors.json unchanged: no deliverable prints these years and the ids resolve | MODELED | - |

## 3. Evidence matrix (`literature/evidence_matrix.md`)

| Check | Result | Label | Evidence |
|---|---|---|---|
| Every id printed in a row equals the id stored in anchors.json | 40 rows, 51 anchor references, 0 problems | MEASURED | `matrix_ids_check.json`, `matrix_ids_check_final_stdout.txt` |
| Structure (10 cells per row) and truth labels in guarantee and not-guaranteed cells | 40 rows, 0 problems | MEASURED | `matrix_label_check_final_stdout.txt` |
| Guarantee column against paper sources | Rewritten for the rows whose guarantee is a theorem. 10 arXiv e-prints were re-fetched on 2026-09-29 (HTTP 200, sha256 recorded) and each theorem is stated with its assumptions | MEASURED (fetch), REPORTED (content) | `eprint_fetch.json`, `eprint_fetch_stdout.txt`, before copy `evidence_matrix_before_fix.md` |
| Label mix after the rewrite | 9 rows cite [R-src] (19 tags); 46 [R-abs], 60 [M], 1 [C] | MEASURED (count on the file) | `literature/evidence_matrix.md` |
| Table integrity | Unescaped `\|` inside formulas had split cells; pipes escaped | MEASURED | `matrix_label_check_final_stdout.txt` |
| Builder re-run after the fix | 40 rows; ADOPT 15, ADAPT 17, REJECT 8; no non-VERIFIED anchor used | MEASURED | `build_evidence_matrix_rerun_stdout.txt`, before copy `build_evidence_matrix_before_fix.py.txt` |
| [M] cells against full texts | Not checked | UNKNOWN | - |

## 4. REFORMS items (`literature/reforms_items.json`)

| Check | Result | Label | Evidence |
|---|---|---|---|
| Item ids and module names against the checklist table of the captured e-print | 32 of 32 ids, 8 of 8 modules equal | MEASURED | `reforms_check.json`, `reforms_check_final_stdout.txt` |
| Captured e-print is the latest version | Archive sha256 equals the v2 e-print re-fetched on 2026-09-29 | MEASURED | `reforms_check.json` |
| Copying check (flag rule set before looking: shared run of 6 or more words, or 3-gram share above 0.50) | 0 flagged; longest run 4 words (item) and 5 (whole source); highest 3-gram share 0.27 | MEASURED | `reforms_check_final_stdout.txt` |
| Reworded items | 3d, 5c, 6c | MEASURED | `reforms_items_before_fix.json`, `build_reforms_items_rerun_stdout.txt` |
| Source license | arXiv non-exclusive distribution license, which grants no reuse, so every item stays our own paraphrase | REPORTED (arXiv abstract page) | `logs/C/reforms/abs.html` (local only) |

## 5. Leaders (`literature/leaders.md`)

| Check | Result | Label | Evidence |
|---|---|---|---|
| "Reading list only. No outreach." at the top | Present | MEASURED | `final_scan.json` (`leaders_md.disclaimer`) |
| Identity by OpenAlex author id | 99 person rows: 96 with an author id (95 distinct ids), 3 anchor first authors with no id on the OpenAlex work record; 0 unparsed rows | MEASURED | `final_scan.json` |
| Fresh author lookups (free singleton calls, one per id) | 95 ids, 0 non-200, 0 display-name mismatches, 0 institution mismatches with the page | MEASURED | `leaders_check.json`, `leaders_check_stdout.txt` |
| Institution types added | Shown: education 61, company 16, nonprofit 6, facility 4, other 2, government 1. 4 institutions (OpenAlex type `healthcare` or clinical-looking) appear as `[EXTERNAL-PARTY]` | MEASURED | `leaders_apply_stdout.txt` |
| Merged or ambiguous author ids: signals S0 to S6 (rules fixed before the first run) | 22 of 95 ids flagged | MEASURED | `leaders_ids_check.json`, `leaders_ids_check_stdout.txt` |
| Hand review of 26 ids (the 22 flagged plus 4 added by reading, 2 of them from the prior verifier) | 3 MERGED_LIKELY, 8 AMBIGUOUS, 6 MOVED_OR_DIFFERENT, 9 CLEARED | MEASURED (signals), CONJECTURE (classes) | `leaders_id_review.json` |
| Flags on the page | The `ID check` column carries the class. Institutions of the 11 MERGED_LIKELY and AMBIGUOUS ids are withheld (UNKNOWN); 0 withheld rows print an institution | MEASURED | `final_scan.json` |
| Clinical-provider institutions printed | 0, re-checked with the extended helper of section 7 | MEASURED | `final_scan.json` (`clinical` = 0), `redact_extended_stdout.txt` |
| Names on the page | Bibliographic metadata only (the display name OpenAlex attaches to each author id). The CONTRACT asks for this page and names seed authors itself | MODELED | - |

## 6. Contact address and other e-mail addresses

| Check | Result | Label | Evidence |
|---|---|---|---|
| Contact address in `literature/`, `scripts/`, `logs/C/` (read from an inline env var, never written) | 0 occurrences; its local part next to `@` or in a mailto: 0. Final run 2026-09-29T21:04:46Z over 6018 files | MEASURED | `email_grep.json`, `email_grep_stdout.txt`; run 1 `email_grep_run1.json` |
| Leaks to remove in Workstream C paths | None found, so nothing was removed | MEASURED | same |
| C scripts read the mailto only from env `SZL_CONTACT_EMAIL` | Consistent with 0 occurrences in `scripts/` | MEASURED | same |
| Other workstreams' local-only metadata copies of the contact address | 214 occurrences outside Workstream C (GitHub run metadata, git output, local clones); listed in `HARVEST_NOTES.md` section 8; not modified | MEASURED | same |
| Third-party addresses (authors' addresses in API metadata or paper sources) | 265 in 49 raw capture files under `logs/C/`; 0 in `literature/` and `scripts/`; 0 in the commit set. Local-only raw evidence, never to be quoted, published, uploaded or pushed | MEASURED | `email_grep.json`, `final_scan.json` |

## 7. Clinical institutions (CONTRACT section 0, Isolation)

| Check | Result | Label | Evidence |
|---|---|---|---|
| Raw API captures made by C-verify | 169 clinical-looking values in 8 files replaced by `[EXTERNAL-PARTY]`; re-run changes 0 | MEASURED | `redact_raw.json`, `redact_raw_rerun_stdout.txt` |
| Other Workstream C raw captures | 2406 values in 35 files replaced. Cause: `build_leaders.py` `redact()` skipped `affiliations[].raw_affiliation_string` (no `type` key). Fixed and tested | MEASURED | `redact_c_logs.json`, `build_leaders_redact_test_stdout.txt`, before copy `build_leaders_before_fix.py.txt` |
| Manual review of all 801 distinct affiliation strings in `literature/*.json` and `logs/C/verify/*.json` | The keyword heuristic had missed care providers, health authorities, clinical societies and clinical research bodies (non-English words, accents, names with no keyword) | REPORTED (C-verify session; the reviewed list is not saved because it holds clinical names), MODELED (classification) | effect measured in `extended_helper_delta_stdout.txt` |
| Helper extended (`scripts/szl_redact.py`) | Generic words added; the proper-name keywords it used to spell out became sha256 digests of word n-grams; 16 full-name digests added. No clinical institution name is written in the helper | MEASURED (file content) | `scripts/szl_redact.py`; before copy kept local only (it spells those keywords) |
| Effect of the extension on the reviewed strings | 40 of 801 newly flagged, 0 unflagged | MEASURED | `extended_helper_delta_stdout.txt` |
| Re-sanitize every JSON and Atom XML file under `literature/` and `logs/C/` | 111 distinct flagged strings (held in memory only), 40 files and 333 values changed, including `literature/raw.json` (57 institutions) and `literature/leaders_verified.json` (7 institution `type` values). `leaders.md` unchanged apart from its time stamp | MEASURED | `redact_extended.json`, `redact_extended_stdout.txt` |
| Literal occurrences of the flagged strings in the commit set | 98 before, 41 after, all in 5 JSON files. The 41 are not in affiliation fields: those 5 files hold 0 flagged affiliation strings (MEASURED), so the matches sit in other fields such as titles, venues and URLs (MODELED, by exclusion) | MEASURED, MODELED | `redact_extended.json`, `flagged_title_venue_check_stdout.txt` |
| Titles that name a clinical body | 3 titles in `literature/raw.json` (off-topic results of the reference query) replaced by a withheld marker; the rows keep DOI and OpenAlex/arXiv ids. Re-check: 0 such titles left in raw.json. The 1 match left in `leaders_verified.json` is the study-design word "Multicenter", not a named body | MEASURED (counts), MODELED (the last reading) | `title_redact.json` (run 1 record reconstructed: its before-hash comes from `redact_extended.json`), `title_redact_stdout.txt`, `flagged_title_venue_check_after_stdout.txt` |
| Scans with the extended helper | `logs/C/`: 583 JSON files, 0 flagged affiliation strings; `literature/`: 8 JSON files, 0 | MEASURED | `clinical_scan_extended_logs_C_stdout.txt`, `clinical_scan_extended_literature_stdout.txt` |
| Another workstream's folder, read only | `logs/verify/C-literature/raw/`: 17 flagged affiliation strings in 7 files; not modified; its owner decides | MEASURED | `clinical_scan_extended_prior_verifier_readonly_stdout.txt` |
| Completeness | A keyword and digest heuristic cannot prove absence; a clinical name with no keyword that the review missed would pass | MODELED | - |

## 8. What is committed and what stays local

- MEASURED (`final_scan.json`, list built by `commit_set()` in `scripts/verify_c_final_scan.py`): the C-verify commit
  holds `literature/` (12 files including this one) and the 22 Workstream C scripts (harvest, anchors, REFORMS,
  leaders, matrix, code links, abstracts, the shared redaction helper and the C-verify checks). It also holds the
  top-level files of `logs/C/verify/` except the two local-only copies listed below.
- MODELED, local only (never commit, publish or upload as they stand):
  - `logs/C/verify/raw/`: API captures. They are redacted, so not byte-exact (sha256 before and after are in the
    committed JSON records). One file holds third-party e-mail addresses. The folder also holds the REFORMS e-print
    binary, whose license grants no redistribution.
  - `logs/C/verify/src/`: arXiv LaTeX sources, which are third-party text; 7 files hold third-party e-mail addresses.
  - `logs/C/verify/leaders_before_fix.md`: the page before post-processing. It names institutions the page withholds.
  - `logs/C/verify/szl_redact_before_fix.py.txt`: the helper before the fix. It spells clinical-institution keywords.
  - All other raw folders under `logs/C/` (anchors, leaders, code, reforms, probe and the earlier runs). They are
    raw evidence with third-party addresses and are outside this commit's list.

## 9. Final scan of the commit set

| Check | Result | Label | Evidence |
|---|---|---|---|
| Contact address, third-party e-mail addresses, credential-shaped strings | 0, 0, 0 | MEASURED | `final_scan.json`, `final_scan_final_stdout.txt` (earlier runs `final_scan_run1..4_stdout.txt`) |
| Name of the party in the CONTRACT isolation section (read from CONTRACT.md at run time, never written) | 0 | MEASURED | same |
| Terms the CONTRACT keeps out of publishable text (read from CONTRACT.md at run time) | 0 | MEASURED | same |
| Clinical affiliation strings (JSON affiliation fields, leaders-style tables) | 0 | MEASURED | same |
| Clinical-content words (HL7, FHIR, patient, specimen) | 20 hits. 8 are words inside bibliographic titles of API results (3 literature JSON files); 6 are regex patterns in 3 scripts that screen such content out; 6 are this file's own description of the check. No PHI, HL7 or FHIR content | MEASURED (counts), MODELED (reading) | same |
| Truth labels on every bullet of `HARVEST_NOTES.md` and this file | 0 bullets without a label | MEASURED | `final_scan.json` (`labels`) |
| Size and sha256 of every committed file | Recorded | MEASURED | `final_scan.json` |

## 10. Fixes made by C-verify

| # | Fix | Files changed | Label | Evidence or before copy |
|---|---|---|---|---|
| 1 | Guarantee column rewritten from arXiv sources, assumptions stated, pipes escaped, cells labelled | `literature/evidence_matrix.md`, `scripts/build_evidence_matrix.py` | MEASURED | `evidence_matrix_before_fix.md`, `build_evidence_matrix_before_fix.py.txt` |
| 2 | REFORMS items 3d, 5c, 6c reworded | `literature/reforms_items.json`, `scripts/build_reforms_items.py` | MEASURED | `reforms_items_before_fix.json` |
| 3 | Leaders page: fresh institutions, types, ID check column, 11 ids withheld, extra clinical pattern | `literature/leaders.md` | MEASURED | `leaders_apply_stdout.txt`, `leaders_apply_run1_stdout.txt` |
| 4 | Redaction gap in the leaders builder closed | `scripts/build_leaders.py` | MEASURED | `build_leaders_before_fix.py.txt`, `build_leaders_redact_test_stdout.txt` |
| 5 | Clinical affiliation strings redacted in raw captures | `logs/C/**` JSON and XML (local only) | MEASURED | `redact_raw.json`, `redact_c_logs.json`, `redact_extended.json` |
| 6 | Helper extended; spelled-out names replaced by digests | `scripts/szl_redact.py` | MEASURED | `extended_helper_delta_stdout.txt` |
| 7 | 57 institution strings in raw.json and 7 type values in leaders_verified.json redacted | `literature/raw.json`, `literature/leaders_verified.json` | MEASURED | `redact_extended.json` |
| 8 | 3 titles naming a clinical body withheld | `literature/raw.json` | MEASURED | `title_redact.json` |
| 9 | Notes: C-verify pointers, privacy notes, labels | `literature/HARVEST_NOTES.md` | MEASURED | `HARVEST_NOTES_before_fix.md` |

## 11. Open items

- UNKNOWN: guarantees beyond the captured abstracts and the 10 re-fetched arXiv sources; [M] cells were not checked
  against full texts.
- CONJECTURE: the author-id classes (MERGED_LIKELY, AMBIGUOUS, MOVED_OR_DIFFERENT, CLEARED) are readings of OpenAlex
  records, not facts about any person.
- UNKNOWN: whether the published journal version of REFORMS words its items like the arXiv v2 checklist.
- MEASURED: the 2 Crossref year differences of section 2 are kept as they are in anchors.json.
- MODELED: raw captures under `logs/C/` are no longer byte-exact API output (redacted); sha256 before and after are
  recorded.
- UNKNOWN (owner decision pending): the other workstreams' local copies of the contact address (`HARVEST_NOTES.md`
  section 8) and the flagged affiliation strings in the prior verifier's folder (section 7). C-verify did not modify
  them.
- MODELED: before anything from Workstream C is published, Stephen should decide whether leaders.md keeps the name
  column; the renderer can emit an id-only version.
