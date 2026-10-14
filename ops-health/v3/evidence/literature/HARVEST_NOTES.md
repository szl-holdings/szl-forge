# Workstream C harvest notes

Status: final for this session (2026-09-29). Reading material only; nothing here was posted, sent or uploaded.
Verified and fixed by C-verify on 2026-09-29: see `literature/VERIFICATION.md` (checks, evidence, fixes, verdict).

Every statement carries a truth label. MEASURED = ran this session, output under `logs/C/`.
REPORTED = stated by a source whose id and retrieval are cited. MODELED = our analysis.
CONJECTURE = unverified guess. UNKNOWN = not established.

## Deliverables

| File | What it is |
|---|---|
| `literature/anchors.json` | 51 VERIFIED anchors (11 CONTRACT seeds, 5 synthetic-tabular-evaluation papers, 35 method papers); 2 non-seed anchors dropped |
| `literature/reforms_items.json` | 32 REFORMS items (module, id, own paraphrase), source arXiv 2308.07832v2 |
| `literature/evidence_matrix.md` | 40 rows: guarantee, limits, OAC applicability, official code license, verdict (ADOPT 15 / ADAPT 17 / REJECT 8) |
| `literature/leaders.md` | Reading list per topic (14 topics), people identified by OpenAlex author id. No outreach. |
| `literature/official_code_map.json` | 12 repositories that a paper names as its own code, with licenses |
| `literature/leaders_verified.json`, `literature/synthetic_eval_candidates.json`, `literature/synthetic_eval_selected.json` | Intermediate data behind leaders.md and the synthetic anchors |
| `literature/raw.json`, `literature/leaders.json` | Reference-implementation outputs. leaders.json is not usable (section 1). |
| `literature/VERIFICATION.md` | C-verify record: every check, its evidence file, every fix, final verdict |

Scripts (all under `scripts/`): `lit_harvest.py`, `resolve_anchors.py`, `fetch_reforms.py`,
`build_reforms_items.py`, `build_leaders.py`, `render_leaders_md.py`, `code_links.py`,
`extract_abstracts.py`, `build_evidence_matrix.py`; shared redaction helper `szl_redact.py`.
C-verify scripts: `verify_c_literature.py`, `verify_c_reforms.py`, `verify_c_matrix_labels.py`,
`verify_c_leaders.py`, `verify_c_leaders_ids.py`, `verify_c_leaders_apply.py`, `verify_c_email_grep.py`,
`verify_c_redact_raw.py`, `verify_c_redact_extended.py`, `verify_c_title_redact.py`, `verify_c_clinical_scan.py`,
`verify_c_final_scan.py` (outputs under `logs/C/verify/`).

## 1. raw.json and leaders.json (lit_harvest.py)

- MEASURED: `literature/raw.json` holds 560 records: 14 queries x (25 OpenAlex + 15 arXiv), 0 error rows
  (count re-checked 2026-09-29). Run log `logs/C/lit_harvest_requests.json`: 28 requests, all HTTP 200,
  2026-09-26T22:04:25Z to 22:06:32Z; stdout `logs/C/lit_harvest_stdout.txt`. No query failed, so nothing was
  re-run this session.
- MEASURED: two earlier runs lost arXiv results (run 1: 14 arXiv errors, run 2: 13;
  `logs/C/lit_harvest_run{1,2,3}_stdout.txt`). export.arxiv.org answered Python urllib with HTTP 406 while curl
  with the same User-Agent got 200 (`logs/C/probe/arxiv_406_*`). All arXiv traffic now goes through the system
  curl binary with the same honest User-Agent and at least 3.1 s between calls.
- MEASURED: 209 clinical-looking institution names in raw.json were written as `[EXTERNAL-PARTY]`
  (keyword heuristic in `scripts/szl_redact.py`; it can over- and under-redact). C-verify found that it had
  under-redacted and extended it (section 8); raw.json now holds 266 `[EXTERNAL-PARTY]` entries in 97 rows
  (MEASURED 2026-09-29, `logs/C/verify/redact_extended.json`), and 3 titles that name a clinical society are
  withheld (section 8).
- MEASURED (`logs/C/raw_relevance_check.txt`): the reference query style (OpenAlex full-text `search`, sorted
  by citation count) returns mostly off-topic works. 303 of 350 OpenAlex records share no content word with
  their query in the title (arXiv: 17 of 210). As a result `literature/leaders.json` is led by authors of
  mega-cited unrelated papers (top 5 all at about 119,870 citations from 2 records). **Do not use
  leaders.json as a leader list.** It is kept only because the CONTRACT names it as an output; leaders.md
  replaces it (section 6).

## 2. Contact address handling

- MEASURED: every Workstream C script reads the mailto only from env `SZL_CONTACT_EMAIL` and writes only a
  boolean `user_agent_has_mailto`. The raw.json run had it unset (`false`); the anchor, leaders and code-link
  runs set it inline on the command line. curl receives the User-Agent as a process argument (not written
  to disk).
- MEASURED (C-literature grep of `literature/`, `logs/C/`, `scripts/`, `progress/` at about 2026-09-29T10:43Z): no match.
  The address did appear in files owned by other workstreams, all as git/GitHub metadata rather than a
  User-Agent leak: `logs/E/ci/*_runs*.json` (15 files; commit author `email`/`login` fields returned by the
  GitHub API) and `logs/G/main_drift_20260929_patches.txt` (one patch line). Workstream C did not modify
  them because they are not its paths. Flagged to the orchestrator. Superseded by the whole-workspace
  C-verify scan in section 8 (later state; the G patch file no longer contains it).

## 3. Anchors (literature/anchors.json)

- MEASURED: 51 anchors, all VERIFIED under matching rules v2 (at least 2 sources whose titles agree, with the
  expected surname). Source count per anchor: 2 sources 35, 3 sources 11, 4 sources 5. All 11 CONTRACT seed
  anchors are VERIFIED (model cards, REFORMS, datasheets, Data Cards, TRIPOD+AI, leakage, shortcut learning,
  underspecification, calibration of modern NNs, gentle conformal intro, DeepMutation), plus the REFORMS
  arXiv preprint as a separate record.
- MEASURED: the requested method families are covered: split conformal (`split_conformal_lei`), isotonic
  (`isotonic_zadrozny`), Brier (`brier_score`), ECE (`ece_naeini`, `nixon_measuring_calibration`,
  `kumar_verified_calibration`), selective classification (`selective_geifman`, `reject_chow`), metamorphic
  testing of ML (`metamorphic_xie`), bootstrap CIs (`bootstrap_bca_efron`, `bootstrap_ci_diciccio`).
  `mondrian_conditional_validity` was added because v2/PREREGISTRATION.md uses Mondrian conformal.
- MEASURED, dropped (non-seed, fewer than 2 matching sources in the final run): `platt_scaling` (Platt 1999;
  no DOI; only Semantic Scholar matched in the final run. OpenAlex had matched W1618905105 in an earlier v2
  pass when S2 was rate-limited, so the two sources never agreed within one run; by the rules it stays
  dropped) and `selective_foundations_elyaniv` (0 sources). Platt scaling as a method is still covered through
  `niculescu_caruana` and `guo_calibration`, whose abstracts discuss it.
- MEASURED: synthetic tabular evaluation papers were chosen from API results only
  (`--discover`, `literature/synthetic_eval_candidates.json`), preferring evaluation frameworks over
  generators and non-clinical venues: SynthEval, holdout-based assessment, structured evaluation, Anonymeter
  privacy risk, data-centric benchmark (`literature/synthetic_eval_selected.json`).
- MEASURED: run 1 (matching rules v1) is archived in `logs/C/anchors_run1/`. It produced one false VERIFIED
  (`chandola_anomaly_survey` matched "Anomaly Detection for Discrete Sequences: A Survey", not the intended
  survey) and three MISMATCHes caused by related-but-different works (Wild-Time, a 2026 conformal risk control
  follow-up, an isolation-forest extension). Rules v2 (stricter two-way title match, Crossref subtitle, an
  id-based lookup or cross-source agreement may accept a subtitle-less title prefix) fix all four. The rules
  are in the `resolve_anchors.py` docstring and in anchors.json `status_rules`.
- MEASURED, run segmentation of v2 (stdout files `logs/C/anchors_v2_part{1..5}_stdout.txt`; request log
  `logs/C/anchors_requests.json`, 384 requests, 199 non-200, most of them rate-limit 429s or logged skips):
  part 1 plain run, stopped by us after the OpenAlex budget ran out; part 2 `--resume`, crashed on an
  `anchors.json` write (OSError EINVAL, a Windows file-sharing clash); part 3 `--resume`, crashed the same way
  on the request-log write, losing the request records of its last anchor (`numenta_nab`, then re-resolved
  with `--only`); part 4 `--resume` plus `--only numenta_nab`; part 5 `--only` on the two dropped keys to
  restore the dropped list that an `--only` run had erased (bug fixed). Writes are now atomic with retry.
  Raw responses: `logs/C/anchors/` (v2; file names shared with run 1 were overwritten, run 1 copies are in
  `logs/C/anchors_run1/raw/`).
- MEASURED: Semantic Scholar (no API key) answered HTTP 429 to most calls; a circuit breaker skips it after
  3 consecutive 429s.
- MEASURED: OpenAlex now meters anonymous use. Response headers showed `X-RateLimit-Limit-USD: 0.1` and
  `X-RateLimit-Cost-Required-USD: 0.001` for a search, while singleton lookups cost 0. The daily budget ran
  out at about 2026-09-29T10:26Z (HTTP 429, Retry-After 48,674 s). The resolver then skipped OpenAlex
  searches and used free singleton lookups by DOI, including arXiv DOIs (`10.48550/arxiv.<id>`).
- MODELED: OpenAlex ingests arXiv, so an "arXiv + OpenAlex" pair is weaker independence than two unrelated
  registries. 35 anchors rest on exactly 2 sources; see `sources_matched` per anchor.
- MEASURED: abstracts captured for 48 of 51 anchors (`logs/C/abstracts.json`, via
  `scripts/extract_abstracts.py`); none for `brier_score`, `metamorphic_xie`, `page_cusum`.

## 4. REFORMS items (literature/reforms_items.json)

- MEASURED: 32 items in 8 modules, parsed from the checklist table of the arXiv e-print 2308.07832 (latest
  version v2, 2023-09-19 per the abs page; e-print and main.tex sha256 recorded in the file). Built offline by
  `scripts/build_reforms_items.py` (`logs/C/build_reforms_items_stdout.txt`, 0 problems).
- REPORTED (arXiv abs page, `logs/C/reforms/abs.html`): license
  `http://arxiv.org/licenses/nonexclusive-distrib/1.0/`, which grants no reuse. Every item is therefore our own
  paraphrase (at most 17 words), not the source wording. Source text stays under `logs/C/reforms/` for audit.
- REPORTED (main.tex, Box 1 and scope section): the paper scopes out pure predictive analytics. OAC is closer
  to that than to ML-based science, so the matrix marks REFORMS as ADAPT (audit checklist), not ADOPT.
- UNKNOWN: whether the published Science Advances version (doi 10.1126/sciadv.adk3452, VERIFIED) words the
  items the same way. Not compared.

## 5. Evidence matrix (literature/evidence_matrix.md)

- MEASURED: 40 rows built by `scripts/build_evidence_matrix.py`; ids and statuses are read from anchors.json
  at build time, not typed by hand (`logs/C/build_evidence_matrix_stdout.txt`).
- MEASURED (C-verify label check, `logs/C/verify/matrix_label_check_final_stdout.txt`: 40 rows, 0 problems): guarantee cells are labelled per claim: [R-abs] = stated in the captured abstract, [R-src] = stated in
  captured arXiv source, [M] = our reading. No full texts were read except the REFORMS source.
- The applicability column and the verdicts are MODELED. They use OAC v1 as its Hub card describes it
  (REPORTED: `src/szl-forge/clinical-gateway/huggingface/model/oac-system-health-v1/README.md`, szl-forge
  9a6fdd2c) and v2/PREREGISTRATION.md as of 2026-09-29. All OAC metrics are SYNTHETIC.
- Official code licenses: MEASURED via `gh api repos/<owner>/<repo>` (GET only) on 2026-09-29, only for links
  the paper itself states (contexts in `logs/C/code/<key>__links.json`, reviewed by hand;
  `literature/official_code_map.json`). Found: MIT for 9 repositories, GPL-2.0 for `ryantibs/conformal`
  (split conformal, weighted conformal), no license file for the shortcut-learning toy code, and
  `statice/anonymeter` reported by GitHub as NOASSERTION while its LICENSE.md text is the Clear BSD license.
  Other GitHub links found in the same sources (baselines, cited tools) are listed in
  `logs/C/code_links_summary.json` but are not the papers' own code. "none stated" in the matrix means no link
  was found in the captured sources, not that no code exists.
- MODELED, main takeaways for v2: the pre-registered choices (Platt/isotonic, Mondrian split conformal with
  ABSTAIN, Brier/ECE, percentile bootstrap, mutation suite, shift regimes) all map to VERIFIED anchors. The
  cheapest additions are: a plug-in-ECE bias caveat with a bootstrap interval (Kumar, Nixon), a Chow
  reject-rule baseline next to conformal ABSTAIN, a trivial-rule baseline next to headline metrics
  (benchmark-flaw critique), metamorphic relations and program mutation on the stdlib kernel, and a
  datasheet stating that synthetic-data privacy/fidelity metrics are not applicable (no real reference data).

## 6. Leaders (literature/leaders.md)

- MEASURED: 14 topic filters (quoted phrases and boolean OpenAlex syntax) re-queried OpenAlex
  (`scripts/build_leaders.py`, `logs/C/build_leaders_stdout.txt`, 17 requests, all 200). A first pass with the
  plain lit_harvest query strings still returned off-topic works; it is archived in `logs/C/leaders_run1/`.
- MEASURED: authors are aggregated by OpenAlex author id. Institutions are the `last_known_institutions` of
  that id, looked up by id (batched in build_leaders.py; free singleton lookups in render_leaders_md.py after
  the budget ran out, cache `logs/C/leaders/authors_render_cache.json`). The affiliation text is REPORTED
  (OpenAlex), not independently verified; some records look stale or mis-disambiguated.
- MEASURED: 96 table rows with an OpenAlex author id, 3 anchor first authors whose OpenAlex work record carries
  no author id (affiliation UNKNOWN), 23 rows whose OpenAlex record has no institution (UNKNOWN). Clinical
  institutions are never named; 5 rows show `[EXTERNAL-PARTY]` next to a non-clinical institution.
- MEASURED (C-verify, `logs/C/verify/leaders_check.json`, `leaders_ids_check.json`, `leaders_id_review.json`): all 95
  author ids re-read by free singleton lookup (0 non-200; names and institutions equal the page). leaders.md now shows the
  OpenAlex institution type, an `ID check` column and withholds the institutions of 11 author ids whose OpenAlex record
  probably pools different people or carries a misattributed work (3 MERGED_LIKELY, 8 AMBIGUOUS; classes are CONJECTURE).
  After this, 21 rows have no institution on the record and 3 rows show `[EXTERNAL-PARTY]`. Details: `literature/VERIFICATION.md`.
- MEASURED: title screen: works whose titles match clinical or off-domain patterns are excluded per topic before ranking.
  A backspace-character bug in the first version of that regex (three patterns silently inactive) was found and
  fixed before the final render.
- MODELED: bias notes are in the page: co-author citation bias (a `top-work share` column shows when one paper carries
  a person's whole count), citation-sort bias toward older work, query bias, seed bias.
- MEASURED: "Reading list only. No outreach." is stated at the top. Names appear only as bibliographic metadata from
  the APIs. MODELED interpretation: the CONTRACT asks for this page and names seed authors itself, so
  bibliographic author names inside `literature/` are treated as allowed while clinical institution names are
  not. Stephen may prefer an id-only version; the renderer can drop the name column.

## 7. Limits and open items

- UNKNOWN: full-text verification of any guarantee beyond abstracts (only REFORMS source was read).
- UNKNOWN: whether the v1 synthetic label is a deterministic function of the 8 input fields
  (evidence matrix row on feature legitimacy is marked CONJECTURE until someone checks the v1 generator).
- MEASURED: Semantic Scholar and, after 10:26Z, OpenAlex search were rate-limited; anchors resolved after that point
  rest on Crossref, arXiv and free OpenAlex singleton lookups.
- MEASURED (location), CONJECTURE (identity): `[EXTERNAL-PARTY]` `logs/C/anchors_run1/raw/kumar_verified_calibration__search__openalex.json:7626` and
  `:7646`: an author name in an unrelated OpenAlex search result shares a given name with the party named in
  the CONTRACT isolation section. Almost certainly a different person (CONJECTURE); left untouched as raw API
  data and not referenced in any deliverable. Flagged for Stephen.
- UNKNOWN (owner decision pending): the non-C files containing the contact address (section 8) need a decision by their owners.

## 8. Privacy notes (C-verify, 2026-09-29)

- MEASURED (`scripts/verify_c_email_grep.py`, run 1 at 2026-09-29T20:25:58Z: `logs/C/verify/email_grep_run1.json`; final
  run at 21:04:46Z, after the raw redaction below: `logs/C/verify/email_grep.json` and `email_grep_stdout.txt`, 6018 files
  scanned, 0 skipped; the address is read from an inline env var and only paths and counts are written): the contact
  address occurs 0 times in `literature/`, `scripts/` and `logs/C/`, and its local part occurs 0 times next to an `@` or in
  a mailto there. Nothing had to be removed there. Scan scope: the whole workspace except `.git/`, `__pycache__/` and
  `v2/data/sealed/` (never read).
- MEASURED, other workstreams' local-only metadata copies that contain the contact address. Workstream C did not modify
  them; their owners decide. None of them may be published as they stand.
  - `logs/E/ci/*_runs*.json`: 15 files, 140 occurrences (GitHub Actions run metadata, commit-author fields).
  - `logs/verify/G-rename-branch/git_checks.txt`: 1 file, 1 occurrence (git metadata output).
  - `scratch/G-rename-branch/szl-forge/`: 8 files, 10 occurrences (working copy of szl-forge files).
  - `src/lutar-lean/` 47 files / 47 occurrences, `src/ouroboros/` 2 / 2, `src/szl-forge/` 8 / 10, `src/szl-kernels/` 3 / 4:
    local clones of SZL repositories; the address is part of the cloned files (whether each upstream copy is public was
    not checked: UNKNOWN).
  - `logs/G/main_drift_20260929_patches.txt`, listed in section 2, no longer contains the address.
- MEASURED (final run): third-party e-mail addresses (authors' addresses inside API metadata or paper sources) occur 265
  times in 49 raw capture files under `logs/C/` and 0 times in `literature/` and `scripts/`: `logs/C/anchors/` 9 files / 25
  hits, `logs/C/anchors_run1/` 10 / 27, `logs/C/code/` 1 / 4, `logs/C/leaders/` 10 / 72, `logs/C/leaders_run1/` 9 / 122,
  `logs/C/reforms/` 2 / 2, `logs/C/verify/raw/` 1 / 4, `logs/C/verify/src/` 7 / 9. Run 1 (before the `--c-logs` raw
  redaction below) counted 344 in 52 files; the whole drop is in `logs/C/leaders/` and `logs/C/leaders_run1/`, the folders
  that redaction changed (MODELED cause: some redacted affiliation strings carried an address). They are local-only raw
  evidence: never quote, publish, upload or push them. The C-verify commit leaves `logs/C/verify/raw/` and
  `logs/C/verify/src/` out (`literature/VERIFICATION.md`, section 7); the other raw folders under `logs/C/` are not committed.
- MEASURED: clinical-looking affiliation strings left in raw captures were replaced by `[EXTERNAL-PARTY]`: 169 values in 8
  C-verify files (`logs/C/verify/redact_raw.json`) and 2406 values in 35 C-literature files under `logs/C/leaders/`,
  `logs/C/leaders_run1/` and `logs/C/probe/` (`logs/C/verify/redact_c_logs.json`; counts include institution `type` values
  "healthcare"). Cause of the second group: `scripts/build_leaders.py` `redact()` only replaced `raw_affiliation_string` when
  the enclosing object had a `type` key, which `affiliations[]` entries lack. Fixed in the script (copy before the fix:
  `logs/C/verify/build_leaders_before_fix.py.txt`; test: `logs/C/verify/build_leaders_redact_test_stdout.txt`). Re-scan:
  0 files left (`logs/C/verify/clinical_affiliation_scan_after_stdout.txt`). These raw files are therefore no longer
  byte-exact API output; before/after sha256 are in the two JSON records.
- MEASURED (C-verify manual review, 2026-09-29): all 801 distinct affiliation strings in `literature/*.json` and
  `logs/C/verify/*.json` were read by hand. Care providers, health authorities, clinical societies and clinical
  research bodies had passed the keyword heuristic (non-English words, accented spellings, names with no keyword).
  The extended helper flags 40 of the 801 strings that the original did not, and unflags none (`logs/C/verify/extended_helper_delta_stdout.txt`,
  counted on copies taken just before the re-sanitizing; the 40 include some deliberate over-redaction by the broad words). `scripts/szl_redact.py` was extended with generic words and with sha256 digests (16 full names; the
  proper-name keywords the helper used to spell out were moved to word n-gram digests), so no clinical institution
  name is written in the helper any more (copy before the fix, local only: `logs/C/verify/szl_redact_before_fix.py.txt`).
  `scripts/verify_c_redact_extended.py` then re-sanitized every JSON and Atom XML file under `literature/` and
  `logs/C/`: 111 distinct flagged strings (held in memory only), 40 files and 333 values changed, including
  `literature/raw.json` (57) and `literature/leaders_verified.json` (7 institution `type` values); `leaders.md` did not
  change. Re-scan with the extended helper: 0 flagged affiliation strings in `logs/C/` (583 JSON files) and
  `literature/` (8) (`logs/C/verify/clinical_scan_extended_*_stdout.txt`). MODELED: the heuristic still cannot
  prove absence; a name with no keyword that the review missed would pass.
- MEASURED: titles are not affiliation fields, so a follow-up check looked at flagged titles and venues in the
  literature JSON (`logs/C/verify/flagged_title_venue_check_stdout.txt`): `literature/raw.json` held paper titles
  that name a clinical society (off-topic results of the reference query). `scripts/verify_c_title_redact.py`
  replaced 3 such titles with a withheld marker; the rows keep their DOI and OpenAlex/arXiv ids
  (`logs/C/verify/title_redact.json`, `title_redact_stdout.txt`; re-check `flagged_title_venue_check_after_stdout.txt`).
- MEASURED, another workstream's local-only copies (read only, not modified by C-verify): the prior verifier's folder
  `logs/verify/C-literature/raw/` holds 17 affiliation strings in 7 JSON files that the extended helper flags
  (`logs/C/verify/clinical_scan_extended_prior_verifier_readonly_stdout.txt`). Its owner decides; it must not be
  published as it stands.
