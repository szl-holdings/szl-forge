# Leaders by topic (reading list)

**Reading list only. No outreach.** Nobody listed here is to be contacted, tagged, emailed or mentioned in any SZL artifact outside this literature folder. This page exists so Stephen can decide what to read.

Generated 2026-09-29T10:40:33Z by `scripts/render_leaders_md.py` from `literature/leaders_verified.json` (OpenAlex, generated 2026-09-29T10:06:51Z) and `literature/anchors.json`.

Post-processed 2026-09-29T21:27:48Z by `scripts/verify_c_leaders_apply.py` (C-verify): institutions re-read from fresh singleton OpenAlex author lookups (`logs/C/verify/leaders_check.json`, 2026-09-29T16:22:34Z, 95 ids, 0 non-200), institution types added, author-id review applied. Checks and verdict: `literature/VERIFICATION.md`.

## How to read this

- **Identity** (MEASURED: ids taken from OpenAlex work authorships): each person is an OpenAlex author id (`A...`), not a name string. Names are the display names OpenAlex attaches to that id.
- **Affiliation**: `last_known_institutions` of that OpenAlex author id, looked up by id this session. Truth label **REPORTED** (OpenAlex record, retrieval date above). Not independently verified; OpenAlex author disambiguation and institution matching can be wrong or stale.
- **Basis** (MEASURED: counts from the OpenAlex topic query, see Provenance): `n works / citations` counts only works in the topic's top-25 most-cited OpenAlex matches (filter shown) after the title screen below. `anchor` means first author of a VERIFIED anchor in `anchors.json`.
- **Title screen** (MEASURED: both regexes are in the script): works whose title matches a clinical/off-domain pattern are excluded (isolation rule and relevance), and each topic keeps only titles matching a topic pattern (both regexes are in the script). Dropped-work counts are shown per topic.
- **Institution type**: the OpenAlex institution `type` is shown in brackets (education, company, facility, nonprofit, government, other). Institutions of type `healthcare`, clinical-looking names and names matching `psychiatr|diagnos|clinical` are never printed; they appear as `[EXTERNAL-PARTY]`. REPORTED (OpenAlex).
- **ID check** (signals MEASURED, class CONJECTURE): `ok` = no signal; `CLEARED (Sx)` = signal explained by the record; `MOVED_OR_DIFFERENT (S5)` = last-known country differs from the cited work while the topics fit, affiliation marked `(?)`; `AMBIGUOUS` or `MERGED_LIKELY` = the OpenAlex id probably pools different people or carries a misattributed work, so its institutions are withheld (UNKNOWN). `hand` = added by reading, not by the fixed rules. Signal definitions: `scripts/verify_c_leaders_ids.py`; per-id reasons: `logs/C/verify/leaders_id_review.json`.
- **People whose OpenAlex institutions are all clinical-looking** appear only as `[EXTERNAL-PARTY]` with a pointer to the data file (CONTRACT section 0, Isolation).

## Known biases (MODELED; read before trusting any ranking)

- **Co-author citation bias**: every co-author of a heavily cited paper inherits its full citation count, so one large-team paper can fill a topic with tied names. The `top-work share` column shows how much of a person's count comes from a single work (1.00 = all of it). Treat any row with share near 1.00 as 'author of one important paper', not 'leader of the area'.
- **Citation-sort bias**: sorting by citations favours older work and large communities; recent frontier work (2024-2026) is under-represented. Venue, field-size and self-citation are not normalised.
- **Query bias**: topic filters are our own phrasings; groups using different vocabulary are missed.
- **Seed bias**: anchors come from the CONTRACT seed list, so their authors are included by construction.

## model reporting

OpenAlex filter: `title.search:"model cards" OR "model card" OR "model reporting"` - works kept 4, dropped by title screen 21.

Anchor works (VERIFIED): `model_cards` 10.1145/3287560.3287596 arXiv:1810.03993; `factsheets` 10.1147/jrd.2019.2942288 arXiv:1808.07261

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Ben Hutchinson | [A5118595929](https://openalex.org/A5118595929) | Google (United States) [company] | 1 works / 2072 cites | 1.00 | ok |
| Anamaria Crisan | [A5022433344](https://openalex.org/A5022433344) | University of Waterloo [education] (?) | 1 works / 103 cites | 1.00 | MOVED_OR_DIFFERENT (S5) |
| Margaret Drouhard | [A5054008161](https://openalex.org/A5054008161) | Salesforce (United States) [company] | 1 works / 103 cites | 1.00 | ok |
| Jesse Vig | [A5044735229](https://openalex.org/A5044735229) | Salesforce (United States) [company] | 1 works / 103 cites | 1.00 | ok |
| Nazneen Fatema Rajani | [A5052900932](https://openalex.org/A5052900932) | UNKNOWN (no institution on OpenAlex record) | 1 works / 103 cites | 1.00 | CLEARED (S6) |
| Margaret Mitchell | none on the OpenAlex work record | UNKNOWN (no author id to verify) | anchor `model_cards` | n/a | no author id |
| Matthew G. Arnold | [A5017107088](https://openalex.org/A5017107088) | UNKNOWN (author id flagged AMBIGUOUS; OpenAlex institutions withheld) | anchor `factsheets` | n/a | AMBIGUOUS (S1,S2,S6) |

Groups (an institution shared by two or more listed people, withheld and [EXTERNAL-PARTY] entries excluded): Salesforce (United States) (2)

## ML-based-science reporting standards

OpenAlex filter: `title_and_abstract.search:"machine learning" AND ("reporting standards" OR "reporting checklist" OR "reproducibility checklist")` - works kept 6, dropped by title screen 19.

Anchor works (VERIFIED): `reforms` 10.1126/sciadv.adk3452; `reforms_arxiv` 10.48550/arxiv.2308.07832 arXiv:2308.07832

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Joëlle Pineau | [A5080591144](https://openalex.org/A5080591144) | UNKNOWN (no institution on OpenAlex record) | 1 works / 204 cites | 1.00 | ok |
| Philippe Vincent‐Lamarre | [A5002591360](https://openalex.org/A5002591360) | UNKNOWN (no institution on OpenAlex record) | 1 works / 204 cites | 1.00 | ok |
| Koustuv Sinha | [A5109576489](https://openalex.org/A5109576489) | UNKNOWN (no institution on OpenAlex record) | 1 works / 204 cites | 1.00 | ok |
| Vincent Larivière | [A5068542997](https://openalex.org/A5068542997) | Université du Québec à Montréal [education]; Stellenbosch University [education]; Érudit [other]; Université de Montréal [education]; National Research Foundation [nonprofit] | 1 works / 204 cites | 1.00 | CLEARED (S4) |
| Alina Beygelzimer | [A5078135763](https://openalex.org/A5078135763) | UNKNOWN (no institution on OpenAlex record) | 1 works / 204 cites | 1.00 | ok |
| Sayash Kapoor | [A5063345981](https://openalex.org/A5063345981) | Princeton University [education] | anchor `reforms` | n/a | ok |

## dataset documentation

OpenAlex filter: `title.search:"datasheets for datasets" OR "data cards" OR "dataset documentation" OR "data documentation"` - works kept 20, dropped by title screen 5.

Anchor works (VERIFIED): `datasheets` 10.1145/3458723 arXiv:1803.09010; `data_cards` 10.1145/3531146.3533231 arXiv:2204.01075

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Jennifer Wortman Vaughan | [A5043117896](https://openalex.org/A5043117896) | UNKNOWN (author id flagged AMBIGUOUS; OpenAlex institutions withheld) | 3 works / 2147 cites | 0.96 | AMBIGUOUS (hand) |
| Hanna Wallach | [A5046348432](https://openalex.org/A5046348432) | Microsoft (United States) [company]; Microsoft Research New York City (United States) [company] | 3 works / 2147 cites | 0.96 | ok |
| Timnit Gebru | [A5089275971](https://openalex.org/A5089275971) | DAIR Institute [nonprofit] | 2 works / 2089 cites, anchor `datasheets` | 0.99 | ok |
| Jamie Morgenstern | [A5045592180](https://openalex.org/A5045592180) | University of Washington [education] | 2 works / 2089 cites | 0.99 | ok |
| Briana Vecchione | [A5069421536](https://openalex.org/A5069421536) | Data & Society Research Institute [nonprofit] | 2 works / 2089 cites | 0.99 | ok |
| Mahima Pushkarna | [A5007386215](https://openalex.org/A5007386215) | Google DeepMind (United Kingdom) [company] (?) | anchor `data_cards` | n/a | MOVED_OR_DIFFERENT (S5) |

Groups: none after the author-id review (no institution shown for two or more listed people).

## synthetic tabular data evaluation

OpenAlex filter: `title_and_abstract.search:"synthetic data" AND tabular AND (evaluation OR fidelity OR utility)` - works kept 12, dropped by title screen 13.

Anchor works (VERIFIED): `synth_syntheval` 10.1007/s10618-024-01081-4 arXiv:2404.15821; `synth_holdout_assessment` 10.3389/fdata.2021.679939; `synth_structured_eval` 10.48550/arxiv.2403.10424 arXiv:2403.10424; `synth_privacy_risk` 10.56553/popets-2023-0055 arXiv:2211.10459; `synth_datacentric_benchmark` 10.52202/075280-1466 arXiv:2310.16981

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Álvaro Figueira | [A5029477501](https://openalex.org/A5029477501) | Universidade do Porto [education]; INESC TEC [nonprofit] | 1 works / 409 cites | 1.00 | ok |
| Bruno Vaz | [A5008604331](https://openalex.org/A5008604331) | Universidade do Porto [education] | 1 works / 409 cites | 1.00 | ok |
| João Fonseca | [A5085754936](https://openalex.org/A5085754936) | UNKNOWN (no institution on OpenAlex record) | 1 works / 126 cites | 1.00 | ok |
| Fernando Bação | [A5053763560](https://openalex.org/A5053763560) | Universidade Nova de Lisboa [education] | 1 works / 126 cites | 1.00 | ok |
| Zilong Zhao | [A5016185099](https://openalex.org/A5016185099) | Liaoning Technical University [education] (?) | 1 works / 111 cites | 1.00 | MOVED_OR_DIFFERENT (S5) |
| Anton D. Lautrup | [A5093229388](https://openalex.org/A5093229388) | UNKNOWN (no institution on OpenAlex record) | anchor `synth_syntheval` | n/a | ok |
| Michael Platzer | [A5086683790](https://openalex.org/A5086683790) | Vienna University of Economics and Business [education] | anchor `synth_holdout_assessment` | n/a | ok |
| Scott Cheng‐Hsin Yang | [A5065866283](https://openalex.org/A5065866283) | UNKNOWN (no institution on OpenAlex record) | anchor `synth_structured_eval` | n/a | ok |
| M. Giomi | [A5086017246](https://openalex.org/A5086017246) | UNKNOWN (author id flagged AMBIGUOUS; OpenAlex institutions withheld) | anchor `synth_privacy_risk` | n/a | AMBIGUOUS (S1,S2,S6) |
| Lasse Hansen | [A5127821191](https://openalex.org/A5127821191) | UNKNOWN (no institution on OpenAlex record) | anchor `synth_datacentric_benchmark` | n/a | ok |

Groups (an institution shared by two or more listed people, withheld and [EXTERNAL-PARTY] entries excluded): Universidade do Porto (2)

## calibration

OpenAlex filter: `title.search:calibration AND ("neural networks" OR classifier OR classifiers OR "probability estimates" OR "calibration error")` - works kept 8, dropped by title screen 17.

Anchor works (VERIFIED): `guo_calibration` 10.48550/arxiv.1706.04599 arXiv:1706.04599; `kumar_verified_calibration` 10.48550/arxiv.1909.10155 arXiv:1909.10155; `nixon_measuring_calibration` 10.48550/arxiv.1904.01685 arXiv:1904.01685; `ece_naeini` 10.1609/aaai.v29i1.9602; `niculescu_caruana` 10.1145/1102351.1102430; `isotonic_zadrozny` 10.1145/775047.775151; `gneiting_scoring_rules` 10.1198/016214506000001437

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Chuan Guo | [A5069524591](https://openalex.org/A5069524591) | Snap (United States) [company] | 1 works / 2595 cites, anchor `guo_calibration` | 1.00 | ok |
| Geoff Pleiss | [A5065203625](https://openalex.org/A5065203625) | UNKNOWN (no institution on OpenAlex record) | 1 works / 2595 cites | 1.00 | ok |
| Yu Sun | [A5100772262](https://openalex.org/A5100772262) | UNKNOWN (author id flagged AMBIGUOUS; OpenAlex institutions withheld) | 1 works / 2595 cites | 1.00 | AMBIGUOUS (S5,hand) |
| Kilian Q. Weinberger | [A5006380227](https://openalex.org/A5006380227) | Cornell University [education] | 1 works / 2595 cites | 1.00 | ok |
| Sunil Thulasidasan | [A5028031083](https://openalex.org/A5028031083) | Los Alamos National Laboratory [facility] | 1 works / 433 cites | 1.00 | ok |
| Ananya Kumar | [A5014379995](https://openalex.org/A5014379995) | UNKNOWN (author id flagged AMBIGUOUS; OpenAlex institutions withheld) | anchor `kumar_verified_calibration` | n/a | AMBIGUOUS (hand) |
| Jeremy Nixon | [A5110890879](https://openalex.org/A5110890879) | UNKNOWN (no institution on OpenAlex record) | anchor `nixon_measuring_calibration` | n/a | ok |
| Mahdi Pakdaman Naeini | [A5000623954](https://openalex.org/A5000623954) | Harvard University [education] | anchor `ece_naeini` | n/a | ok |
| Alexandru Niculescu-Mizil | [A5029480393](https://openalex.org/A5029480393) | NEC (United States) [company] | anchor `niculescu_caruana` | n/a | ok |
| Bianca Zadrozny | [A5006347550](https://openalex.org/A5006347550) | IBM (United States) [company] | anchor `isotonic_zadrozny` | n/a | ok |
| Tilmann Gneiting | [A5046150465](https://openalex.org/A5046150465) | UNKNOWN (no institution on OpenAlex record) | anchor `gneiting_scoring_rules` | n/a | ok |

## conformal prediction

OpenAlex filter: `title.search:"conformal prediction" OR "conformal predictors"` - works kept 18, dropped by title screen 7.

Anchor works (VERIFIED): `gentle_conformal` 10.48550/arxiv.2107.07511 arXiv:2107.07511; `split_conformal_lei` 10.1080/01621459.2017.1307116 arXiv:1604.04173; `venn_abers` 10.48550/arxiv.1211.0025 arXiv:1211.0025; `conformal_covariate_shift` 10.48550/arxiv.1904.06019 arXiv:1904.06019; `conformal_beyond_exchangeability` 10.1214/23-aos2276 arXiv:2202.13415; `learn_then_test` 10.1214/24-aoas1998; `conformal_risk_control` 10.48550/arxiv.2208.02814 arXiv:2208.02814

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Vladimir Vovk | [A5063978613](https://openalex.org/A5063978613) | University of London [education]; Royal Holloway University of London [education] | 5 works / 1303 cites, anchor `venn_abers` | 0.48 | CLEARED (S6) |
| Ulf Norinder | [A5067014994](https://openalex.org/A5067014994) | Uppsala University [education]; Stockholm University [education]; Örebro University [education] | 3 works / 368 cites | 0.60 | ok |
| Anastasios N. Angelopoulos | [A5019661690](https://openalex.org/A5019661690) | UNKNOWN (no institution on OpenAlex record) | 2 works / 731 cites, anchor `gentle_conformal`, anchor `learn_then_test`, anchor `conformal_risk_control` | 0.90 | ok |
| Stephen Bates | [A5057257463](https://openalex.org/A5057257463) | UNKNOWN (author id flagged AMBIGUOUS; OpenAlex institutions withheld) | 2 works / 731 cites | 0.90 | AMBIGUOUS (hand) |
| Harris Papadopoulos | [A5008815275](https://openalex.org/A5008815275) | Frederick University [education] | 2 works / 350 cites | 0.54 | CLEARED (S6) |
| Jing Lei | [A5085887576](https://openalex.org/A5085887576) | University of Michigan [education]; [EXTERNAL-PARTY]; Carnegie Mellon University [education] | anchor `split_conformal_lei` | n/a | ok |
| Ryan J. Tibshirani | [A5076708141](https://openalex.org/A5076708141) | University of California, Berkeley [education] | anchor `conformal_covariate_shift` | n/a | ok |
| Rina Foygel Barber | [A5048658972](https://openalex.org/A5048658972) | University of Chicago [education] | anchor `conformal_beyond_exchangeability` | n/a | ok |

## selective classification / abstention

OpenAlex filter: `title.search:"selective classification" OR "reject option" OR "selective prediction" OR "learning to defer"` - works kept 20, dropped by title screen 5.

Anchor works (VERIFIED): `selective_geifman` 10.48550/arxiv.1705.08500 arXiv:1705.08500; `reject_chow` 10.1109/tit.1970.1054406; `hendrycks_msp_baseline` 10.48550/arxiv.1610.02136 arXiv:1610.02136

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Marten H. Wegkamp | [A5057473211](https://openalex.org/A5057473211) | Cornell University [education] | 3 works / 623 cites | 0.52 | ok |
| Ran El‐Yaniv | [A5025669300](https://openalex.org/A5025669300) | Technion – Israel Institute of Technology [education] | 3 works / 463 cites | 0.38 | ok |
| Giorgio Fumera | [A5021089553](https://openalex.org/A5021089553) | University of Cagliari [education] | 3 works / 399 cites | 0.43 | ok |
| Fabio Roli | [A5065359946](https://openalex.org/A5065359946) | University of Cagliari [education]; University of Genoa [education] | 3 works / 399 cites | 0.43 | ok |
| Yonatan Geifman | [A5061409282](https://openalex.org/A5061409282) | UNKNOWN (no institution on OpenAlex record) | 2 works / 293 cites, anchor `selective_geifman` | 0.61 | ok |
| C. Chow | none on the OpenAlex work record | UNKNOWN (no author id to verify) | anchor `reject_chow` | n/a | no author id |
| Dan Hendrycks | [A5020400986](https://openalex.org/A5020400986) | UNKNOWN (no institution on OpenAlex record) | anchor `hendrycks_msp_baseline` | n/a | ok |

Groups (an institution shared by two or more listed people, withheld and [EXTERNAL-PARTY] entries excluded): University of Cagliari (2)

## shortcut learning

OpenAlex filter: `title.search:"shortcut learning" OR "spurious correlations"` - works kept 1, dropped by title screen 24.

Anchor works (VERIFIED): `shortcut_learning` 10.1038/s42256-020-00257-z arXiv:2004.07780

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Robert Geirhos | [A5043113119](https://openalex.org/A5043113119) | Google DeepMind (United Kingdom) [company]; University of Tübingen [education] | 1 works / 2144 cites, anchor `shortcut_learning` | 1.00 | ok |
| Jörn-Henrik Jacobsen | [A5109463535](https://openalex.org/A5109463535) | Apple (United States) [company] (?) | 1 works / 2144 cites | 1.00 | MOVED_OR_DIFFERENT (S5) |
| Claudio Michaelis | [A5076080916](https://openalex.org/A5076080916) | Tübingen AI Center [education]; University of Tübingen [education] | 1 works / 2144 cites | 1.00 | ok |
| Richard S. Zemel | [A5000111344](https://openalex.org/A5000111344) | Columbia University [education] (?) | 1 works / 2144 cites | 1.00 | MOVED_OR_DIFFERENT (S5) |
| Wieland Brendel | [A5086801305](https://openalex.org/A5086801305) | Max Planck Institute for Intelligent Systems [facility]; Tübingen AI Center [education]; ELLIS Institute Tübingen [other] | 1 works / 2144 cites | 1.00 | ok |

Groups (an institution shared by two or more listed people, withheld and [EXTERNAL-PARTY] entries excluded): Tübingen AI Center (2); University of Tübingen (2)

## leakage and reproducibility

OpenAlex filter: `title_and_abstract.search:"data leakage" AND "machine learning" AND reproducibility` - works kept 9, dropped by title screen 16.

Anchor works (VERIFIED): `leakage_kapoor` 10.1016/j.patter.2023.100804; `kaufman_leakage` 10.1145/2382577.2382579

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Sayash Kapoor | [A5063345981](https://openalex.org/A5063345981) | Princeton University [education] | 2 works / 1223 cites, anchor `leakage_kapoor` | 0.88 | ok |
| Arvind Narayanan | [A5058102069](https://openalex.org/A5058102069) | Princeton University [education] | 2 works / 1223 cites | 0.88 | ok |
| Matthew Rosenblatt | [A5060428905](https://openalex.org/A5060428905) | Broad Institute [nonprofit]; Harvard University [education]; Yale University [education]; [EXTERNAL-PARTY]; [EXTERNAL-PARTY] | 1 works / 210 cites | 1.00 | CLEARED (S4) |
| Link Tejavibulya | [A5044673394](https://openalex.org/A5044673394) | Yale University [education]; [EXTERNAL-PARTY] | 1 works / 210 cites | 1.00 | CLEARED (S1,S2) |
| Rongtao Jiang | [A5013717612](https://openalex.org/A5013717612) | Beijing Normal University [education]; Yale University [education]; State Key Laboratory of Cognitive Neuroscience and Learning [facility] | 1 works / 210 cites | 1.00 | CLEARED (S1,S2) |
| Shachar Kaufman | [A5048628578](https://openalex.org/A5048628578) | UNKNOWN (no institution on OpenAlex record) | anchor `kaufman_leakage` | n/a | ok |

Groups (an institution shared by two or more listed people, withheld and [EXTERNAL-PARTY] entries excluded): Yale University (3); Princeton University (2)

## underspecification

OpenAlex filter: `title_and_abstract.search:underspecification AND "machine learning"` - works kept 6, dropped by title screen 19.

Anchor works (VERIFIED): `underspecification` 10.48550/arxiv.2011.03395 arXiv:2011.03395

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Alexander D’Amour | [A5060694111](https://openalex.org/A5060694111) | Google DeepMind (United Kingdom) [company] | 1 works / 424 cites, anchor `underspecification` | 1.00 | ok |
| Katherine Heller | [A5014018142](https://openalex.org/A5014018142) | UNKNOWN (no institution on OpenAlex record) | 1 works / 424 cites | 1.00 | ok |
| DAN I. MOLDOVAN | [A5043094101](https://openalex.org/A5043094101) | UNKNOWN (author id flagged MERGED_LIKELY; OpenAlex institutions withheld) | 1 works / 424 cites | 1.00 | MERGED_LIKELY (S0) |
| Ben Adlam | [A5048167471](https://openalex.org/A5048167471) | UNKNOWN (no institution on OpenAlex record) | 1 works / 424 cites | 1.00 | ok |
| Babak Alipanahi | [A5026092995](https://openalex.org/A5026092995) | UNKNOWN (author id flagged AMBIGUOUS; OpenAlex institutions withheld) | 1 works / 424 cites | 1.00 | AMBIGUOUS (S1,S2) |

## ML mutation testing

OpenAlex filter: `title.search:"mutation testing" AND (deep OR neural OR "machine learning")` - works kept 23, dropped by title screen 2.

Anchor works (VERIFIED): `deepmutation` 10.1109/issre.2018.00021 arXiv:1805.05206; `deepcrime` 10.1145/3460319.3464825; `mutation_survey_jia` 10.1109/tse.2010.62

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Lei Ma | [A5101468661](https://openalex.org/A5101468661) | University of Alberta [education]; The University of Tokyo [education] | 3 works / 520 cites, anchor `deepmutation` | 0.71 | ok |
| Yang Liu | [A5100355692](https://openalex.org/A5100355692) | UNKNOWN (author id flagged MERGED_LIKELY; OpenAlex institutions withheld) | 3 works / 520 cites | 0.71 | MERGED_LIKELY (S0,S3,S4) |
| Jianjun Zhao | [A5065190767](https://openalex.org/A5065190767) | Kyushu University [education]; State Grid Corporation of China (China) [company] | 3 works / 520 cites | 0.71 | CLEARED (hand) |
| Nargiz Humbatova | [A5076282301](https://openalex.org/A5076282301) | Università della Svizzera italiana [education] | 3 works / 136 cites, anchor `deepcrime` | 0.90 | ok |
| Gunel Jahangirova | [A5085228874](https://openalex.org/A5085228874) | King's College London [education] | 3 works / 136 cites | 0.90 | ok |
| Yue Jia | [A5100725712](https://openalex.org/A5100725712) | UNKNOWN (author id flagged AMBIGUOUS; OpenAlex institutions withheld) | anchor `mutation_survey_jia` | n/a | AMBIGUOUS (S4,S5) |

## metamorphic testing of ML

OpenAlex filter: `title.search:"metamorphic testing" AND ("machine learning" OR deep OR neural OR classifiers)` - works kept 24, dropped by title screen 1.

Anchor works (VERIFIED): `metamorphic_xie` 10.1016/j.jss.2010.11.920; `metamorphic_review_chen` 10.1145/3143561; `metamorphic_survey_segura` 10.1109/tse.2016.2532875; `ml_testing_survey` 10.1109/tse.2019.2962027 arXiv:1906.10742

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Christian Murphy | [A5002151898](https://openalex.org/A5002151898) | Swarthmore College [education] | 3 works / 668 cites | 0.60 | ok |
| Gail E. Kaiser | [A5090171418](https://openalex.org/A5090171418) | UNKNOWN (no institution on OpenAlex record) | 3 works / 668 cites | 0.60 | ok |
| Tsong Yueh Chen | [A5035107113](https://openalex.org/A5035107113) | Swinburne University of Technology [education] | 3 works / 527 cites, anchor `metamorphic_review_chen` | 0.76 | CLEARED (S6) |
| Faqeer Ur Rehman | [A5005747707](https://openalex.org/A5005747707) | Intel (United States) [company] | 3 works / 30 cites | 0.43 | ok |
| Xiaoyuan Xie | [A5100746280](https://openalex.org/A5100746280) | Wuhan University [education] | 2 works / 508 cites, anchor `metamorphic_xie` | 0.79 | ok |
| Sergio Segura | [A5057070358](https://openalex.org/A5057070358) | Universidad de Sevilla [education] | anchor `metamorphic_survey_segura` | n/a | ok |
| Jie M. Zhang | [A5088708850](https://openalex.org/A5088708850) | King's College London [education] | anchor `ml_testing_survey` | n/a | ok |

## operational anomaly detection

OpenAlex filter: `title.search:"anomaly detection" AND ("time series" OR multivariate OR telemetry OR microservice OR logs)` - works kept 24, dropped by title screen 1.

Anchor works (VERIFIED): `numenta_nab` 10.1109/icmla.2015.141 arXiv:1510.03336; `ts_anomaly_benchmarks_flawed` 10.1109/tkde.2021.3112126 arXiv:2009.13807; `isolation_forest` 10.1109/icdm.2008.17; `chandola_anomaly_survey` 10.1145/1541880.1541882; `page_cusum` 10.1093/biomet/41.1-2.100

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Ya Su | [A5028459386](https://openalex.org/A5028459386) | UNKNOWN (author id flagged MERGED_LIKELY; OpenAlex institutions withheld) | 2 works / 2096 cites | 0.80 | MERGED_LIKELY (S0,S2,S4) |
| Youjian Zhao | [A5101491459](https://openalex.org/A5101491459) | Beijing Institute of Technology [education]; Tsinghua University [education] | 2 works / 2096 cites | 0.80 | ok |
| Dan Pei | [A5046419834](https://openalex.org/A5046419834) | Chinese Academy of Sciences [government]; Beijing National Research Center for Information Science and Technology [facility]; Tsinghua University [education] | 2 works / 2096 cites | 0.80 | ok |
| Ángel Conde | [A5055393052](https://openalex.org/A5055393052) | Universidad Rey Juan Carlos [education] | 2 works / 1522 cites | 0.72 | ok |
| Congrui Huang | [A5027878846](https://openalex.org/A5027878846) | Microsoft (United States) [company] (?) | 2 works / 1325 cites | 0.55 | MOVED_OR_DIFFERENT (S5) |
| Alexander Lavin | [A5000311824](https://openalex.org/A5000311824) | Pasteur Labs (United States) [company] | anchor `numenta_nab` | n/a | ok |
| Renjie Wu | none on the OpenAlex work record | UNKNOWN (no author id to verify) | anchor `ts_anomaly_benchmarks_flawed` | n/a | no author id |
| Fei Tony Liu | [A5042335244](https://openalex.org/A5042335244) | UNSW Sydney [education] | anchor `isolation_forest` | n/a | ok |
| Varun Chandola | [A5003851078](https://openalex.org/A5003851078) | University at Buffalo, State University of New York [education] | anchor `chandola_anomaly_survey` | n/a | ok |
| E. S. Page | [A5042292094](https://openalex.org/A5042292094) | University of Hull [education] | anchor `page_cusum` | n/a | ok |

Groups (an institution shared by two or more listed people, withheld and [EXTERNAL-PARTY] entries excluded): Tsinghua University (2)

## distribution shift evaluation

OpenAlex filter: `title.search:("distribution shift" OR "distribution shifts" OR "dataset shift") AND (benchmark OR evaluation OR robustness)` - works kept 21, dropped by title screen 4.

Anchor works (VERIFIED): `wilds` 10.48550/arxiv.2012.07421 arXiv:2012.07421

| Person | OpenAlex author id | OpenAlex last-known institution(s) | Basis | Top-work share | ID check |
|---|---|---|---|---|---|
| Chelsea Finn | [A5005431772](https://openalex.org/A5005431772) | University of Washington [education]; Stanford University [education] | 3 works / 324 cites | 0.88 | ok |
| Pang Wei Koh | [A5079904764](https://openalex.org/A5079904764) | University of Washington [education]; Allen Institute for Artificial Intelligence [nonprofit] | 2 works / 306 cites, anchor `wilds` | 0.93 | ok |
| Sergey Levine | [A5026322200](https://openalex.org/A5026322200) | Princeton University [education] | 2 works / 302 cites | 0.94 | ok |
| Rohan Taori | [A5017940830](https://openalex.org/A5017940830) | UNKNOWN (no institution on OpenAlex record) | 2 works / 242 cites | 0.70 | ok |
| Achal Dave | [A5045047925](https://openalex.org/A5045047925) | UNKNOWN (no institution on OpenAlex record) | 2 works / 242 cites | 0.70 | ok |

Groups (an institution shared by two or more listed people, withheld and [EXTERNAL-PARTY] entries excluded): University of Washington (2)

## Provenance

- Topic works: `logs/C/leaders/works_qNN.json` (OpenAlex /works, raw, clinical institution names redacted).
- Author records: `logs/C/leaders/authors_bNN.json` (batched, from build_leaders.py) and `logs/C/leaders/authors_render_cache.json` (singleton lookups; request log `authors_render_requests.json`).
- Request log: `logs/C/leaders_requests.json`; run stdout: `logs/C/build_leaders_stdout.txt`.
- A first run with the 14 plain lit_harvest queries returned mostly off-topic works; it is archived in `logs/C/leaders_run1/` and not used here.
- C-verify (2026-09-29): fresh author records `logs/C/verify/raw/authors/<id>.json`, check `logs/C/verify/leaders_check.json`, id signals `logs/C/verify/leaders_ids_check.json`, id review `logs/C/verify/leaders_id_review.json`, page before this post-processing `logs/C/verify/leaders_before_fix.md`.
