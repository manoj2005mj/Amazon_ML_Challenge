# Session report - Amazon ML Challenge 2026, business entity resolution

Covers the work of 2026-09-26 (about 04:30 to 16:00) on top of the v1 pipeline built on
2026-09-25. It records what was done, in which order, the intuition behind every step, the
exact parameters used, the numbers measured, and what is still open. Companion files:
`docs/EDA_REPORT.md` (full EDA with figures), `HANDOFF.md` (operational state and commands),
`docs/BLOCKING_EVALUATION.md` (earlier blocking benchmark).

---

## 1. The problem

Three sources of noisy business records (`entity_id, business_name, business_address,
country`). Source 1 is the deduplicated reference. For every Source-1 record, predict the set
of Source-2 and Source-3 records that describe the same business (zero, one or many).

- Metric: macro F0.5 per Source-1 entity. Precision is weighted twice as much as recall. An
  entity with no true matches scores 1.0 if the prediction is empty and 0.0 otherwise.
- Countries: US and India in train and test; France only in test (15% of test entities).
- Deliverables: `output/matching_results.tsv` (scored), `output/candidate_pairs.tsv`
  (audited), the code folder with README and requirements, a filled
  `Documentation_template.md`. Rules: no external data or APIs, model under an MIT or Apache
  licence and under 8B parameters. `student_resource/utils/validate_submission.py` must pass.
- Leaderboard: 5 uploads per day, best score counts.

### 1.1 What the metric implies (the intuitions that drove every decision)

1. Blocking recall is a hard ceiling. A true match the blocker never retrieves can never be
   predicted, and because the average is per entity, an entity with all links missed costs a
   full point. The ceiling is computed per entity as `mean(1.25 r / (0.25 + r))`, never from
   mean recall (the function is concave, so the mean-recall shortcut flatters every strategy
   by about a point).
2. Precision counts double, so the decision rule should be conservative on uncertain pairs,
   but an entity emptied by mistake loses its whole point. Emptying French entities in v1 was
   therefore the single most expensive behaviour.
3. Each Source-2/3 record belongs to at most one Source-1 entity (verified in the ground
   truth). Two consequences: an exclusive one-owner rule at inference is exact, and "how does
   this candidate score against every other Source-1 record" is evidence of ownership (the
   competition features).
4. Singletons are 5.6% of entities. A prediction file that empties much more than that is
   over-emptying; one that empties much less is over-predicting.

---

## 2. Timeline of the session

| Time (2026-09-26) | Step | Outcome |
| --- | --- | --- |
| before (09-25) | v1 pipeline: raw data, `token_idf_fast` blocker k=20, 74-feature LightGBM | validation F0.5 0.940, leaderboard 0.902 |
| 04:30-05:15 | Full EDA on the raw files, France probe scripts, charts | `docs/EDA_REPORT.md`; France diagnosed as the largest loss |
| 05:15 | Goal set: cleaned dataset, better candidates (recall 98-99%), better matcher (F0.5 > 0.98), serial compute, adversarial review | working plan |
| 05:15-05:45 | Cleaning stage designed and written (`cleaning/`), user decisions taken | code ready |
| 05:45-06:00 | Dictionary learned from the training ground truth (946 s) | `cleaning/dictionary.json` |
| 06:00-06:20 | First cleaned pass; blocking measured on the 1% sample: old blocker vs cleaned data vs new `union_passes` | 0.8930 -> 0.9165 -> 0.9695 pair completeness |
| 06:20-06:55 | Adversarial review of the cleaner (16 findings), fixes, full re-clean of both splits | 24.2M rows, 0 errors, ~25 min |
| 07:45-09:30 | Full export with `union_passes` into `candidates_v2/` (test all entities, train 50% of entities) | 103.9M test pairs, 66.2M train pairs |
| 09:30-12:07 | Matcher v2: entities, competition stats, sample, features, training | validation F0.5 0.9755 |
| 12:07-13:52 | Prediction on 104M test pairs, validator | `output_v2/` PASS |
| 13:52-14:03 | France probes built (France emptied, gamma 4) | probe files |
| 14:20 | Matcher v3 launched (4000 rounds, lr 0.05, feature fraction 0.6) | validation 0.9759 (not shipped) |
| 15:00 | Adversarial review of v2 (5 findings), fixes in code, `HANDOFF.md` written | code fixed; `candidates_v2` predates the fix |
| 15:09 | Threshold probes for France (0.3, 0.9) | probe files |
| ~15:10 | v4 export launched (fixed scorer) | 2 France parts written, then stopped |
| afternoon | Leaderboard: v2 = 0.95, France-emptied probe = 0.83 | France F0.5 ~ 0.86, US+India ~ 0.967 |
| end | User: "stop all the process" | everything stopped, nothing running |

---

## 3. Working constraints and process decisions

- Hardware: Windows 11 laptop, 16 GB RAM (1-5 GB free in practice), RTX 3050 4 GB (unused;
  LightGBM ran on CPU with 12 threads), Python 3.12, spawn-based multiprocessing.
- Heavy jobs run one at a time. Every matrix is streamed in chunks; nothing loads the full
  pair table. `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` is set for every command (cp1252 console).
- Environment variables select the "world" a matcher command works in: `ER_ROOT`,
  `ER_DATASET` (raw or cleaned copy), `ER_CANDIDATES` (which export), `ER_WORK` (where
  intermediate files go). This was added so v1, v2 and v4 could coexist on disk.
- Agents: after rate limits were hit, at most one sub-agent or workflow at a time, sub-agents
  on Sonnet, main work done directly. Two adversarial reviews were run (cleaning, v2).
- Submissions: 5 per day; after v2 and the France probe, 2 remained for the day.
- Another session edits the same `src/` folder (it added `matching/france.py` and edited
  `prep_entities.py` and `features.py` at 04:46); files are checked before being changed.
- Kaggle runner (`kaggle/kjob.py`) exists but is 2.8x slower for LightGBM and was not used.
- All processes were stopped on the user's instruction at the end; nothing runs now.

---

## 4. Exploratory data analysis (what we learned)

Scripts: `docs/charts/eda_profile.py` (streams the raw TSVs), `eda_pairs.py` (validation
split, test features, predictions), `france_probe.py`, `eda_charts.py`. Figures:
`docs/charts/fig1_dataset_structure.png`, `fig2_model_and_france.png`.

### 4.1 Inventory and integrity

| File | Rows | US | India | France |
| --- | ---: | ---: | ---: | ---: |
| train S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 | 60% | 40% | - |
| train ground truth | 7,638,365 links | | | |
| test S1 | 1,732,544 | 663,106 | 809,986 | 259,452 |
| test S2 / S3 | 4,887,273 / 5,082,316 | | | 703,378 / 731,615 |

- No empty names anywhere; empty addresses in 2.3-3.7% of S2/S3 records, never in S1.
- Source 1 has zero duplicate (name, address) pairs; S2/S3 have 0.45-0.73% exact duplicates,
  and both copies can be true matches, so duplicates are never collapsed.
- Ground truth is clean: every S1 id present once, no duplicate links, no S2/S3 record linked
  to two S1 entities, no cross-country links, no leakage in ids or row order (correlations
  0.0001), zero train/test id overlap, 29 + 19 shared (name, address) strings out of 5M.
- Id numeric parts collide across sources (26,801 S2/S3 collisions), so the prefix is kept;
  ids are encoded as `source * 10^10 + number` int64 everywhere past loading.

### 4.2 Ground-truth structure

Per Source-1 entity: mean 3.461 matches (S2 1.674, S3 1.788), 5.58% singletons, median 3,
p95 6, max 11, at most 5 from S2 and 6 from S3. 80.5% have matches in both sources. 26.6% of
S2 and 25.4% of S3 records are orphans. US and India are identical to three decimals, so one
generator produced both; France is assumed to follow the same generator (singleton rate
about 5.6%).

### 4.3 Train-to-test shift

S2+S3 records per S1 entity: train 4.67, test US 5.76, test India 5.82, test France 5.53.
The test pools are 23% denser. Either test entities have more matches (about 4.3 each) or
more orphans (about 40%); labels cannot tell, only the leaderboard can. Everything else
(lengths, scripts, legal forms, formats) is identical between train and test for US/India.

### 4.4 Noise patterns

Names: about 55% of true matches differ from Source 1 only by case, punctuation, injected
accents, legal form or word order; 25-30% by a token insertion/drop or a typo; 4-7% have an
unrelated name (the address carries those). Indic scripts in 23.5% of India S2 names and
13.2% of S3 names (Devanagari 13.4%, Telugu 2.0, Kannada 1.8, Tamil 1.7, Gujarati 1.5,
Bengali 1.5, Malayalam 0.9, Odia 0.4, Gurmukhi 0.3); 22-24% of India S2/S3 addresses are
Indic. Injected honorifics (Mr/Dr/Smt/Shri/Sri/M/s at about 1.3% each), "(ID: n)" tags
(0.3%), "X formerly/dba/aka Y" aliases (0.6-1%), "(India)"/"(France)" tags (5-8%), dotted
legal forms "S.A.S." (4-6%), legal form moved to the front (2.5-4.8%), websites as names
(3.4-4.4%), ALL CAPS (15-22% in S2), double spaces (10%), junk prefixes ("-- ", "<< ").

Addresses: Source 2 writes US addresses in capitals with abbreviated street types and drops
unit numbers; Source 3 spells US states out but abbreviates Indian ones; Source 1 always
writes the full Indian state and the US two-letter code. House numbers differ in 12-16% of
true matches, are dropped in 1-15%, and carry leading zeros in 1.6-4.4% of records. Postal
codes are almost absent (11% US, ~0% India, 0.4% France). Units appear in 16% of US S1
addresses, 0.1% of S2, 8% of S3. Indian addresses are long (median 11 tokens), landmark
based, with repeated city/district components. In the joint view 27% of US/S3 true matches
and 48% of India/S3 true matches differ in both name and address.

### 4.5 Where v1 lost score

- Blocking (`token_idf_fast`, top 20 per source, raw data): 89.35% of links retrieved,
  F0.5 ceiling 0.954 (US 0.977, India 0.921). True match at rank 1 only 44.5% of the time.
- Matcher: AUC 0.9999, pair precision 0.994, pair recall 0.974 inside candidates; macro F0.5
  inside candidates 0.985 versus 0.940 actual, so 4.5 points were lost to blocking and 1.4 to
  the matcher. 35% of false negatives were candidates with an empty address; a legal-form
  conflict acted as a near veto (15% FN rate). Every decision rule tried was within 0.2 points.
- Normaliser defects: `squeeze` collapsed digit runs ("1100" -> "10", "0029" -> "029"), dotted
  legal forms became single letters, "(ID:)", "Formerly", "DBA" were not handled, `postal_eq`
  was undefined on 94-100% of pairs.

### 4.6 France diagnosis

French test records: 3 regions, about 18 cities, so city and region tokens carry almost no
IDF; Source 1 ends with the region in 87% of records while S2/S3 end with the region
(28-30%), the department (27-28%: Nord, Pas-de-Calais, Gironde, Loire-Atlantique) or the
city (33-34%); 23% abbreviated street types (R., BD, AV, ALL, CH, IMP); bis/ter in 5-6%;
"N°"/"No."/"#"/"(41)" number markers in 14.5%; no postcodes; 67% of names end in a legal form
(SARL, SAS, SASU, EURL, SA, SCI, SNC, EI), often dotted or bracketed; 64% of names contain a
generic word (association, amicale, club, comité, école).

v1 emptied 16.6% of French entities (13.7% had no candidate above p 0.1). Evidence gathered:
the competition features were shifted (French blocking scores half the US level, 16% of
pairs at margin exactly zero versus 3% US), the model gave lower probability at equal
similarity (0.08 vs 0.33 at token-set similarity 0.8-0.9), the legal veto was stronger, and
accents were not the problem (non-ASCII candidates scored slightly higher). Estimated France
F0.5 about 0.73-0.80, costing 1.5-2 points overall.

### 4.7 The "missing things" list that the pipeline had to handle

Empty candidate addresses, absent postcodes, unit numbers, leading-zero house numbers, "(ID:
n)" tags, "Formerly/DBA" aliases, country tags, dotted and bracketed legal forms, legal
forms at the front, Indic scripts in names and addresses, honorifics, placeholders ("null",
"n/a"), department-vs-region in France, abbreviated street types, state spelled out versus
abbreviated, repeated components in Indian addresses, websites as names, and the France
absence from training.

---

## 5. Cleaning stage

### 5.1 Decisions (taken with the user)

1. Output as cleaned copies of the TSVs in `student_resource/dataset_clean/{train,test}`,
   same schema and ids, plus one sidecar parquet per file holding everything that was
   stripped or inferred. Reason: every downstream stage (blocking harness, exporter, matcher)
   reads TSVs by path, so pointing `ER_DATASET` at the clean folder changes nothing else.
2. More than 12 hours were available, which justified a full-rule cleaner rather than
   patches inside the normaliser.
3. Transliterate everything (NFKC + `unidecode`) and learn a dictionary from the training
   ground truth to repair transliteration artefacts.
4. Strip legal forms, honorifics, aliases and id tags from the text but keep them in the
   sidecar so the matcher can still use them as features.

### 5.2 Intuition

Clean once, upstream, so the blocker and the matcher see the same keys, rather than
normalising differently in each stage. The generator's noise is systematic (fixed lists of
honorifics, tags, abbreviations, transliteration rules), so deterministic rules recover
most of it. Anything removed is evidence, so it is kept in a sidecar rather than lost.

### 5.3 Dictionary learner (`cleaning/learn_dictionary.py`)

Reads only the provided files. Every ground-truth pair's names and addresses are cleaned by
the rule-only cleaner, tokenised and aligned (position-wise when the lengths agree,
otherwise `difflib` on the token lists). A mapping source token -> Source-1 token is kept when
it is frequent (`--min-count 25`), dominant for its source token (`--purity 0.6`) and
string-similar (`--sim 40`, rapidfuzz ratio). Abbreviations (`ln` -> `lane`) require the
subsequence test instead of similarity. Every 8th ASCII-only link is used for abbreviation
learning; all cross-script links are used for token maps. 2,337,859 links aligned in 946 s
with 8 workers.

Output `cleaning/dictionary.json`: 508 Indic-origin name-token maps (`tteknolonjiij` ->
`technologies`, `kulloopl` -> `global`, `proddktts` -> `products`), 26 address abbreviations
(`ln, ct, dr, st, rd, ave, blvd, trl, cir, pkwy, ter, cv, tpke, rdg, aly` and Indian state
codes `ka, dl, mh, tg, kl, gj, hr, rj, br, pb`), 17 address components (Indic state names ->
canonical), 15 French city -> region maps (from test Source 1, no labels), a 116,953-token
Source-1 name vocabulary for website segmentation. Two learned entries were pruned by hand
as wrong (`ce` -> `care`, city "maison des associations").

### 5.4 Name pipeline (`Cleaner.clean_name`, all countries, in this order)

1. NFKC normalisation, zero-width characters removed, whitespace collapsed.
2. Tags to the sidecar: `(ID: n)` -> `id_tag`; `(The)`; `(France)`/`(India)`/`(USA)` ->
   `country_tag`; bracketed legal forms `[LLC]`, `(Limited)`.
3. Aliases `X formerly / dba / d/b/a / aka / f/k/a / t/a Y`: the new name Y is kept, the
   whole alias recorded in `alias_new`.
4. Junk edge punctuation (`-- `, `<< `, trailing `|`), website suffixes (`| www.x.com`).
5. Website names (`tristateguild.com`, `MSCÓFFEE.COM`): the host is segmented into words by
   unigram Viterbi over the Source-1 vocabulary (`cleaning/segment.py`); a string that is
   itself a vocabulary token stays whole; single letters carry a -12 log penalty; if no full
   segmentation exists the host is kept as one token. `web` flag set.
6. Dotted legal forms (`L.L.C.`, `S.A.S.`) recognised as legal forms; Indic legal phrases
   (`प्राइवेट लिमिटेड`, `પ્રા. લિ.`) and their transliterations (`praivet limited`,
   `elelpi`) recognised.
7. Transliteration with `unidecode`, then the learned name-token dictionary.
8. Honorifics stripped (`mr, mrs, dr, smt, shri, sri, m/s, messrs`; `shree` and `ms` were
   deliberately excluded after the dry run because they are name words).
9. Legal forms stripped and recorded as canonical codes in `legal` (`LTD+PVT`) with their
   position in `legal_pos`; `CO, SA, EI, LP, PC` are only recognised at the end, `SA, EI` also
   at the start for France; French `Ets/Établissements/Sté` normalised to one spelling.
10. Leading article "The", possessives (`'s`, upper or lower case), apostrophes, punctuation.
11. Case: title case when the input was all upper or all lower, otherwise unchanged.

### 5.5 Address pipeline (`Cleaner.clean_address`)

Common: placeholder components (`null`, `<null>`, `n/a`, `-`, `none`) dropped;
transliteration plus learned component/token dictionary; number markers removed and the
number kept (`H.No`, `House No.`, `Plot No`, `Shop No`, `Door No`, `Flat`, `Gali No-`, `#`,
`N°`, `(9)`, trailing `6445-`); leading zeros removed from numbers of 1-4 digits (5-digit
codes untouched); `bis/ter/b/t` after a number normalised; street abbreviations expanded per
country; components de-duplicated; canonical component order (street first, then locality,
then state/region last); case as for names; state codes kept upper case.

US: unit/suite/floor/apt/PO box/`#x` moved to the sidecar `unit`; state canonicalised to the
two-letter code with "last wins" when two states appear, and moved to the end
(`state_moved`); "City of X", "X County", "Town of X" reduced to the name; street types
expanded from the `US_STREET` table with a guard so state codes (`MT`, `CT`) are never
expanded as street words.

India: state to the full canonical name from abbreviations (`KA`, `TG`), variants and Indic
spellings; city aliases (`Bangalore` -> `Bengaluru`, `Gurgaon` -> `Gurugram`, `Calcutta` ->
`Kolkata`, `Bombay` -> `Mumbai`, `Madras` -> `Chennai`); `Po` -> `Post`, `Opp` -> `Opposite`,
`Flr` -> `Floor`, `Apt` -> `Apartment`; repeated city/district components removed.

France: department -> region (`Nord`, `Pas-de-Calais` -> `Hauts-de-France`; `Gironde` ->
`Nouvelle-Aquitaine`; `Loire-Atlantique`, `Vendée` -> `Pays de la Loire`); region added from
the city when missing (`region_added`); street abbreviations (`R.`, `AV`, `BD`, `ALL`, `CH`,
`PL`, `IMP`, some only when leading a component); `ST.-HERBLAIN` -> `Saint-Herblain`;
5-digit postcodes removed (Source 1 has none); floor/apartment/cedex/BP components removed;
`N°` handled before tokenising.

### 5.6 Sidecar schema

`entity_id, country, script, name_nonascii, legal, legal_pos, honorific, alias_new, id_tag,
country_tag, web, name_rules, addr_nonascii, unit, state_raw, state, region_added,
placeholders, addr_rules`. `prep_entities` reads `legal`, `name_nonascii` and `web` from it
(the cleaned text has no legal form left to detect) and verifies the row order matches the
TSV.

### 5.7 Adversarial review and fixes

Dry-run defects found by inspection: `Shree` stripped as an honorific; state codes expanded
by the street table (`MT` -> `Mount`); `N°` -> `ndeg`; `ST.-HERBLAIN` untouched; `H.NO A-26`
marker not removed; state abbreviations title-cased. Review findings (16) that changed the
code before the final run: upper-case possessives, `#`-units, leading zeros of any length
up to 4 digits, alias forms `dba/aka/f/k/a/t/a`, US state last-wins ordering, Indic legal
spellings, PO boxes, `(The)` tag, country tags in brackets, K-style bracket legal forms.
Also fixed on the way: `learn_dictionary` crash on empty addresses; an Arrow
`binary_join_element_wise` type mismatch (large_string vs string separator).

### 5.8 Run

`python -m cleaning.clean --split {train,test} --workers 10`, both splits, 24.2M rows, 0
per-row errors, about 25 min total (train S1 75 s, S2 183 s). Reports:
`_cleaning_report.json` (rule counts per file and country) and `_cleaning_examples.md`
(before/after samples per rule).

Selected rule counts (rows affected):

| File / country | Rows | Notable rules |
| --- | ---: | --- |
| train S1 US | 1,323,633 | legal form 752k, unit 208k, city suffix 150k, state moved 88k, dotted legal 23k |
| train S1 India | 883,188 | state canonical 883k, number marker 405k, city alias 253k, street abbrev 162k, dedupe 152k, country tag 44k |
| train S2 India | 2,017,799 | number marker 1.15M, case 864k, Indic state 478k, dictionary translit 459k, Indic legal 422k, honorific 154k, leading zero 126k, bracket legal 74k, placeholder 60k, web segmented 56k, id tag 5k |
| train S2 US | 3,016,817 | street abbrev 1.44M, case 853k, punct 400k, junk edge 298k, state moved 194k, leading zero 152k, web segmented 121k, placeholder 116k, unit 103k, bracket legal 91k, dotted legal 77k, id tag 10k |
| train S3 US | 3,170,056 | full state name 2.91M, street abbrev 1.45M, unit 475k, alias 116k, web segmented 122k |
| test S1 France | 259,452 | legal form 180k, translit 73k, country tag 21k, street abbrev 20k, Ets 10k, bis/ter 7k |
| test S2 France | 703,378 | legal form 422k, street abbrev 277k, region added 230k, department to region 227k, translit 131k, number marker 129k, bis/ter 37k, dotted legal 32k, web segmented 25k, Saint 12k |

Examples: `राम मार्केटिंग प्राइवेट लिमिटेड` -> `Ram Marketing`; `Quonovi t/a United Software
Private Limited` -> `United Software`; `wegmanpinkertonlive.com` -> `Wegman Pinkerton Live`;
`Mack Rd, Haltom City, Texas` -> `Mack Road, Haltom, TX`; `91 AVE LOUIS BRAILLE, Tourcoing,
Nord` -> `91 Avenue Louis Braille, Tourcoing, Hauts-de-France`; `5139 ROCK SPRINGS RD, <NULL>,
FAUQUIER COUNTY, VA` -> `5139 Rock Springs Road, Fauquier, VA`.

### 5.9 Effect and a related fix

Same old blocker, same 1% sample: pair completeness 0.8930 on raw data -> 0.9165 on cleaned
data (India 0.851, US 0.960), F0.5 ceiling 0.9530 -> 0.9619. The normaliser's `_RUNS` regex in
`blocking_strategies/harness/textnorm.py` was changed to collapse repeated letters only,
because collapsing digits turned "1100" into "10"; changing textnorm invalidates every
`_norm_cache`.

---

## 6. Candidate generation (blocking)

### 6.1 Background

Earlier work (branch `sparse-dot-topn-blocking`, `docs/BLOCKING_EVALUATION.md`): the
shipped exact-key baseline recovered 30% of links; rare-token blocking with `unidecode`
transliteration and the address field lifted it to 87-89%; the address, not the name,
carries the recall (name-only cross-script recall 0.38 vs 0.85 with the address). The
engine is a sparse matrix product: candidates x rare tokens (0/1) against queries x rare
tokens (`idf * field weight`), `sparse_dot_topn` keeps the top k per row across all cores.
A token is "rare" when its document frequency in the pool is at most `max_df` (0.5%).

### 6.2 The recall audit that shaped the new blocker

The 813,701 training links missed by the single top-20 pass were bucketed:
- 84% were true matches that were outranked: they share a rare token, but a competitor
  shares more. Typical causes: the candidate's name changed while the address matches (34%);
  the candidate has no address and its 2-3 word name loses to address-sharing competitors
  (20%); an identical generic name pushed out by look-alikes (15%).
- 16% shared only common tokens (short house numbers, generic words).
- Under 1% shared nothing.

Intuition: the misses are not absent from the index, they lose the competition. So add
passes that change who competes (address only, name only), passes whose tokens are rare even
when every unigram is common (bigrams), and exact keys, then fuse.

### 6.3 `union_passes` design (`strategies/union_passes.py`, registry key `union_passes`)

Five passes on the cleaned `name_key` / `addr_key`, tokens of length >= 2 (digits always
kept):

| Pass | Index | Query weight | k |
| --- | --- | --- | --- |
| base | name + address unigrams | name 1.0, address 0.6 | 30 |
| addr | address unigrams only | 1.0 | 15 |
| name | name unigrams only | 1.0 | 10 |
| bigram | all adjacent name bigrams + address bigrams containing a number | 1.0 | 10 |
| exact | K1 = `name_key|last address token` (needs non-empty name and address); K2 = whole address key (>= 2 tokens); buckets larger than `max_bucket` 20 are skipped | - | all |

Fusion: reciprocal rank fusion, `sum over passes of w_pass / (rrf_k + rank)` with `rrf_k`
20 and weights base 1.0, addr 0.8, name 0.6, bigram 0.7, exact 1.0; dense rank within each
pass per Source-1 row. Cap 30 candidates per (Source-1 entity, source) by fused rank, ties
broken by score. Every kept pair is rescored with each pass's own index and the maximum is
exported (`score + 1e-3 * rrf`), so the matcher's score and competition features keep a
meaning for pairs found only by a secondary pass.

Parameters of the shipped configuration: `k_base 30, k_addr 15, k_name 10, k_bigram 10,
max_bucket 20, cap 30, max_df 0.005, min_len 2, address_weight 0.6, rrf_k 20`, threads =
cores - 2.

### 6.4 Measurements (1% hash sample of train S1, full pools, cleaned data)

| Run | Pair completeness | India / US | F0.5 ceiling | Pairs per entity | Time |
| --- | ---: | --- | ---: | ---: | ---: |
| old blocker, raw data | 0.8930 | - | 0.9530 | 40 | - |
| old blocker (`token_idf_fast` k=20, max_df 0.005, min token 3), cleaned | 0.9165 | 0.851 / 0.960 | 0.9619 | 40 | 91 s |
| union_passes, shipped config | 0.9695 | 0.954 / 0.980 | 0.9897 | 60 | 289 s |
| union_passes without bigrams | 0.9596 | 0.935 / 0.976 | 0.9861 | 60 | 219 s |
| union_passes cap 40 (k_base 40) | 0.9724 | 0.959 / 0.981 | 0.9908 | 80 | 294 s |

22,123 evaluation entities, 76,949 true links, 0 entities without candidates, 90.2% of
entities fully recovered (old blocker 79.0%), peak RSS 4.6 GB. Cap 40 was judged not worth
+33% pairs for +0.3 points; the bigram pass is worth 1 point and was kept.

### 6.5 Export (`blocking_strategies/export_candidates.py`, into `candidates_v2/`)

Options added this session: `--strategy`, `--param KEY=VALUE`, `--s1-fraction` (train only;
refused on test after the review), `--no-tsv`. One Parquet part per (country, source),
written atomically and skipped on rerun. Columns `s1_id, cand_id, source, country, score,
rank` plus `label` for train.

| Part | S1 entities | Pool | Pairs | S1 without candidates | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| test France S2 / S3 | 259,452 | 703,378 / 731,615 | 7.76M / 7.76M | 3 / 4 (v1: 834) | 67 / 72 |
| test India S2 / S3 | 809,986 | 2,312,565 / 2,405,000 | 24.28M / 24.29M | 1 / 0 | 908 / 944 |
| test US S2 / S3 | 663,106 | 1,871,330 / 1,945,701 | 19.89M / 19.89M | 0 / 0 | 548 / 573 |
| train India S2 / S3 (50%) | 441,435 | 2,017,799 / 2,115,547 | 13.23M / 13.24M | 0 | 480 / 506 |
| train US S2 / S3 (50%) | 661,476 | 3,016,817 / 3,170,056 | 19.84M / 19.84M | 0 | 992 / 1035 |

Test total 103.9M pairs (60 per entity) in 53 min; train total 66.2M pairs with 3.70M
positives found (recall 0.970) in 51 min. Train was exported for a 50% hash sample of
Source-1 entities to fit the time budget, which later caused a validation artefact (7.9).

### 6.6 Defect found in review and fixed in code

The export rescored every pair with the base index only, so about 10% of pairs (found by
the address, name or bigram pass) carried a near-zero score, which distorts the score,
`cand_margin` and `cand_rel` features. The code now exports the maximum over all pass
scores and breaks cap ties on it. `candidates_v2/` predates the fix; v4 is the re-export
with the fixed scorer (only the two France test parts were written before the stop).

### 6.7 What limits recall now

About 15% of the remaining misses changed both the house number and the name, which no
lexical method can bridge, so the lexical ceiling is about 98.4%. Untested cheap levers,
each to be measured on the 1% sample (about 5 min per run): cap 40, larger `k_addr`,
`max_df` 0.01, `max_bucket` 40. `missed_links.py` (bucket misses by cause) was written but
never run.

---

## 7. Matcher

### 7.1 Pipeline (`matching/`, all steps stream in chunks)

1. `prep_entities`: per-record attributes once per split (7.2).
2. `cand_stats`: per-candidate competition statistics over the whole candidate file (7.3).
3. `sample`: 10% of Source-1 entities with all their pairs, 20% of them flagged validation;
   `n_true` counts every ground-truth link including those the blocker missed.
4. `features`: 74 features per pair, sharded per country (`--shards 4` train, `2` test,
   `--workers 8` for the pure-Python token loop).
5. `train_full`: hard-negative mining, streaming LightGBM, rule tuning on validation.
6. `predict`: score every test pair, apply the rule per country, write the two files.

### 7.2 Entity preprocessing (`prep_entities.py`, `common.py`, `france.py`)

Per record: `name_key` (squeezed transliterated name, legal tokens removed; French records
through `france.py`), `name_base` (a-z0-9 only), `addr_c` (squeezed address with US state
names -> codes, Indian state variants unified, single-token abbreviations `r/bd/all/imp/che/
rte/fg/crs` expanded, `saint` mapped onto the same symbol as `st`), `addr_base`, `legal`
(bitmask over 22 legal-form groups, taken from the cleaning sidecar), `nonascii`, `web`. IDF
tables of name and address tokens are computed from Source 1 of the split.

`france.py` (written by the parallel session; test only): rejoins dotted legal forms,
maps `5ARL/5AS` typos, `Frs` -> `Freres`, `Ets/Cie` spellings, drops suffix noise words that
are far more frequent in S2/S3 than in S1 (`Participations` 5,865x, `Holding` 342x,
`Trading` 120x, `Distribution` 91x, `Associes` 67x, `International` 29x, `Developpement`
10x, `Groupe` 4.5x), drops French stop words and `EI`; for addresses drops region and
department components, keeps house numbers unsqueezed, strips `N°`/`bis`/`ter`/5-digit
postcodes, expands `CH/Q/PSG`, repairs one-edit typos of long street types.

### 7.3 Competition statistics (`cand_stats.py`)

For every Source-2/3 candidate over the full candidate file: `cand_max` (best blocking score
against any Source-1 record), `cand_second` (second best; ties at the max make the second
equal to the max, meaning no unique owner), `cand_n` (number of lists it appears in). No
labels are used, so the same code runs on test.

### 7.4 Features (74, `features.py`; no feature uses `country`)

Blocking (13): `is_s3, score, rank, g_max` (best score in the entity's list for that
source), `score_rel = score/g_max, score_gap, n_above` (strictly better candidates), `tie_n,
g_n` (list size), `cand_rel = score/cand_max, cand_best` (score >= cand_max - 1e-4),
`cand_margin` = (score - cand_second)/cand_max when best else (score - cand_max)/cand_max,
`cand_n = log1p(count)`.

Entity (13): `legal1, legal2, legal_eq` (same non-zero mask), `legal_conflict` (both set, no
shared bit), `nonascii2, web2, nm_empty2, ad_empty1, ad_empty2, nm_len1, nm_len2, ad_len1,
ad_len2`.

String similarity, rapidfuzz in C++ (13): `nm_ratio, nm_partial, nm_tsort, nm_tset, nm_jw,
nmb_ratio` (ratio on baseline keys), `ad_ratio, ad_partial, ad_tsort, ad_tset, ad_jw,
adb_ratio, all_tset` (token-set ratio on name + address concatenated, robust to swapped
fields and website names).

Token loop (28, -1 when undefined): `nm_jacc, nm_contain` (overlap / smaller set), `nm_idf`
(IDF-weighted Jaccard), `nm_idf_cov1` (weighted coverage of the S1 name), `nm_first_eq,
nm_acronym, nm_ntok1, nm_ntok2, nm_rare_shared` (max IDF of a shared token), `nm_rare_diff`
(max IDF of an unshared token), `ad_jacc, ad_contain, ad_idf, ad_idf_cov1, ad_idf_cov2,
ad_rare_shared, ad_rare_miss1` (rarest S1 address token absent from the candidate),
`num_jacc, num_inter, num_n1, num_n2, num_miss1` (fraction of S1 numbers missing),
`num_first_eq, postal_eq` (shared 5-6 digit token), `ad_last_eq, ad_ntok1, ad_ntok2, all_idf`
(pooled name+address weighted Jaccard).

Group-relative (7): `nm_tset, ad_tset, all_tset, nm_idf, ad_idf, all_idf, num_jacc` each minus
the best value in the same Source-1 entity's list (`_dbest`), so the model sees "is this the
best candidate here".

### 7.5 Training-set construction (`train_full.py`)

The full training split (88M pairs x 74 float32 features, about 26 GB) does not fit in
memory, so:
- Hard-negative mining: a negative is hard when `cand_best == 1` or `rank <= 3` or
  `cand_rel >= 0.8` (keeps 14.9% of negatives and covers 98.7% of the negatives the first
  model found hard). All positives and hard negatives are kept; easy negatives are sampled at
  `--easy-rate 0.1` with weight 10 so probabilities stay calibrated.
- Held out: `val` = the sample10 validation entities (rule tuning and reported score), `es` =
  1% of the other entities (`--es-pct 1`, early stopping only).
- The downsampled set is written to Parquet and fed through `lightgbm.Sequence`, so only
  LightGBM's binned copy (1 byte per value, `max_bin 255`) is resident.

Assembled sets:

| Model | Positives | Hard negatives | Easy negatives kept (of) | ES rows | Val rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 (raw, 100% train) | 6,620,612 | 11,737,022 | 6,722,309 (67.2M) | 867,511 | 1,763,912 |
| v2 / v3 (cleaned, 50% train) | 3,590,582 | 17,587,445 | 4,302,265 (43.0M) | 650,291 | 1,322,518 |

### 7.6 LightGBM parameters

Base (`train_eval.LGB_PARAMS`): `objective binary, learning_rate 0.08, num_leaves 127,
min_data_in_leaf 200, feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2
1.0, max_bin 255, num_threads 12`; early stopping after 50 rounds without improvement of the
ES binary log-loss; CPU only. CLI overrides added this session: `--rounds, --learning-rate,
--num-leaves, --feature-fraction, --min-data-in-leaf, --drop <features>` (the model records
its feature names, and `predict` reads them from the booster, so a model without the
competition features can be shipped).

| Model | Rounds | Learning rate | Feature fraction | Best iteration | Train time |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 2000 | 0.08 | 0.8 | 1999 (hit the cap) | 2,796 s |
| v2 | 2000 | 0.08 | 0.8 | 1532 | 3,488 s |
| v3 | 4000 | 0.05 | 0.6 | 3081 | 3,593 s |

### 7.7 Validation design

Sampling is by Source-1 entity, never by pair, because group features are only correct
when the whole list is present. Every reported score is the macro F0.5 over all validation
entities including singletons, entities without candidates and links the blocker missed
(`n_true` from the ground truth). The candidate "ceiling" is the score of a perfect matcher
on the same candidates.

### 7.8 Decision rules (`train_eval.apply_rule`)

- `threshold`: match every candidate with `p >= t`; grid 0.05-0.95 step 0.025.
- `expected_f` (per entity): sort candidates by `p`; `S_k` = sum of the k largest;
  `T = sum(p) * (1 + miss)` estimates the true link count including links never retrieved;
  choose the k maximising `1.25 * S_k / (0.25 * T + k)`; predict empty when
  `gamma * prod(1 - p)` beats the best k. Grid `miss` in {0, 0.05, 0.1, 0.15, 0.2}, `gamma` in
  {0.5, 1, 1.5, 2, 3}.
- `exclusive`: before either rule, each candidate keeps only its highest-`p` Source-1 entity
  (exact at full scale because each S2/S3 record has one owner; on a sample its true owner
  may be unsampled, so it looks weaker there).

### 7.9 Model versions and results

| Model | Data | Validation F0.5 (India / US) | Candidate ceiling | Rule shipped | Status |
| --- | --- | --- | ---: | --- | --- |
| v1 | raw, old blocker k=20, 100% train | 0.9403 (0.905 / 0.964) | 0.9544 | expected_f miss 0.15 gamma 2.0, no exclusive | uploaded, leaderboard 0.902 |
| v2 | cleaned, union_passes, 50% train | 0.9755 (0.970 / 0.979) | 0.9899 | expected_f miss 0.15 gamma 2.0 + exclusive | uploaded, leaderboard 0.95 |
| v3 | as v2, 4000 rounds, lr 0.05, ff 0.6 | 0.9759 (0.970 / 0.980) | 0.9899 | tuned: miss 0.2 gamma 3.0 (threshold 0.725 gives 0.9757) | trained, not predicted |

Validation artefact: `train_full` printed 0.514 for v2 because the train export covered 50%
of Source-1 entities while sample10's validation entities span all of Source 1, so half of
them had no candidate rows. The score was recomputed on the 22,046 validation entities that
have candidates; `sample.py` now restricts itself to exported entities and
`sample10/s1.parquet` was fixed on disk (original in `s1_all.parquet`). The tuned rule on
the broken validation read miss 0.1; the prediction used miss 0.15 (v1's tuned value) with
the exclusive rule on. Rules differ by under 0.2 points in every experiment.

### 7.10 Feature importance (share of LightGBM gain)

| Rank | v1 | v2 | v3 |
| --- | --- | --- | --- |
| 1 | cand_margin 0.615 | all_tset 0.620 | all_tset 0.537 |
| 2 | cand_rel 0.147 | all_tset_dbest 0.134 | all_tset_dbest 0.170 |
| 3 | num_miss1 0.047 | num_miss1 0.047 | cand_margin 0.091 |
| 4 | all_tset 0.030 | cand_margin 0.047 | num_miss1 0.034 |
| 5 | num_jacc 0.018 | all_idf 0.026 | all_idf 0.018 |

Reading: on raw data the model leaned on the competition features (61% of gain in
`cand_margin`), which are exactly the features that shift on France and on the denser test
pools. After cleaning, the pooled token-set similarity and its group-relative version carry
75% of the gain. This is the main reason v2 stopped emptying French entities.

### 7.11 Timings (16 cores, 16 GB)

Cleaning 25 min; union export 53 min (test) and 51 min (train 50%); `prep_entities` 16 min;
features 3.3 h total (test 104M pairs about 2 h); training 50-60 min; prediction over 104M
pairs about 1 h (about 10k pairs/s); validator about 10 min.

---

## 8. Submissions, probes and the leaderboard

### 8.1 Files and their behaviour

| File | Leaderboard | Empty overall | Matches / entity | France empty / per entity | India | US |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `output/` (v1) | 0.902 | 9.33% | 2.93 | 16.85% / 2.50 | 9.63% / 2.83 | 6.02% / 3.22 |
| `output_v2/` (v2 main) | 0.95 | 5.43% | 3.33 | 3.28% / 3.69 | 5.91% / 3.21 | 5.69% / 3.33 |
| `output_v2_probes/..._france_empty.tsv` | 0.83 | 19.92% | 2.78 | 100% / 0 | as v2 | as v2 |
| `output_v2_g4/` and `..._france_conservative.tsv` (gamma 4) | - | 5.70% | 3.33 | 3.89% / 3.69 | 6.16% / 3.22 | 5.85% / 3.33 |
| `..._france_t03.tsv` (France threshold 0.3) | - | 5.33% | 3.38 | 2.61% / 4.01 | as v2 | as v2 |
| `..._france_t09.tsv` (France threshold 0.9) | - | 5.70% | 3.28 | 5.04% / 3.37 | as v2 | as v2 |

All probe files differ from v2 only in the French rows; `candidate_pairs.tsv` is shared. The
v2 pair and the France-empty probe passed the validator; the t09 validator run was
interrupted by the stop and must be re-run before upload.

### 8.2 Decomposing the leaderboard score

France is 259,452 of 1,732,544 test entities (14.98%), so
`F_total = 0.850 * F_USIN + 0.150 * F_FR`. Emptying every French row scores the French
singleton rate `s_FR` (about 0.056) on those rows, hence `F_v2 - F_probe = 0.150 * (F_FR -
s_FR)`. With 0.95 and 0.83: `F_FR ~ 0.857`, and then `F_USIN ~ 0.967`.

### 8.3 Where the remaining 4-5 points are (approximate, per point of total score)

| Loss | Size | Evidence |
| --- | ---: | --- |
| France (0.857 vs 0.967 on 15% of entities) | 1.7 | probe algebra above |
| Matcher inside candidates (ceiling 0.990 vs 0.9755 on 85%) | 1.2-1.4 | validation |
| Missing candidates (ceiling 0.990 vs 1) | 0.9-1.0 | validation ceiling |
| Density shift (validation 0.9755 vs leaderboard 0.967 for US+India) | 0.7-0.9 | leaderboard minus validation |

### 8.4 Submission plan agreed for the day

Two uploads were left after the probe. Next: `matching_results_france_t09.tsv` (France
appears to over-predict: 3.3% empty against an expected 5.6% singletons and 3.69 matches per
entity, more than US/India; `gamma` barely moves France because its probabilities are
bimodal, a threshold does). The last upload is reserved for v4.

---

## 9. France, end to end

1. Diagnosis (4.6): v1 emptied 16.6% of French entities because the competition features
   were flat and the legal-form veto strong; accents were not the cause.
2. Cleaning: department -> region, region completion, street abbreviations, Saint, bis/ter,
   postcodes and floors removed, `Ets/Sté` spellings, French legal forms including `EI`,
   dotted and bracketed forms, `(France)` tags.
3. Matcher preprocessing: `france.py` rules (7.2) for French `name_key` and `addr_c`.
4. Result in v2: French empty rate 3.3% (from 16.6%), 3.69 matches per entity; only 7 French
   entities without candidates (from 834). Leaderboard implies France F0.5 about 0.86, so
   France now looks over-predicted rather than under-predicted.
5. Measurement without labels: probes that change only French rows, read through the
   algebra of 8.2. Threshold probes at 0.3 and 0.9 are ready; 0.9 is the recommended one.
6. Not yet done: a France-specific model trained with `--drop cand_margin cand_rel cand_best
   cand_n` (no competition features), selected by US -> India transfer, applied to French
   rows only.

---

## 10. Adversarial reviews (what they found)

Cleaning review (before the final run): see 5.7; every finding was fixed and the data
re-cleaned.

v2 review (Sonnet reviewer, 15:00):
1. `union_passes` exported base-only scores, about 10% of pairs near zero -> fixed to the
   maximum of pass-native scores (6.6); `candidates_v2` predates the fix.
2. Exact key K1 fired on records without an address -> now needs a non-empty name and address.
3. The exporter accepted `--s1-fraction` on test -> now refused.
4. Validation covered unexported entities (the 0.514 artefact) -> `sample.py` restricted.
5. Open: the 50% train export thins `cand_stats` competition counts relative to test; France
   may over-predict; test density shift unresolved.

---

## 11. State on disk (`C:\Users\manoj\Downloads\amazon ML resource`)

- `student_resource/dataset_clean/{train,test}`: cleaned TSVs, sidecars, reports, examples.
- `candidates/`: v1 export (locked by another process, cannot be moved); `candidates_v2/`:
  the export used by v2/v3; `candidates_v4/test/parts`: France S2/S3 only (stopped).
- `work/match/`: v1 world; `work/match_v2/`: entities, cand_stats, idf, sample10 (restricted
  `s1.parquet`, original `s1_all.parquet`), `full_train`, `full_test/pred_full`, `full`
  (v2 model, `model.txt`, `rule.json`, `results.json`), `full_v3`; `work/match_v4/` empty;
  `work/match_fr/` (other session's France-normalisation probe); logs `work/*.log`.
- `output/` (v1), `output_v2/` (v2), `output_v2_g4/`, `output_v2_probes/` (4 files),
  `output_probes/` (earlier-session probes A-E), `output_v4/` empty.
- `code/business_entity_resolution/src`: `cleaning/`, `blocking_strategies/` (with
  `strategies/union_passes.py`, `export_candidates.py`, `missed_links.py`), `matching/`,
  `docs/`, `runs/clean/*.json`, `HANDOFF.md`. `matching/`, `cleaning/`, `union_passes.py`,
  `docs/EDA_REPORT.md` and this file are untracked and need committing.
- Memory notes for future sessions live outside the repo.

---

## 12. Open issues and next steps

1. Nothing runs until the stop is lifted. On resume: finish the v4 export (fixed scorer;
   test then train at 100%), then with `ER_CANDIDATES=candidates_v4 ER_WORK=work/match_v4`:
   copy entities and idf from v2, `cand_stats` for both splits, `sample`, features (train 4
   shards, test 2), `train_full` (rounds 3500, lr 0.05, feature fraction 0.6), `predict` with
   the exclusive rule into `output_v4/`, validator. About 9 hours serial.
2. Upload `matching_results_france_t09.tsv` after re-running the validator; keep the last
   upload for v4.
3. France model without competition features if France still lags after the threshold probe.
4. Recall levers to measure on the 1% sample: cap 40, larger `k_addr`, `max_df` 0.01,
   `max_bucket` 40; run `missed_links.py` to bucket the remaining misses.
5. Matcher levers beyond 0.98: sidecar alias/unit/honorific features, pass-provenance bits
   for each pair, a two-seed ensemble, a cross-encoder on the uncertain probability band,
   name-only handling for empty-address candidates, softer legal-form conflict.
6. Fill `Documentation_template.md` (EDA from `docs/EDA_REPORT.md`, blocking from section 6,
   model from section 7, results from section 8), copy the final pair of files to `output/`,
   build the zip, commit the untracked code.

---

## Appendix A. Parameter quick reference

| Component | Parameter | Value |
| --- | --- | --- |
| Dictionary learner | min_count / purity / sim / ascii_every / workers | 25 / 0.6 / 40 / 8 / 8 |
| Cleaner | workers, leading-zero width, honorifics, articles | 10; 1-4 digits; mr mrs dr smt shri sri m/s messrs; the |
| Normaliser | `_RUNS` | letters only |
| Blocker | passes | base, addr, name, bigram, exact |
| Blocker | k_base / k_addr / k_name / k_bigram | 30 / 15 / 10 / 10 |
| Blocker | max_df / min_len / address_weight | 0.005 / 2 / 0.6 |
| Blocker | rrf_k / weights base, addr, name, bigram, exact | 20 / 1.0, 0.8, 0.6, 0.7, 1.0 |
| Blocker | cap per (entity, source) / max_bucket | 30 / 20 |
| Export | train s1 fraction (v2) | 0.5 |
| Sample | fraction / validation share / es share | 0.10 / 0.20 / 1% of the rest |
| Hard negatives | cand_best or rank <= 3 or cand_rel >= 0.8; easy rate 0.1, weight 10 | |
| LightGBM | objective, learning_rate, num_leaves, min_data_in_leaf | binary, 0.08 (v3 0.05), 127, 200 |
| LightGBM | feature_fraction, bagging_fraction, bagging_freq, lambda_l2, max_bin | 0.8 (v3 0.6), 0.8, 1, 1.0, 255 |
| LightGBM | rounds, early stopping, threads | 2000 (v3 4000), 50, 12 |
| Rule | expected_f miss / gamma / exclusive (v2 shipped) | 0.15 / 2.0 / on |

## Appendix B. Feature list (74)

Blocking: is_s3, score, rank, g_max, score_rel, score_gap, n_above, tie_n, g_n, cand_rel,
cand_best, cand_margin, cand_n.
Entity: legal1, legal2, legal_eq, legal_conflict, nonascii2, web2, nm_empty2, ad_empty1,
ad_empty2, nm_len1, nm_len2, ad_len1, ad_len2.
Fuzz: nm_ratio, nm_partial, nm_tsort, nm_tset, nm_jw, nmb_ratio, ad_ratio, ad_partial,
ad_tsort, ad_tset, ad_jw, adb_ratio, all_tset.
Loop: nm_jacc, nm_contain, nm_idf, nm_idf_cov1, nm_first_eq, nm_acronym, nm_ntok1, nm_ntok2,
nm_rare_shared, nm_rare_diff, ad_jacc, ad_contain, ad_idf, ad_idf_cov1, ad_idf_cov2,
ad_rare_shared, ad_rare_miss1, num_jacc, num_inter, num_n1, num_n2, num_miss1, num_first_eq,
postal_eq, ad_last_eq, ad_ntok1, ad_ntok2, all_idf.
Relative: nm_tset_dbest, ad_tset_dbest, all_tset_dbest, nm_idf_dbest, ad_idf_dbest,
all_idf_dbest, num_jacc_dbest.

## Appendix C. Commands (run from `src/` with the environment variables of section 3)

```bash
python -m cleaning.learn_dictionary --data-dir "$ER_ROOT/student_resource/dataset" --workers 8
python -m cleaning.clean --split train --in-dir "$ER_ROOT/student_resource/dataset/train" --out-dir "$ER_DATASET/train" --workers 10
python -m blocking_strategies.harness.runner union_passes --data-dir "$ER_DATASET/train" --sample-fraction 0.01 --param cap=30 --out runs/clean/x.json
python -m blocking_strategies.export_candidates --split test --strategy union_passes --data-dir "$ER_DATASET/test" --out-dir "$ER_CANDIDATES/test" --no-tsv
python -m matching.prep_entities --split train
python -m matching.cand_stats --split train
python -m matching.sample --fraction 0.10 --name sample10
python -m matching.features --split train --full --shards 4 --workers 8
python -m matching.train_full --features "full_train/features_*.parquet" --tag full --rounds 3500 --learning-rate 0.05 --feature-fraction 0.6
python -m matching.predict --model "$ER_WORK/full/model.txt" --rule '{"name":"expected_f","exclusive":true,"miss":0.15,"gamma":2.0}' --tag full --out-dir "$ER_ROOT/output_v4"
python utils/validate_submission.py --matching ../output_v4/matching_results.tsv --candidate ../output_v4/candidate_pairs.tsv --test-dir dataset/test
```
