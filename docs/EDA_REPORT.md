# Exploratory data analysis — Amazon ML Challenge 2026 (business entity resolution)

Date: 2026-09-26. Data: `student_resource/dataset/{train,test}` (full files, 22.2M records,
7.64M ground-truth links). Scripts and figures: `docs/charts/` (`eda_profile.py` streams the
raw TSVs; `eda_pairs.py` works on the matcher's validation split, test features and
predictions; `france_probe.py` prints French examples; `eda_charts.py` draws the figures).

Figures: `charts/fig1_dataset_structure.png`, `charts/fig2_model_and_france.png`.

---

## 1. Inventory

| File | Rows | US | India | France | Size |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_source1 | 2,206,821 | 1,323,633 | 883,188 | – | 210 MB |
| train_source2 | 5,034,616 | 3,016,817 | 2,017,799 | – | 489 MB |
| train_source3 | 5,285,603 | 3,170,056 | 2,115,547 | – | 504 MB |
| train_ground_truth | 2,206,821 rows, 7,638,365 links | | | | |
| test_source1 | 1,732,544 | 663,106 | 809,986 | 259,452 | 175 MB |
| test_source2 | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 | 510 MB |
| test_source3 | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 | 506 MB |

Columns are exactly `entity_id, business_name, business_address, country`. No row has a
tab or double quote inside a field (0.05% of French names contain a quote, harmless with
`quoting=QUOTE_NONE`). Every `entity_id` is unique within its file and carries the correct
`S1-/S2-/S3-` prefix.

## 2. Data quality and what is missing

| Check | Result |
| --- | --- |
| Empty `business_name` | none, anywhere |
| Placeholder values (`nan`, `n/a`, `-`, …) | none (0.01% of French names) |
| Empty `business_address` | Source 1: never. Source 2/3: 3.7%/3.5% US train, 2.9%/3.1% India train; test 2.9%/2.8% US, 2.3%/2.5% India, 3.1%/2.9% France |
| Duplicate `(name, address)` inside a file | Source 1: **0** (it really is deduplicated). Source 2: 36,650 rows (0.73%) train, 32,012 test. Source 3: 24,031 (0.45%) train, 21,856 test. Exact-duplicate S2/S3 rows can *both* be true matches of the same S1 record (seen in samples), so never collapse them. |
| Ground-truth integrity | all 2,206,821 S1 ids present exactly once; every matched id exists in its source file; 0 duplicate links; **0 candidates linked to more than one S1 entity**; **0 cross-country links** |
| Leakage | none: Pearson correlation between linked ids is 0.0001 (same as shuffled), between file row positions 0.0001; the ground-truth file order is unrelated to the source-1 file order (Spearman 0.0003). Ids are random 9-digit numbers. |
| Id numeric parts | collide across sources (26,801 S2/S3 collisions in train, 11,001 S1/S2). Always keep the prefix. |
| Train/test overlap | 0 shared ids; 29 (S2) and 19 (S3) shared `(name, address)` strings out of ~5M each; 0 for France. Test is a disjoint draw from the same generator. |

**Missing things the pipeline must handle** (each is quantified in later sections):

1. Empty candidate addresses (3% of S2/S3): these pairs are name-only and are where 35%
   of the matcher's false negatives come from (section 6).
2. Postal codes are almost absent: 11% of US addresses carry a 5-digit ZIP, India ~0%
   (PIN codes essentially never appear), France 0.4%. `postal_eq` is undefined on 94–100%
   of pairs and is dead weight.
3. Unit / suite numbers: 16% of US Source-1 addresses have one, Source 2 keeps it in 0.1%
   of records, Source 3 in 8%. Unit tokens are noise for Source-2 comparisons.
4. Leading-zero house numbers ("0029 R BLANCHE", "062 BOULEVARD", "00202"): 4.4% of US
   S2/S3 addresses, 2.9% France, 1.6% India. The normaliser keeps them (and `squeeze`
   turns "0029" into "029", and "1100" into "10"), so numeric agreement fails.
5. Name suffixes injected by the generator: "(ID: 93967)" in 0.3% of S2/S3 names,
   "Formerly <old name>" in 0.6% of S3 names, "DBA" in 1% of S3 names, "(India)" /
   "(France)" in 5–8% of names in those countries, "(The)" and "[EURL]"-style brackets.
6. Dotted legal forms ("S.A.S.", "E.U.R.L.", "L.L.C.", "P.C."): 4–6% of S2/S3 names.
   `fold()` turns them into single letters ("s a s"), so they are neither stripped nor
   recognised as a legal form.
7. Legal form moved to the front ("LLC Moncada Learning Center", "SCI Ptit Amicale"):
   2.5–4.8% of S2/S3 names; harmless for token-set features, harmful for `nm_ratio`.

## 3. Ground-truth structure (train)

Per Source-1 entity (identical for US and India to three decimals, which says the two
countries were generated with the same parameters):

| | value |
| --- | --- |
| mean matches | 3.461 (S2 1.674 + S3 1.788) |
| singletons | 5.58% (123,247 entities) |
| median / p95 / p99 / max | 3 / 6 / 8 / 11 |
| max per source | 5 from S2, 6 from S3 |
| has S2 and S3 matches | 80.5%; S2-only 6.5%; S3-only 7.5% |
| distribution 0..8 | 5.6, 5.4, 17.0, 24.1, 21.9, 14.6, 7.5, 2.9, 0.8 % |

26.6% of Source-2 and 25.4% of Source-3 records match nothing (orphans), again the same
in both countries. Each S2/S3 record belongs to at most one S1 entity, so an exclusivity
rule at inference is exact (it changes nothing on validation but removes 1–4k conflicting
pairs per country on test).

## 4. Train → test shift

| | train US | train India | test US | test India | test France |
| --- | ---: | ---: | ---: | ---: | ---: |
| S2 records per S1 entity | 2.28 | 2.28 | 2.82 | 2.86 | 2.71 |
| S3 records per S1 entity | 2.39 | 2.40 | 2.93 | 2.97 | 2.82 |
| S2+S3 per S1 | 4.67 | 4.68 | 5.76 | 5.82 | 5.53 |

The test pools are **23% denser** than train. Either test entities have more matches
(≈4.3 per entity if the 74% matched-record rate of train holds) or the orphan share is
higher (≈40% if 3.46 matches per entity holds). This cannot be resolved without labels;
it matters because the shipped model predicts 3.22 (US), 2.83 (India) and 2.51 (France)
matches per entity, all below the train mean. One leaderboard probe settles it (section 8).

Record-level distributions (lengths, scripts, legal forms, address formats) are
otherwise identical between train and test for US and India, so the covariate shift
is confined to (a) pool density and (b) France.

## 5. Noise patterns

### 5.1 Names

| pattern | US S1 | US S2 | US S3 | IN S1 | IN S2 | IN S3 | FR S1 | FR S2 | FR S3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ASCII only | 100 | 93.3 | 93.2 | 100 | 72.1 | 81.5 | 84.3 | 75.5 | 76.1 |
| Latin with diacritics | 0 | 6.7 | 6.8 | 0 | 4.4 | 5.3 | 15.7 | 24.5 | 23.9 |
| Indic script | 0 | 0 | 0 | 0 | 23.5 | 13.2 | 0 | 0 | 0 |
| has legal form | 52 | 47 | 46 | 85 | 70 | 72 | 67 | 59 | 58 |
| legal form first | 0 | 2.5 | 2.4 | 0 | 4.0 | 4.5 | 0 | 4.8 | 4.7 |
| ALL CAPS | 0 | 21.6 | 3.2 | 0 | 14.9 | 2.6 | 0 | 20.8 | 5.6 |
| all lower case | 0 | 6.7 | 6.9 | 0 | 4.3 | 5.3 | 0 | 6.5 | 6.7 |
| double spaces | 0 | 11.6 | 11.1 | 0 | 10.1 | 10.6 | 0 | 9.1 | 7.9 |
| junk prefix ("-- ", "<< ") | 0.1 | 1.4 | 1.3 | 0 | 1.0 | 1.0 | 0 | 0 | 0 |
| website as name | 0 | 4.4 | 4.2 | 0 | 3.4 | 3.7 | 0 | 3.5 | 3.5 |
| parenthetical | 0 | 3.4 | 3.5 | 5.1 | 7.2 | 7.4 | 8.1 | 8.7 | 8.5 |

Indic scripts in India S2 names: Devanagari 13.4%, Telugu 2.0, Kannada 1.8, Tamil 1.7,
Gujarati 1.5, Bengali 1.5, Malayalam 0.9, Odia 0.4, Gurmukhi 0.3 (S3 about half of each).
India S2/S3 **addresses** are Indic in 23.7% / 22.5% of records, more often than the names
in S3. Accents in US/India names ("Génomic", "Próject") are injected noise, not language.

**How a true match's name differs from its Source-1 name** (49k US and 31k India links,
categories are the first normalisation level at which the two agree):

| category | US/S2 | US/S3 | IN/S2 | IN/S3 |
| --- | ---: | ---: | ---: | ---: |
| identical | 6.0 | 5.7 | 2.7 | 2.9 |
| case only | 7.8 | 7.2 | 3.7 | 3.7 |
| punctuation / spacing only | 11.5 | 13.0 | 8.2 | 9.9 |
| injected accents only | 4.7 | 4.7 | 2.6 | 3.0 |
| legal suffix / stop-word only | 23.2 | 19.4 | 25.9 | 24.6 |
| word order | 3.3 | 3.2 | 0.8 | 0.8 |
| token added or dropped | 13.2 | 18.2 | 10.8 | 16.0 |
| small typo (ratio ≥ 85) | 13.9 | 12.6 | 8.4 | 8.8 |
| typo / abbreviation, medium | 10.0 | 9.8 | 7.7 | 8.1 |
| Indic script | – | – | 23.3 | 13.2 |
| website as name | 0.9 | 0.8 | 0.7 | 0.8 |
| different name | 4.3 | 4.0 | 4.2 | 6.7 |

So about 55% of true matches differ from Source 1 only by case, punctuation, accents,
legal form or word order; 25–30% by a token insertion/drop or a typo; 4–7% have an
unrelated name (the address carries those).

### 5.2 Addresses

| pattern | US S1 | US S2 | US S3 | IN S1 | IN S2 | IN S3 | FR S1 | FR S2 | FR S3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL CAPS | 0 | 89.8 | 0 | 0 | 23.9 | 0 | 0 | 28.8 | 0 |
| leading house number | 86 | 75 | 75 | 25 | 22 | 21 | 86 | 68 | 69 |
| 5-digit code | 10.9 | 10.4 | 10.5 | 0.3 | 1.1 | 1.0 | 0.4 | 0.5 | 0.5 |
| ends with 2-letter state | 86 | 83 | 4 | 0 | 0 | 57 | 0 | 0 | 0 |
| full state name | 2.9 | 2.8 | 92 | 100 | 76 | 20 | – | – | – |
| street type abbreviated | 3 | 48 | 45 | 4 | 3 | 3 | 2 | 23 | 23 |
| unit / suite / floor | 16 | 0.1 | 8 | 22 | 20 | 16 | 0.1 | 0.1 | 0.2 |
| landmark ("near", "opp") | 0 | 0 | 0 | 14 | 12 | 9 | 0 | 0 | 0 |
| mean length (chars) | 35 | 32 | 38 | 78 | 68 | 59 | 50 | 40 | 40 |
| median comma components | 3 | 3 | 3 | 5 | 5 | 4 | 3 | 3 | 3 |

Source conventions are systematic: Source 2 writes US addresses in capitals with
abbreviated street types and no unit numbers; Source 3 writes the full state name for
the US but the two-letter abbreviation for India ("KA", "TG"); Source 1 always writes the
full Indian state and the US abbreviation. 6% of US addresses put the state first
("IA, Iowa City, 1064 Newton Rd"). Indian addresses are long (median 11 tokens),
landmark-based, and 22% of S2/S3 Indian addresses are in Indic script.

**How a true match's address differs from Source 1:**

| category | US/S2 | US/S3 | IN/S2 | IN/S3 |
| --- | ---: | ---: | ---: | ---: |
| identical / case only | 11 | 5 | 10 | 4 |
| abbreviation or state format only | 13 | 19 | 2 | 0 |
| component reorder (± abbreviation) | 11 | 10 | 4 | 2 |
| component dropped (subset) | 7 | 6 | 29 | 6 |
| house number dropped | 15 | 9 | 7 | 1 |
| **house number differs** | 12 | 12 | 14 | 16 |
| small typo | 12 | 17 | 17 | 25 |
| extra or missing tokens | 11 | 14 | 11 | 35 |
| one side empty | 5 | 5 | 4 | 4 |
| different address | 1 | 1 | 1 | 6 |

Note the house number *differs* in 12–16% of true matches ("2915" vs "915", "6445" vs
"6445-", "No 26" vs "H.NO A-26"). Only 8% of US true pairs share a postal code, so ZIP
agreement is rare evidence, not a requirement.

Joint view (US/S3): name-same & address-same 16%, name-same & address-different 35%,
name-different & address-same 18%, **both different 27%**, address missing 5%. India/S3:
both different 48%. A large share of true matches therefore needs soft evidence from
both fields, which is why the blocking score and its competition features dominate the
model.

## 6. Where the current pipeline loses score

Blocking (`token_idf_fast`, top-20 per source) retrieves 89.35% of links; the ceiling of
the candidate file is F0.5 = 0.954 (US 0.977, **India 0.921**). Inside the top-20 lists,
the true match is at rank 1 only 44.5% of the time, within the top 3 86.9%, top 5 94.9%,
top 10 98.0%.

Matcher on the 44,122-entity validation split (1.76M pairs, 136,845 positives):

| | value |
| --- | --- |
| AUC of the model probability | 0.9999 (US and India alike); single best raw feature `cand_margin` 0.986, `all_tset` 0.985 |
| calibration | excellent (bin 0.5–0.6 → 55% positives, 0.9–1.0 → 99.8%) |
| pair precision / recall inside candidates | 0.994 / 0.974 (764 FP, 3,533 FN) |
| macro F0.5 inside candidates | 0.985 → actual 0.940, i.e. ≈4.5 points are lost to blocking recall and ≈1.4 to the matcher |
| entities | 82–85% perfect, 8% partial, 0.4% all missed, 0.2–0.3% false merge on a singleton |
| decision rules | every rule tried (thresholds 0.5–0.8, expected-F variants, exclusive) is within 0.2 points; the rule is not the lever |

False negatives: 35% of them are candidates with an **empty address** (37% FN rate on
those pairs versus 1.7% otherwise); the FN rate climbs with rank (1.2% at rank 1, 5.5%
at ranks 4–5, 14–17% at ranks 6–20); pairs with a **legal-form conflict** have a 15% FN
rate (0.4% of true pairs conflict, but the model treats a conflict as a near veto).
False positives are rare and concentrated on rank ≥ 6 and empty-address candidates.

Conclusions for US/India: the next gains are (1) blocking recall, especially India
(cascade fusion ceiling 0.977 vs 0.953 was measured but not exported), (2) name-only
handling for empty-address candidates, (3) softening the legal-form veto.

## 7. Test predictions per country (shipped model + rule)

| | US | India | France |
| --- | ---: | ---: | ---: |
| candidate pairs | 26.5M | 32.3M | 10.3M |
| entities whose best candidate has p < 0.1 | 4.2% | 7.7% | **13.7%** |
| predicted empty | 6.0% | 9.6% | **16.6%** |
| predicted matches per entity | 3.22 | 2.83 | 2.51 |
| sum of p per entity (expected matches if calibrated) | 3.34 | 2.92 | 2.60 |
| pairs with p > 0.5 | 8.2% | 7.3% | 6.5% |

Train truth: 5.6% singletons, 3.46 matches per entity. Alternative rules move France's
mean from 2.46 to 2.57 and its empty rate from 16.1% to 17.2%: the model's probabilities
are bimodal (92.5% of French pairs below 0.05, 5.9% above 0.95), so no rule can
recover French matches. The fix has to be upstream of the rule.

## 8. France

### 8.1 What French records look like

- Geography is tiny: **3 regions and about 18 cities** (Bordeaux, Nantes, Lille,
  Tourcoing, Dunkerque, Roubaix, Calais, Saint-Nazaire, Pessac, La Teste-de-Buch,
  Mérignac, Lège-Cap-Ferret, Pornic, La Baule-Escoublac, Saint-Herblain, …). City and
  region tokens therefore carry almost no IDF; the street name and house number are the
  only address evidence.
- Source 1 always writes three components, street / city / region, with the **region
  last in 87%** of records (components are reordered in the rest). Source 2/3 end with
  the region in 28–30%, the **department** (Nord, Pas-de-Calais, Gironde,
  Loire-Atlantique) in 27–28%, and the city in 33–34%; 3% are empty. The pipeline has no
  department→region mapping, so the last-token feature `ad_last_eq` is 0.16 in France
  against 0.63 for true US pairs.
- Street types are abbreviated in 23% of S2/S3 addresses ("R.", "R", "BD", "AV", "ALL",
  "CH", "IMP"); 5–6% carry "bis"/"ter"/"B"/"T" after the number; 14.5% carry a prefix
  marker ("N°", "Nº", "No.", "#", "(41)", "52 - "); 2.9% have leading zeros.
- No postcodes (0.4%).
- Names: 67% of Source-1 names end in a legal form (SARL, SAS, SASU, EURL, SA, SCI, SNC,
  EI); S2/S3 move it to the front in 4.8% and write it dotted ("S.A.S.", "E.U.R.L.") in
  5.5% or bracketed in 1.7%. 64% of names contain a generic word (association, amicale,
  club, comité, école, groupe, fils, …) so name vocabulary is low-entropy. Accents are
  present in 16% of S1 and 24% of S2/S3 names, often injected into the wrong letter
  ("Àmicale", "Thê", "Sâinte"); "(France)" appears inside 8% of names; "Ets"/"Cie"
  abbreviations in 6%.
- The generator otherwise behaves as for US/India: 26% of French candidate pairs have a
  legal-form conflict (US 18%), 25% of French candidates are non-ASCII (US 7%).

### 8.2 Why the model under-matches France (evidence)

1. **The competition features are shifted.** `cand_margin` (61% of LightGBM gain) and
   `cand_rel` (15%) compare blocking scores between rival Source-1 entities. French
   blocking scores are half the US level (mean 8.7 vs 12.9, entity best 14 vs 26) because
   the vocabulary is low-IDF, and **16% of French pairs sit at a margin of exactly zero**
   (ties) against 3% in the US and 5% in India. 33% of French pairs have
   `score_rel ≥ 0.9` versus 11% in the US: candidate lists are flat, so the model sees
   "ambiguous" everywhere.
2. **Same similarity, lower probability.** At name+address token-set similarity
   0.8–0.9 the mean probability is 0.08 in France vs 0.33 in the US; at 0.9–0.95 it is
   0.41 vs 0.54. Only above 0.95 are they equal (0.77 vs 0.75).
3. **Clear cases are fine.** For pairs with an identical normalised name, the same
   street tokens and the best blocking score (91k French pairs), 90.7% get p > 0.5,
   *higher* than the US (82%). The loss is in the middle: a generic token inserted into
   the name ("NU Theatre SARL" → "NU Thëatre Distribution SARL") combined with a changed
   house number ("14 Rue du Flocon" → "16 R. Du Flocon, Tourcoing, Nord") gets p = 0.000
   even when the candidate is the unique best (margin 1.0). More examples of the same
   shape: "Let & Frères SAS, 6 Avenue Louis Renault" vs "Let & Frères International SAS,
   17 AVENUE LOUIS RENAULT" (p 0.000); "Liste Primaire, 134 Rue Mondenard" vs "Liste
   Primaire France, 141 R. MONDENARD" (p 0.004); "Ubaye Primaire, 193 Rue de la
   Gilarderie" vs "Ubaye Primaire SAS, 198 RUE DE LA GILARDERIE" (p 0.43).
4. **The legal-form veto is stronger in France.** Pairs with `all_tset ≥ 0.9` and a
   legal conflict get mean p 0.006 (US 0.04). Most of those are genuinely different
   businesses on the same street, but French forms are many (SA/SAS/SASU/SARL/EURL/SCI/
   SNC/EI) and dotted or bracketed variants are not recognised, so any generator noise on
   the legal form is unrecoverable.
5. **Address similarity is inflated.** 1.79M French pairs have `ad_tset ≥ 0.9` with a
   name similarity below 0.5 (same street or just same city/region, different business).
   These are correctly rejected, but they dilute IDF and blocking margins for everyone.
6. Accents are *not* the problem: French candidates flagged non-ASCII get slightly
   higher probabilities (0.63 vs 0.59 at equal similarity).

Size of the problem: 13.7% of French entities have no candidate above p = 0.1 and 16.6%
are submitted empty. If France has the generator's 5.6% singleton rate, roughly 8–11% of
French entities (21–28k) are wrongly emptied. Each costs a full point, so France is
plausibly scoring about 0.80 against 0.94 elsewhere, which is 1.5–2 points of the overall
macro F0.5 (France is 15% of test entities).

### 8.3 Options, ranked by expected value per hour

1. **Measure France on the leaderboard (no labels exist anywhere else).** Submit files
   that differ only in the French rows: (a) current; (b) France emptied; (c) France with
   the rank-1 candidate accepted whenever `all_tset ≥ 0.8` or p ≥ 0.05; (d) US+India
   emptied with France as-is. With F_overall = 0.85·F_USIN + 0.15·F_FR and an empty
   country scoring its singleton rate, (a) minus (b) gives 0.15·(F_FR − s_FR), (d) gives
   F_FR up to the assumed 5.6% singleton rate, and (c) tells whether recall is the
   problem. Two to four submissions; also use one to test denser predictions for all
   countries (add rank-2 candidates with p ≥ 0.3) to settle the pool-density question.
2. **France-only normalisation fixes, no retraining** (recompute French test features
   ≈15 min, rescore ≈5 min): map departments to regions (Nord, Pas-de-Calais →
   Hauts-de-France; Gironde → Nouvelle-Aquitaine; Loire-Atlantique, Vendée → Pays de la
   Loire) or drop region/department tokens entirely for France; expand R./BD/AV/ALL/CH/
   IMP/PL/CRS; normalise bis/ter/B/T; strip N°/Nº/No./#/parentheses and leading zeros from
   house numbers; treat dotted and bracketed legal forms as legal tokens; add EI, Ets,
   Cie. These also apply to US/India (leading zeros, dotted forms) but changing train
   features means retraining (~50 min locally, ~2.2 h on Kaggle).
3. **Reduce dependence on the shifted competition features.** Train a second model
   without `cand_margin`/`cand_rel`/`score`/`g_max` (or with them rank-normalised per
   country) and choose by **cross-country transfer** (train US, validate India, and the
   reverse), which is the only measurable proxy for France. Use the transfer-best model
   for France only; keep the full model for US/India. ~1 h per model.
4. **Soften the legal veto for France**: cap the effect by zeroing `legal_conflict`
   when the conflict is between forms of the same family (SA/SAS/SASU; SARL/EURL), or
   drop the feature for French inference. Cheap; verify with option 1.
5. Self-training on high-confidence French pairs: low expected value, because it
   reinforces the current blind spots (changed house numbers, inserted tokens).

Sanity note on the blocking side: only 834 French entities have no candidates, and
French lists are flat rather than empty, so recall inside the top-20 is probably
comparable to US/India; nothing here suggests re-blocking France.

## 9. Prioritised list of gaps

| # | Gap | Evidence | Cost | Expected effect |
| --- | --- | --- | --- | --- |
| 1 | France under-matched (16.6% empty) | §7–8 | leaderboard probes + normalisation fixes | up to +1.5–2 points overall |
| 2 | Pool density shift: are test entities denser? | §4 | 1 leaderboard probe | tells whether to predict more matches everywhere |
| 3 | Blocking recall 89.4% (India ceiling 0.921) | §6 | cascade export ≈ 90 min per split | up to +2 points (ceiling 0.977 vs 0.953) |
| 4 | Empty-address candidates → 35% of FNs | §6 | name-only feature path / separate threshold | ≈ +0.3–0.5 |
| 5 | Leading zeros, digit squeeze, dotted legal forms, "(ID:)", "Formerly", "DBA" | §2 | normaliser edits (+ retrain for train-side consistency) | small but broad |
| 6 | Legal-form conflict treated as a veto | §6, §8.2 | feature tweak | small for US/India, larger for France |
| 7 | `postal_eq` dead (undefined on 94–100% of pairs) | §2 | drop or replace with generic numeric-code agreement | none, cleanliness |
| 8 | Exclusivity rule not applied | §3 | flag in rule.json | removes 1–4k conflicting pairs per country, neutral on validation |
