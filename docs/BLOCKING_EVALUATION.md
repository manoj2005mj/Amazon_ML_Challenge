# Candidate generation: strategies and measured results

The shipped baseline (`blocking_strategies/build_block_set.py`) recovers about
**30%** of ground-truth links. Because the leaderboard metric is computed per
Source-1 entity and then averaged, that number is a hard ceiling on the final
score no matter how good the downstream matcher is. This document records the
strategies tried to lift that ceiling, how they were measured, and what the
measurements actually say.

## 1. The metric that matters

Blocking is conventionally reported with *pair completeness* (what fraction of
true links survived) and *reduction ratio* (how much of the cartesian product was
discarded). Both are necessary and neither is sufficient here, because the
challenge scores a **macro** average over Source-1 entities:

```
F_0.5 = (1.25 x P x R) / (0.25 x P + R)      per entity, then averaged
```

A block set that recovers 95% of all links while completely missing 10% of
entities scores worse than its pair completeness suggests. So the headline metric
used throughout is the **F0.5 ceiling**: the macro F0.5 this candidate set would
achieve *if the downstream matcher were perfect*. A perfect matcher keeps every
true link present in the candidates and rejects every false one, so per-entity
precision is 1.0 and recall is the fraction of that entity's links that survived
blocking:

```
f05_ceiling = mean over entities of  1.25 r / (0.25 + r)      (r = per-entity recall)
```

Singletons contribute 1.0 — a perfect matcher predicts the empty set for them
regardless of how many candidates blocking offered. This makes the number a
genuine upper bound on the leaderboard score for a given block set, and directly
comparable across strategies.

It is computed per entity and then averaged, never from the mean recall. The
function is concave, so `mean(f(r))` is strictly below `f(mean(r))` — using the
latter would flatter every strategy by roughly one point.

## 2. Evaluation design

**The query side is sampled; the index side is complete.** A strategy is asked to
find candidates for a deterministic hash-sample of Source-1 entities, but it
searches the *full* Source-2/Source-3 pool for that country. Sampling both sides
would inflate recall by removing distractors and would make the reduction ratio
meaningless.

Sampling uses blake2b of the entity id, not Python's `hash()`, whose string seed
is randomized per process. A benchmark whose sample silently changed between runs
would make every comparison worthless.

Each strategy runs in its **own subprocess** (`blocking_strategies/benchmark.py`).
This is deliberate: a single process looping over strategies never returns run
*n*'s TF-IDF matrices and index arrays to the OS before run *n+1* allocates, and
on a 16 GB machine that is the difference between finishing and swapping. Runs
are therefore strictly sequential — the full candidate pool for one country
partition is up to 3.17M records and two concurrent runs do not fit.

### The dev fixture, and how far to trust it

`blocking_strategies/harness/make_fixture.py` builds a 1/25-scale dataset that
preserves ground-truth linkage: it samples Source-1 entities, keeps *every*
Source-2/3 record they link to, then adds an independent random sample of
unrelated records as distractors. Sampling the three sources independently would
break the linkage and make measured recall pure noise.

The fixture runs in seconds instead of minutes, which is what makes iteration
possible. **But its numbers are optimistic for any top-K method**, and the gap is
not small:

| `token_idf` k=10, df<=0.005 | F0.5 ceiling | Pair completeness |
| --- | ---: | ---: |
| Fixture (141k-record pool) | 0.9721 | 0.9315 |
| Full data (3.17M-record pool) | 0.9438 | 0.8742 |

Selecting the top 10 of 3.17M candidates is a harder problem than the top 10 of
141k. The exact-match baseline, which does no top-K at all, transfers almost
exactly (0.2987 fixture vs 0.3007 full) — so the fixture is trustworthy for
*exact-key* methods and merely indicative for *ranked* ones. Every headline number
in this document is from the full dataset.

`reduction_ratio` and `pairs_per_entity` on the fixture are not comparable to full
scale at all, since the candidate pool is 25x smaller.

## 3. What the data actually required

Two findings drove the design more than any choice of retrieval algorithm.

### 3.1 The baseline normalization erases non-Latin scripts

The baseline key is `casefold()` then delete everything outside `[a-z0-9]`. Source
1 is 100% ASCII, but Source 2 and Source 3 are 11-15% non-Latin — and not only
Devanagari. Real examples from the training data:

| Script | Source-2 name |
| --- | --- |
| Odia | `ଶ୍ୟାମ ଅଲ୍ ଫାଇନାନ୍ସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍` |
| Devanagari | `टेक कंसल्टिंग प्राइवेट लिमिटेड` |
| Telugu | `Tirupati టెక్నాలజీ LLP` |

Under the baseline normalization every one of those names becomes the empty
string: **9.05% of Source-2 and 4.61% of Source-3 records have no name key at
all**, so they can never be retrieved by name. Measured consequence: 21.6% of
Source-1 entities receive *zero* candidates from the shipped baseline, and India
scores far below the US (F0.5 ceiling 0.448 vs 0.589).

**13.92% of all ground-truth links join an ASCII Source-1 name to a non-ASCII
candidate name.** No character n-gram or shingle method can match across scripts,
because the two sides share no characters. `harness/textnorm.py` therefore
transliterates to ASCII (`unidecode`, a static table — not an external data
lookup) and then collapses repeated-character runs, because transliteration
lengthens vowels and doubles consonants:

```
राम मार्केटिंग प्राइवेट लिमिटेड
  -> unidecode        raam maarkettiNg praaivett limittedd
  -> collapse runs    ram marketing praivet limited
  -> strip legal      ram marketing
```

which is now within character-n-gram reach of the Source-1 spelling
`Ram Marketing Private Limited`.

### 3.2 The address field, not the name, carries the recall

Ablating the address field out of rare-token blocking (fixture, k=10):

| Fields used | Cross-script link recall | Same-script link recall |
| --- | ---: | ---: |
| name + address | **0.8494** | **0.9449** |
| name only | 0.3764 | 0.6697 |

Addresses contain house numbers, plot numbers and PIN codes, which are far more
discriminative than 2-3 generic name words — and crucially, records with
Indic-script *names* generally still carry **ASCII addresses**, so the address is
the bridge that makes cross-script pairs findable at all.

Transliteration still earns its place: name-only cross-script recall is 0.3764
rather than the ~0 it would be without it. But the address is the larger effect,
and any strategy that ignores it is leaving most of the recall on the table.

## 4. Strategies implemented

All are O(n) or O(n log n) in record count — no all-pairs comparison — and all are
partitioned by `country` before candidate generation, which is both semantically
correct (ground truth never crosses countries) and what caps the working set at
3.17M rows instead of 10.3M.

| Module | Idea |
| --- | --- |
| `exact_baseline.py` | The shipped baseline, re-implemented against the harness so comparisons share sampling and scoring. |
| `token_idf.py` | Inverted index over tokens rare enough to be evidence (`df <= max_df`); score = summed IDF of shared rare tokens; top-K per entity. |
| `tfidf_topn.py` | TF-IDF character n-grams + `sparse_dot_topn` top-K cosine, over name, address, or a weighted fusion of both. |
| `sorted_neighbourhood.py` | Multi-pass sorted neighbourhood (forward name, reversed name, sorted tokens, address) with a sliding window. |
| `minhash_lsh.py` | Vectorized banded MinHash LSH over character shingles. |
| `simhash_lsh.py` | Random-hyperplane (SimHash) LSH with banded bit-substring buckets. |
| `cascade.py` | Union of several of the above, re-ranked by Reciprocal Rank Fusion and capped per entity. |

### Why RRF for the cascade

The components emit incomparable quantities — summed IDF (unbounded, scales with
corpus size), cosine similarity (0..1), and a pass-count integer. Normalizing
them against one another needs arbitrary calibration that would have to be
re-tuned whenever a component changes. Reciprocal Rank Fusion discards magnitudes
and keeps only each strategy's *ordering*, which is the comparable part:

```
fused(i, j) = sum over strategies  weight_s / (rrf_k + rank_s(i, j))
```

A candidate ranked highly by two independent strategies outranks one ranked
highly by a single strategy — exactly the agreement signal worth keeping under a
tight per-entity cap. `min_strategies` makes that agreement a hard requirement,
which is the main precision lever.

## 5. An engineering note that mattered as much as the algorithms

The first `token_idf` implementation stored the index as `dict[str, list[int]]`
built from a per-record `list[list[tuple]]`. At full scale that is three million
Python lists of tuples of interned strings: it reached 7.7 GB private bytes,
pushed the machine into swap, and was heading for hours per run.

Rewriting the index as flat NumPy arrays in CSR layout
(`postings_indptr` / `postings_record`), built in two streaming passes with no
per-record intermediates, and aggregating each query row with
`np.unique` + `np.add.at` instead of a Python dict, made it **2x faster** and
brought peak RSS to 4.5 GB. Same output, same recall.

One subtlety found by regression-testing the rewrite: applying the field weight
on *both* the query and candidate side squares the discount on an address-only
match (0.6 x 0.6) and costs about 3 points of pair completeness. Given finding
3.2, penalizing address evidence twice is precisely backwards. Scoring uses the
query-side weight only.

## 6. Reproducing

```bash
# one-time: build the normalized column cache (~4 min per source file)
python -c "from pathlib import Path; from blocking_strategies.harness.dataio import load_source; \
  load_source(Path('<train_dir>/train_source1.tsv'))"

# build the dev fixture for fast iteration
python -m blocking_strategies.harness.make_fixture \
    --train-dir "<train_dir>" --out-dir data/dev_fixture

# score one strategy
python -m blocking_strategies.harness.runner token_idf \
    --data-dir "<train_dir>" --sample-fraction 0.01 --param k=10 --param max_df=0.005

# run a whole sweep, sequentially, resumable
python -m blocking_strategies.benchmark \
    --sample-fraction 0.01 --runs-dir runs/full --sweep-file runs/sweep_full.json
```

Set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` first: the Windows console is
cp1252 and any strategy that prints an Indic-script key will crash the process on
encode rather than on logic.
