# Project handoff

Self-contained state of this project. Read this first in a new session; it
assumes no prior context.

Last updated: 2026-09-25. Branch: `sparse-dot-topn-blocking`.

---

## 1. What we are trying to achieve

**Amazon ML Challenge 2026 — Business Entity Resolution.**

Given business records from three independent, noisy sources, decide which
records refer to the same real-world business. Source 1 is the deduplicated
reference; for every Source-1 record we must find all matching Source-2 and
Source-3 records. A Source-1 entity may match zero, one, or many.

**Scoring — this shapes every decision:**

```
F_0.5 = (1.25 x P x R) / (0.25 x P + R)     computed PER Source-1 entity,
                                            then macro-averaged over ALL of them
```

- Precision is weighted **2x** over recall. False merges hurt more than misses.
- **Singletons count.** An entity with no true matches scores **1.0** if you
  correctly predict an empty list, and **0.0** if you predict anything. 5.58% of
  entities are singletons, so this is ~5.6 points of free score — or free loss.
- Because it is a *macro* average, an entity you completely miss costs as much
  as an entity with nine matches. Per-entity coverage matters more than total
  link coverage.

**Deliverables (both go in `output/` of the submission zip):**

| File | Purpose |
| --- | --- |
| `matching_results.tsv` | Final matches. **The only file scored on the leaderboard.** |
| `candidate_pairs.tsv` | The blocking output — the exact set fed to the model for inference. Not scored, but audited. |

Format: tab-separated, one row per Source-1 entity, `matched_entity_ids` is a
comma-separated list (empty for singletons). Every test Source-1 entity must
appear exactly once. Validate with `student_resource/utils/validate_submission.py`
before submitting.

**Hard constraints:**

- **No external data lookup.** No entity-resolution APIs, no geocoding, no
  business registries, no internet augmentation. Disqualification if found.
  Static library tables (e.g. `unidecode`'s character map) are fine — they are
  not a data lookup.
- Final model must be MIT/Apache-2.0 licensed and <= 8B parameters.

---

## 2. The data

Location: `C:\Users\manoj\Downloads\amazon ML resource\student_resource\dataset\`

| File | Rows | Countries |
| --- | ---: | --- |
| `train/train_source1.tsv` | 2,206,821 | US 1.32M / India 0.88M |
| `train/train_source2.tsv` | 5,034,616 | US 3.02M / India 2.02M |
| `train/train_source3.tsv` | 5,285,603 | US 3.17M / India 2.12M |
| `train/train_ground_truth.tsv` | 2,206,821 | — |
| `test/test_source1.tsv` | **1,732,544** | US 0.66M / India 0.81M / **France 0.26M** |
| `test/test_source2.tsv` | ~5M | |
| `test/test_source3.tsv` | ~5M | |

**Ground truth:** 7,638,365 links, mean 3.46 matches per entity, 5.58%
singletons, 99.9% of entities have <= 9 matches.

**Quirks that matter:**

- Matches **never cross country**. Partitioning by country is both correct and
  what keeps memory tractable (largest partition 3.17M instead of 10.3M).
- **Test contains France (259,452 entities) with zero French training data.**
- Source 1 is **100% ASCII**. Sources 2/3 are **11-15% non-Latin** — and not just
  Devanagari: Odia and Telugu appear too.
- **13.92% of all ground-truth links join an ASCII Source-1 name to a
  non-ASCII candidate name.**

---

## 3. Current state — READ THIS CAREFULLY

### What is DONE

Blocking / candidate generation is solved and measured. On full training data
(22,123-entity sample, searched against the complete 10.3M candidate pool):

| Strategy | F0.5 ceiling | Pair compl. | pairs/entity | no cands | Time | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cascade` k=12 min=1 | **0.9769** | 0.9407 | 24.0 | 0.00% | 88 min | 7.5 GB |
| `token_idf` k=20 | 0.9530 | 0.8925 | 40.0 | 0.01% | 5.2 min | 4.1 GB |
| `cascade` k=12 min=2 | 0.9495 | 0.8832 | 9.33 | 0.52% | 87 min | 7.3 GB |
| **`token_idf` k=10** | 0.9438 | 0.8742 | 20.0 | 0.01% | **5.6 min** | 4.5 GB |
| `token_idf` k=10 shared>=2 | 0.9426 | 0.8739 | 18.0 | 0.47% | 4.4 min | 4.4 GB |
| `tfidf_addr` k=12 | 0.9400 | 0.8581 | 20.4 | 0.02% | 51 min | 6.1 GB |
| `token_idf` k=5 | 0.9313 | 0.8460 | 10.0 | 0.01% | 5.7 min | 4.1 GB |
| `minhash_lsh` | 0.8132 | 0.7082 | 20.0 | 0.00% | 8 min | 4.3 GB |
| **shipped baseline** | 0.5331 | 0.3007 | 10.7 | **21.56%** | 1.1 min | 3.9 GB |
| `simhash_lsh` (untuned) | 0.1169 | 0.0374 | 7.1 | 59.02% | 15 min | 4.3 GB |

"F0.5 ceiling" = the macro F0.5 this candidate set would achieve **if the
matcher were perfect**. It is a hard upper bound on the leaderboard score for a
given block set. Raw JSON reports are in `runs/full/`.

**Recall went 30% -> 94%. Entities receiving zero candidates went 21.6% -> 0%.**

### What is NOT done

This is the important part. Do not assume more exists than this.

1. **No block set exists on disk.** Nothing. The harness computes candidate
   matrices, scores them, and discards them — `BlockingScorer` keeps only
   per-entity counters by design, so memory is constant regardless of block-set
   size. `runs/` contains **metric JSON only**, never pairs.
   (The `blocking_strategies/block_set/` directory is the OLD exact-match
   baseline from before this work — the 30%-recall one. Not useful.)
2. **No export path exists.** Nothing writes `candidate_pairs.tsv` or
   `matching_results.tsv`. This code must be written.
3. **Nothing has ever run on the test set.** The only reference to
   `test_source1.tsv` in the codebase is a row-count constant.
4. **There is no matcher.** Current leaderboard score would be 0.
5. All benchmarks used a **1% query sample**. Statistically that is ample for
   *choosing* a strategy (standard error ~0.002 against gaps of ~0.03), but the
   submission requires **all 1,732,544 test entities**.
6. `simhash_lsh` is untuned — 59% of entities get no candidates. Its 0.1169 is a
   broken configuration, not a verdict.

---

## 4. Key findings (the intuitions worth keeping)

**1. The bottleneck was normalization, not the algorithm.**
The baseline key is `casefold()` then delete everything outside `[a-z0-9]`. That
turns every Indic-script name into an **empty string**: 9.05% of Source-2 and
4.61% of Source-3 records had no name key at all and could never be retrieved by
name. Combined with the 13.92% cross-script links, this alone explains most of
the baseline's 21.6% zero-candidate rate.

The fix (`harness/textnorm.py`): transliterate to ASCII, then collapse repeated
characters, because transliteration lengthens vowels and doubles consonants.

```
राम मार्केटिंग प्राइवेट लिमिटेड
  -> unidecode        raam maarkettiNg praaivett limittedd
  -> collapse runs    ram marketing praivet limited
  -> strip legal      ram marketing
```

`praivet` vs `private` is now within character-n-gram reach.

**2. The address field carries the recall, not the name.** Measured ablation:

| Fields | cross-script recall | same-script recall |
| --- | ---: | ---: |
| name + address | 0.8494 | 0.9449 |
| name only | 0.3764 | 0.6697 |

Addresses carry house numbers, plot numbers and PIN codes — high entropy.
Business names are 2-5 generic words. Crucially, records with Indic-script
*names* usually still have **ASCII addresses**, so the address is the bridge
across the script gap. Any feature set that underweights the address is wrong.

**3. Word-level rare tokens beat character n-grams — a vocabulary effect.**
Word vocabulary is in the millions with a Zipfian tail, so filtering to rare
tokens leaves very short posting lists. The char 2-3-gram vocabulary is bounded
at ~50,000, so with 3.17M documents *every* posting list averages ~4,900 entries
and there is no sparse tail to select. Small vocabulary = uniformly dense
postings. This is a property of the representation, not the engine.

**4. Requiring agreement beats lowering K.** `cascade min=2` (two independent
strategies must propose the pair) reaches 0.9495 at 9.33 candidates/entity,
strictly better than `token_idf k=5` at 0.9313/10.0 on both axes.

**5. Data structures mattered as much as algorithms.** The first `token_idf`
index used `dict[str, list[int]]` over 3M records: 7.7 GB, swapped, heading for
hours per run. Flat NumPy arrays in CSR layout: 4.5 GB, 2x faster, identical
output.

### Two conclusions that were WRONG and got corrected

- **"LSH is not competitive."** Drawn from dev-fixture numbers. MinHash goes
  0.4123 (fixture) -> **0.8132** (full data), because LSH bucket occupancy
  depends on pool density and a 1/25 sample starves the buckets. It beats the
  shipped baseline by 28 points. It still loses to `token_idf`, but the fixture
  badly misrepresented it.
- **"`sparse_dot_topn` will win on speed."** It lost 9x. `tfidf_addr` reaches
  0.9400 in 51 min; `token_idf` reaches a higher 0.9438 in 5.6 min. The matmul
  is fast; TF-IDF vectorization over 3.17M documents per partition dominates,
  and see finding 3 for why the representation is weaker.

**Therefore: the dev fixture is unreliable for ranked/top-K methods in BOTH
directions.** It transfers almost exactly for exact-key methods (baseline 0.2987
fixture vs 0.3007 full). Use it for iteration, never for a verdict.

---

## 5. Code map

Canonical location: GitHub, branch `sparse-dot-topn-blocking`.
`https://github.com/manoj2005mj/Amazon_ML_Challenge`

> The working clone used during development lives under a **temp scratchpad
> directory and is ephemeral**. Clone fresh from GitHub.

```
blocking_strategies/
  harness/
    textnorm.py     normalization variants (transliterate, squeeze, legal-strip)
    dataio.py       RecordSet, Arrow-backed columns, Parquet normalization cache
    metrics.py      BlockingScorer, the F0.5-ceiling metric
    runner.py       Strategy protocol + single-strategy CLI + REGISTRY
    make_fixture.py builds the 1/25 dev fixture
  strategies/
    exact_baseline.py        the shipped baseline, re-implemented to the contract
    token_idf.py             rare-token inverted index      <- BEST PRACTICAL
    tfidf_topn.py            char n-gram + sparse_dot_topn (name/addr/fused)
    sorted_neighbourhood.py  multi-pass sorted neighbourhood
    minhash_lsh.py           vectorized banded MinHash
    simhash_lsh.py           random-hyperplane LSH (UNTUNED)
    cascade.py               RRF fusion of several strategies  <- BEST CEILING
  benchmark.py      sequential sweep driver (subprocess per run)
docs/
  BLOCKING_EVALUATION.md    full methodology + results + corrections
runs/full/          9 full-data metric reports (JSON)
runs/fixture/       14 fixture metric reports (JSON)
```

**Strategy contract** — to add one, implement:

```python
class MyStrategy:
    columns = ("name_key", "addr_key")   # which cached columns to load
    name: str
    params: dict
    def block(self, s1, cand, country) -> scipy.sparse.csr_matrix:
        """len(s1) x len(cand). Entry (i,j) means cand[j] is a candidate
        for s1[i]. Value = your score. Top-K/threshold is YOUR job."""
```

then add it to `REGISTRY` in `runner.py`. Registry keys currently available:
`exact_baseline`, `token_idf`, `tfidf_name`, `tfidf_addr`, `tfidf_fused`,
`sorted_neighbourhood`, `minhash_lsh`, `simhash_lsh`, `cascade`.

---

## 6. Environment and gotchas

- **Windows 11, 16 GB RAM, 16 logical cores, Python 3.12.3.** RAM is the binding
  constraint; typically only 3-6 GB free.
- **Always set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` before running.** The
  console is cp1252 and any strategy printing an Indic-script key crashes the
  process on *encode*, which looks like a logic bug and is not.
- **Run benchmarks one at a time.** `benchmark.py` uses a subprocess per run
  deliberately: an in-process loop never returns the previous run's matrices to
  the OS, and two concurrent full-scale runs do not fit in 16 GB.
- **Normalization cache persists and is worth protecting**: 824 MB at
  `dataset/train/_norm_cache/*.parquet`. Building it takes ~4 min per source
  file. It is keyed on a hash of `textnorm.py`, so **editing any normalizer
  silently invalidates it and forces a 12-minute rebuild**. Intentional, but
  budget for it.
- The dev fixture (`data/dev_fixture/`, 82 MB) is gitignored and rebuildable.
- Installed beyond the usual: `sparse_dot_topn` 1.2.0, `Unidecode`, `datasketch`,
  `rapidfuzz`, `psutil`, `pyarrow`.

### Commands

```bash
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1     # always

# score one strategy on full training data (1% query sample)
python -m blocking_strategies.harness.runner token_idf \
    --data-dir "<dataset>/train" --sample-fraction 0.01 \
    --param k=20 --param max_df=0.005

# build the fast dev fixture (seconds per run instead of minutes)
python -m blocking_strategies.harness.make_fixture \
    --train-dir "<dataset>/train" --out-dir data/dev_fixture

# run a sweep sequentially; resumable (completed runs are skipped)
python -m blocking_strategies.benchmark \
    --sample-fraction 0.01 --runs-dir runs/full --sweep-file runs/sweep_full.json
```

---

## 7. What to do next, in priority order

**1. Write the export path and produce a test-set candidate file. START HERE.**
This is the only genuinely non-optional artifact and it does not exist. It needs
a mode that streams candidate pairs to disk instead of scoring them in memory,
run over `test/` for all 1,732,544 entities. Estimated 5-8 hours with
`token_idf k=20` as-is. **Run it overnight** — it is the one thing a deadline
cannot absorb.

Generate at a **generous k (20), not a tight one.** You can always prune a large
candidate set down later (by score or top-N); you cannot expand a small one
without paying the whole run again. The optimal k depends on how precise the
matcher turns out to be, which is not yet known.

**2. Build the matcher.** The ceiling is 0.977 and the actual score is 0. Every
point here is worth far more than further blocking work. Suggested features: the
blocking scores themselves (token-IDF overlap, name cosine, address cosine),
`rapidfuzz` ratios, and **numeric-token agreement** (house numbers, PIN codes) —
finding 2 predicts that last one will be strong. Gradient boosting on pair
features is the obvious first model. Training on blocking's own output gives
hard negatives for free and matches the inference distribution.

**3. Tune the decision rule for F0.5, not accuracy.** A single global threshold
is the wrong shape. Exploit: singletons are 5.58% of entities and a correct
empty prediction is worth a full 1.0, so an explicit no-match decision is needed;
and mean matches per entity is 3.46, so predicting sets of roughly the right
*size* matters more than calibrating individual pair scores.

**4. France.** 259,452 mandatory test rows, zero French training data, nothing
validated on them. Unhedged risk.

**5. Speed, if needed.** Three levers, cheapest first:
   - `max_df` is the dominant cost term (retrieval is linear in it). Currently
     `0.005` = 15,850 postings/token against 3.17M — very loose. Finding 3 says
     the tokens you would drop first are the least discriminative.
   - `np.add.at` in `token_idf` retrieval is a known NumPy slow path;
     `np.bincount(inverse, weights=...)` computes the same thing in C.
   - **Reframe rare-token scoring as a sparse matmul.** `score = Q @ P` where Q
     is queries x tokens (IDF-weighted) and P is tokens x candidates (binary) —
     which means `sparse_dot_topn` can run it with an OpenMP C++ inner loop and
     built-in top-K, replacing the per-row Python loop. This does not contradict
     the finding above: that was about the *representation* (char n-grams),
     this reuses the *engine* with the better representation. Plausibly 5-20x.
   - Note on multiprocessing: awkward here. Windows has no `fork`, so each
     worker copies the ~4 GB index. Threads are better (NumPy releases the GIL
     on sort/bincount/argpartition and they share the index for free), but the
     matmul reframing subsumes this since `sparse_dot_topn` is already threaded.

**6. Optional.** Tune `simhash_lsh`; re-run the top strategies at 5-10% to firm
up the tails (not needed for the headline ranking); analyse the ~6% of links
still missed to decide what blocking pass would catch them.
