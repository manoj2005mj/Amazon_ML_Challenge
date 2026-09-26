> Repository root corresponds to `code/business_entity_resolution/src/` of the submission layout described below, so the `src/` prefix in the paths is dropped here. Earlier dataset-workspace notes: [docs/DATASET_WORKSPACE.md](docs/DATASET_WORKSPACE.md). Project state: [HANDOFF.md](HANDOFF.md). Full session record: [docs/SESSION_REPORT.md](docs/SESSION_REPORT.md).

# Business Entity Resolution — reproduction guide

End-to-end pipeline: **raw TSVs → normalization → blocking (candidates) →
pair features → LightGBM matcher → F0.5 decision rule → submission files.**

Everything runs on a 16 GB Windows laptop, CPU only. No external data, APIs or
geocoding are used. Transliteration uses `Unidecode`'s built-in character table.

## Layout

```
src/
  blocking_strategies/   candidate generation (token-IDF blocker), normalization
    harness/textnorm.py  transliteration + squeeze + legal/abbreviation handling
    export_candidates.py writes candidates/<split>/parts/<country>_<source>.parquet
  matching/              the matcher
    common.py            paths, id codec, address canonicalization, metric
    prep_entities.py     per-record attributes (once per split)
    cand_stats.py        candidate-side competition stats over the full candidate file
    sample.py            10% Source-1 sample for experiments
    features.py          74 pair features (sample or full split)
    train_eval.py        model comparison on the sample (LogReg vs LightGBM)
    train_full.py        final model: full train split, easy-negative downsampling
    predict.py           score test pairs, apply the rule, write output/*.tsv
  cleaning/              data cleaning stage (raw TSVs -> cleaned TSVs + sidecars)
    rules.py             static tables: legal forms, honorifics, street/state/region maps
    clean.py             the cleaner + CLI (parallel, streaming)
    learn_dictionary.py  Indic-transliteration / abbreviation dictionaries learned from
                         training ground truth only; writes cleaning/dictionary.json
    segment.py           website-name segmentation ("tristateguild.com" -> "tri state guild")
  docs/EDA_REPORT.md     the EDA that motivated every cleaning rule (charts in docs/charts/)
```

## Setup

```bash
pip install -r requirements.txt
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1       # Windows console is cp1252
export ER_ROOT="<folder containing student_resource/>"   # default: the author's path
cd src
```

`ER_ROOT` must contain `student_resource/dataset/{train,test}/`. Intermediate
files go to `$ER_ROOT/work/match/` (override with `ER_WORK`); candidates to
`$ER_ROOT/candidates/`.

## Run

```bash
# 0. Cleaning: cleaned copies of the six source files (same schema, same ids) plus a
#    sidecar parquet per file with what was stripped (legal form, honorific, alias,
#    "(ID: n)" tag, unit, state, French region). Ground truth is copied unchanged.
#    The dictionary is learned from TRAIN ground truth only and applied to both splits.
python -m cleaning.learn_dictionary --data-dir "$ER_ROOT/student_resource/dataset" --workers 8
python -m cleaning.clean --split train --in-dir "$ER_ROOT/student_resource/dataset/train" --out-dir "$ER_ROOT/student_resource/dataset_clean/train" --workers 10
python -m cleaning.clean --split test  --in-dir "$ER_ROOT/student_resource/dataset/test"  --out-dir "$ER_ROOT/student_resource/dataset_clean/test"  --workers 10
#    Every later step reads the cleaned folder: pass it as --data-dir below and set
#    ER_DATASET="$ER_ROOT/student_resource/dataset_clean" for the matching steps.

# 1. Blocking: top-20 candidates per Source-1 record from each of S2 and S3
python -m blocking_strategies.export_candidates --split train --data-dir "$ER_ROOT/student_resource/dataset/train" --out-dir "$ER_ROOT/candidates/train"
python -m blocking_strategies.export_candidates --split test  --data-dir "$ER_ROOT/student_resource/dataset/test"  --out-dir "$ER_ROOT/candidates/test"

# 2. Per-record attributes and candidate competition stats
python -m matching.prep_entities --split train
python -m matching.prep_entities --split test
python -m matching.cand_stats --split train
python -m matching.cand_stats --split test

# 3. Validation sample (10% of Source-1 entities; 20% of those held out)
python -m matching.sample --fraction 0.10 --name sample10

# 4. Pair features for every train and test pair
python -m matching.features --split train --full --shards 4 --workers 8
python -m matching.features --split test  --full --shards 2 --workers 8

# 5. Final model (validation entities are never trained on)
python -m matching.train_full --features "full_train/features_*.parquet" --tag full

# 6. Score test and write output/matching_results.tsv + output/candidate_pairs.tsv
python -m matching.predict --model "$ER_ROOT/work/match/full/model.txt" --rule "$(cat $ER_ROOT/work/match/full/rule.json)" --tag full

# 7. Validate the submission format
cd "$ER_ROOT/student_resource"
python utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

Optional experiments on the sample: `python -m matching.features --sample sample10 --workers 6`
then `python -m matching.train_eval --sample sample10`.

Approximate run times on the reference laptop: blocking ~55 min, prep ~15 min,
full features ~80 min, training ~40 min, prediction ~10 min.
