# Amazon ML Challenge 2026 dataset workspace

This project turns the official multi-source entity-resolution files into ordinary,
model-ready binary pair datasets. The raw download stays unchanged in Downloads; the
scripts create reproducible samples inside this workspace.

## What is created

- `train_pairs.tsv`: labeled pairs for fitting a model.
- `validation_pairs.tsv`: a labeled 10% Source-1 holdout for local accuracy and F0.5 testing.
- `metadata.json`: the exact configuration, row counts, label meaning, and split rules.

Each pair file is tab-separated and contains raw names and addresses for both sides,
country, candidate source, and a binary `label`. It can be loaded by pandas, Polars,
Spark, scikit-learn, XGBoost, PyTorch, TensorFlow, or a custom pipeline.

## Build a quick sample

```powershell
python src/prepare_dataset.py --config config/quickstart.json
```

The quick-start config selects approximately 1% of Source-1 entities and reserves exactly 10% of those
entities for validation. It still scans the raw Source-2/3 files once so that every
positive record can be found.

## Build the default 10% sample

```powershell
python src/prepare_dataset.py --config config/dataset.json
```

Change `data_fraction` in `config/dataset.json` to any value above 0 and at most 1.
For example, use `0.02` for 2%, `0.25` for 25%, or `1.0` for the full training corpus.
Keep `validation_fraction` at `0.10` to preserve the requested 10% labeled holdout.
Hash sampling makes `data_fraction` reproducible and statistically close to the requested
percentage; the validation split is assigned as an exact fraction of the selected entities.

You can also override values without editing the file:

```powershell
python src/prepare_dataset.py --config config/dataset.json --data-fraction 0.05 --output-dir data/prepared/five-percent
```

## Split design

The split is made by `source1_entity_id`, not by individual pair. Consequently, a
business and all its matches/non-matches stay entirely in train or validation. The
official unlabeled competition test set is never used for local validation.

## Evaluate model scores

Write a tab-separated prediction file with these columns:

```text
source1_entity_id    candidate_entity_id    score
```

Then run:

```powershell
python src/evaluate_predictions.py --validation data/prepared/quickstart/validation_pairs.tsv --predictions predictions.tsv --threshold 0.5
```

The evaluator reports ordinary pair accuracy and the challenge's entity-level macro
F0.5. Prefer macro F0.5 when selecting a final threshold.

## Important modeling note

The generated random negatives make hypothesis tests fast, but they are not a final
blocking strategy. After a baseline works, add hard negatives with similar names or
addresses and measure candidate recall before training the competition model.

See `docs/DATASET_OVERVIEW.md` for the source profile and interpretation.

## Build the first candidate block set

The deterministic country/name/address baseline and its generated compressed
candidate shards live in `blocking_strategies/`. It uses exact normalized-name
and normalized-address indexes within country, unions both candidate streams,
and records the rule that produced each pair.

See `blocking_strategies/README.md` for the algorithm, reproduction command,
full-data counts, verification command, and recall limitation.
