# Blocking strategies

This folder contains the first candidate-generation baseline for the Amazon ML
Challenge entity-resolution data. It creates a block set; it does not decide
whether a pair is a true match.

## Baseline idea

For every Source-1 record, build two inverted indexes:

1. `(country, normalized business name)`
2. `(country, normalized business address)`

For every Source-2 and Source-3 record, retrieve records from both indexes,
union the results, and emit each candidate pair once. The `blocking_rules`
column records whether the pair came from `name`, `address`, or both.

Normalization uses Unicode case folding and removes every character except
ASCII letters and digits. Empty keys never generate candidates.

## Generate the block set

```powershell
python blocking_strategies/build_block_set.py `
  --reference "C:\path\to\train_source1.tsv" `
  --source2 "C:\path\to\train_source2.tsv" `
  --source3 "C:\path\to\train_source3.tsv" `
  --output-dir blocking_strategies/block_set
```

The output is partitioned by candidate source and country and rotated every
five million rows. Each gzip file is an ordinary tab-separated table with:

```text
source1_entity_id  candidate_entity_id  candidate_source  country  blocking_rules
```

`metadata.json` contains row counts, timings, file sizes, and SHA-256 hashes.

Verify a generated copy with:

```powershell
python blocking_strategies/verify_block_set.py blocking_strategies/block_set
```

## Full training result

The complete training run produces 23,413,014 unique candidate pairs from
10,320,219 Source-2/3 records and 2,206,821 Source-1 records.

| Rule contribution | Candidate pairs |
| --- | ---: |
| Exact normalized name | 22,742,308 |
| Exact normalized address | 777,333 |
| Found by both | 106,627 |
| Union after deduplication | 23,413,014 |

The three rule rows overlap: the union is `name + address - both`.
In the emitted `blocking_rules` column, the mutually exclusive counts are
22,635,681 `name`, 670,706 `address`, and 106,627 `name+address` rows.

## Limitation

This is a fast baseline, not the final blocking design. On the prepared 1%
sample, exact name-or-address equality recovered 22,826 of 76,430 labeled
positive pairs, or 29.87% recall. Later blocking passes should add rare tokens,
postal/PIN keys, phonetic keys, and character n-gram LSH, then union all passes
and measure candidate recall before model training.
