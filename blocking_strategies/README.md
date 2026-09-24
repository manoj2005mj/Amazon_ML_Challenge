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

## Exact intersection mode

To require the same Source-1/candidate pair to have both an exact normalized
name and an exact normalized address in the same country, use:

```powershell
python blocking_strategies/build_block_set.py `
  --reference "C:\path\to\train_source1.tsv" `
  --source2 "C:\path\to\train_source2.tsv" `
  --source3 "C:\path\to\train_source3.tsv" `
  --match-mode intersection `
  --output-dir blocking_strategies/intersection_block_set
```

This is the set intersection of the exact-name pair set and exact-address pair
set. It is intentionally strict: a company with an address variation will not
be emitted even if its normalized name is identical.

On the complete training data, intersection mode produces 106,627 pairs in
298.89 seconds:

| Candidate partition | Pairs |
| --- | ---: |
| Source 2, India | 25,249 |
| Source 2, US | 81,053 |
| Source 3, India | 325 |
| Source 3, US | 0 |
| **Total** | **106,627** |

Compared with all 7,638,365 labeled positive links, 106,626 intersection pairs
are positive and one is negative. That is 99.9991% precision but only 1.3959%
recall. Intersection is therefore useful as a very high-confidence exact-match
subset, but it is too restrictive to be the only candidate-generation rule.

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
