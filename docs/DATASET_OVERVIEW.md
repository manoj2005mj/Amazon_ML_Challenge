# Amazon ML Challenge 2026 dataset overview

## Task

The challenge is business entity resolution. Source 1 is the deduplicated reference.
For every Source-1 business, the solution must identify zero or more matching records
from Source 2 and Source 3. Names and addresses contain abbreviations, spelling noise,
formatting differences, transliterations, reordered components, and missing address
values.

The competition metric is macro F0.5. It weights precision more heavily than recall,
and it gives full credit for correctly predicting no matches for a singleton entity.

## Raw training data profile

| File | Rows | Notes |
| --- | ---: | --- |
| train_source1.tsv | 2,206,821 | 1,323,633 US; 883,188 India; no missing names or addresses |
| train_source2.tsv | 5,034,616 | 3,016,817 US; 2,017,799 India; 168,967 missing addresses |
| train_source3.tsv | 5,285,603 | 3,170,056 US; 2,115,547 India; 175,916 missing addresses |
| train_ground_truth.tsv | 2,206,821 | 7,638,365 positive links; average 3.4613 links per Source-1 entity |

There are 123,247 labeled singletons with zero matches. The positive links contain
3,693,619 Source-2 records and 3,944,746 Source-3 records.

The official test set is unlabeled and also includes France. Country must therefore
remain an open string feature rather than a hard-coded US/India category.

## Prepared pair format

The preparation script creates `train_pairs.tsv` and `validation_pairs.tsv`. Each row
compares one Source-1 entity with one Source-2 or Source-3 candidate. `label=1` means
the records describe the same business; `label=0` means they are a sampled non-match.

The sample and split are deterministic. Sampling happens at the Source-1 entity level,
then exactly 10% of the selected Source-1 entities are held out by default. This keeps every
pair for a business in only one split and prevents leakage.

Random negatives come from the same country as the Source-1 entity. They are useful
for quick model experiments but are easier than near-duplicate hard negatives. A
production model should later add blocking-based hard negatives.
