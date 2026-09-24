import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from blocking_strategies.build_block_set import build_block_set, normalize_text
from blocking_strategies.verify_block_set import verify


HEADER = ["entity_id", "business_name", "business_address", "country"]


def write_tsv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)


class BlockingStrategyTests(unittest.TestCase):
    def test_normalize_text(self) -> None:
        self.assertEqual(normalize_text(" ACME, Inc. "), "acmeinc")
        self.assertEqual(normalize_text(""), "")

    def test_build_and_verify_small_block_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "source1.tsv"
            source2 = root / "source2.tsv"
            source3 = root / "source3.tsv"
            output = root / "block_set"
            write_tsv(
                reference,
                [
                    ["A1", "Acme", "1 Main", "US"],
                    ["A2", "Acme", "2 Main", "US"],
                    ["A3", "Beta", "1 Main", "US"],
                    ["A4", "Acme", "1 Main", "India"],
                ],
            )
            write_tsv(
                source2,
                [
                    ["B1", "Acme", "9 Side", "US"],
                    ["B2", "Gamma", "1 Main", "US"],
                    ["B3", "Acme", "1 Main", "US"],
                ],
            )
            write_tsv(source3, [["C1", "Acme", "1 Main", "India"]])

            manifest = build_block_set(
                reference,
                [("S2", source2), ("S3", source3)],
                output,
                max_rows_per_shard=3,
                compression_level=1,
                progress_every=0,
            )

            self.assertEqual(manifest["total_pairs"], 8)
            self.assertEqual(
                manifest["rule_counts"],
                {"name": 3, "address": 3, "name+address": 2},
            )
            self.assertEqual(verify(output), {"shards": 4, "rows": 8})

            rules = []
            for shard in manifest["shards"]:
                with gzip.open(
                    output / shard["file"], "rt", encoding="utf-8", newline=""
                ) as handle:
                    rules.extend(row["blocking_rules"] for row in csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(rules.count("name+address"), 2)


if __name__ == "__main__":
    unittest.main()

