import unittest

from src.prepare_dataset import (
    Entity,
    assign_splits,
    choose_negatives,
    unit_hash,
    validate_config,
)


class PrepareDatasetTests(unittest.TestCase):
    def test_hash_is_deterministic_and_bounded(self):
        first = unit_hash("S1-1", 2026, "sample")
        second = unit_hash("S1-1", 2026, "sample")
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)
        self.assertLess(first, 1.0)

    def test_validation_fraction_rejects_zero(self):
        config = {
            "data_fraction": 0.1,
            "validation_fraction": 0,
            "negative_to_positive_ratio": 1,
            "minimum_negatives_per_source1": 2,
            "negative_pool_size_per_source_country": 10,
        }
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_validation_split_is_exact_and_disjoint(self):
        splits = assign_splits((f"S1-{i}" for i in range(100)), 0.10, 2026)
        validation = {key for key, value in splits.items() if value == "validation"}
        training = {key for key, value in splits.items() if value == "train"}
        self.assertEqual(len(validation), 10)
        self.assertEqual(len(training), 90)
        self.assertFalse(validation & training)

    def test_negatives_do_not_include_true_match(self):
        s1 = Entity("S1-1", "A", "X", "US")
        true_target = Entity("S2-1", "A", "X", "US")
        other = Entity("S2-2", "B", "Y", "US")
        pools = {("S2", "US"): [true_target, other]}
        chosen = choose_negatives(s1, {"S2-1"}, pools, 1, 2026)
        self.assertEqual([item.entity_id for item in chosen], ["S2-2"])


if __name__ == "__main__":
    unittest.main()
