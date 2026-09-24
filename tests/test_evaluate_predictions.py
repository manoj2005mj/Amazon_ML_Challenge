import unittest

from src.evaluate_predictions import macro_f05


class EvaluatePredictionsTests(unittest.TestCase):
    def test_macro_f05_rewards_correct_singleton(self):
        truth = {"S1-1": {"S2-1"}, "S1-2": set()}
        predicted = {"S1-1": {"S2-1"}, "S1-2": set()}
        self.assertEqual(macro_f05(truth, predicted), 1.0)

    def test_macro_f05_penalizes_false_singleton_merge(self):
        truth = {"S1-1": set()}
        predicted = {"S1-1": {"S2-9"}}
        self.assertEqual(macro_f05(truth, predicted), 0.0)


if __name__ == "__main__":
    unittest.main()
