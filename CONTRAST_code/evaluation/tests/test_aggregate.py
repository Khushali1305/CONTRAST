import unittest

from evaluation.aggregate import item_macro, item_scores, stratified_item_bootstrap


class AggregateTests(unittest.TestCase):
    def test_item_macro_weights_items_not_probe_rows(self):
        rows = [
            {"source_dataset": "A", "canonical_item_id": "i1", "metric": "1"},
            {"source_dataset": "A", "canonical_item_id": "i1", "metric": "1"},
            {"source_dataset": "B", "canonical_item_id": "i2", "metric": "0"},
        ]
        self.assertEqual(item_macro(rows, "metric"), 0.5)

    def test_stratified_bootstrap_is_deterministic(self):
        rows = [
            {"source_dataset": "A", "canonical_item_id": "i1", "metric": "1"},
            {"source_dataset": "A", "canonical_item_id": "i2", "metric": "0"},
            {"source_dataset": "B", "canonical_item_id": "i3", "metric": "1"},
        ]
        scores = item_scores(rows, "metric")
        first = stratified_item_bootstrap(scores, replicates=200, seed=17)
        second = stratified_item_bootstrap(scores, replicates=200, seed=17)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

