import unittest

from evaluation.aggregate import (
    apply_contract_version,
    item_macro,
    item_scores,
    stratified_item_bootstrap,
)


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

    def test_v11_excludes_only_direct_will_to_shall_invariance_pairs(self):
        base = {
            "analysis_family": "invariance",
            "operation": "rule_modal_synonym",
            "source_base_first_modal_class": "2",
            "source_probe_first_modal_class": "3",
            "source_eligible": "1",
        }
        same_class = {**base, "source_probe_first_modal_class": "2"}
        directional = {**base, "analysis_family": "directional"}
        rows = [base, same_class, directional]
        v11 = apply_contract_version(rows, "1.1")
        self.assertEqual([row["source_eligible"] for row in v11], ["0", "1", "1"])
        self.assertEqual(apply_contract_version(rows, "1.0"), rows)


if __name__ == "__main__":
    unittest.main()

