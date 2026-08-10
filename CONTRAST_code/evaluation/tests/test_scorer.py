import unittest

from evaluation.scorer import (
    collateral_scores,
    ears_struct_v1,
    first_modal_class,
    score_pair,
    source_eligibility,
)


class ScorerTests(unittest.TestCase):
    def test_first_modal_not_highest_modal(self):
        self.assertEqual(first_modal_class("The device may retry but must stop after 3 attempts."), 1)

    def test_directional_equality_fails(self):
        record = {
            "analysis_family": "directional",
            "operation": "modal_substitution",
            "direction": "weaken",
            "original_modal": "shall",
            "new_modal": "may",
            "base_source": "The service shall retain the record.",
            "probe_source": "The service may retain the record.",
            "base_rewrite": "The service shall retain the record.",
            "probe_rewrite": "The service shall retain the record.",
        }
        scores = score_pair(record)
        self.assertEqual(scores["focal_relation_pass"], 0)
        self.assertEqual(scores["focal_exact_pass"], 0)

    def test_directional_exact_pass(self):
        record = {
            "analysis_family": "directional",
            "operation": "modal_substitution",
            "direction": "weaken",
            "original_modal": "shall",
            "new_modal": "may",
            "base_source": "The service shall retain the record.",
            "probe_source": "The service may retain the record.",
            "base_rewrite": "The service shall retain the record.",
            "probe_rewrite": "The service may retain the record.",
        }
        eligible, _ = source_eligibility(record)
        scores = score_pair(record)
        self.assertTrue(eligible)
        self.assertEqual(scores["focal_relation_pass"], 1)
        self.assertEqual(scores["focal_exact_pass"], 1)
        self.assertEqual(scores["complete_exact_primary"], 1)

    def test_numbers_and_negation_are_exact(self):
        collateral = collateral_scores(
            "The service shall not retry more than 3 times.",
            "The service must not retry more than 3 times.",
            "The service shall not retry more than 3 times.",
            "The service must retry more than 4 times.",
        )
        self.assertEqual(collateral.number_consistency_pass, 0)
        self.assertEqual(collateral.negation_consistency_pass, 0)
        self.assertEqual(collateral.collateral_primary_pass, 0)

    def test_addition_is_descriptive(self):
        record = {
            "analysis_family": "addition",
            "operation": "distractor_insertion",
            "base_source": "The service shall retain the record.",
            "probe_source": "The service shall retain the record with military-grade encryption.",
            "distractor": "military-grade encryption",
            "base_rewrite": "The service shall retain the record.",
            "probe_rewrite": "The service shall retain the record with military-grade encryption.",
        }
        scores = score_pair(record)
        self.assertEqual(scores["addition_uptake_70"], 1)
        self.assertEqual(scores["addition_probe_only_exact_phrase"], 1)

    def test_ears_struct_is_only_a_form_comparator(self):
        self.assertEqual(ears_struct_v1("The service shall erase every record."), 1.0)
        self.assertLess(ears_struct_v1("Erase every record"), 1.0)


if __name__ == "__main__":
    unittest.main()

