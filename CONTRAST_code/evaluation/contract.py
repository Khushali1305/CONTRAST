"""Frozen constants for CONTRAST-SRS-CR-v1.0.0 and its v1.1.0 eligibility amendment."""

CONTRACT_ID = "CONTRAST-SRS-CR-v1.0.0"
FROZEN_DATE = "2026-07-31"

# Version 1.1.0 (the version reported in the paper) changes source eligibility only.
# It excludes the 166 meaning-preserving pairs that directly substitute "will" -> "shall".
# The frozen modal scale assigns "will" (moderate, class 2) and "shall" (strong, class 3)
# to different classes, so these pairs conflict with the scale. The rule reads only
# source-side fields and operation metadata; scoring, thresholds, and aggregation are unchanged.
CONTRACT_ID_V11 = "CONTRAST-SRS-CR-v1.1.0"
DEFAULT_CONTRACT_VERSION = "1.1"
CONTRACT_VERSIONS = ("1.0", "1.1")


def is_v11_excluded(row: dict[str, str]) -> bool:
    """Return True for a direct will->shall meaning-preserving pair excluded in v1.1.0."""
    return (
        row.get("analysis_family") == "invariance"
        and row.get("operation") == "rule_modal_synonym"
        and row.get("source_base_first_modal_class") == "2"
        and row.get("source_probe_first_modal_class") == "3"
    )

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_731

MODEL_ORDER = [
    "flan_t5_large",
    "flan_t5_xl",
    "llama_8b",
    "llama_70b",
    "qwen_32b",
    "gemma_27b",
    "gpt4",
    "rule_baseline",
]

MODEL_METADATA = {
    "flan_t5_large": ("FLAN-T5-Large", "evaluated_neural_system", "short_user_only"),
    "flan_t5_xl": ("FLAN-T5-XL", "evaluated_neural_system", "short_user_only"),
    "llama_8b": ("Llama-3.1-8B", "evaluated_neural_system", "full_system_plus_user"),
    "llama_70b": ("Llama-3.1-70B", "evaluated_neural_system", "full_system_plus_user"),
    "qwen_32b": ("Qwen2.5-32B", "evaluated_neural_system", "full_system_plus_user"),
    "gemma_27b": ("Gemma-3-27B", "evaluated_neural_system", "full_system_plus_user"),
    "gpt4": ("GPT-4.1", "evaluated_neural_system", "full_system_plus_user"),
    "rule_baseline": ("Rule Baseline", "calibration_baseline", "deterministic_rule_baseline"),
}

EXPECTED_ELIGIBLE_PAIRS = {
    "invariance": 1_286,
    "directional": 762,
    "addition": 891,
}

EXPECTED_ELIGIBLE_PAIRS_V11 = {
    "invariance": 1_120,
    "directional": 762,
    "addition": 891,
}
V11_EXCLUDED_PAIRS_PER_MODEL = 166

MODAL_CLASSES = {
    "may": 1,
    "can": 1,
    "should": 2,
    "will": 2,
    "shall": 3,
    "must": 3,
}

CONTENT_STOPWORDS = frozenset(
    "a an and are as at be been being by for from had has have he her hers him his "
    "i in into is it its of on or our ours she that the their theirs them they this to "
    "was we were with you your yours system when if while where".split()
)

NEGATION_TOKENS = frozenset({"not", "no", "never"})
CONTENT_F1_THRESHOLD = 0.80
ANCHOR_RECALL_THRESHOLD = 0.50

