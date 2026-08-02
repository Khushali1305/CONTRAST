"""
gec_probe.config.settings
-------------------------
Central configuration for the GEC CONTRAST pipeline.
All API keys come from environment variables. No hardcoded secrets.
"""
import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("GEC_PROBE_ROOT", Path.cwd()))
DATA_DIR = PROJECT_ROOT / "data_artifacts"
RAW_DIR = DATA_DIR / "raw" / "bea2019"
NORMALIZED_DIR = DATA_DIR / "normalized"
PROBES_DIR = DATA_DIR / "probes"
INFERENCE_DIR = DATA_DIR / "inference"
JUDGE_DIR = DATA_DIR / "judge"
SCORES_DIR = DATA_DIR / "scores"

for d in (DATA_DIR, RAW_DIR, NORMALIZED_DIR, PROBES_DIR,
          INFERENCE_DIR, JUDGE_DIR, SCORES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
BEA2019_SOURCE_URL = (
    "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"
)
# Splits: ABC.train.gold.bea19.m2, ABCN.dev.gold.bea19.m2
SPLIT_FILES = {
    "train": "wi+locness/m2/ABCN.train.gold.bea19.m2",
    "dev":   "wi+locness/m2/ABCN.dev.gold.bea19.m2",
}

# Target ProbeCore size after stratified sampling
TARGET_BASE_ITEMS = 900

# Stratification axes
CEFR_LEVELS = ["A", "B", "C", "N"]   # N = native (LOCNESS)
ERROR_TYPE_BUCKETS = [
    "M:DET", "R:DET", "U:DET",
    "M:PREP", "R:PREP", "U:PREP",
    "R:VERB:SVA", "R:VERB:TENSE", "R:VERB:FORM",
    "R:MORPH", "R:ORTH", "R:WO",
    "R:NOUN:NUM", "R:OTHER",
]
LENGTH_QUARTILES = 4

# ------------------------------------------------------------------
# Probe construction
# ------------------------------------------------------------------
PROBE_FAMILIES = ["invariance", "directional", "overcorrection"]

# Per family, max probes per base item
PROBES_PER_ITEM = {
    "invariance":     2,
    "directional":    3,   # add / remove / swap
    "overcorrection": 2,
}

# Overcorrection cue banks (replaces SRS-style "shortcut" family for GEC)
OVERCORRECTION_CUES = {
    "register":   ["henceforth", "hitherto", "aforementioned",
                   "notwithstanding", "thereupon"],
    "discourse":  ["actually", "in fact", "as it happens",
                   "to be honest"],
    "topicalize": ["It is well known that", "As is often noted,",
                   "It goes without saying that"],
}

# Sentinel value for invalid / unscorable probes
SENTINEL_INVALID = -1

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
# Correctors-under-test (same pool as SRS for cross-domain comparability)
CORRECTOR_MODELS = [
    {"id": "flan_t5_large",  "family": "flan_t5",  "host": "local"},
    {"id": "flan_t5_xl",     "family": "flan_t5",  "host": "local"},
    {"id": "llama_3_1_8b",   "family": "llama",    "host": "groq"},
    {"id": "llama_3_1_70b",  "family": "llama",    "host": "groq"},
    {"id": "qwen_2_5_32b",   "family": "qwen",     "host": "local"},
    {"id": "gemma_3_27b",    "family": "gemma",    "host": "local"},
    {"id": "gpt_4_1",        "family": "openai",   "host": "openai"},
    {"id": "gector_baseline","family": "rule",     "host": "local"},
]

# Judge constraint: ZERO family overlap with CORRECTOR_MODELS.
JUDGE_MODELS = [
    {"id": "claude_haiku_4_5", "family": "anthropic",
     "host": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    {"id": "mistral_large_3",  "family": "mistral",
     "host": "mistral",   "api_key_env": "MISTRAL_API_KEY"},
]

# Probe-generator and paraphrase models.
# Family chosen to be DISJOINT from both corrector families AND judge families.
# Native DeepSeek V4 Flash (deepseek family) satisfies this — NOT the
# Groq-hosted "deepseek-r1-distill-llama-70b", which is architecturally Llama.
GENERATOR_MODELS = {
    "paraphraser":    {"id": "deepseek_v4_flash", "host": "deepseek"},
    "error_injector": {"id": "deepseek_v4_flash", "host": "deepseek"},
}
GENERATOR_FAMILY = "deepseek"

# ------------------------------------------------------------------
# API credentials (env vars only)
# ------------------------------------------------------------------
API_KEYS = {
    "openai":    os.environ.get("OPENAI_API_KEY"),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
    "mistral":   os.environ.get("MISTRAL_API_KEY"),
    "groq":      os.environ.get("GROQ_API_KEY"),
    "deepseek":  os.environ.get("DEEPSEEK_API_KEY"),
}

# ------------------------------------------------------------------
# ERRANT
# ------------------------------------------------------------------
ERRANT_LANG = "en"
ERRANT_TIMEOUT_S = 30

# ------------------------------------------------------------------
# Family-membership guard
# ------------------------------------------------------------------
def assert_judge_corrector_disjoint() -> None:
    """Hard assertion: judge families and corrector families must not overlap."""
    corrector_families = {m["family"] for m in CORRECTOR_MODELS}
    judge_families = {m["family"] for m in JUDGE_MODELS}
    overlap = corrector_families & judge_families
    if overlap:
        raise RuntimeError(
            f"Judge/corrector family overlap detected: {overlap}. "
            "Pick judges from disjoint families."
        )


def assert_generator_disjoint_from_judges_and_correctors() -> None:
    """
    Hard assertion: probe-generator family must be disjoint from BOTH
    corrector families and judge families. This blocks:
      - same-family generator/corrector bias (probes easier for kin models)
      - same-family generator/judge bias (self-recognition / self-preference)
    """
    corrector_families = {m["family"] for m in CORRECTOR_MODELS}
    judge_families = {m["family"] for m in JUDGE_MODELS}
    if GENERATOR_FAMILY in corrector_families:
        raise RuntimeError(
            f"Generator family {GENERATOR_FAMILY!r} overlaps corrector pool "
            f"{corrector_families}. Pick a disjoint generator family."
        )
    if GENERATOR_FAMILY in judge_families:
        raise RuntimeError(
            f"Generator family {GENERATOR_FAMILY!r} overlaps judge pool "
            f"{judge_families}. Pick a disjoint generator family."
        )


def assert_api_keys_present(needed: list) -> None:
    missing = [k for k in needed if not API_KEYS.get(k)]
    if missing:
        raise RuntimeError(f"Missing API keys for: {missing}")
