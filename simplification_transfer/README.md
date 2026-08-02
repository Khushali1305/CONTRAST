# Simplification Transfer — Readability-Controlled Text Simplification

**Transfer domain of the CONTRAST paper (RQ4).** Tests whether the same
counterfactual probe methodology used for SRS correction generalizes to a
second norm-governed revision task: rewriting text to a target readability
level (CEFR A2/B1) while preserving meaning and critical detail.

- **RQ4** — does the framework transfer from SRS correction to a second
  norm-governed revision task?

The key methodological twist versus SRS: for the **shortcut** probe family,
reliable behavior is the *opposite* of SRS. In SRS, a reliable corrector
must **filter** a tempting distractor cue. In simplification, a reliable
simplifier must **preserve** a critical inserted detail (e.g. "especially
chest pain") rather than deleting it while making the sentence shorter.

---

## 1. Datasets

| Dataset | Role | Source |
|---|---|---|
| **TSAR 2025** | Main controlled-simplification backbone — B2+ → A2/B1 CEFR rewriting task, explicitly defined in these terms | [huggingface.co/datasets/cardiffnlp/TSAR2025_SharedTask_RCTS_Trial-Data](https://huggingface.co/datasets/cardiffnlp/TSAR2025_SharedTask_RCTS_Trial-Data) |
| **OneStopEnglish** | Aligned difficulty probes — same content exists at multiple reading levels; ingested as **full articles**, not 2–3 sentence excerpts | [huggingface.co/datasets/SetFit/onestop_english](https://huggingface.co/datasets/SetFit/onestop_english) |
| **ASSET** | Stress-tests the "multiple valid rewrites" side of the framework | [huggingface.co/datasets/facebook/asset](https://huggingface.co/datasets/facebook/asset) |

Raw copies of these are included under `data/transfer_data/` for
reproducibility convenience — check each dataset's own license before
redistributing.

## 2. Pipeline — Exact Run Order

```
ingestion/  →  data/  (master table + probecore)  →  pipeline/  →  results/
```

### Stage 1 — Ingestion
```bash
cd simplification_transfer/ingestion
python ingest_datasets.py
#   reads:  ../data/transfer_data/{tsar2025,onestop_english,asset_*}.json
#   writes: ../data/simp_master_table.jsonl   (normalized base items)
#           simp_data_card_v2.json, simp_transfer_examples_v2.json
```

Each base item looks like:
```json
{
  "item_id": "ose_001",
  "dataset": "OneStopEnglish",
  "original": "The text to simplify...",
  "target_level": "a2",
  "ose_level": "advanced",
  "tier": 1,
  "text_length_words": 120
}
```

Probes generated from these base items (`data/simp_probecore_v1.jsonl`)
look like:
```json
{
  "probe_id": "probe_inv_001",
  "base_item_id": "ose_001",
  "probe_family": "invariance",
  "probe_subtype": "paraphrase",
  "probe_text": "A paraphrased version of the original...",
  "base_text": "The original text...",
  "target_level": "a2",
  "tier": 1,
  "dataset": "OneStopEnglish",
  "generator_model": "",
  "critical_detail": ""
}
```
`probe_family` ∈ `{invariance, directional, shortcut}`.

`pipeline/patch_semantic_probes.py` is a one-off balancing pass used when
per-model probe coverage was uneven (originally: flan-t5 models had ~2035
probes each vs. as few as 397 for llama-3.1-8b) — it tops up every model
to equal coverage by generating additional paraphrase probes. Not part of
the main happy-path pipeline; only needed if you add a new model whose
probe coverage is imbalanced.

### Stage 2 — Run + Score (single-machine)
```bash
cd simplification_transfer/pipeline

# Full pipeline (all 4 phases)
python run_and_score.py \
  --input-dir  ../data \
  --output-dir ../results \
  --phase all

# Or run phases individually:
python run_and_score.py --input-dir ../data --output-dir ../results --phase run      # 1. generate simplifications
python run_and_score.py --input-dir ../data --output-dir ../results --phase score    # 2. rule-based scoring
python run_and_score.py --input-dir ../data --output-dir ../results --phase judge    # 3. LLM judge evaluation
python run_and_score.py --input-dir ../data --output-dir ../results --phase results  # 4. aggregate into 13 CSV/JSON files

# Run specific models only
python run_and_score.py --input-dir ../data --output-dir ../results --models llama-3.1-70b,gemma-3-27b
```

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | *(required)* | Directory with input JSONL files |
| `--output-dir` | *(required)* | Directory for all outputs |
| `--phase` | `all` | `run` \| `score` \| `judge` \| `results` \| `all` |
| `--models` | all models | Comma-separated model names to run |
| `--delay` | `0.5` | Seconds between sequential API calls |
| `--api-threads` | `4` | Concurrent threads for API models |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |

### Stage 2 (alternative) — Sharded / multi-machine run

For the actual paper run, judge validation was parallelized across three
shards (three Lightning Studios / API-key sets), because judge calls are
the throughput bottleneck. This is why `results/raw_shards/` exists — see
§5 below. If you want to reproduce that setup:

```bash
# Split the probecore into N chunks, one per shard/machine
python shared_split.py   # see script for exact chunking logic

# On each shard machine, generate model outputs for its chunk:
python run_chunk.py \
  --chunk-file  <shard_dir>/splits/probes_chunk_1_of_2.jsonl \
  --master      ../data/simp_master_table.jsonl \
  --model-name  llama-3.1-70b \
  --out-file    <shard_dir>/llama-3.1-70b_chunk1.jsonl

# On each shard, run judge workers (supports multiple API keys in parallel
# via --key-id / --num-keys, round-robin over probes):
MISTRAL_API_KEY=key1 python run_judge_worker.py \
    --probecore  <shard_dir>/simp_probecore_v1.jsonl \
    --out-dir    <shard_dir>/simp_judge_outputs \
    --judge-name mistral-large-3 \
    --provider   mistral \
    --model-id   mistral-large-latest \
    --delay      2.5
#   add --key-id 0 --num-keys 3 (and a second/third terminal with key-id 1/2)
#   to fan a single shard across 3 API keys

# After all shard workers finish, merge judge decisions:
python merge_judge_output.py \
    --base-dir  <results>/out \
    --final-dir <results>/final_judge_outputs

# Then combine all shards and run phase_results on the combined data:
python combine_results.py \
    --shard-out-dirs results/out/shard1 results/out/shard2 results/out/shard3 \
    --combined-dir   results/out/combined \
    --probecore      results/shard1/simp_probecore_v1.jsonl \
    --judge-names    mistral-large-3 claude-judge
```

## 3. Models Under Evaluation

| name | family | provider | model_id | max_tokens |
|---|---|---|---|---|
| `flan-t5-large` | Google | local | `google/flan-t5-large` | 512 |
| `flan-t5-xl` | Google | local | `google/flan-t5-xl` | 512 |
| `llama-3.1-8b` | Meta | ollama | `llama3.1:8b-instruct-q8_0` | 2048 |
| `llama-3.1-70b` | Meta | ollama | `llama3.1:70b` | 2048 |
| `qwen2.5-32b` | Alibaba | ollama | `qwen2.5:32b` | 2048 |
| `gemma-3-27b` | Google | ollama | `gemma3:27b` | 2048 |
| `gpt-4.1` | OpenAI | openai | `gpt-4.1` | 2048 |

**Judges** (probe-validation only — never used to score evaluated models):

| name | family | provider | model_id | threads |
|---|---|---|---|---|
| `claude-haiku` | Anthropic | anthropic | `claude-haiku-4-5-20251001` | 2 |
| `mistral-large-3` | Mistral | mistral | `mistral-large-2411` | 1 |

`JUDGE_POLICY = "agree_only"` — a probe only counts as validated if both
judges agree it's semantically equivalent/valid; disagreements are logged
to `judge_disagreement_analysis.csv` for manual review, not auto-resolved.

To add a new model, edit `common/config.py` → `EXPERIMENTAL_MODELS`, and
(for Ollama models) add a matching entry to `OLLAMA_MODEL_CONFIGS` inside
`pipeline/run_and_score.py` to tune `num_ctx`, `num_predict`, and worker
concurrency for your GPU.

## 4. GPU Requirements — Ollama on H100 80GB

| Model | Quantization | VRAM | Parallel workers |
|---|---|---|---|
| `llama3.1:70b` | q4_K_M | ~40 GB | 6 |
| `gemma3:27b` | q4_K_M | ~17 GB | 6 |
| **Total** | | **~70 GB** | — |

Both models run **simultaneously** on one H100 80GB. Expected wall time:
**~90 minutes** for 10,814 probes. FLAN-T5-Large/XL run locally via
HuggingFace `transformers` (small enough to not need quantization);
Qwen2.5-32B also runs via Ollama alongside the two models above if VRAM
allows, otherwise run it in a separate pass. GPT-4.1 is API-only, no GPU
needed.

## 5. Prompts

All prompts live in `common/config.py`.

**Paraphrase (invariance probe generation):**
```
SYSTEM: You are a paraphrasing engine. Rewrite input using different words
and structure while preserving EXACTLY the same meaning and difficulty.
Output ONLY the rewritten text.

USER: Rewrite using different words and structure. Keep exact same meaning.

TEXT: {text}

REWRITTEN:
```

**Simplification (the actual task under test):**
```
SYSTEM: You are a text simplification system. Rewrite to make easier to
read. Preserve ALL important information including medical warnings,
conditions, numbers, names. Do NOT drop details to shorten. Output ONLY
simplified text.

USER (CEFR-targeted): Simplify to {target} reading level. Preserve all
important information.

TEXT: {text}

SIMPLIFIED:

USER (generic, no explicit target level): Simplify to make easier to read.
Preserve all important information.

TEXT: {text}

SIMPLIFIED:
```

**Judge (probe validation only):**
```
SYSTEM: You are a semantic equivalence judge. Compare original and
transformed text.
Check: AGENTS (same entities?), ACTIONS (same events?), MODALS
(shall/should/may preserved?), NEGATION (preserved?), CONDITIONS
(preserved?), CONTENT (all facts/numbers preserved?).
Respond ONLY in JSON:
{"equivalent":true/false,"confidence":"high/medium/low","failures":["CRITERION"],"reason":"one sentence"}

USER: ORIGINAL: {source}
TRANSFORMED: {probe}

Semantically equivalent? JSON only.
```

## 6. Scoring Thresholds (`common/config.py`)

| Constant | Value | Meaning |
|---|---|---|
| `MIN_WORDS_ASSET` | 12 | minimum word count for ASSET-sourced items |
| `OSE_ALIGN_THRESHOLD` | 5 | alignment threshold for OneStopEnglish cross-level matching |
| `AUDIT_SAMPLE_SIZE` | 120 | number of items in the manual audit sample |
| `LEXICAL_RECALL_THRESHOLD` | 0.6 | minimum lexical overlap for a valid probe |
| `MAX_LENGTH_RATIO` / `MIN_LENGTH_RATIO` | 1.6 / 0.5 | acceptable output/input length ratio |
| `BERTSCORE_THRESHOLD` | 0.75 | semantic similarity floor |
| `READABILITY_DELTA_MAX` | 3.0 | max allowed readability-score drift for invariance probes |

Shortcut probes are built from four cue-phrase banks (`medical`, `safety`,
`qualifying`, `numerical`) in `common/config.py::SHORTCUT_TEMPLATES` —
e.g. `", especially if experiencing chest pain"`. A reliable simplifier
must **retain** these phrases; a reliable SRS corrector, by contrast, must
**filter** structurally analogous cues. `SYNONYMS` and `PROTECTED_WORDS`
in the same file drive the rule-based lexical-invariance probe generator
(protected words like modal verbs and negation are never swapped for a
synonym, since that would change meaning rather than just wording).

## 7. Audit Process

`audit/` holds the manual quality-control sample used to validate the
probe generation pipeline before freezing `data/simp_probecore_v1.jsonl`:
- `simp_audit_sample.jsonl` — the 120-item stratified sample drawn per
  `AUDIT_SAMPLE_SIZE`
- `simp_audit_form.csv` — blank annotation form
- `simp_audit_completed.csv` — completed manual annotations
- `simp_data_card.json`, `simp_ose_alignments.json`, `simp_probecore_stats.json`
  — dataset/alignment statistics referenced in the paper's data card
- `early_snapshots/` — an earlier pass of the data card / OSE alignment
  files, kept for provenance; superseded by the root-level files above

## 8. Results Directory — What's Authoritative

```
results/
├── final/            # ⭐ AUTHORITATIVE — the 13 files described below,
│                      #   used for every paper table/figure in this domain
├── model_outputs/     # raw per-model generations (one .jsonl per model)
├── judge_outputs/      # final merged judge decisions + agreement stats
└── raw_shards/         # intermediate sharded-run artifacts (shard1/2/3,
                         # out_combined, patch_temp, splits, a partial
                         # llama-3.1-70b chunk) — NOT required to reproduce
                         # the paper's numbers, kept only for provenance
                         # of the distributed run described in §2 above
```

`results/final/` contains 13 files:

| File | Description |
|---|---|
| `instance_level_results.csv` | Per-probe score for every model |
| `reliability_summary.csv` / `.json` | Per-model summary (invariance / directional / shortcut / global) |
| `reliability_by_dataset.csv` | Reliability broken down by dataset |
| `reliability_by_subtype.csv` | Reliability broken down by probe subtype |
| `reliability_by_tier.csv` | Reliability broken down by tier |
| `transform_quality_comparison.csv` | Transform quality across subtypes |
| `pipeline_overview.csv` | High-level pipeline counts |
| `paper_table_1_main_results.csv` | Camera-ready main results table |
| `paper_table_2_per_dataset.csv` | Camera-ready per-dataset table |
| `paper_table_3_tier_comparison.csv` | Camera-ready tier-comparison table |
| `judge_decisions_detail.csv` | Every judge decision with confidence & failures |
| `judge_disagreement_analysis.csv` | Probes where judges disagreed |
| `judge_agreement_stats.csv` | Inter-judge agreement rate |

## 9. Folder Reference

```
simplification_transfer/
├── ingestion/           # raw dataset JSON → normalized master table
├── pipeline/             # run/score/judge/aggregate + sharding helpers
├── data/                 # master table, frozen probecore, raw transfer_data/
├── audit/                # manual QC sample + completed annotations
├── results/               # see §8 above
└── legacy/                # step4_run_and_score_v1.py — superseded pipeline version
```
