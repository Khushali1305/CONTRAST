# SRS-ProbeCore — Software Requirements Specification Correction

**Primary domain of the CONTRAST paper.** Evaluates whether an LLM
reliably rewrites a free-text software requirement into an **EARS**
(Easy Approach to Requirements Syntax) structured template, answering:

- **RQ1** — do counterfactual probes reveal reliability failures that
  ordinary template-conformance evaluation misses?
- **RQ2** — which of the three probe families (invariance, directional,
  shortcut) is most diagnostic of unreliable behavior?

Full methodology and worked examples: see the "Methodology" and
"Evaluation Settings" sections of the paper. The summary needed to run
this code is below.

---

## 1. Datasets

| Dataset | Role | Size used | Source |
|---|---|---|---|
| **PURE** | Realistic, messy free-text requirements | 587 items in frozen benchmark | [zenodo.org/records/7118517](https://zenodo.org/records/7118517) — 79 public requirements docs, 34,268 sentences total |
| **Großer benchmark** | `(free_text, EARS_template)` aligned pairs — best source for invariance probes and clean-reference scoring | 219 items | [zenodo.org/records/10974836](https://zenodo.org/records/10974836) — 249 free-text forms, 1,764 requirement instances |
| **PROMISE_exp** | Wording diversity + FR/NFR subgroup analysis (not used as conformance gold) | 85 items | [PMC8974737](https://pmc.ncbi.nlm.nih.gov/articles/PMC8974737/) — 444 FR + 525 NFR instances |

None of these raw datasets are redistributed in this repo (check their
licenses) — `ingestion/` scripts expect you to download them yourself and
point `--input-dir` at the unpacked folder.

## 2. The Frozen Benchmark

`data/SRS-ProbeCore-v5-frozen.json` is the **frozen, do-not-regenerate**
benchmark used for all reported numbers:

- **891 base items** (587 PURE + 219 Großer + 85 PROMISE_exp)
- **3,068 probe pairs**: 1,286 invariance, 891 directional, 891 shortcut
- A standalone supplementary mini-benchmark, `data/reports/nfr_shortcut_minibench_v1.json`
  (18 probes across 8 NFR categories), is generated alongside the main
  benchmark but **not** injected into it.

`data/reports/` also holds intermediate artifacts from building the
benchmark (per-dataset ingestion reports, probe-generation QA reports).
`legacy/probecore_snapshots/` holds the v1–v4 iterations of the benchmark
kept for audit-trail purposes — **not** used by any current script.

## 3. Pipeline — Exact Run Order

```
ingestion/  →  probe_generation/  →  evaluation/
```

### Stage 1 — Ingestion
```bash
cd srs_probecore/ingestion

python parse_pure.py \
    --input-dir  /path/to/PURE_raw/ \
    --output-dir ../data/

python parse_grosser.py     # produces grosser_requirements.json / grosser_report.json
python inspect_promise.py   # inspects PROMISE_exp ARFF/schema, produces promise_exp.json
```
`diagnose.py` / `diagnose_pure.py` / `pure_dataset_inspection.py` are QA
utilities for spot-checking ingestion output — not required for the main
pipeline, useful if a dataset's format changes.

### Stage 2 — Probe Generation
```bash
cd srs_probecore/probe_generation

# 1. Generate invariance/directional/shortcut probes per base item
export OPENAI_API_KEY=your_key
python generate_probes.py \
    --input  <ingested_base_items.json> \
    --output ../data/reports/srs_probecore_v4_probed.json
#   --dry-run --limit 20   # to test without API calls first

# 2. Apply the final freeze pass (balances probe counts, fills
#    generation_method metadata, splits off the NFR mini-benchmark)
python build_srs_probecore.py
#   reads:  ../data/reports/srs_probecore_v4_probed_final.json
#   writes: ../data/SRS-ProbeCore-v5-frozen.json   (main, frozen)
#           ../data/reports/nfr_shortcut_minibench_v1.json
#   Set DRY_RUN = True inside the script to preview changes before writing.
```

**Probe generation models:**
- Generation: `o4-mini` (reasoning model, for faithful paraphrasing)
- Verification: `gpt-4.1-mini` (binary YES/NO quality check)

Per base item, `generate_probes.py` produces: 2 invariance probes
(hybrid rule+LLM, expected relation `stable`), 1 directional probe (modal
or condition-scope change, expected relation `directional`), 1 shortcut
probe (hallucination-targeted distractor, expected relation `no_shortcut`).

### Stage 3 — Model Evaluation
```bash
cd srs_probecore/evaluation

export OPENAI_API_KEY=your_key     # for gpt4
export HF_TOKEN=your_hf_token      # for Llama and Gemma (gated HF models)

# First run
python run_models.py

# Resume after a GPU session timeout (safe to re-run — skips completed jobs)
python run_models.py --resume

# Run only specific models
python run_models.py --models gpt4

# Smoke test — 20 jobs per model
python run_models.py --smoke --models rule_baseline,gpt4
```

This generates a rewrite for every model on every base item **and** every
probe variant, writing `../data/week4_model_outputs.json`.

```bash
# Score everything against the methodology's scoring protocol
export ANTHROPIC_API_KEY=your_key   # required for the shortcut-probe judge
python score_models.py
#   --no_llm_judge   # rule-based shortcut fallback, no Anthropic API needed
#   --tau 0.80       # override the BERTScore invariance threshold (default 0.85)

python make_figures.py   # camera-ready figures from the scored results
```

## 4. Models Under Evaluation

| Key | Model | Provider |
|---|---|---|
| `rule_baseline` | Deterministic EARS restructurer (regex/keyword pattern, no ML) | local |
| `flan_t5_large` | `google/flan-t5-large` | local (HF `transformers`) |
| `flan_t5_xl` | `google/flan-t5-xl` | local (HF `transformers`) |
| `llama_8b` | `meta-llama/Llama-3.1-8B-Instruct` | local (HF, gated — needs `HF_TOKEN`) |
| `llama_70b` | `meta-llama/Llama-3.1-70B-Instruct` | local (HF, gated, 4-bit quantized) |
| `qwen_32b` | `Qwen/Qwen2.5-32B-Instruct` | local (HF, 4-bit quantized) |
| `gemma_27b` | `google/gemma-3-27b-it` | local (HF, gated, 4-bit quantized) |
| `gpt4` | `gpt-4.1` | OpenAI API |

`llama_8b` is run and scored, but excluded from the main paper results
table per project decision (kept here for completeness/comparison).

**Shortcut-probe judge:** `claude-sonnet-4-6` (chosen over Opus because the
binary PASS/FAIL task doesn't need frontier reasoning; chosen over Haiku
because it needs reliable detection of paraphrased distractors). This
judge is used **only** during scoring of the shortcut probe family — it
never sees or influences probe construction for this domain.

## 5. GPU Requirements

- **1× A100 80GB or H100 80GB** recommended.
- FLAN-T5-Large/XL run unquantized (small enough).
- Llama-3.1-70B, Qwen2.5-32B, Gemma-3-27B use **4-bit quantization**
  (`bitsandbytes`) via HuggingFace `transformers` so they fit on a single
  80GB GPU. `dtype` (not the deprecated `torch_dtype`) is used explicitly
  in `run_models.py`'s model loader.
- `HF_TOKEN` must be set in the environment (not just passed as an
  argument) because some versions of the HF Hub library only read gated
  model tokens from the environment.
- GPT-4.1 and the shortcut judge are API-only — no GPU needed for those.

## 6. Prompts

**Rewriting prompt** (used for every non-rule-baseline model in
`run_models.py`):

```
SYSTEM:
You are a requirements engineering assistant.
Your task is to rewrite a free-text software requirement into a structured
EARS (Easy Approach to Requirements Syntax) template form.

EARS templates use:
  Ubiquitous  : The <system> shall <action>.
  Event-driven: WHEN <trigger>, the <system> shall <action>.
  Unwanted    : IF <condition>, the <system> shall <action>.
  State-driven: WHILE <state>, the <system> shall <action>.
  Optional    : WHERE <feature>, the <system> shall <action>.

Rules you must follow:
1. Preserve the original obligation strength (shall/should/may/must).
   Do NOT upgrade "may" or "should" to "shall" unless the original says so.
2. Preserve all conditions, exceptions, and actors from the original.
3. Do NOT invent new behavior, add new clauses, or remove existing ones.
4. Output ONLY the rewritten requirement. No explanation, no commentary.

USER:
Rewrite the following requirement into EARS format:

Requirement: {requirement}

EARS rewrite:
```

**Rule-based baseline** does *not* call an LLM — it's a deterministic
regex classifier over modal verbs (`shall`/`must`/`should`/`will`/`may`/`can`)
and trigger keywords (`if`/`when`/`while`/`once`/`after`), implemented in
`evaluation/run_models.py::RuleBasedEARSBaseline`.

## 7. Scoring Protocol

From `evaluation/score_models.py`:

| Probe family | Pass condition |
|---|---|
| Invariance | BERTScore F1(f(x), f(x′)) ≥ `tau_invariance` (default **0.85**) |
| Directional | Modal strength shifts in the correct direction (scale: `shall`/`must`=3, `should`/`will`=2, `may`/`can`=1) |
| Shortcut | `claude-sonnet-4-6` judge confirms the distractor was **not** acted on |

```
invariance_reliability  = PASS_inv / N_inv
directional_sensitivity = PASS_dir / N_dir
shortcut_resistance     = PASS_shc / N_shc
R(f) = (invariance_reliability + directional_sensitivity + shortcut_resistance) / 3
```

Ordinary-evaluation baseline (for RQ1's comparison): EARS structural
conformance (regex check for `WHEN`/`IF`/`WHILE`/`WHERE` + a modal verb)
and BERTScore F1 between rewrite and source (meaning preservation).

## 8. Known Issues / Fixed Bugs (kept here for audit transparency)

- An earlier version of the directional scorer used `SequenceMatcher` and
  under-counted failures (3 vs. a corrected 438) — fixed in the current
  `score_models.py`.
- The shortcut judge originally called OpenAI instead of Anthropic, and
  its API-key fallback check pointed at the wrong env var — both fixed.
- Records with `run_status=empty` are now correctly excluded from the
  scoring index (previously silently included, deflating denominators).

## 9. Folder Reference

```
srs_probecore/
├── ingestion/            # raw dataset → normalized JSON
├── probe_generation/     # normalized JSON → frozen probe benchmark
├── evaluation/           # run models on frozen benchmark, score, plot
├── data/
│   ├── SRS-ProbeCore-v5-frozen.json   # ⭐ the frozen benchmark
│   ├── week4_model_outputs.json       # raw model rewrites (base + probes)
│   └── reports/                       # per-dataset + per-stage QA reports
└── legacy/                            # archived v1-v4 script/data iterations (not used by current pipeline)
```
