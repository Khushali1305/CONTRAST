# CONTRAST — Counterfactual Probes for Norm-Governed Text Revision

CONTRAST is a counterfactual probe protocol for evaluating whether an LLM
**reliably** revises text under an explicit norm — rewriting a software
requirement into a structured template, simplifying text to a target
readability level, or correcting grammatical errors — rather than merely
producing an output that *looks* correct.

Instead of scoring a single output against a single reference, CONTRAST
builds **paired inputs** (a base item and a controlled perturbation of it)
and checks whether the model's *behavioral relation* between the two
outputs matches what the perturbation should produce. Three probe families
are used throughout every domain in this repo:

| Probe family | What it does | Expected behavior |
|---|---|---|
| **Invariance** | Paraphrase the input; meaning is unchanged | Output should stay stable |
| **Directional** | Change one meaning-bearing feature (e.g. `should`→`may`, add/remove an error) | Output should shift in the same direction |
| **Shortcut** | Insert a tempting local cue (a distractor phrase, an overcorrection trigger) | Output should not overreact to the cue — reliability here is *task-conditioned*: SRS correction and GEC should **filter** the cue; simplification should **preserve** it if it's a critical detail |

This repository contains the code and data for all three domains studied in
the paper, reorganized for public release:

1. **`srs_probecore/`** — primary domain: Software Requirements Specification
   (SRS) correction into EARS-style templates. RQ1 (hidden failures) and RQ2
   (which probe family is most diagnostic).
2. **`simplification_transfer/`** — transfer domain: readability-controlled
   text simplification. RQ4 (does the framework transfer to a second
   norm-governed revision task).
3. **`gec_extension/`** — second transfer domain: grammatical error
   correction (GEC) on BEA-2019, added as a post-submission generalization
   extension.
4. **`cross_domain_analysis/`** — RQ3: does probe-based reliability predict
   expert/gold-quality judgments better than ordinary conformance metrics.

Each domain folder has its **own detailed README** with exact run commands,
prompts, and file-by-file explanations. This top-level README covers the
shared setup, the overall methodology, and a map of the repo.

---

## Repository Structure

```
CONTRAST/
├── README.md                      # you are here
├── environment/
│   ├── requirements.txt           # merged Python deps, all domains
│   └── .env.example                # copy to .env at repo root, fill in keys
├── common/
│   ├── api_client.py               # unified call_model() for every provider (SRS + Simplification)
│   └── config.py                   # shared model list, prompts, thresholds (Simplification domain)
├── srs_probecore/                  # PRIMARY DOMAIN — see srs_probecore/README.md
├── simplification_transfer/        # TRANSFER DOMAIN 1 — see simplification_transfer/README.md
├── gec_extension/                  # TRANSFER DOMAIN 2 — see gec_extension/README.md
└── cross_domain_analysis/          # RQ3 proxy analysis — see cross_domain_analysis/README.md
```

---

## Quick Start

```bash
git clone <this-repo-url> CONTRAST
cd CONTRAST

# 1. Install dependencies
pip install -r environment/requirements.txt
# (some hosted environments require: pip install --break-system-packages -r environment/requirements.txt)

# 2. Set API keys
cp environment/.env.example .env
# edit .env with your real keys, then:
export $(grep -v '^#' .env | xargs)

# 3. Go into whichever domain you want to reproduce and follow its README:
cd srs_probecore            # or simplification_transfer, or gec_extension
cat README.md
```

Every domain README documents its own exact stage order (ingest → generate
probes → run models → score), because each domain has a different
frozen benchmark and a different set of models under test.

---

## Models Evaluated (across domains)

| Model | Family | Role |
|---|---|---|
| Rule/template baseline | — (deterministic, no ML) | Corrector-under-test (SRS only) |
| FLAN-T5-Large / FLAN-T5-XL | `flan_t5` (Google) | Corrector-under-test (all domains) |
| Llama-3.1-8B-Instruct | `llama` (Meta) | Corrector-under-test (all domains; excluded from main SRS paper tables per project decision) |
| Llama-3.1-70B-Instruct | `llama` (Meta) | Corrector-under-test (all domains) |
| Qwen2.5-32B-Instruct | `qwen` (Alibaba) | Corrector-under-test (all domains) |
| Gemma-3-27B-it | `gemma` (Google) | Corrector-under-test (all domains) |
| GPT-4.1 | `openai` | Corrector-under-test (all domains) |
| Claude Haiku 4.5 | `anthropic` | Judge (probe-validation only, Simplification + GEC) |
| Mistral Large (`mistral-large-2411`) | `mistral` | Judge (probe-validation only, Simplification + GEC) |
| DeepSeek V4 Flash | `deepseek` | Probe generator (GEC only — native API, `api.deepseek.com/v1`) |
| claude-sonnet-4-6 | `anthropic` | Shortcut-probe judge (SRS scoring stage only) |

**Family-disjointness principle:** the model that generates/paraphrases a
probe, the models that judge whether a generated probe is valid, and the
models being evaluated as correctors must belong to three disjoint model
families. This prevents self-preference bias (a model favoring outputs
from its own family) from contaminating either probe construction or
scoring. The GEC pipeline (`gec_extension/gec_probe/config/settings.py`)
enforces this with a hard assertion at import time; the same principle
was applied by hand when choosing models for the SRS and Simplification
domains.

---

## Compute & GPU Requirements

| Domain | Recommended GPU | Notes |
|---|---|---|
| SRS correction | 1× A100 80GB or H100 80GB | FLAN-T5-Large/XL run unquantized; Llama-3.1-70B / Qwen2.5-32B / Gemma-3-27B use 4-bit `bitsandbytes` quantization via HuggingFace `transformers` to fit on a single GPU |
| Simplification transfer | 1× H100 80GB | Uses **Ollama** for local model serving (`llama3.1:8b-instruct-q8_0`, `llama3.1:70b`, `qwen2.5:32b`, `gemma3:27b`) alongside API-hosted GPT-4.1; FLAN-T5 runs locally via `transformers` |
| GEC extension | 1× H100 80GB (A100 80GB also supported; avoid A10G — Gemma-27B/Qwen-32B will not fit) | Uses **vLLM** (continuous batching, higher throughput than Ollama), serving **one model at a time** and tearing down before loading the next — see `gec_extension/gec_probe/SETUP.md` |
| Everything else (ingestion, probe generation, scoring, judges) | CPU only | All judge/generator calls in this stage go through hosted APIs (OpenAI, Anthropic, Mistral, Groq, DeepSeek) |

Expected wall-clock for the GEC pipeline end-to-end on H100 80GB: **6–8
hours** (see `gec_extension/gec_probe/SETUP.md` for a stage-by-stage
breakdown). The SRS and Simplification pipelines have similar per-model
runtimes; budget roughly 30–90 minutes per local model for generation, plus
judge/scoring passes that are API-bound rather than GPU-bound.

## Compute & API Budget (rough)

| Provider | Used for | Approx. cost for a full run |
|---|---|---|
| OpenAI | GPT-4.1 corrector (all domains) | ~$5–10 |
| Anthropic | Claude Haiku judge (Simplification + GEC) + claude-sonnet-4-6 shortcut judge (SRS) | ~$2–4 |
| Mistral | Mistral Large judge (Simplification + GEC) | ~$2–4 |
| Groq | Llama-3.1-8B/70B inference | free tier typically sufficient |
| DeepSeek | Probe generator (GEC only) | ~$1 (new accounts get 5M free tokens) |
| HuggingFace | Gated model access (Llama, Gemma) — token only, no cost | — |

**Total: roughly $10–20** for a full reproduction across all three domains.
Dropping GPT-4.1 from the corrector pool cuts this to ~$5.

---

## Reproducibility Notes

- **Frozen benchmarks.** Each domain has a benchmark that is frozen after
  manual audit and should not be regenerated casually:
  - SRS: `srs_probecore/data/SRS-ProbeCore-v5-frozen.json` — **891 base
    items**, **3,068 probe pairs** (1,286 invariance, 891 directional, 891
    shortcut) across PURE (587), Großer (219), PROMISE_exp (85).
  - Simplification: `simplification_transfer/data/simp_probecore_v1.jsonl`.
  - GEC: `gec_extension/data_artifacts/probes/gec_probecore_base.jsonl`.
  If you need to regenerate a benchmark from scratch, the corresponding
  `ingestion/` and `probe_generation/` scripts in each domain folder will
  do it — but note this will not reproduce the exact frozen item IDs used
  in the paper's tables unless you also fix the random seeds documented in
  those scripts.
- **Judges are for probe construction only.** In every domain, LLM judges
  (Claude Haiku, Mistral Large, claude-sonnet-4-6) are used only to filter
  or validate *candidate probes* before they enter the frozen benchmark.
  Final model scores always come from a deterministic, rule-based scorer
  — never from an LLM judge. This separation avoids judge-induced
  circularity between evaluation and scoring.
- **Known methodological limitations** (documented in full in each domain's
  README and, for GEC, in `gec_extension/gec_probe/METHODOLOGY.md`):
  - GEC's directional probe family is dominated by a single sub-operation
    (`remove_error`); GEC has no RQ3-equivalent (gold/expert-alignment)
    validation; the `gector_baseline` corrector is an identity passthrough,
    not an actual GECToR model, and must not be described as one.
  - RQ3 (probe-based reliability vs. expert/gold alignment) is currently
    an **unsupported hypothesis** on the available proxy data
    (Spearman ρ = −0.24, p = 0.57, non-significant) — see
    `cross_domain_analysis/README.md` for the honest framing of this result.
  - The deterministic SRS/GEC scorers depend on modal-verb extraction, cue
    inventories, and content-overlap thresholds; they are auditable and
    reproducible, but can miss valid paraphrases that fall outside their
    rule coverage.
- **`*/legacy/` and `*/results/raw_shards/` folders.** These are kept for
  provenance (earlier script versions, intermediate sharded pipeline runs)
  but are **not** required to reproduce the paper's reported numbers. The
  authoritative final results for each domain live under that domain's
  `results/final/` (or equivalent) folder — see each domain README for
  exactly which file/table maps to which paper table.

---

## Security Note

While reorganizing this codebase, two hardcoded credentials (an OpenAI API
key and a HuggingFace token) were found as fallback default values inside
`srs_probecore/evaluation/run_models.py` and have been **removed** — the
script now reads exclusively from `OPENAI_API_KEY` / `HF_TOKEN` environment
variables, with an empty-string fallback. If those original keys were ever
live and this code was shared with anyone before this cleanup, we'd
recommend rotating them. No other hardcoded secrets were found in a full
scan of the codebase.

---

## Data Sources & Licenses

| Dataset | Domain | Source |
|---|---|---|
| PURE | SRS | [zenodo.org/records/7118517](https://zenodo.org/records/7118517) |
| Großer benchmark | SRS | [zenodo.org/records/10974836](https://zenodo.org/records/10974836) |
| PROMISE_exp | SRS | [PMC8974737](https://pmc.ncbi.nlm.nih.gov/articles/PMC8974737/) |
| TSAR 2025 | Simplification | [huggingface.co/datasets/cardiffnlp/TSAR2025_SharedTask_RCTS_Trial-Data](https://huggingface.co/datasets/cardiffnlp/TSAR2025_SharedTask_RCTS_Trial-Data) |
| OneStopEnglish | Simplification | [huggingface.co/datasets/SetFit/onestop_english](https://huggingface.co/datasets/SetFit/onestop_english) |
| ASSET | Simplification | [huggingface.co/datasets/facebook/asset](https://huggingface.co/datasets/facebook/asset) |
| BEA-2019 W&I+LOCNESS | GEC | [cl.cam.ac.uk/research/nl/bea2019st](https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz) |

Check each dataset's own license before redistributing it. This repo
includes raw copies of the smaller ones (BEA-2019, transfer_data JSONs)
under `*/data*` for reproducibility convenience — remove them if your
target license doesn't permit redistribution.

---

## Citation

If you use this benchmark or code, please cite the CONTRAST paper (see the
submission for full bibliographic details — this repo intentionally
contains code and data only, no paper source).
