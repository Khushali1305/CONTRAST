# GEC CONTRAST Pipeline

Third evaluation domain for the CONTRAST framework: a probe-based reliability
evaluation of LLM correctors for grammatical error correction (GEC) on
BEA-2019 W&I+LOCNESS, mirroring the SRS and simplification pipelines.

## Read in this order

1. **`METHODOLOGY.md`** — design rationale, probe-family adaptations,
   family-disjointness principle, what changed from the SRS pipeline and
   why. Read this first if you want to understand the science.
2. **`SETUP.md`** — exact command sequence to run end-to-end on Lightning AI,
   from account creation through inspecting outputs. Read this second if you
   want to run the pipeline.
3. **This file** — module-level overview and the file tree.

## Architecture

```
gec_probe/
├── METHODOLOGY.md                # design rationale (read first)
├── SETUP.md                      # exact run commands (read second)
├── README.md                     # this file
├── run_pipeline.py               # stage orchestrator with skip-if-exists
├── config/
│   └── settings.py               # paths, models, env-var keys,
│                                 #   family-disjointness assertions
├── data/
│   ├── ingest_bea2019.py         # BEA-2019 M2 -> normalized JSONL
│   └── stratified_sample.py      # ProbeCore base set (FROZEN OUTPUT)
├── probes/
│   ├── errant_utils.py           # ERRANT wrappers + span helpers
│   ├── invariance.py             # paraphrase preserving error
│   ├── directional.py            # ADD / REMOVE / SWAP errors
│   └── overcorrection.py         # replaces SRS "shortcut" family
├── rule_filters/
│   └── filters.py                # deterministic per-family checks
├── judges/
│   └── two_judge_validator.py    # Claude Haiku + Mistral Large
├── inference/
│   ├── llm_client.py             # provider router (incl. native DeepSeek)
│   ├── checkpoint.py             # verifies actual file existence
│   └── run_correctors.py         # runs all 8 correctors with checkpointing
├── scoring/
│   ├── probe_scoring.py          # reliability per family + bootstrap CIs
│   └── conformance.py            # ERRANT F0.5 + GLEU+
├── analysis/
│   └── cross_domain_rho.py       # SRS-Simp-GEC Spearman triangle
└── scripts/
    ├── run_lightning.sh          # one-shot end-to-end runner
    └── run_vllm_for_one_model.sh # serve one model, infer, tear down
```

## Three roles, three disjoint model families

| Role | Model | Family |
|---|---|---|
| Probe generator | DeepSeek V4 Flash (native API) | `deepseek` |
| Judge 1 | Claude Haiku 4.5 | `anthropic` |
| Judge 2 | Mistral Large 3 | `mistral` |
| Correctors (8) | FLAN-T5-Large/XL, Llama-3.1-8B/70B, Qwen-32B, Gemma-27B, GPT-4.1, GECToR | `flan_t5, llama, qwen, gemma, openai, rule` |

The disjointness is enforced at import time by
`assert_generator_disjoint_from_judges_and_correctors()`. Edit the config
to violate it and the judge module refuses to load.

## Stage flow

`ingest → stratify → probes → judge → infer → score → analyze`

Each stage writes JSONL; downstream stages read those JSONLs. Re-running
the launch script after a crash skips completed stages and resumes
inference at the exact (model, probe_id) pair that didn't finish.

## Total cost and time

- **API spend:** ~$10–20 (dominated by GPT-4.1 corrector calls; ~$1 for
  DeepSeek probe generation, often $0 with the 5M-token new-account grant)
- **Compute:** 1× H100 80GB or A100 80GB, ~6–8 hours end-to-end
- **Storage:** ~2 GB for all intermediate and final outputs

## Quick start

```bash
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... MISTRAL_API_KEY=... \
       GROQ_API_KEY=... DEEPSEEK_API_KEY=... HF_TOKEN=...
bash gec_probe/scripts/run_lightning.sh
```

For the full setup walkthrough see **`SETUP.md`**.
