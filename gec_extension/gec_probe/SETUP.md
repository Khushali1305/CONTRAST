# GEC CONTRAST — Setup and Run

This is the exact command sequence to run the pipeline end-to-end on a
Lightning AI Studio with one GPU (H100 80GB or A100 80GB recommended).

For methodology and design rationale, see `METHODOLOGY.md`.
For architecture and module-level details, see `README.md`.

---

## Step 0 — One-time account setup (5 min)

Create accounts and get API keys for these five providers:

| Provider | What you need | URL | Cost for this pipeline |
|---|---|---|---|
| OpenAI | API key | https://platform.openai.com/api-keys | ~$5–10 for GPT-4.1 corrector |
| Anthropic | API key | https://console.anthropic.com/settings/keys | ~$2–4 for Haiku judge |
| Mistral | API key | https://console.mistral.ai/api-keys/ | ~$2–4 for Large judge |
| Groq | API key | https://console.groq.com/keys | Free tier covers Llama correctors |
| DeepSeek | API key | https://platform.deepseek.com/api_keys | ~$1 for probe generator (5M free tokens for new accounts) |
| HuggingFace | Token with `read` access | https://huggingface.co/settings/tokens | Free — required for gated models (Llama, Gemma) |

**Total expected API spend: ~$10–20.** Most of that is GPT-4.1. If you
drop GPT-4.1 from the corrector pool, the spend drops to ~$5.

---

## Step 1 — Launch a Lightning AI Studio

1. Go to https://lightning.ai/studios
2. Create a new Studio (or open an existing one).
3. Switch the compute to **1× H100 80GB** (preferred) or **1× A100 80GB**.
   Avoid A10G — Gemma-27B and Qwen-32B will not fit.
4. Open a terminal in the Studio.

---

## Step 2 — Upload the gec_probe code

You can either upload from a local zip or clone from your git repo.

**Option A: upload zip**
1. In Lightning Studio, drag-and-drop the `gec_probe` folder (or its zip)
   into `/teamspace/studios/this_studio/`.
2. Unzip if needed: `unzip gec_probe.zip -d /teamspace/studios/this_studio/`

**Option B: clone from git**
```bash
cd /teamspace/studios/this_studio
git clone <YOUR_GEC_PROBE_REPO_URL> gec_probe_run
mv gec_probe_run/gec_probe gec_probe   # if nested
```

After this step you should have:
```
/teamspace/studios/this_studio/gec_probe/
  ├── config/
  ├── data/
  ├── probes/
  ├── ...
  └── scripts/run_lightning.sh
```

---

## Step 3 — Export all API keys

Paste this into the terminal, **replacing each `<...>` with your actual key**:

```bash
export OPENAI_API_KEY="<your-openai-key>"
export ANTHROPIC_API_KEY="<your-anthropic-key>"
export MISTRAL_API_KEY="<your-mistral-key>"
export GROQ_API_KEY="<your-groq-key>"
export DEEPSEEK_API_KEY="<your-deepseek-key>"
export HF_TOKEN="<your-hf-token>"
```

Optional but recommended — persist them for future sessions:
```bash
cat >> ~/.bashrc <<'EOF'
export OPENAI_API_KEY="<your-openai-key>"
export ANTHROPIC_API_KEY="<your-anthropic-key>"
export MISTRAL_API_KEY="<your-mistral-key>"
export GROQ_API_KEY="<your-groq-key>"
export DEEPSEEK_API_KEY="<your-deepseek-key>"
export HF_TOKEN="<your-hf-token>"
EOF
source ~/.bashrc
```

Verify they're all set:
```bash
for k in OPENAI ANTHROPIC MISTRAL GROQ DEEPSEEK HF; do
  varname="${k}_API_KEY"
  [ "${k}" = "HF" ] && varname="HF_TOKEN"
  if [ -z "${!varname:-}" ]; then echo "MISSING: $varname"; else echo "OK: $varname"; fi
done
```

All six should print `OK`.

---

## Step 4 — Make the working directory environment

```bash
# This is where the pipeline reads/writes everything.
export GEC_PROBE_ROOT="/teamspace/studios/this_studio/gec_probe_run"
mkdir -p "${GEC_PROBE_ROOT}"

# Point Python at the package.
export PYTHONPATH="/teamspace/studios/this_studio:${PYTHONPATH:-}"
```

Verify the package imports cleanly:
```bash
cd /teamspace/studios/this_studio
python -c "from gec_probe.config.settings import GENERATOR_FAMILY; print('Generator family:', GENERATOR_FAMILY)"
```
Expected output: `Generator family: deepseek`

---

## Step 5 — Run the pipeline

Two equivalent ways. **Use Option A** unless you want manual stage control.

### Option A — One-shot launch script (recommended)

```bash
cd /teamspace/studios/this_studio/gec_probe
chmod +x scripts/run_lightning.sh
bash scripts/run_lightning.sh 2>&1 | tee "${GEC_PROBE_ROOT}/full_run.log"
```

This runs all seven stages in order:
1. ingest
2. stratify
3. probes
4. judge
5. infer (sequentially for each local model under vLLM)
6. score
7. analyze

Expected total wall-clock on H100 80GB: **6–8 hours**.

### Option B — Stage-by-stage manual control

If you want to inspect outputs between stages:

```bash
cd /teamspace/studios/this_studio

# Install dependencies (one-time)
pip install --upgrade pip
pip install "vllm>=0.6.3" "errant>=3.0.0" "openai>=1.40.0" \
            "anthropic>=0.40.0" "mistralai>=1.0.0" "groq>=0.11.0" requests
python -m spacy download en_core_web_sm

# Download BEA-2019 (one-time)
mkdir -p "${GEC_PROBE_ROOT}/data_artifacts/raw/bea2019"
cd "${GEC_PROBE_ROOT}/data_artifacts/raw/bea2019"
curl -L -O https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
tar xzf wi+locness_v2.1.bea19.tar.gz

# Stage 1: ingest M2 -> normalized JSONL (~2 min)
cd /teamspace/studios/this_studio
python -m gec_probe.run_pipeline ingest

# Stage 2: stratified sampling -> ProbeCore base set (~30 sec)
python -m gec_probe.run_pipeline stratify

# Stage 3: probe generation (DeepSeek API; ~30-45 min for ~7k LLM calls)
python -m gec_probe.run_pipeline probes

# Stage 4: two-judge validation (Anthropic + Mistral APIs; ~30-60 min)
python -m gec_probe.run_pipeline judge

# Stage 5: corrector inference — one local model at a time via vLLM.
# Run this for EACH local model you want to evaluate:
bash gec_probe/scripts/run_vllm_for_one_model.sh flan_t5_large google/flan-t5-large \
     "--dtype bfloat16 --max-model-len 512"
bash gec_probe/scripts/run_vllm_for_one_model.sh flan_t5_xl google/flan-t5-xl \
     "--dtype bfloat16 --max-model-len 512"
bash gec_probe/scripts/run_vllm_for_one_model.sh qwen_2_5_32b Qwen/Qwen2.5-32B-Instruct-AWQ \
     "--quantization awq --dtype half --max-model-len 4096 --gpu-memory-utilization 0.92"
bash gec_probe/scripts/run_vllm_for_one_model.sh gemma_3_27b google/gemma-3-27b-it \
     "--dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.92"
bash gec_probe/scripts/run_vllm_for_one_model.sh llama_3_1_70b_local hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
     "--quantization awq --dtype half --max-model-len 4096 --gpu-memory-utilization 0.94"

# Stage 6: scoring (~5 min)
python -m gec_probe.run_pipeline score

# Stage 7: cross-domain Spearman triangle
# (point this at your existing SRS and Simp reliability tables)
python -m gec_probe.analysis.cross_domain_rho
```

---

## Step 6 — Inspect outputs

All outputs land under `${GEC_PROBE_ROOT}/data_artifacts/`:

```bash
# Probe survival rates
cat "${GEC_PROBE_ROOT}/data_artifacts/judge/gec_judge_decisions.jsonl" | \
  python -c "
import json, sys, collections
c = collections.Counter()
for ln in sys.stdin: c[json.loads(ln)['final_verdict']] += 1
print(dict(c))
"

# Final reliability table
cat "${GEC_PROBE_ROOT}/data_artifacts/scores/rq1_reliability_gec.jsonl" | \
  python -m json.tool

# Cross-domain triangle (after analyze stage)
cat "${GEC_PROBE_ROOT}/data_artifacts/scores/cross_domain_rho_triangle.csv"
```

---

## Step 7 — Resuming after a crash

The pipeline is fully idempotent. If the run crashes or you stop it:

```bash
# Just re-run the launch script.
bash scripts/run_lightning.sh 2>&1 | tee -a "${GEC_PROBE_ROOT}/full_run.log"
```

The orchestrator skips any stage whose primary output file already exists.
The inference checkpoint layer (`inference/checkpoint.py`) verifies the
*actual existence of each output file on disk* — not just a flag — before
declaring a (model, probe_id) pair complete. Crashed runs that left no
output file are re-run automatically.

To force re-running a single stage:
```bash
python -m gec_probe.run_pipeline probes --force probes
```

---

## Troubleshooting

### vLLM never reports healthy

```bash
tail -200 "${GEC_PROBE_ROOT}/vllm_serve.log"
```

Most common causes:
- **HF_TOKEN missing or wrong** for a gated model (Llama, Gemma) → 401 in the log
- **Out of GPU memory** → reduce `--gpu-memory-utilization` to 0.88 or
  reduce `--max-model-len`
- **AWQ quantization not supported by your vLLM version** → `pip install -U vllm`

### "RuntimeError: Generator family overlaps corrector pool"

You've edited `CORRECTOR_MODELS` or `JUDGE_MODELS` in a way that violates
disjointness. Check `GENERATOR_FAMILY` in `config/settings.py` is
`"deepseek"` and that no corrector or judge has `family: "deepseek"`.

### DeepSeek API rate limit

Switch to off-peak hours (16:30–00:30 UTC, which is 10 PM–6 AM IST):
it's both 50–75% cheaper and faster.

### "ModuleNotFoundError: No module named 'gec_probe'"

`PYTHONPATH` doesn't include the parent of `gec_probe/`. Re-run:
```bash
export PYTHONPATH="/teamspace/studios/this_studio:${PYTHONPATH:-}"
```
