#!/usr/bin/env bash
# scripts/run_lightning.sh
# ------------------------------------------------------------------
# End-to-end runner for the GEC CONTRAST pipeline on Lightning AI.
# Designed for 1× H100 80GB or 1× A100 80GB.
# Switches local-model serving from Ollama -> vLLM for ~5–20× throughput.
# ------------------------------------------------------------------
set -euo pipefail

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
# Lightning AI persistent storage path; survives studio restarts.
: "${WORKDIR:=/teamspace/studios/this_studio/gec_probe_run}"
: "${VLLM_PORT:=8000}"
: "${VLLM_LOG:=${WORKDIR}/vllm_serve.log}"

# Required API keys (must be exported before running)
REQUIRED_KEYS=(OPENAI_API_KEY ANTHROPIC_API_KEY MISTRAL_API_KEY GROQ_API_KEY DEEPSEEK_API_KEY HF_TOKEN)

# Local models we will serve with vLLM, in order.
# (model_id_for_our_code, hf_repo, vllm_flags)
LOCAL_MODELS=(
  "flan_t5_large|google/flan-t5-large|--dtype bfloat16 --max-model-len 512"
  "flan_t5_xl|google/flan-t5-xl|--dtype bfloat16 --max-model-len 512"
  "qwen_2_5_32b|Qwen/Qwen2.5-32B-Instruct-AWQ|--quantization awq --dtype half --max-model-len 4096 --gpu-memory-utilization 0.92"
  "gemma_3_27b|google/gemma-3-27b-it|--dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.92"
  "llama_3_1_70b_local|hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4|--quantization awq --dtype half --max-model-len 4096 --gpu-memory-utilization 0.94"
)
# NOTE: llama_3_1_70b is normally hit via Groq API for speed. Listed above
# only if you want to run it locally too; comment out otherwise.

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------
echo "==> Checking required env vars..."
for k in "${REQUIRED_KEYS[@]}"; do
  if [ -z "${!k:-}" ]; then
    echo "ERROR: missing env var $k"
    exit 1
  fi
done

echo "==> Checking GPU..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no nvidia-smi found; need a GPU studio"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

export GEC_PROBE_ROOT="${WORKDIR}"
export LOCAL_INFERENCE_URL="http://localhost:${VLLM_PORT}/v1"

# ------------------------------------------------------------------
# 1. Repo + Python deps
# ------------------------------------------------------------------
if [ ! -d gec_probe ]; then
  echo "==> Copying gec_probe package to ${WORKDIR}/gec_probe ..."
  # If you uploaded the package zip to the studio, unzip it here.
  # Otherwise, clone from your git remote:
  if [ -n "${GEC_PROBE_GIT_URL:-}" ]; then
    git clone "${GEC_PROBE_GIT_URL}" gec_probe
  else
    echo "ERROR: no gec_probe/ in ${WORKDIR} and no GEC_PROBE_GIT_URL set"
    exit 1
  fi
fi

echo "==> Installing Python deps..."
pip install --quiet --upgrade pip
pip install --quiet \
    "vllm>=0.6.3" \
    "errant>=3.0.0" \
    "openai>=1.40.0" \
    "anthropic>=0.40.0" \
    "mistralai>=1.0.0" \
    "groq>=0.11.0" \
    requests
python -m spacy download en_core_web_sm

# ------------------------------------------------------------------
# 2. BEA-2019 download (idempotent)
# ------------------------------------------------------------------
RAW_DIR="${WORKDIR}/data_artifacts/raw/bea2019"
mkdir -p "${RAW_DIR}"
if [ ! -d "${RAW_DIR}/wi+locness" ]; then
  echo "==> Downloading BEA-2019 W&I+LOCNESS..."
  cd "${RAW_DIR}"
  curl -L -O https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
  tar xzf wi+locness_v2.1.bea19.tar.gz
  cd "${WORKDIR}"
else
  echo "==> BEA-2019 already downloaded"
fi

# ------------------------------------------------------------------
# 3. CPU/API-only stages (no vLLM yet)
# ------------------------------------------------------------------
echo "==> Stage: ingest"
python -m gec_probe.run_pipeline ingest

echo "==> Stage: stratify"
python -m gec_probe.run_pipeline stratify

echo "==> Stage: probes (uses OpenAI API for paraphrase + error injection)"
python -m gec_probe.run_pipeline probes

echo "==> Stage: judge (uses Anthropic + Mistral APIs)"
python -m gec_probe.run_pipeline judge

# ------------------------------------------------------------------
# 4. vLLM serving + inference, one model at a time
# ------------------------------------------------------------------
start_vllm() {
  local code_id="$1"
  local repo="$2"
  shift 2
  local flags="$*"
  echo "==> Starting vLLM for ${repo} (served as ${code_id}) ..."
  # Kill any previous server.
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 3
  # Serve. We use --served-model-name so the API model field matches the
  # code-side identifier regardless of HF repo (e.g., AWQ variant).
  nohup python -m vllm.entrypoints.openai.api_server \
      --model "${repo}" \
      --served-model-name "${code_id}" \
      --port "${VLLM_PORT}" \
      --host 0.0.0.0 \
      ${flags} \
      > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  # Wait for the /v1/models endpoint to become healthy.
  for i in $(seq 1 120); do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null; then
      echo "    vLLM healthy (PID=${VLLM_PID})"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: vLLM never became ready. Tail of log:"
  tail -50 "${VLLM_LOG}"
  return 1
}

stop_vllm() {
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 5
}

run_single_local_model() {
  local code_id="$1"
  local repo="$2"
  local flags="$3"
  start_vllm "${code_id}" "${repo}" ${flags}

  # Point our code to ONLY run this one local model on this pass.
  # API-hosted models (GPT-4.1, Groq, GECToR-CPU) are NOT restricted by this
  # env var and are run on the first pass; checkpointing skips them after.
  GEC_ACTIVE_LOCAL_MODEL="${code_id}" \
    python -m gec_probe.run_pipeline infer

  stop_vllm
}

# API-hosted models (GPT-4.1, Groq Llama, GECToR-CPU) run anytime — we
# fold them into the first vLLM pass.
for entry in "${LOCAL_MODELS[@]}"; do
  IFS='|' read -r code_id repo flags <<< "${entry}"
  echo ""
  echo "###################################################################"
  echo "# Local model: ${code_id} (${repo})"
  echo "###################################################################"
  run_single_local_model "${code_id}" "${repo}" "${flags}"
done

# ------------------------------------------------------------------
# 5. Scoring + analysis
# ------------------------------------------------------------------
echo ""
echo "==> Stage: score"
python -m gec_probe.run_pipeline score

echo ""
echo "==> Pipeline finished."
echo "    Reliability:   ${WORKDIR}/data_artifacts/scores/rq1_reliability_gec.jsonl"
echo "    Conformance:   ${WORKDIR}/data_artifacts/scores/rq3_conformance_gec.jsonl"
echo ""
echo "Run cross_domain_rho.py manually with your SRS + Simp tables:"
echo "  python -m gec_probe.analysis.cross_domain_rho"
