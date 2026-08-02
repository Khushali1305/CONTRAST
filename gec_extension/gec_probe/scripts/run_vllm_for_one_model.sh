#!/usr/bin/env bash
# scripts/run_vllm_for_one_model.sh
# ------------------------------------------------------------------
# Helper for Option B (manual stage-by-stage execution).
# Starts vLLM for one local model, runs the inference stage with the
# active-model filter set, then tears down vLLM.
#
# Usage:
#   bash scripts/run_vllm_for_one_model.sh <code_id> <hf_repo> "<vllm_flags>"
#
# Example:
#   bash scripts/run_vllm_for_one_model.sh qwen_2_5_32b \
#        Qwen/Qwen2.5-32B-Instruct-AWQ \
#        "--quantization awq --dtype half --max-model-len 4096 --gpu-memory-utilization 0.92"
# ------------------------------------------------------------------
set -euo pipefail

CODE_ID="${1:?missing code_id}"
HF_REPO="${2:?missing hf_repo}"
VLLM_FLAGS="${3:-}"

: "${VLLM_PORT:=8000}"
: "${GEC_PROBE_ROOT:?GEC_PROBE_ROOT must be exported}"
: "${VLLM_LOG:=${GEC_PROBE_ROOT}/vllm_serve.log}"

export LOCAL_INFERENCE_URL="http://localhost:${VLLM_PORT}/v1"

echo "==> Starting vLLM for ${HF_REPO} (served as ${CODE_ID})"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3

nohup python -m vllm.entrypoints.openai.api_server \
    --model "${HF_REPO}" \
    --served-model-name "${CODE_ID}" \
    --port "${VLLM_PORT}" \
    --host 0.0.0.0 \
    ${VLLM_FLAGS} \
    > "${VLLM_LOG}" 2>&1 &

# Wait for /v1/models to become healthy (max 10 min).
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "    vLLM healthy"
    break
  fi
  sleep 5
  if [ "$i" -eq 120 ]; then
    echo "ERROR: vLLM never became ready. Tail of log:"
    tail -50 "${VLLM_LOG}"
    exit 1
  fi
done

echo "==> Running inference stage with active model = ${CODE_ID}"
GEC_ACTIVE_LOCAL_MODEL="${CODE_ID}" \
  python -m gec_probe.run_pipeline infer

echo "==> Tearing down vLLM"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 5
echo "==> Done with ${CODE_ID}"
