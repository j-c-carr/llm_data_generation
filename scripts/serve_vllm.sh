#!/bin/bash
# Start a vLLM server in the background and wait until it is ready.
# Meant to be *sourced* so that $VLLM_PID stays available in your shell:
#
#   source scripts/serve_vllm.sh                      # default model
#   MODEL=Qwen/Qwen2.5-7B-Instruct source scripts/serve_vllm.sh
#   ...
#   kill $VLLM_PID                                     # when done
#
# No `set -e` here: in a sourced script it would close your interactive shell on any error.

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PORT=${PORT:-8000}
VLLM_LOG=${VLLM_LOG:-logs/vllm-${SLURM_JOB_ID:-local}.log}

# Model weights are large; keep the Hugging Face cache on scratch rather than $HOME.
# export HF_HOME=/path/to/scratch/hf_cache

mkdir -p "$(dirname "$VLLM_LOG")"

# vLLM runs in its own isolated environment (it pins its own torch/CUDA).
# Pin a version for reproducibility, e.g. --from "vllm==X.Y.Z".
uvx --from vllm vllm serve "$MODEL" --port "$PORT" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM to load $MODEL (log: $VLLM_LOG)..."
until curl -sf "http://localhost:${PORT}/health" > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited early; see $VLLM_LOG"
        return 1 2>/dev/null || exit 1
    fi
    sleep 10
done
echo "vLLM is ready at http://localhost:${PORT}/v1 (PID $VLLM_PID)"

export LLM_BASE_URL="http://localhost:${PORT}/v1"
export LLM_MODEL="$MODEL"
