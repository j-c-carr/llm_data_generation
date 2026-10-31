#!/bin/bash
# Start a vLLM server on a GPU node, wait until it is ready, run generation, then shut it down.
#
# Usage (from the repo root):
#   mkdir -p logs && sbatch scripts/generate_cluster.slurm
#   MODEL=Qwen/Qwen2.5-7B-Instruct INPUT=data/my_prompts.jsonl sbatch scripts/generate_cluster.slurm
#
# Partition/account names depend on your cluster; adjust the #SBATCH lines.
# For interactive use (salloc), see "Interactive session" in the README.
#SBATCH --job-name=llm-datagen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

INPUT=${INPUT:-data/example_prompts.jsonl}
OUTPUT=${OUTPUT:-outputs/responses.jsonl}
WORKERS=${WORKERS:-32}

# Starts vLLM (MODEL, PORT, HF_HOME are configured there) and sets VLLM_PID, LLM_BASE_URL, LLM_MODEL.
source scripts/serve_vllm.sh
trap 'kill $VLLM_PID 2>/dev/null' EXIT

uv run python -m llm_datagen.generate \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --workers "$WORKERS"
