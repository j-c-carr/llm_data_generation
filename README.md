# llm_data_generation

Synthetic data generation with a locally hosted LLM.

The code talks to an **OpenAI-compatible HTTP server** running on the same machine:

| Where   | Server | Base URL                    | Example model               |
|---------|--------|-----------------------------|-----------------------------|
| Laptop  | Ollama | `http://localhost:11434/v1` | `qwen2.5:0.5b`              |
| Cluster | vLLM   | `http://localhost:8000/v1`  | `HuggingFaceTB/SmolLM2-135M-Instruct` (default, for testing) |

The Python code is the same on both machines. Only `--base-url` and `--model` change.

## Flow

```
data/*.jsonl  ──►  generate.py  ──HTTP──►  Ollama / vLLM  ──►  outputs/*.jsonl
{"id","prompt"}     (LLMClient)                              {"id","prompt","response",...}
```

- `src/llm_datagen/client.py`: `LLMClient.generate(prompt, system=None)` sends one prompt and returns the response.
- `src/llm_datagen/generate.py`: batch runner that reads JSONL, sends concurrent requests, and appends results as they finish. If you rerun it, it skips ids that are already in the output file.
- `scripts/generate_cluster.slurm`: starts vLLM on a GPU node, waits until it is ready, then runs `generate.py`.

## Setup

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Laptop (Ollama)

```bash
brew install ollama          # or download the app from ollama.com
ollama serve                 # in a separate terminal (not needed if the app is running)
ollama pull qwen2.5:0.5b     # small model, ~400MB

uv run python -m llm_datagen.client      # single-prompt smoke test
uv run python -m llm_datagen.generate    # data/example_prompts.jsonl -> outputs/responses.jsonl
```

## Cluster (vLLM + SLURM)

### One-time setup

Install uv in your home directory (no admin rights needed; check `module avail uv` first in case the cluster provides it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # or: pip install --user uv
```

This installs `uv` and `uvx` into `~/.local/bin`. Then add the following to `~/.bashrc` and run `source ~/.bashrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"                 # if `which uvx` finds nothing
export UV_CACHE_DIR=/path/to/scratch/$USER/uv_cache  # vLLM + torch are several GB
export HF_HOME=/path/to/scratch/$USER/hf_cache       # model weights are larger still
export UV_LINK_MODE=copy                             # silences warnings when cache and repo are on different filesystems
```

Home directories usually have small quotas, so keep both caches on scratch. Use the scratch path from your cluster's documentation; on FASRC it is usually under `/n/netscratch/<your_lab>/`.

Then, from the repo root on the login node:

```bash
uv sync --locked                          # install the exact versions from uv.lock
uvx --from vllm vllm --version            # pre-download vLLM into the uv cache
```

If compute nodes have no internet access, also download the model on the login node, e.g. `uvx --from huggingface_hub hf download HuggingFaceTB/SmolLM2-135M-Instruct`.

Notes:
- Installing and downloading on the login node is fine, but run vLLM and generation only inside `salloc`/`sbatch`. Login nodes are shared.
- **Without uv** (e.g. if your cluster recommends conda/mamba): install the project with `pip install -e .` and vLLM with `pip install vllm`, ideally in separate environments since vLLM pins its own torch. Then replace `uvx --from vllm vllm serve` with `vllm serve` in `scripts/serve_vllm.sh`, and `uv run python` with `python` in `scripts/generate_cluster.slurm`.

### Batch job (sbatch)

```bash
mkdir -p logs
sbatch scripts/generate_cluster.slurm
# override defaults with environment variables:
MODEL=Qwen/Qwen2.5-7B-Instruct INPUT=data/my_prompts.jsonl OUTPUT=outputs/run1.jsonl \
    sbatch scripts/generate_cluster.slurm
```

Before the first run, edit the `#SBATCH` partition and resources for your cluster. Gated models such as Llama also need `HF_TOKEN`.

The first run is slow while vLLM loads the model. Its progress goes to `logs/vllm-<jobid>.log` rather than the terminal, so you can follow it with `tail -f logs/vllm-*.log`.

### Interactive session (salloc)

Useful for iterating on prompts: load the model once, then run generation as often as you like.

```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00
# (on clusters where salloc leaves you on the login node, follow with: srun --pty bash)

cd /path/to/llm_data_generation
source scripts/serve_vllm.sh                 # or: MODEL=... source scripts/serve_vllm.sh
uv run python -m llm_datagen.client          # quick single-prompt check
uv run python -m llm_datagen.generate --input data/my_prompts.jsonl --output outputs/try1.jsonl --workers 32

kill $VLLM_PID                               # stop the server when done
```

`serve_vllm.sh` exports `LLM_BASE_URL` and `LLM_MODEL`, so the Python commands pick up the server automatically.

## Using the client directly

```python
from llm_datagen.client import LLMClient

client = LLMClient(base_url="http://localhost:11434/v1", model="qwen2.5:0.5b")
out = client.generate("Write a product review.", system="Be concise.")
print(out["response"])
```
