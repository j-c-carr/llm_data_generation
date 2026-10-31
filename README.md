# llm_data_generation

Synthetic data generation with a locally hosted LLM.

The code talks to an **OpenAI-compatible HTTP server** running on the same machine:

| Where   | Server | Base URL                    | Example model               |
|---------|--------|-----------------------------|-----------------------------|
| Laptop  | Ollama | `http://localhost:11434/v1` | `qwen2.5:0.5b`              |
| Cluster | vLLM   | `http://localhost:8000/v1`  | `Qwen/Qwen2.5-7B-Instruct`  |

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

```bash
mkdir -p logs
sbatch scripts/generate_cluster.slurm
# override defaults with environment variables:
MODEL=Qwen/Qwen2.5-7B-Instruct INPUT=data/my_prompts.jsonl OUTPUT=outputs/run1.jsonl \
    sbatch scripts/generate_cluster.slurm
```

Before the first run, edit the `#SBATCH` partition and resources for your cluster, and set `HF_HOME` (in `scripts/serve_vllm.sh`) to a scratch directory. Gated models such as Llama also need `HF_TOKEN`.

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
