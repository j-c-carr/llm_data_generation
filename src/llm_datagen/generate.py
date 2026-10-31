"""Batch generation: read prompts from JSONL, write responses to JSONL.

Input lines:  {"id": "...", "prompt": "...", "system": "..."}   (system is optional)
Output lines: the input record plus "response", "finish_reason", "usage", "model".

Results are appended as they complete, and ids already in the output file are
skipped, so a job that gets killed can be resubmitted and will resume.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_datagen.client import LLMClient


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/example_prompts.jsonl"))
    p.add_argument("--output", type=Path, default=Path("outputs/responses.jsonl"))
    p.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"))
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen2.5:0.5b"))
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent requests. Keep at 1-2 on a laptop; use 32+ with vLLM on a GPU.")
    args = p.parse_args()

    client = LLMClient(args.base_url, args.model, temperature=args.temperature,
                       max_tokens=args.max_tokens, seed=args.seed)

    records = read_jsonl(args.input)
    done = {r["id"] for r in read_jsonl(args.output)} if args.output.exists() else set()
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(records)} prompts, {len(done)} already done, {len(todo)} to generate")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(client.generate, r["prompt"], r.get("system")): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            record = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"[{i}/{len(todo)}] {record['id']} failed: {e}")
                continue
            out.write(json.dumps({**record, **result, "model": args.model}) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {record['id']} done")


if __name__ == "__main__":
    main()
