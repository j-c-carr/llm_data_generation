"""Minimal client for any OpenAI-compatible server.

Locally this is Ollama (http://localhost:11434/v1); on the cluster it is vLLM
(http://localhost:8000/v1). Only the base URL and model name differ.
"""

from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",  # local servers ignore it, but the SDK requires one
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: int | None = None,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def generate(self, prompt: str, system: str | None = None) -> dict:
        """Send one prompt and return the response text plus metadata."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        choice = resp.choices[0]
        return {
            "response": choice.message.content,
            "finish_reason": choice.finish_reason,  # "length" means it hit max_tokens
            "usage": resp.usage.model_dump() if resp.usage else None,
        }


if __name__ == "__main__":
    # Smoke test: python -m llm_datagen.client
    import os

    client = LLMClient(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        model=os.environ.get("LLM_MODEL", "qwen2.5:0.5b"),
    )
    print(client.generate("Write one sentence about the ocean.")["response"])
