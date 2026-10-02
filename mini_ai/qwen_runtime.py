"""Tiny local runtime for the distilled Mini-AI model."""
from __future__ import annotations

import os
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = os.getenv(
    "MINI_AI_MODEL",
    "techwithsergiu/Qwen3.5-text-0.8B-bnb-4bit",
)


class MiniQwen:
    """Load a compact Qwen3.5 text model from Hugging Face on demand."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        max_memory: dict[int, str] | None = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        kwargs = {"trust_remote_code": True}
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        if max_memory:
            kwargs["max_memory"] = max_memory

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    @torch.inference_mode()
    def chat(
        self,
        messages: Iterable[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        messages = list(messages)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        do_sample = temperature > 0
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update({"temperature": temperature, "top_p": top_p})

        outputs = self.model.generate(**inputs, **kwargs)
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="*", help="User message")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip() or "Hello! What can you do?"
    model = MiniQwen(args.model)
    print(model.chat([{"role": "user", "content": prompt}], max_new_tokens=args.tokens))


if __name__ == "__main__":
    main()
