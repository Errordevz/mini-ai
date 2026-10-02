#!/usr/bin/env python3
"""Merge a Mini-AI LoRA adapter into the text-only bf16 base model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE = "techwithsergiu/Qwen3.5-text-0.8B"
DEFAULT_ADAPTER = "artifacts/mini-ai-lora"
DEFAULT_OUTPUT = "artifacts/mini-ai-merged"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs.update({"torch_dtype": dtype, "device_map": "auto"})

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, **kwargs)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    print(f"merged_model={output}")


if __name__ == "__main__":
    main()
