#!/usr/bin/env python3
"""Response-level knowledge distillation: Qwen3.5-2B teacher -> JSONL dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TEACHER = "techwithsergiu/Qwen3.5-text-2B"


def read_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Each JSONL row needs a non-empty string 'prompt'.")
        prompts.append(prompt.strip())
    return prompts


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Answer accurately and helpfully. Be concise unless the task requires detail. "
                "Do not reveal hidden reasoning or private chain-of-thought."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="data/distill_prompts.jsonl")
    parser.add_argument("--output", default="data/distilled.jsonl")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    prompts = read_prompts(Path(args.prompts))[:args.limit]
    if not prompts:
        raise ValueError("No prompts found.")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        load_kwargs.update({"device_map": "auto", "torch_dtype": dtype})

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **load_kwargs)
    model.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, 1):
            response = generate_response(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
            if not response:
                print(f"skip={i}: empty teacher response")
                continue

            f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"generated={i}/{len(prompts)}")

    print(f"saved={out}")


if __name__ == "__main__":
    main()
