#!/usr/bin/env python3
"""QLoRA SFT for the compact text-only Qwen3.5 student."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_STUDENT = "techwithsergiu/Qwen3.5-text-0.8B-bnb-4bit"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]


def read_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("prompt") and row.get("response"):
                rows.append(row)
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    return rows


def tokenize_row(tokenizer, row: dict[str, str], max_length: int) -> Example:
    messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["response"]},
    ]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    prefix = tokenizer(
        prefix_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    labels = list(full["input_ids"])
    prefix_len = min(len(prefix["input_ids"]), len(labels))
    labels[:prefix_len] = [-100] * prefix_len
    return Example(full["input_ids"], labels)


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, examples: list[Example]) -> dict[str, torch.Tensor]:
        max_len = max(len(x.input_ids) for x in examples)
        input_ids, labels, mask = [], [], []

        for ex in examples:
            pad = max_len - len(ex.input_ids)
            input_ids.append(ex.input_ids + [self.pad_id] * pad)
            labels.append(ex.labels + [-100] * pad)
            mask.append([1] * len(ex.input_ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def split_rows(rows: list[dict[str, str]], val_fraction: float, seed: int):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    return rows[n_val:], rows[:n_val]


@torch.no_grad()
def evaluate(model, loader, device: str, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            enabled=torch.cuda.is_available(),
        ):
            out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/distilled.jsonl")
    parser.add_argument("--student-model", default=DEFAULT_STUDENT)
    parser.add_argument("--output", default="artifacts/mini-ai-lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training needs a CUDA GPU. Use a Hugging Face GPU Job or a CUDA machine.")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "quantization_config": quant,
    }
    model = AutoModelForCausalLM.from_pretrained(args.student_model, **load_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    rows = read_rows(Path(args.data))
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)

    train_examples = [tokenize_row(tokenizer, r, args.max_length) for r in train_rows]
    val_examples = [tokenize_row(tokenizer, r, args.max_length) for r in val_rows]

    from torch.utils.data import DataLoader

    collator = Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val = float("inf")
    model.train()

    for epoch in range(math.ceil(args.epochs)):
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            warmup_scale = min(1.0, (global_step + 1) / max(args.warmup_steps, 1))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup_scale

            with torch.autocast(
                device_type="cuda",
                dtype=compute_dtype,
                enabled=True,
            ):
                loss = model(**batch).loss / args.grad_accum

            loss.backward()

            if (batch_index + 1) % args.grad_accum == 0 or batch_index == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 25 == 0:
                    val_loss = evaluate(model, val_loader, model.device)
                    print(
                        f"step={global_step}/{total_steps} "
                        f"train_loss={loss.item() * args.grad_accum:.4f} "
                        f"val_loss={val_loss:.4f}"
                    )
                    if val_loss < best_val:
                        best_val = val_loss
                        model.save_pretrained(out)
                        tokenizer.save_pretrained(out)

        if global_step >= total_steps:
            break

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"saved_adapter={out}")
    print(f"best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
