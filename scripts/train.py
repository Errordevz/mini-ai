#!/usr/bin/env python3
"""Train Mini-AI from scratch."""
from pathlib import Path
import argparse
import math
import random

import torch

from mini_ai.data import read_corpus, make_blocks, split_blocks, batcher
from mini_ai.model import MiniAI, MiniConfig
from mini_ai.tokenizer import MiniTokenizer


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, x, y, batch_size, device):
    model.eval()
    total = 0.0
    count = 0
    for i in range(0, x.size(0), batch_size):
        xb = x[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        _, loss = model(xb, yb)
        total += float(loss) * xb.size(0)
        count += xb.size(0)
    model.train()
    return total / max(count, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--out", default="checkpoints/mini-ai.pt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tok = MiniTokenizer(args.tokenizer)
    ids = tok.encode(read_corpus(args.corpus))
    x, y = make_blocks(ids, args.block_size)
    train_x, train_y, val_x, val_y = split_blocks(
        x, y, val_fraction=args.val_fraction, seed=args.seed
    )

    config = MiniConfig(
        vocab_size=8192,
        block_size=args.block_size,
    )
    model = MiniAI(config).to(device)
    print(f"trainable_parameters={model.param_count:,} ({model.param_count/1e6:.3f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    warmup = max(20, args.steps // 20)
    train_iter = batcher(
        train_x, train_y, args.batch_size, device, shuffle=True, seed=args.seed
    )

    best_val = math.inf
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        lr_scale = min(1.0, step / warmup)
        if step > warmup:
            progress = (step - warmup) / max(1, args.steps - warmup)
            lr_scale *= 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale

        xb, yb = next(train_iter)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_x, val_y, args.batch_size, device)
            print(
                f"step={step:5d} train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config.__dict__,
                        "tokenizer_vocab_size": tok.vocab_size,
                        "step": step,
                        "val_loss": val_loss,
                    },
                    out,
                )

    print(f"best checkpoint: {out} (val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
