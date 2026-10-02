#!/usr/bin/env python3
from pathlib import Path
import argparse
import torch

from mini_ai.model import MiniAI, MiniConfig
from mini_ai.tokenizer import MiniTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/mini-ai.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--prompt", default="The future of technology")
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = MiniConfig(**ckpt["config"])
    model = MiniAI(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = MiniTokenizer(args.tokenizer)
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
