#!/usr/bin/env python3
"""Train the 8k BPE tokenizer used by Mini-AI."""
from pathlib import Path
import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--output", default="artifacts/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=8192)
    args = p.parse_args()

    corpus = Path(args.corpus)
    files = sorted(corpus.rglob("*.txt")) if corpus.is_dir() else [corpus]
    if not files:
        raise FileNotFoundError(f"No .txt files found under {corpus}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL,
        show_progress=True,
    )
    tokenizer.train([str(f) for f in files], trainer)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    print(f"saved {out} with vocab_size={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
