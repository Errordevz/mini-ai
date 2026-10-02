"""Datasets and batching for next-token language-model training."""
from pathlib import Path
import random
import torch


def read_corpus(path: str | Path) -> str:
    root = Path(path)
    if root.is_file():
        return root.read_text(encoding="utf-8")
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under {root}")
    return "\n\n".join(p.read_text(encoding="utf-8") for p in files)


def make_blocks(token_ids: list[int], block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) < block_size + 1:
        raise ValueError("Corpus is too small for the selected block_size.")
    data = torch.tensor(token_ids, dtype=torch.long)
    usable = (data.numel() - 1) // block_size * block_size
    x = data[:usable].view(-1, block_size)
    y = data[1:usable + 1].view(-1, block_size)
    return x, y


def split_blocks(x, y, val_fraction=0.05, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.size(0), generator=g)
    n_val = max(1, int(x.size(0) * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def batcher(x, y, batch_size, device, shuffle=True, seed=42):
    rng = random.Random(seed)
    indices = list(range(x.size(0)))
    while True:
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            if len(ids) < batch_size:
                continue
            yield x[ids].to(device), y[ids].to(device)
