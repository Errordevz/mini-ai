#!/usr/bin/env python3
from mini_ai.model import MiniAI, MiniConfig


model = MiniAI(MiniConfig(vocab_size=8192))
count = model.param_count
print(f"trainable_parameters={count:,}")
print(f"millions={count/1_000_000:.3f}M")
assert 9_900_000 <= count <= 10_200_000, count
