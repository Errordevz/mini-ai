# Mini-AI 🧠

A deliberately small ~10 million parameter decoder-only language model built from scratch in PyTorch.

The goal is not to compete with large models. The goal is to have a complete, understandable training stack that you can actually experiment with:

- model architecture
- BPE tokenizer training
- text dataset loading and chunking
- next-token pretraining
- validation loss
- checkpointing
- text generation
- parameter-count verification
- GitHub Actions smoke checks

## Architecture

With the default configuration (vocab_size=8192) the model has 10,060,800 trainable parameters (~10.06M).

| Setting | Value |
|---|---:|
| Vocabulary size | 8,192 |
| Context length | 256 |
| Transformer layers | 10 |
| Attention heads | 8 |
| Hidden size | 256 |
| MLP size | 1,024 |
| Attention | causal SDPA |
| Output head | tied to token embeddings |
| Parameters | 10,060,800 |

The attention implementation uses PyTorch's scaled dot-product attention primitive, which can dispatch to optimized kernels when the installed backend supports them.

## Quick start

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    python -m scripts.count_params

    mkdir -p data/corpus
    cp data/demo_corpus.txt data/corpus/demo.txt

    python -m scripts.train_tokenizer --corpus data/corpus
    python -m scripts.train --corpus data/corpus --tokenizer artifacts/tokenizer.json --steps 200
    python -m scripts.sample --checkpoint checkpoints/mini-ai.pt --tokenizer artifacts/tokenizer.json --prompt "Mini-AI"

The demo corpus is intentionally tiny. It is useful for proving the pipeline works, not for producing a broadly capable model.

## Training on a real corpus

Put UTF-8 .txt files under data/corpus/ and run:

    python -m scripts.train_tokenizer --corpus data/corpus
    python -m scripts.train \
      --corpus data/corpus \
      --tokenizer artifacts/tokenizer.json \
      --steps 20000 \
      --batch-size 32 \
      --block-size 256

For serious training, use a substantially larger corpus, deduplicate it, keep validation data separate, and record dataset provenance/licensing.

## What this is

This is a base language model, not an instruction-tuned assistant. It learns next-token prediction from text. A later stage could add instruction tuning, tool use, retrieval, or an agent loop on top.

## Repository layout

    mini-ai/
    ├── mini_ai/
    │   ├── __init__.py
    │   ├── model.py
    │   ├── tokenizer.py
    │   └── data.py
    ├── scripts/
    │   ├── train_tokenizer.py
    │   ├── train.py
    │   ├── sample.py
    │   └── count_params.py
    ├── data/
    │   └── demo_corpus.txt
    ├── .github/workflows/ci.yml
    ├── requirements.txt
    ├── pyproject.toml
    └── README.md

## Design notes

- The model uses pre-norm Transformer blocks.
- The feed-forward expansion is 4x the hidden size.
- Token and output embeddings are weight-tied.
- Training uses AdamW, gradient clipping, a short warmup, and cosine learning-rate decay.
- Checkpoints contain model weights, config, training step, and validation loss.
- No pretrained model weights are required.

## License

MIT.
