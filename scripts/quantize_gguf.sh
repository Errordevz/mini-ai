#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-artifacts/mini-ai-merged}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
OUT_DIR="${2:-artifacts/gguf}"
QUANT="${QUANT:-Q4_K_M}"

if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
  echo "Missing llama.cpp at $LLAMA_CPP_DIR" >&2
  echo "Set LLAMA_CPP_DIR or clone/build llama.cpp first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR"   --outfile "$OUT_DIR/mini-ai-f16.gguf"   --outtype f16

"$LLAMA_CPP_DIR/build/bin/llama-quantize"   "$OUT_DIR/mini-ai-f16.gguf"   "$OUT_DIR/mini-ai-${QUANT}.gguf"   "$QUANT"

echo "quantized=$OUT_DIR/mini-ai-${QUANT}.gguf"
