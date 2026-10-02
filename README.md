# Mini-AI 🧠

Mini-AI started as a deliberately small ~10M-parameter Transformer built from scratch.

It now has a second, much more capable path: **Qwen3.5 distillation + QLoRA fine-tuning + compact GGUF export**. The original 10M model is kept as the educational baseline; the Qwen path is the recommended assistant build.

## The tiny-but-capable plan

The production recipe is:

**Qwen3.5 2B teacher → response distillation → Qwen3.5 0.8B student → QLoRA → merge → Q4_K_M GGUF**

Qwen3.5 currently has an official 0.8B and 2B family member, rather than an official 1B release. The official models are Apache-2.0 licensed.

For text-only training, this repo uses the community-maintained text-only Qwen3.5 derivatives from Hugging Face:

- teacher: `techwithsergiu/Qwen3.5-text-2B`
- student: `techwithsergiu/Qwen3.5-text-0.8B-bnb-4bit`

The text-only variants remove the vision tower before training/inference, avoiding multimodal storage and runtime overhead for a text-only assistant.

The final GGUF should **not** be committed to Git. Keep the repo code/config small and publish the model artifact on a model hub instead.

## Why 0.8B instead of 2B for the final model?

A 2B model is much heavier on disk. The official Qwen3.5-2B repository is listed at roughly 4.57 GB in its released safetensors form, while a community Q4_K_M quantized text-only 0.8B build is about **528 MB**. That is the sweet spot for Mini-AI: enough capacity to be useful, but still small enough to distribute and run locally.

The teacher can be large because it is used only during the offline data-generation stage. The deployed model is the 0.8B student.

## Response distillation

Mini-AI uses **response-level distillation** rather than shipping a large teacher or storing teacher logits.

1. Give the teacher a set of high-value prompts.
2. Save its clean responses as JSONL.
3. Fine-tune the 0.8B student on those teacher-generated examples.
4. Merge the LoRA adapter into the text-only bf16 student.
5. Convert the merged model to GGUF.
6. Quantize to Q4_K_M for the final compact artifact.

This is intentionally simple and reproducible. It transfers useful teacher behavior without requiring the deployed model to contain the teacher.

## Quick start: build the dataset

Install the optional training dependencies:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-training.txt

Generate distilled answers with the 2B teacher:

    python -m scripts.distill_responses       --prompts data/distill_prompts.jsonl       --output data/distilled.jsonl       --teacher-model techwithsergiu/Qwen3.5-text-2B       --limit 256

For a serious run, replace/extend `data/distill_prompts.jsonl` with thousands of diverse, licensed prompts.

## QLoRA fine-tuning

The student is loaded in 4-bit NF4 and adapted with LoRA.

    python -m scripts.train_qlora       --data data/distilled.jsonl       --student-model techwithsergiu/Qwen3.5-text-0.8B-bnb-4bit       --output artifacts/mini-ai-lora       --epochs 2       --batch-size 2       --grad-accum 8       --max-length 1024

This step needs a CUDA GPU. A Hugging Face GPU Job is a good place to run it.

The repository intentionally keeps the base model and generated weights outside Git.

## Merge the adapter

After training:

    python -m scripts.merge_lora \
      --base techwithsergiu/Qwen3.5-text-0.8B \
      --adapter artifacts/mini-ai-lora \
      --output artifacts/mini-ai-merged

## Quantize to a tiny GGUF

Install/build llama.cpp separately, then:

    LLAMA_CPP_DIR=./llama.cpp \
    bash scripts/quantize_gguf.sh \
      artifacts/mini-ai-merged \
      artifacts/gguf

Default quantization:

    Q4_K_M

A Q4_K_M build of this model family is around the ~500 MB range instead of multi-gigabyte bf16/fp32 storage.

## Tiny runtime

Install the optional runtime dependencies:

    pip install -r requirements-qwen.txt

The runtime downloads the model on demand instead of bundling weights into this repository:

    python -m mini_ai.qwen_runtime "Explain what an MCP server is"

Or select your own Hub-hosted fine-tuned model:

    MINI_AI_MODEL=YOUR_HF_MODEL_ID \
    python -m mini_ai.qwen_runtime "Write a Python function that reverses a linked list"

## Existing 10M educational model

The original Mini-AI remains in:

    mini_ai/model.py
    mini_ai/tokenizer.py
    mini_ai/data.py
    scripts/train.py
    scripts/train_tokenizer.py
    scripts/sample.py
    scripts/count_params.py

Its default configuration is still:

| Setting | Value |
|---|---:|
| Vocabulary size | 8,192 |
| Context length | 256 |
| Transformer layers | 10 |
| Attention heads | 8 |
| Hidden size | 256 |
| MLP size | 1,024 |
| Parameters | 10,060,800 |

That path is useful for learning how a Transformer works from scratch. It is not expected to match the capabilities of the Qwen3.5 student.

## Repository layout

    mini-ai/
    ├── mini_ai/
    │   ├── model.py
    │   ├── tokenizer.py
    │   ├── data.py
    │   └── qwen_runtime.py
    ├── scripts/
    │   ├── train_tokenizer.py
    │   ├── train.py
    │   ├── sample.py
    │   ├── count_params.py
    │   ├── distill_responses.py
    │   ├── train_qlora.py
    │   ├── merge_lora.py
    │   └── quantize_gguf.sh
    ├── data/
    │   ├── demo_corpus.txt
    │   └── distill_prompts.jsonl
    ├── requirements.txt
    ├── requirements-training.txt
    ├── pyproject.toml
    └── README.md

## Storage rules

Do not commit:

- base model weights
- generated teacher responses at production scale
- merged bf16 checkpoints
- GGUF releases

Keep those in Hugging Face or another model-artifact store. Git should contain the code, configs, small example data, and reproducible commands.

## License

MIT for this repository.

The Qwen-based models remain subject to their upstream model licenses and terms; check the exact Hub model card before redistribution.
