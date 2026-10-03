# Mini-AI web chat

The repository root contains a Vercel-ready static chat page.

The current browser build uses the public ONNX Qwen3.5 0.8B model through Transformers.js/WebGPU as the bootstrap runtime. This avoids API keys and keeps inference in the browser.

When the distilled Mini-AI student is trained and exported, replace MODEL_ID in app.js with an ONNX-compatible export of the trained model.

The model weights are not committed to this repository. The browser downloads and caches them from Hugging Face.
