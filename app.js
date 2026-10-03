import {
  AutoProcessor,
  Qwen3_5ForConditionalGeneration,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@next";

const MODEL_ID = "onnx-community/Qwen3.5-0.8B-ONNX";
const MAX_TOKENS = 384;

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const inputEl = $("input");
const sendEl = $("send");
const clearEl = $("clear");
const statusEl = $("status");
const loadingEl = $("loading");
const loadingText = $("loadingText");
const runtimeEl = $("runtime");

let processor = null;
let model = null;
let history = [];
let generating = false;

function setLoading(show, text) {
  loadingEl.classList.toggle("hidden", !show);
  loadingText.textContent = text || "Downloading the local model…";
}

function render() {
  messagesEl.innerHTML = "";
  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.innerHTML = "<div><h2>Mini-AI</h2><p>A tiny local assistant. Your conversation stays in this browser.</p></div>";
    messagesEl.appendChild(empty);
    return;
  }

  for (const message of history) {
    const row = document.createElement("div");
    row.className = "msg " + message.role;
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = message.role === "user" ? "You" : "M";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = String(message.content || "");
    row.append(avatar, bubble);
    messagesEl.appendChild(row);
  }

  window.scrollTo({top: document.body.scrollHeight, behavior: "smooth"});
}

function save() {
  localStorage.setItem("mini-ai-history", JSON.stringify(history));
}

function load() {
  try {
    history = JSON.parse(localStorage.getItem("mini-ai-history") || "[]");
    if (!Array.isArray(history)) history = [];
  } catch {
    history = [];
  }
}

function normalizeConversation(items) {
  return items.slice(-10).map((m) => ({
    role: m.role,
    content: [{type: "text", text: m.content}],
  }));
}

async function init() {
  load();
  render();

  const webgpu = "gpu" in navigator;
  runtimeEl.textContent = webgpu ? "WebGPU available" : "WASM fallback";
  statusEl.textContent = "loading model…";
  setLoading(true);

  try {
    processor = await AutoProcessor.from_pretrained(MODEL_ID);

    model = await Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: {
        embed_tokens: "q4",
        vision_encoder: "fp16",
        decoder_model_merged: "q4",
      },
      device: webgpu ? "webgpu" : "wasm",
    });

    statusEl.textContent = webgpu ? "ready · WebGPU" : "ready · WASM";
    setLoading(false);
    inputEl.focus();
  } catch (error) {
    console.error(error);
    statusEl.textContent = "model failed to load";
    setLoading(true, "Could not load the local model. Check the browser console for details.");
    sendEl.disabled = true;
  }
}

async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed || generating || !model) return;

  history.push({role: "user", content: trimmed});
  history.push({role: "assistant", content: ""});
  const assistantIndex = history.length - 1;
  save();
  render();

  generating = true;
  sendEl.disabled = true;
  inputEl.disabled = true;
  statusEl.textContent = "thinking…";

  try {
    const conversation = normalizeConversation(history.slice(0, assistantIndex));
    const prompt = processor.apply_chat_template(conversation, {
      add_generation_prompt: true,
      enable_thinking: false,
    });

    const inputs = await processor(prompt);
    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (token) => {
        history[assistantIndex].content += token;
        render();
      },
    });

    const outputs = await model.generate({
      ...inputs,
      max_new_tokens: MAX_TOKENS,
      do_sample: true,
      temperature: 0.7,
      top_p: 0.9,
      streamer,
    });

    if (!history[assistantIndex].content) {
      const promptTokens = inputs.input_ids.dims.at(-1);
      const generated = outputs.slice(null, [promptTokens, null]);
      history[assistantIndex].content =
        processor.batch_decode(generated, {skip_special_tokens: true})[0] ||
        "I couldn't generate a response.";
    }

    save();
    render();
    statusEl.textContent = runtimeEl.textContent.startsWith("WebGPU")
      ? "ready · WebGPU"
      : "ready · WASM";
  } catch (error) {
    console.error(error);
    history[assistantIndex].content =
      "Generation error: " + (error && error.message ? error.message : "unknown error");
    save();
    render();
    statusEl.textContent = "generation error";
  } finally {
    generating = false;
    sendEl.disabled = false;
    inputEl.disabled = false;
    inputEl.focus();
  }
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = inputEl.value;
  inputEl.value = "";
  inputEl.style.height = "auto";
  sendMessage(text);
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});

clearEl.addEventListener("click", () => {
  if (generating) return;
  history = [];
  localStorage.removeItem("mini-ai-history");
  render();
  inputEl.focus();
});

init();
