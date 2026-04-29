import os
import time

import gradio as gr
from config import (
    FLASH_ATTN,
    HEADER_INFO,
    KV_CACHE_TYPE,
    MAX_TOKENS,
    MIN_P,
    N_CTX,
    PRESENCE_PENALTY,
    REPEAT_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    model_zoo,
    system_prompt,
)
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ──────────────────────────── Constants ───────────────────────────────

_KV_TYPE: dict[str, int] = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
    "q8_0": 8,
}

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_METRICS_SEP = "\n"

N_CPU = os.cpu_count() or 4
N_PHYS = max(1, N_CPU // 2)

_DEFAULT_MODEL = next(iter(model_zoo))
_loaded: dict[str, Llama] = {}


# ──────────────────────────── Think stripping ─────────────────────────


class ThinkStripper:
    """Streaming filter that removes <think>…</think> blocks."""

    def __init__(self) -> None:
        self.in_think = False
        self.buf = ""

    def feed(self, text: str) -> str:
        self.buf += text
        out: list[str] = []

        while self.buf:
            if self.in_think:
                end = self.buf.find(_THINK_CLOSE)
                if end == -1:
                    self.buf = ""
                    break
                self.buf = self.buf[end + len(_THINK_CLOSE) :]
                self.in_think = False
                continue

            start = self.buf.find(_THINK_OPEN)
            end = self.buf.find(_THINK_CLOSE)

            if start == -1 and end == -1:
                out.append(self.buf)
                self.buf = ""
            elif start == -1:
                out.append(self.buf[:end])
                self.buf = self.buf[end + len(_THINK_CLOSE) :]
            else:
                out.append(self.buf[:start])
                self.buf = self.buf[start + len(_THINK_OPEN) :]
                self.in_think = True

        return "".join(out)


# ──────────────────────────── Model loading ───────────────────────────


def _load_model(name: str) -> Llama:
    cfg = model_zoo[name]
    path = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["model_file"])

    base = {
        "model_path": path,
        "n_ctx": N_CTX,
        "n_batch": 1024,
        "n_ubatch": 1024,
        "n_threads": N_PHYS,
        "n_threads_batch": N_CPU,
        "flash_attn": bool(FLASH_ATTN),
        "use_mmap": True,
        "use_mlock": False,
        "verbose": False,
    }

    kv = _KV_TYPE.get(KV_CACHE_TYPE)
    try:
        model = Llama(**base, type_k=kv, type_v=kv)
        print(f"KV cache type: {KV_CACHE_TYPE}")
    except ValueError:
        print(f"KV cache '{KV_CACHE_TYPE}' unsupported on this backend, using default.")
        model = Llama(**base)
    return model


print(f"Loading {_DEFAULT_MODEL} …")
_loaded[_DEFAULT_MODEL] = _load_model(_DEFAULT_MODEL)
think_stripper = ThinkStripper()
print("Model ready.")


# ──────────────────────────── History helpers ─────────────────────────


def _to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _strip_think(text: str) -> str:
    return think_stripper.feed(text)


def _strip_metrics(text: str) -> str:
    """Drop the trailing metrics line we appended to assistant messages."""
    return text.split(_METRICS_SEP)[0] if _METRICS_SEP in text else text


def _display_content(turn: dict) -> str:
    """User-visible content (without metrics line) of a history turn."""
    return _strip_metrics(_to_str(turn.get("content", "")))


def _pick_feed_content(disp_turn: dict, raw_turn: dict | None) -> str:
    """
    Choose the content to feed back into the model for a given turn.

    Prefer the raw version (which keeps <think>…</think>) so the KV-cache
    prefix can be reused; if the user clearly edited the message via
    `editable=True`, fall back to the displayed version instead.
    """
    disp = _display_content(disp_turn)

    if not (
        isinstance(raw_turn, dict) and raw_turn.get("role") == disp_turn.get("role")
    ):
        return disp

    raw = _to_str(raw_turn.get("content", ""))

    if disp_turn.get("role") == "assistant":
        # Displayed ≈ _strip_think(raw); if they match, message wasn't edited.
        if _strip_think(raw).strip() == disp.strip():
            return raw
        return disp

    # User / system messages: raw and displayed should be identical.
    return raw if raw.strip() == disp.strip() else disp


# ──────────────────────────── Inference ───────────────────────────────


def respond(
    message: str, history: list[dict], model_name: str, raw_history: list[dict]
):
    # Lazy-load the requested model.
    if model_name not in _loaded:
        print(f"Switching to {model_name} …")
        _loaded[model_name] = _load_model(model_name)
        print(f"{model_name} ready.")
    llm = _loaded[model_name]

    if not isinstance(history, list):
        history = []
    if not isinstance(raw_history, list):
        raw_history = []

    # Build messages from raw history (so the KV prefix can be reused).
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    aligned_raw: list[dict] = []
    for i, turn in enumerate(history):
        if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
            continue
        raw_turn = raw_history[i] if i < len(raw_history) else None
        feed = _pick_feed_content(turn, raw_turn)
        messages.append({"role": turn["role"], "content": feed})
        aligned_raw.append({"role": turn["role"], "content": feed})
    messages.append({"role": "user", "content": message})

    # Stream generation.
    t_start = time.perf_counter()
    n_gen = 0
    raw = ""  # full text incl. <think>
    prev_visible = ""

    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repeat_penalty=REPEAT_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        min_p=MIN_P,
        stream=True,
    ):
        delta = chunk["choices"][0]["delta"].get("content") or ""
        if not delta:
            continue

        raw += delta
        n_gen += 1
        visible = _strip_think(raw)
        if visible != prev_visible:
            # raw_history stays unchanged during streaming.
            yield visible, raw_history
            prev_visible = visible

    total_time = time.perf_counter() - t_start
    overall_tps = n_gen / total_time if total_time > 0 else 0.0
    metrics_line = f"✏️ {n_gen}t | ⏱️ {total_time:.1f}s | 🚀 {overall_tps:.1f}t/s"

    # Rebuild raw_history to match what Gradio will store after this turn.
    new_raw_history = [
        *aligned_raw,
        {"role": "user", "content": message},
        {"role": "assistant", "content": raw},
    ]

    response = _strip_think(raw)
    yield f"{response}{_METRICS_SEP}`{metrics_line}`", new_raw_history


# ──────────────────────────── UI ──────────────────────────────────────

with open("./style.css") as f:
    CSS = f.read()

with gr.Blocks(title="EdgeRazor Playground") as demo:
    gr.Image(
        value="https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/Logo-full.png",
        show_label=False,
        container=False,
        interactive=False,
        elem_classes=["logo-wrap"],
    )
    gr.Markdown(HEADER_INFO, elem_classes=["header-md"])

    current_model = gr.State(_DEFAULT_MODEL)
    raw_history_state = gr.State([])  # raw history with <think> blocks

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=list(model_zoo.keys()),
            value=_DEFAULT_MODEL,
            label="Model",
            interactive=True,
            elem_id="model-selector",
        )

    chat_iface = gr.ChatInterface(
        fn=respond,
        additional_inputs=[current_model, raw_history_state],
        additional_outputs=[raw_history_state],
        additional_inputs_accordion=gr.Accordion(label="", open=False, visible=False),
        editable=True,
        chatbot=gr.Chatbot(label="", height=480),
    )

    def _on_model_change(new_model, cur_model, history):
        # Switching model invalidates raw history; reset chat alongside it.
        # Re-selecting the same model keeps the conversation intact.
        if new_model == cur_model:
            safe_history = history if isinstance(history, list) else []
            return (
                cur_model,
                gr.update(value=cur_model),
                safe_history,
                safe_history,
                [],
            )
        return (
            new_model,
            gr.update(value=new_model),
            [],
            [],
            [],
        )

    model_dd.change(
        fn=_on_model_change,
        inputs=[model_dd, current_model, chat_iface.chatbot_state],
        outputs=[
            current_model,
            model_dd,
            chat_iface.chatbot,
            chat_iface.chatbot_state,
            raw_history_state,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        css=CSS,
        server_name="0.0.0.0",
        server_port=8888,
        ssr_mode=False,
    )
