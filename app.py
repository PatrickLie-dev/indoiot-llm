# app.py — IndoIoT LLM Gradio Interface
# Loads LoRA adapter from HuggingFace Hub (CPU-compatible)

import os
import logging
import torch

# gradio_client bug: _json_schema_to_python_type doesn't handle boolean JSON schemas
# (e.g. additionalProperties: true), causing TypeError on API info generation.
import gradio_client.utils as _gcu
_orig_j2p = _gcu._json_schema_to_python_type
def _patched_j2p(schema, defs):
    if isinstance(schema, bool):
        return "any"
    return _orig_j2p(schema, defs)
_gcu._json_schema_to_python_type = _patched_j2p

import gradio as gr
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import huggingface_hub

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL_ID  = "Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL_ID  = "Pat-L/indoiot-qwen2.5-lora"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16 if DEVICE == "cuda" else torch.float32

SYSTEM_PROMPT = (
    "Kamu adalah asisten IoT expert Indonesia yang menjawab pertanyaan teknis "
    "dengan jelas dan akurat."
)

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        logger.info("HuggingFace login successful")

    logger.info(f"Device: {DEVICE} | Dtype: {DTYPE}")
    logger.info(f"Loading tokenizer from {LORA_MODEL_ID}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_MODEL_ID,
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Pakai 4-bit quantization kalau GPU available (sama seperti inference.py kamu)
    if DEVICE == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading base model with 4-bit quantization (GPU)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
    else:
        logger.info("Loading base model in float16 (CPU)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,     # hemat RAM saat loading
            trust_remote_code=True,
            token=hf_token,
        )

    logger.info(f"Loading LoRA adapter from {LORA_MODEL_ID}...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL_ID,
        token=hf_token,
    )
    
    if DEVICE == "cpu":
        model = model.to("cpu")
    
    model.eval()
    logger.info("Model ready!")
    return model, tokenizer

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> str:
    return (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT + "\n"
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + question + "\n"
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )

@torch.inference_mode()
def generate_response(
    question: str,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    if not question.strip():
        return "Silakan masukkan pertanyaan IoT Anda."

    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Global variables
# ---------------------------------------------------------------------------
model = None
tokenizer = None

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    "Bagaimana cara menghubungkan ESP32 ke broker MQTT?",
    "Apa perbedaan antara QoS 0, 1, dan 2 dalam protokol MQTT?",
    "Bagaimana cara membaca sensor DHT22 dengan ESP32?",
    "Jelaskan perbedaan antara HTTP dan MQTT untuk aplikasi IoT.",
    "Apa yang harus dilakukan jika ESP32 terus restart karena watchdog timer?",
]

def respond(message: str, chat_history: list):
    if not message.strip():
        return "", chat_history
    response = generate_response(message)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    return "", chat_history

# Initialize model SEBELUM launch Gradio
logger.info("Initializing IndoIoT LLM...")
model, tokenizer = load_model()

with gr.Blocks(title="IndoIoT LLM") as demo:
    gr.Markdown(
        "# 🌐 IndoIoT LLM — Asisten IoT Berbahasa Indonesia\n"
        "**Fine-tuned Qwen2.5-3B** dengan QLoRA pada 760 samples IoT Bahasa Indonesia.  \n"
        "Model: [Pat-L/indoiot-qwen2.5-lora](https://huggingface.co/Pat-L/indoiot-qwen2.5-lora)"
    )
    chatbot = gr.Chatbot(height=400, type="messages")
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ketik pertanyaan IoT Anda di sini...",
            label="",
            show_label=False,
            scale=4,
        )
        submit_btn = gr.Button("Kirim", scale=1, variant="primary")
    clear_btn = gr.Button("Hapus Riwayat")
    gr.Examples(examples=EXAMPLES, inputs=msg)

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)