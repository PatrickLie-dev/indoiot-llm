# upload_fix.py
from huggingface_hub import HfApi

api = HfApi()

with open("app.py", "rb") as f:
    app_content = f.read()

_UNUSED = '''# app.py — IndoIoT LLM Gradio Interface
import os
import logging
import torch
import gradio as gr
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import huggingface_hub

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL_ID = "Pat-L/indoiot-qwen2.5-lora"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.float16 if DEVICE == "cuda" else torch.float32

SYSTEM_PROMPT = (
    "Kamu adalah asisten IoT expert Indonesia yang menjawab pertanyaan teknis "
    "dengan jelas dan akurat."
)

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

    logger.info("Loading base model in float32 (CPU)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        token=hf_token,
    )

    logger.info(f"Loading LoRA adapter from {LORA_MODEL_ID}...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL_ID,
        token=hf_token,
    )
    model.eval()
    logger.info("Model ready!")
    return model, tokenizer

def build_prompt(question: str) -> str:
    return (
        "<|im_start|>system\\n"
        + SYSTEM_PROMPT + "\\n"
        + "<|im_end|>\\n"
        + "<|im_start|>user\\n"
        + question + "\\n"
        + "<|im_end|>\\n"
        + "<|im_start|>assistant\\n"
    )

@torch.inference_mode()
def generate_response(question: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    if not question.strip():
        return "Silakan masukkan pertanyaan IoT Anda."

    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")
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

model = None
tokenizer = None

def chat_fn(message: str, history: list) -> str:
    return generate_response(message)

logger.info("Initializing IndoIoT LLM...")
model, tokenizer = load_model()

demo = gr.ChatInterface(
    fn=chat_fn,
    title="🌐 IndoIoT LLM — Asisten IoT Berbahasa Indonesia",
    description=(
        "**Fine-tuned Qwen2.5-3B** dengan QLoRA pada 760 samples IoT Bahasa Indonesia.  \\n"
        "Model: [Pat-L/indoiot-qwen2.5-lora](https://huggingface.co/Pat-L/indoiot-qwen2.5-lora)"
    ),
    examples=[
        "Bagaimana cara menghubungkan ESP32 ke broker MQTT?",
        "Apa perbedaan antara QoS 0, 1, dan 2 dalam protokol MQTT?",
        "Bagaimana cara membaca sensor DHT22 dengan ESP32?",
        "Jelaskan perbedaan antara HTTP dan MQTT untuk aplikasi IoT.",
        "Apa yang harus dilakukan jika ESP32 terus restart karena watchdog timer?",
    ],
)

# FIX: No arguments — HF Spaces handles routing automatically
demo.launch()
'''

api.upload_file(
    path_or_fileobj=app_content,
    path_in_repo="app.py",
    repo_id="Pat-L/indoiot-llm",
    repo_type="space",
)
print("app.py uploaded successfully!")

api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id="Pat-L/indoiot-llm",
    repo_type="space",
)
print("requirements.txt uploaded successfully!")

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id="Pat-L/indoiot-llm",
    repo_type="space",
)
print("README.md uploaded successfully!")