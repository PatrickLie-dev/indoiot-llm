"""
inference.py — Side-by-side comparison of base model vs. fine-tuned LoRA model.

Usage:
    python inference.py                  # base model vs fine-tuned
    python inference.py --finetuned-only # skip base model, faster
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR   = Path("./models/lora-adapter-only")
RESULTS_DIR   = Path("./results")
OUTPUT_FILE   = RESULTS_DIR / "comparison.txt"

SYSTEM_PROMPT = (
    "Kamu adalah asisten IoT expert Indonesia yang menjawab pertanyaan teknis "
    "dengan jelas dan akurat."
)

TEST_QUESTIONS = [
    "Bagaimana cara menghubungkan ESP32 ke broker MQTT menggunakan library PubSubClient?",
    "Apa perbedaan antara QoS 0, 1, dan 2 dalam protokol MQTT?",
    "Bagaimana cara membaca sensor DHT22 dengan ESP32 dan mengirim datanya ke server?",
    "Apa yang harus dilakukan jika ESP32 terus-menerus restart karena watchdog timer?",
    "Jelaskan perbedaan antara HTTP dan MQTT untuk aplikasi IoT.",
]

GEN_CONFIG = dict(
    max_new_tokens=300,
    temperature=0.7,
    do_sample=True,
    repetition_penalty=1.1,
)

DIVIDER  = "=" * 60
SEPARATOR = "-" * 60

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def hf_login() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set in .env")
        sys.exit(1)
    import huggingface_hub
    huggingface_hub.login(token=token, add_to_git_credential=False)
    logger.info("HuggingFace Hub login successful")


def print_gpu_memory(label: str = "") -> None:
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved  = torch.cuda.memory_reserved(device)  / 1024 ** 3
    total     = props.total_memory                   / 1024 ** 3
    logger.info(
        "GPU %s| %s | Total: %.1f GB | Allocated: %.2f GB | Reserved: %.2f GB",
        f"[{label}] " if label else "",
        props.name, total, allocated, reserved,
    )


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_tokenizer() -> AutoTokenizer:
    """Load tokenizer from the saved adapter directory."""
    logger.info("Loading tokenizer from %s", ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for generation
    return tokenizer


def load_base_model() -> AutoModelForCausalLM:
    logger.info("Loading base model '%s' in 4-bit NF4 ...", BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=build_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Base model loaded")
    return model


def load_finetuned_model(base: AutoModelForCausalLM) -> PeftModel:
    """Attach the LoRA adapter to an already-loaded base model."""
    if not ADAPTER_DIR.exists():
        logger.error("Adapter directory not found: %s", ADAPTER_DIR)
        sys.exit(1)
    logger.info("Loading LoRA adapter from %s ...", ADAPTER_DIR)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), autocast_adapter_dtype=False)
    model.eval()
    logger.info("LoRA adapter loaded")
    return model

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def build_prompt(question: str) -> str:
    """Wrap a question in the ChatML format used during training."""
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
def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        **GEN_CONFIG,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_block(question: str, base_answer: str | None, ft_answer: str) -> str:
    lines = [
        DIVIDER,
        f"Q: {question}",
        SEPARATOR,
    ]
    if base_answer is not None:
        lines += ["[BASE MODEL]", base_answer, SEPARATOR]
    lines += ["[FINE-TUNED]", ft_answer, DIVIDER, ""]
    return "\n".join(lines)


def run_comparison(finetuned_only: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()

    print_gpu_memory("before load")
    base_model = load_base_model()
    print_gpu_memory("after base load")

    ft_model: PeftModel | AutoModelForCausalLM
    if finetuned_only:
        ft_model = load_finetuned_model(base_model)
    else:
        # We'll run base first, then attach adapter to the same base model
        ft_model = load_finetuned_model(base_model)

    print_gpu_memory("after adapter load")

    all_output: list[str] = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        logger.info("Question %d/%d ...", i, len(TEST_QUESTIONS))

        base_answer: str | None = None
        if not finetuned_only:
            # Disable adapter to get base model response
            ft_model.disable_adapter_layers()
            base_answer = generate(ft_model, tokenizer, question)
            ft_model.enable_adapter_layers()

        ft_answer = generate(ft_model, tokenizer, question)

        block = format_block(question, base_answer, ft_answer)
        print(block)
        all_output.append(block)

    # Save to file
    output_text = "\n".join(all_output)
    OUTPUT_FILE.write_text(output_text, encoding="utf-8")
    logger.info("Results saved to %s", OUTPUT_FILE)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="IndoIoT LLM inference comparison")
    parser.add_argument(
        "--finetuned-only",
        action="store_true",
        help="Skip base model, only run fine-tuned model (faster)",
    )
    args = parser.parse_args()

    hf_login()
    run_comparison(finetuned_only=args.finetuned_only)


if __name__ == "__main__":
    main()
