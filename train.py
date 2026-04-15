"""
train.py — Fine-tune Qwen2.5-3B-Instruct on IndoIoT dataset using LoRA + 4-bit quantization.

Usage:
    python train.py
"""

import logging
import math
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

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

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = Path("dataset/indoiot_dataset.jsonl")
OUTPUT_DIR = Path("./models/indoiot-qwen2.5-3b-lora")
ADAPTER_DIR = Path("./models/lora-adapter-only")

MAX_SEQ_LENGTH = 512
TRAIN_VAL_SPLIT = 0.9

SYSTEM_PROMPT = (
    "Kamu adalah asisten IoT expert Indonesia yang menjawab pertanyaan teknis "
    "dengan jelas dan akurat."
)

# ---------------------------------------------------------------------------
# bitsandbytes availability check
# ---------------------------------------------------------------------------

try:
    import bitsandbytes  # noqa: F401
    HAS_BNB = True
    logger.info("bitsandbytes available — quantization enabled")
except ImportError:
    HAS_BNB = False
    logger.warning(
        "bitsandbytes not available — model will load WITHOUT quantization. "
        "VRAM usage will be significantly higher."
    )

# ---------------------------------------------------------------------------
# HuggingFace login
# ---------------------------------------------------------------------------


def hf_login() -> None:
    """Load HF_TOKEN from .env and authenticate with HuggingFace Hub."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set in .env — cannot download gated models.")
        sys.exit(1)
    import huggingface_hub
    huggingface_hub.login(token=token, add_to_git_credential=False)
    logger.info("HuggingFace Hub login successful")


# ---------------------------------------------------------------------------
# GPU diagnostics
# ---------------------------------------------------------------------------


def print_gpu_memory() -> None:
    """Print current GPU VRAM usage."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — training will run on CPU (very slow)")
        return
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved  = torch.cuda.memory_reserved(device)  / 1024 ** 3
    total     = props.total_memory                   / 1024 ** 3
    logger.info(
        "GPU: %s | Total VRAM: %.1f GB | Allocated: %.2f GB | Reserved: %.2f GB",
        props.name, total, allocated, reserved,
    )


def print_estimated_time(num_train_samples: int) -> None:
    """Print a rough training time estimate for RTX 4060 Laptop."""
    batch_size       = 2
    grad_accum       = 4
    epochs           = 3
    seconds_per_step = 2.5   # empirical: RTX 4060 Laptop, 4-bit + LoRA
    steps_per_epoch  = math.ceil(num_train_samples / (batch_size * grad_accum))
    total_steps      = steps_per_epoch * epochs
    total_seconds    = total_steps * seconds_per_step
    minutes, secs    = divmod(int(total_seconds), 60)
    logger.info(
        "Estimated training time: ~%d min %d sec  (%d steps x %.1fs/step) — RTX 4060 Laptop estimate",
        minutes, secs, total_steps, seconds_per_step,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def format_sample(sample: dict) -> str:
    """Format one record into the ChatML prompt used for training."""
    return (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT + "\n"
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + sample["instruction"] + "\n"
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + sample["response"] + "\n"
        + "<|im_end|>"
    )


def load_and_split_dataset(path: Path) -> tuple[Dataset, Dataset]:
    """Load JSONL dataset, format samples, and return (train_ds, val_ds)."""
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        sys.exit(1)

    raw = load_dataset("json", data_files=str(path), split="train")
    logger.info("Loaded %d samples from %s", len(raw), path)

    formatted = raw.map(
        lambda example: {"text": format_sample(example)},
        remove_columns=raw.column_names,
    )

    split    = formatted.train_test_split(test_size=1 - TRAIN_VAL_SPLIT, seed=42)
    train_ds: Dataset = split["train"]
    val_ds:   Dataset = split["test"]
    logger.info("Split: %d train / %d val", len(train_ds), len(val_ds))
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_id: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with 4-bit quant; fall back to 8-bit, then fp16 if needed."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Build ordered list of (label, extra_kwargs) attempts
    attempts: list[tuple[str, dict]] = []
    if HAS_BNB:
        attempts.append((
            "4-bit NF4",
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            },
        ))
        attempts.append((
            "8-bit",
            {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)},
        ))
    attempts.append((
        "fp16 (no quantization — high VRAM)",
        {"torch_dtype": torch.float16},
    ))

    base_kwargs: dict = {"device_map": "auto", "trust_remote_code": True}

    for label, extra_kwargs in attempts:
        try:
            logger.info("Loading model with %s ...", label)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, **base_kwargs, **extra_kwargs
            )
            logger.info("Model loaded successfully with %s", label)
            return model, tokenizer
        except Exception as exc:
            logger.warning("Load with %s failed: %s — trying next option", label, exc)

    logger.error("All model loading attempts failed. Exiting.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# LoRA + training config
# ---------------------------------------------------------------------------


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_sft_config(output_dir: Path) -> SFTConfig:
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=False,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_training(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    val_ds: Dataset,
) -> SFTTrainer:
    """Configure and run SFTTrainer, returning the trainer after completion."""
    # Prepare for k-bit training (enables grad checkpointing, casts norms to fp32)
    model = prepare_model_for_kbit_training(model)

    # Wrap with LoRA manually so we can pass autocast_adapter_dtype=False,
    # bypassing the PEFT 0.19 + PyTorch 2.6 float8_e8m0fnu bug in SFTTrainer
    model = get_peft_model(model, build_lora_config(), autocast_adapter_dtype=False)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(OUTPUT_DIR),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # peft_config omitted — model is already wrapped above
        processing_class=tokenizer,
    )
    logger.info("Starting training ...")
    trainer.train()
    return trainer


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_outputs(trainer: SFTTrainer, tokenizer: AutoTokenizer) -> None:
    """Save full fine-tuned model + tokenizer, and LoRA adapter separately."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Saving model + tokenizer to %s ...", OUTPUT_DIR)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    logger.info("Saving LoRA adapter-only to %s ...", ADAPTER_DIR)
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    print("Training complete! Model saved to ./models/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    hf_login()

    logger.info("=== GPU state before model load ===")
    print_gpu_memory()

    train_ds, val_ds = load_and_split_dataset(DATASET_PATH)
    print_estimated_time(len(train_ds))

    model, tokenizer = load_model_and_tokenizer(BASE_MODEL)

    logger.info("=== GPU state after model load ===")
    torch.cuda.empty_cache()
    print_gpu_memory()

    trainer = run_training(model, tokenizer, train_ds, val_ds)
    save_outputs(trainer, tokenizer)


if __name__ == "__main__":
    main()
