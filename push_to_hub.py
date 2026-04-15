"""
push_to_hub.py — Upload LoRA adapter to HuggingFace Hub with auto-generated model card.

Usage:
    python push_to_hub.py
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder

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
# Config
# ---------------------------------------------------------------------------

ADAPTER_DIR = Path("./models/lora-adapter-only")
REPO_NAME   = "indoiot-qwen2.5-lora"   # will become {username}/{REPO_NAME}

MODEL_CARD = """\
---
language:
  - id
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
  - lora
  - peft
  - iot
  - indonesian
  - qwen2.5
  - fine-tuned
pipeline_tag: text-generation
---

# IndoIoT — Qwen2.5-3B-Instruct LoRA (Indonesian IoT Q&A)

LoRA adapter fine-tuned on top of [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
for answering technical IoT questions in **Bahasa Indonesia**.

## Model Details

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Adapter type | LoRA (PEFT) |
| Language | Indonesian (`id`) |
| Domain | IoT — ESP32, MQTT, sensors, protocols, troubleshooting |
| Task | Causal LM / Instruction following |

## Training Data

- **760 synthetic Q&A pairs** generated with Groq (Llama-3.1-8B-Instant)
- Categories: ESP32/Arduino, MQTT, Sensors & Actuators, IoT Protocols, Troubleshooting
- All questions and answers in Bahasa Indonesia

## LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| Alpha (`lora_alpha`) | 32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Dropout | 0.05 |
| Bias | none |
| Trainable params | ~7.4M / 3.09B total (0.24%) |

## Training Details

| Setting | Value |
|---|---|
| Epochs | 3 |
| Batch size | 2 (effective: 8 with grad accum × 4) |
| Learning rate | 2e-4 (cosine schedule) |
| Precision | BF16 |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Hardware | RTX 4060 Laptop 8 GB VRAM |
| Training time | ~28 minutes |
| Final train loss | 0.5833 |
| Final eval loss | 0.5297 |
| Token accuracy | ~85.7% |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_id = "{username}/indoiot-qwen2.5-lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_id, autocast_adapter_dtype=False)
model.eval()

prompt = (
    "<|im_start|>system\\n"
    "Kamu adalah asisten IoT expert Indonesia yang menjawab pertanyaan teknis dengan jelas dan akurat.\\n"
    "<|im_end|>\\n"
    "<|im_start|>user\\n"
    "Bagaimana cara menghubungkan ESP32 ke broker MQTT?\\n"
    "<|im_end|>\\n"
    "<|im_start|>assistant\\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
input_len = inputs["input_ids"].shape[1]
print(tokenizer.decode(output[0][input_len:], skip_special_tokens=True))
```

## Prompt Format

This model uses the **ChatML** format:

```
<|im_start|>system
Kamu adalah asisten IoT expert Indonesia yang menjawab pertanyaan teknis dengan jelas dan akurat.
<|im_end|>
<|im_start|>user
{your question here}
<|im_end|>
<|im_start|>assistant
```
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set in .env")
        sys.exit(1)

    if not ADAPTER_DIR.exists():
        logger.error("Adapter directory not found: %s", ADAPTER_DIR)
        sys.exit(1)

    api = HfApi(token=token)

    # Resolve username
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{REPO_NAME}"
    logger.info("Uploading to: %s", repo_id)

    # Create repo (no-op if it already exists)
    create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    logger.info("Repo ready: https://huggingface.co/%s", repo_id)

    # Write model card into adapter dir temporarily
    readme_path = ADAPTER_DIR / "README.md"
    card_content = MODEL_CARD.replace("{username}", username)
    readme_path.write_text(card_content, encoding="utf-8")
    logger.info("Model card written to %s", readme_path)

    # Upload all files in adapter dir
    logger.info("Uploading files from %s ...", ADAPTER_DIR)
    upload_folder(
        folder_path=str(ADAPTER_DIR),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="Add IndoIoT LoRA adapter + model card",
    )

    # Clean up the temp README (it now lives on the Hub)
    readme_path.unlink()

    model_url = f"https://huggingface.co/{repo_id}"
    print(f"\nModel uploaded successfully!")
    print(f"URL: {model_url}")


if __name__ == "__main__":
    main()
