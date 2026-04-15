"""
save_checkpoint.py — Copy LoRA adapter files from the latest training checkpoint
to ./models/lora-adapter-only/ without reloading any model.

Usage:
    python save_checkpoint.py
"""

import shutil
import sys
from pathlib import Path

CHECKPOINT_BASE = Path("./models/indoiot-qwen2.5-3b-lora")
ADAPTER_OUT = Path("./models/lora-adapter-only")

# Files that constitute the LoRA adapter (plus tokenizer for inference convenience)
ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",        # fallback name used by older PEFT versions
    "tokenizer.json",
    "tokenizer_config.json",
]


def find_latest_checkpoint(base: Path) -> Path:
    checkpoints = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not checkpoints:
        print(f"ERROR: no checkpoint-* directories found in {base}", file=sys.stderr)
        sys.exit(1)
    latest = checkpoints[-1]
    print(f"Found latest checkpoint: {latest}")
    return latest


def copy_adapter(checkpoint: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for filename in ADAPTER_FILES:
        src = checkpoint / filename
        if src.exists():
            shutil.copy2(src, dest / filename)
            copied.append(filename)

    if not copied:
        print("ERROR: no adapter files found in checkpoint.", file=sys.stderr)
        sys.exit(1)

    print(f"Copied {len(copied)} file(s) to {dest}:")
    for f in copied:
        size_kb = (dest / f).stat().st_size / 1024
        print(f"  {f}  ({size_kb:,.0f} KB)")


def main() -> None:
    if not CHECKPOINT_BASE.exists():
        print(f"ERROR: checkpoint base directory not found: {CHECKPOINT_BASE}", file=sys.stderr)
        sys.exit(1)

    checkpoint = find_latest_checkpoint(CHECKPOINT_BASE)
    copy_adapter(checkpoint, ADAPTER_OUT)
    print("Adapter saved successfully")


if __name__ == "__main__":
    main()
